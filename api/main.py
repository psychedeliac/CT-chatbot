"""
api/main.py — public HTTP API behind the website chat widget.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000

Retrieval is built during lifespan startup, before uvicorn accepts connections,
so the first real request never pays the ~40s BM25 + cross-encoder cost.
"""
import logging
import json
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from contextlib import asynccontextmanager

from api.chat import CapacityError, ChatService, TurnTimeout, is_valid_session_id
from config import load_api_config
from warmup import build_shared_config

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("api")

api_config = load_api_config()

# Rate-limit key. Behind a proxy the socket address is the proxy's, so the real
# client IP has to come from a header -- but only one the deployment is known to
# set and overwrite, or a caller can spoof the header and get a fresh quota per
# request. Hence opt-in via CLIENT_IP_HEADER rather than trusting XFF by default.
def _client_key(request: Request) -> str:
    header = api_config.client_ip_header
    if header:
        forwarded = request.headers.get(header, "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=_client_key,
    storage_uri=api_config.rate_limit_storage_uri,
    headers_enabled=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming retrieval before accepting traffic...")
    agent_config = build_shared_config()
    app.state.chat = ChatService(agent_config, api_config)
    logger.info("Ready. CORS origins: %s", list(api_config.allowed_origins))
    yield


app = FastAPI(
    title="Corporate Turnaround Chat API",
    version="1.0.0",
    lifespan=lifespan,
    # The interactive docs describe internals and invite traffic to an endpoint
    # that costs LLM quota per call. Off unless explicitly enabled.
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "").lower() == "true" else None,
    redoc_url=None,
)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(api_config.allowed_origins),
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)


class ChatRequest(BaseModel):
    # max_length is the first line of defence: it caps prompt-injection payload
    # size and the token bill for a single call. Rejected by FastAPI before any
    # of our code runs.
    message: str = Field(min_length=1, max_length=api_config.max_message_chars)
    session_id: str | None = None

    def clean_session_id(self) -> str | None:
        """A malformed id is dropped, not honoured -- the caller gets a new session."""
        if self.session_id and is_valid_session_id(self.session_id):
            return self.session_id
        return None


@app.middleware("http")
async def observability(request: Request, call_next):
    """Request id, latency log, and baseline security headers on every response."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "%s %s -> %s in %.0fms [%s]",
        request.method, request.url.path, response.status_code, elapsed_ms, request_id,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _error(request: Request, status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": message,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(RateLimitExceeded)
async def on_rate_limited(request: Request, exc: RateLimitExceeded):
    return _error(request, 429, "Too many messages. Please wait a moment and try again.")


@app.exception_handler(CapacityError)
async def on_capacity(request: Request, exc: CapacityError):
    return _error(request, 503, "We're handling a lot of conversations right now — please retry shortly.")


@app.exception_handler(TurnTimeout)
async def on_timeout(request: Request, exc: TurnTimeout):
    return _error(request, 504, "That took longer than expected. Please try asking again.")


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    # Detail goes to the logs, never to the caller: a traceback both leaks
    # internals and is useless to someone asking about their business debt.
    logger.exception("Unhandled error [%s]", getattr(request.state, "request_id", None))
    return _error(
        request, 500,
        "Sorry — something went wrong on our end. Please try again, or call us at "
        "1-800-889-0232 and we can help you directly.",
    )


@app.get("/health")
async def health(request: Request):
    """Liveness + readiness. Reports unhealthy until retrieval finished warming."""
    service = getattr(app.state, "chat", None)
    if service is None:
        return _error(request, 503, "warming")
    return {"status": "ok", "active_sessions": service.active_sessions}


@app.post("/api/chat")
@limiter.limit(api_config.rate_limit_burst)
@limiter.limit(api_config.rate_limit_sustained)
async def chat(request: Request, body: ChatRequest):
    result = await app.state.chat.answer(body.message, body.clean_session_id())
    # An explicit Response, not a dict: slowapi injects the X-RateLimit-* headers
    # into the returned object and raises if it isn't a Response.
    return JSONResponse(
        {"session_id": result.session_id, "answer": result.answer, "cached": result.cached}
    )


@app.post("/api/chat/stream")
@limiter.limit(api_config.rate_limit_burst)
@limiter.limit(api_config.rate_limit_sustained)
async def chat_stream(request: Request, body: ChatRequest):
    """
    Server-sent events. The widget should render `delta` text as it arrives but
    replace it with the `done` answer, which is the only guard-checked one.
    """
    async def events():
        try:
            async for event in app.state.chat.stream_turn(body.message, body.clean_session_id()):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            # The response has already started, so the status code is committed.
            # An error event is the only way left to tell the client.
            logger.exception("Stream failed [%s]", request.state.request_id)
            message = (
                "We're busy right now — please retry shortly."
                if isinstance(exc, CapacityError)
                else "That took longer than expected. Please try again."
                if isinstance(exc, TurnTimeout)
                else "Sorry — something went wrong. Please try again, or call 1-800-889-0232."
            )
            yield f"data: {json.dumps({'type': 'error', 'error': message})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which holds the whole
            # stream until completion and defeats streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )
