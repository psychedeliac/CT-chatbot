"""
api/main.py — public HTTP API behind the website chat widget.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000

Retrieval is built during lifespan startup, before uvicorn accepts connections,
so the first real request never pays the ~40s BM25 + cross-encoder cost.
"""
import asyncio
import logging
import json
import os
import statistics
import time
import uuid
from collections import deque
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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


async def _keep_warm(url: str, interval_seconds: int):
    """Hit our own public health URL on a schedule.

    Railway's hobby tier spins an idle instance down; the next visitor then
    pays the full container + BM25 + cross-encoder start (~15s to first token,
    which is what the UX audit measured). Inbound traffic is what resets that
    idle timer, so the ping has to go out through the public URL -- calling the
    handler in-process would keep nothing awake.

    Opt-in via KEEPALIVE_URL because it is only correct on a single always-on
    instance: pointed at a load-balanced deployment it wakes an arbitrary
    replica, and on a paid always-on tier it is pure noise. An external uptime
    pinger does the same job with none of those caveats -- this exists so the
    fix is available without standing one up.
    """
    import httpx

    await asyncio.sleep(interval_seconds)
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            try:
                await client.get(url)
            except Exception as exc:  # a failed ping must never kill the app
                logger.warning("Keep-warm ping failed: %s", exc)
            await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming retrieval before accepting traffic...")
    agent_config = build_shared_config()
    app.state.chat = ChatService(agent_config, api_config)

    warm_task = None
    if api_config.keepalive_url:
        warm_task = asyncio.create_task(
            _keep_warm(api_config.keepalive_url, api_config.keepalive_interval_seconds)
        )
        logger.info(
            "Keep-warm ping every %ss -> %s",
            api_config.keepalive_interval_seconds, api_config.keepalive_url,
        )

    logger.info("Ready. CORS origins: %s", list(api_config.allowed_origins))
    yield

    if warm_task:
        warm_task.cancel()


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


# Rolling window of the last _METRICS_WINDOW /api/chat* calls: latency + status.
# Answers "is the API lagging or erroring right now" without grepping logs by
# hand -- the cheap alternative to standing up Prometheus for one process.
_METRICS_WINDOW = 200
_chat_latencies_ms: deque = deque(maxlen=_METRICS_WINDOW)
_chat_statuses: deque = deque(maxlen=_METRICS_WINDOW)
_CHAT_PATHS = {"/api/chat", "/api/chat/stream"}


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
    if request.url.path in _CHAT_PATHS:
        _chat_latencies_ms.append(elapsed_ms)
        _chat_statuses.append(response.status_code)
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
    """Liveness + readiness.

    Reports unhealthy while retrieval is still warming, and again once the
    last few LLM turns have all failed -- an exhausted key pool or a Gemini
    outage previously left this endpoint reporting "ok" while every real chat
    turn 500'd, which is exactly the state a load balancer needs to see as
    down.
    """
    service = getattr(app.state, "chat", None)
    if service is None:
        return _error(request, 503, "warming")
    if service.is_llm_degraded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "active_sessions": service.active_sessions,
                "reason": "LLM layer failing",
            },
        )
    return {"status": "ok", "active_sessions": service.active_sessions}


WIDGET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "widget", "ct-chat-widget.js",
)


@app.get("/widget.js")
async def widget_js(request: Request):
    """Serve the reference widget from this origin.

    A classic <script> tag is not subject to CORS, so a site can embed the
    widget straight from here without hosting the file or adding a build step:

        <script src="https://<this-host>/widget.js" data-api="https://<this-host>"></script>

    The site's own origin still has to appear in CORS_ALLOWED_ORIGINS -- that
    governs the XHR the widget then makes, which is the part that matters.
    """
    if not os.path.isfile(WIDGET_PATH):
        return _error(request, 404, "Widget not available on this deployment.")
    return FileResponse(
        WIDGET_PATH,
        media_type="application/javascript",
        # Short: long enough to spare repeat visitors the fetch, short enough
        # that a fix reaches embedding sites the same day without a cache bust.
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/metrics")
async def metrics(request: Request):
    """Rolling p50/p95 latency and error rate over the last _METRICS_WINDOW
    /api/chat* calls. Not a Prometheus scrape format -- just enough to answer
    "is the API lagging or erroring right now" without reading logs."""
    latencies = list(_chat_latencies_ms)
    if not latencies:
        return {"sample_size": 0}
    statuses = list(_chat_statuses)
    error_count = sum(1 for status in statuses if status >= 400)
    if len(latencies) >= 2:
        percentiles = statistics.quantiles(latencies, n=100)
        p50, p95 = percentiles[49], percentiles[94]
    else:
        p50 = p95 = latencies[0]
    return {
        "sample_size": len(latencies),
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "max": round(max(latencies), 1)},
        "error_rate": round(error_count / len(statuses), 3),
    }


@app.post("/api/chat")
@limiter.limit(api_config.rate_limit_burst)
@limiter.limit(api_config.rate_limit_sustained)
async def chat(request: Request, body: ChatRequest):
    result = await app.state.chat.answer(body.message, body.clean_session_id())
    # An explicit Response, not a dict: slowapi injects the X-RateLimit-* headers
    # into the returned object and raises if it isn't a Response.
    return JSONResponse(
        {
            "session_id": result.session_id,
            "answer": result.answer,
            "cached": result.cached,
            "suggestions": list(result.suggestions),
            "answer_id": result.answer_id,
        }
    )


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=64)
    verdict: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)


@app.post("/api/feedback")
@limiter.limit(api_config.rate_limit_sustained)
async def feedback(request: Request, body: FeedbackRequest):
    """Rate one answer.

    The client sends only the opaque answer_id it was given; the question, the
    answer and the records retrieved are looked up server-side. That keeps
    retrieval internals off the wire and means a rating cannot be forged to
    describe a turn that never happened.
    """
    service = getattr(app.state, "chat", None)
    if service is None:
        return _error(request, 503, "warming")
    if not service.record_feedback(body.answer_id, body.verdict, body.comment):
        # Expired or never issued. Not worth an alarming message -- the rating
        # is lost either way and the user does not need to know why.
        return _error(request, 404, "That message is no longer available to rate.")
    return JSONResponse({"ok": True})


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
