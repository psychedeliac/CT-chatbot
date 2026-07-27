"""
tests/test_llm_rotation.py — key failover, from the caller's point of view.

The previous version covered the streaming path well and left three things
untested that a user would feel: the non-streaming path (_generate) fails over
through the same helper but wraps the result differently, a mid-stream failure
must NOT restart the answer on another key, and the chunk type _stream yields
is load-bearing downstream. It also never checked what happens with no keys
configured -- the first thing a new deployer hits.

Faked at the network boundary only: the LangChain client. The rotation logic,
the error classification, and the chunk wrapping are all real.
"""
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from core.llms import RotatingGeminiLLM, _first_chunk_and_rest, _is_failover_error


def _Chunk(text):
    """A real AIMessageChunk: ChatGenerationChunk validates what it wraps, so a
    stand-in object would pass the test while the live path still failed."""
    return AIMessageChunk(content=text)


class FakeClient:
    """Streams canned chunks, or raises the given error when used."""

    def __init__(self, chunks=("hello ", "world"), error=None):
        self.chunks = chunks
        self.error = error
        self.stream_calls = 0
        self.invoke_calls = 0

    def stream(self, messages, stop=None, **kwargs):
        self.stream_calls += 1
        if self.error:
            raise self.error
        return (_Chunk(c) for c in self.chunks)

    def invoke(self, messages, stop=None, **kwargs):
        self.invoke_calls += 1
        if self.error:
            raise self.error
        return _Chunk("".join(self.chunks))

    @property
    def calls(self):
        return self.stream_calls + self.invoke_calls


QUOTA_ERROR = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
DENIED_ERROR = Exception("403 PERMISSION_DENIED: project disabled")


def _llm(clients):
    llm = RotatingGeminiLLM(api_keys=[f"key{i}" for i in range(len(clients))])
    by_key = dict(zip(llm.api_keys, clients))
    llm._client = lambda key: by_key[key]
    return llm


def _streamed_text(llm):
    return "".join(chunk.text for chunk in llm._stream([]))


# ── Setup errors a deployer will actually hit ───────────────────────────────

def test_no_configured_keys_says_which_variable_to_set():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        RotatingGeminiLLM(api_keys=[])._stream([]).__next__()


# ── Which errors are worth another key ──────────────────────────────────────

@pytest.mark.parametrize("error", [
    QUOTA_ERROR,
    DENIED_ERROR,
    Exception("Quota exceeded for quota metric"),
    Exception("429 Too Many Requests"),
])
def test_key_level_failures_move_to_the_next_key(error):
    exhausted = FakeClient(error=error)
    healthy = FakeClient(chunks=("we ", "can ", "help"))

    assert _streamed_text(_llm([exhausted, healthy])) == "we can help"
    assert exhausted.stream_calls == 1


@pytest.mark.parametrize("error", [
    ValueError("malformed request"),
    Exception("500 internal error"),
])
def test_a_problem_with_the_request_is_not_retried_across_every_key(error):
    """Burning all three keys on a bug in the request is pure latency, and it
    hides the real error behind whichever key failed last."""
    broken = FakeClient(error=error)
    spare = FakeClient()

    with pytest.raises(Exception):
        list(_llm([broken, spare])._stream([]))
    assert spare.calls == 0


def test_when_every_key_is_exhausted_the_quota_error_is_what_surfaces():
    """The operator needs to see 429, not a generic failure -- it is the
    difference between "add a key" and "debug the app"."""
    clients = [FakeClient(error=QUOTA_ERROR) for _ in range(3)]

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        list(_llm(clients)._stream([]))
    assert all(c.stream_calls == 1 for c in clients)


def test_error_classification_does_not_catch_ordinary_failures():
    assert _is_failover_error(QUOTA_ERROR)
    assert _is_failover_error(DENIED_ERROR)
    assert not _is_failover_error(ConnectionError("connection reset"))
    assert not _is_failover_error(ValueError("bad request"))


# ── The lazy-generator trap ─────────────────────────────────────────────────

def test_a_quota_error_hiding_in_a_lazy_stream_is_pulled_forward():
    """client.stream() returns a generator, so it does no work and raises
    nothing until a chunk is pulled -- by which point the failover loop has
    already returned. The result was a 429 reaching the user with two unused
    keys sitting in .env."""
    def exploding():
        raise QUOTA_ERROR
        yield  # pragma: no cover -- makes this a generator

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        _first_chunk_and_rest(exploding())


def test_pulling_the_first_chunk_forward_does_not_consume_it():
    chunks = list(_first_chunk_and_rest(iter([_Chunk("a"), _Chunk("b")])))
    assert [c.text for c in chunks] == ["a", "b"]


def test_an_empty_stream_is_empty_rather_than_an_error():
    """PEP 479: a bare next() raising StopIteration inside a generator frame
    becomes a RuntimeError."""
    assert list(_first_chunk_and_rest(iter([]))) == []


# ── Streaming behaviour the UI depends on ───────────────────────────────────

def test_tokens_arrive_as_they_are_generated_not_as_one_blob():
    """Without a real _stream, BaseChatModel emits the whole answer as a single
    chunk after generation finishes -- so a "streaming" widget shows nothing
    until the last token and time-to-first-token is full response time."""
    chunks = list(_llm([FakeClient(chunks=("a", "b", "c", "d"))])._stream([]))
    assert len(chunks) == 4


def test_stream_yields_the_chunk_type_langchain_expects():
    """The client yields AIMessageChunk; the _stream contract is
    ChatGenerationChunk. Passing the message straight through blows up
    downstream in _generate_with_cache, far from the cause."""
    chunks = list(_llm([FakeClient(chunks=("a",))])._stream([]))
    assert all(isinstance(c, ChatGenerationChunk) for c in chunks)


def test_a_failure_after_tokens_have_shipped_is_not_retried_on_another_key():
    """Failing over mid-answer would replay the reply from the start, so the
    user watches the first half of one answer followed by a second, different
    one. A mid-stream failure has to propagate instead."""
    class DiesAfterFirstChunk:
        def __init__(self):
            self.stream_calls = 0

        def stream(self, messages, stop=None, **kwargs):
            self.stream_calls += 1

            def gen():
                yield _Chunk("We can ")
                raise QUOTA_ERROR
            return gen()

    dying = DiesAfterFirstChunk()
    spare = FakeClient(chunks=("completely different answer",))
    llm = _llm([dying, spare])

    emitted = []
    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        for chunk in llm._stream([]):
            emitted.append(chunk.text)

    assert emitted == ["We can "]
    assert spare.calls == 0, "the answer was restarted on another key mid-stream"


# ── Non-streaming path ──────────────────────────────────────────────────────

def test_the_non_streaming_path_fails_over_the_same_way():
    """_generate goes through the same helper but wraps the result itself, so
    it can break independently of _stream."""
    exhausted = FakeClient(error=QUOTA_ERROR)
    healthy = FakeClient(chunks=("we ", "can ", "help"))

    result = _llm([exhausted, healthy])._generate([])

    assert result.generations[0].message.text() == "we can help"
    assert exhausted.invoke_calls == 1


# ── Transient blips ─────────────────────────────────────────────────────────

class FlakyOnceClient:
    """One-off network blip, then fine. Not a key problem, so rotating away
    would waste a key's quota to work around a hiccup."""

    def __init__(self, chunks=("recovered",)):
        self.chunks = chunks
        self.calls = 0

    def stream(self, messages, stop=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary network blip")
        return (_Chunk(c) for c in self.chunks)

    def invoke(self, messages, stop=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary network blip")
        return _Chunk("".join(self.chunks))


def test_a_transient_blip_is_retried_once_on_the_same_key():
    """max_retries=0 on the client skips the SDK's 6-retry backoff (tens of
    seconds on a 429) -- but that also left zero retries for a one-off blip."""
    flaky = FlakyOnceClient()
    spare = FakeClient()

    assert _streamed_text(_llm([flaky, spare])) == "recovered"
    assert flaky.calls == 2
    assert spare.calls == 0


def test_a_blip_that_is_not_a_blip_stops_after_one_retry():
    """One retry, no backoff. Retrying a genuinely broken endpoint forever is
    how a fast failure becomes a hung request."""
    class AlwaysBroken:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, stop=None, **kwargs):
            self.calls += 1
            raise ConnectionError("still down")

    broken = AlwaysBroken()
    with pytest.raises(ConnectionError):
        list(_llm([broken])._stream([]))
    assert broken.calls == 2


# ── Rotation state across calls ─────────────────────────────────────────────

def test_consecutive_calls_spread_across_the_key_pool():
    """Free-tier keys are 20 requests/day each. Sending every request to key 0
    until it 429s wastes a round trip per turn once it is exhausted; round-robin
    spends the pool evenly instead."""
    clients = [FakeClient(chunks=(f"from{i}",)) for i in range(3)]
    llm = _llm(clients)

    for _ in range(3):
        _streamed_text(llm)

    assert [c.stream_calls for c in clients] == [1, 1, 1]


def test_an_exhausted_key_is_still_probed_on_later_calls():
    """Documented, not endorsed: rotation advances past the key that worked, so
    a dead key is re-tried roughly once per pass. That costs one failed round
    trip per pass and is the price of picking up a key whose daily quota has
    since reset. Change _next to skip known-dead keys only with a way to let
    them back in.
    """
    dead = FakeClient(error=QUOTA_ERROR)
    healthy = FakeClient(chunks=("ok",))
    llm = _llm([dead, healthy])

    for _ in range(3):
        assert _streamed_text(llm) == "ok"

    assert dead.stream_calls == 3
    assert healthy.stream_calls == 3
