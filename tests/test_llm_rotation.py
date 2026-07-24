"""
Key rotation and streaming failover in RotatingGeminiLLM.

The failure this guards against is subtle: client.stream() returns a lazy
generator, so a quota error on the first key surfaces only when the caller pulls
a chunk -- by which point the failover loop has already returned. The result is
a 429 reaching the user while two unused keys sit in .env.

No network: the clients are fakes.
"""
import pytest
from langchain_core.messages import AIMessageChunk

from core.llms import RotatingGeminiLLM, _first_chunk_and_rest


def _Chunk(text):
    """Real AIMessageChunk: ChatGenerationChunk validates what it wraps, so a
    stand-in object would pass the test while the live path still failed."""
    return AIMessageChunk(content=text)


class FakeClient:
    """Streams canned chunks, or raises the given error when used."""

    def __init__(self, chunks=("hello ", "world"), error=None):
        self.chunks = chunks
        self.error = error
        self.stream_calls = 0

    def stream(self, messages, stop=None, **kwargs):
        self.stream_calls += 1
        if self.error:
            raise self.error
        return (_Chunk(c) for c in self.chunks)

    def invoke(self, messages, stop=None, **kwargs):
        if self.error:
            raise self.error
        return _Chunk("".join(self.chunks))


QUOTA_ERROR = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")


def _llm(clients):
    llm = RotatingGeminiLLM(api_keys=[f"key{i}" for i in range(len(clients))])
    by_key = dict(zip(llm.api_keys, clients))
    llm._client = lambda key: by_key[key]
    return llm


def test_lazy_stream_error_is_pulled_forward_so_failover_can_happen():
    def exploding():
        raise QUOTA_ERROR
        yield  # pragma: no cover -- makes this a generator

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        _first_chunk_and_rest(exploding())


def test_first_chunk_is_replayed_not_swallowed():
    chunks = list(_first_chunk_and_rest(iter([_Chunk("a"), _Chunk("b")])))
    assert [c.text for c in chunks] == ["a", "b"]


def test_empty_stream_yields_nothing_rather_than_raising():
    assert list(_first_chunk_and_rest(iter([]))) == []


def test_streaming_fails_over_to_the_next_key_on_quota():
    exhausted = FakeClient(error=QUOTA_ERROR)
    healthy = FakeClient(chunks=("we ", "can ", "help"))

    text = "".join(c.text for c in _llm([exhausted, healthy])._stream([]))

    assert text == "we can help"
    assert exhausted.stream_calls == 1


def test_streaming_yields_many_chunks_not_one_blob():
    """The whole point of _stream: without it the default emits a single chunk
    after generation completes, so time-to-first-token is full response time."""
    chunks = list(_llm([FakeClient(chunks=("a", "b", "c", "d"))])._stream([]))
    assert len(chunks) == 4


def test_non_quota_errors_are_not_retried_across_keys():
    broken = FakeClient(error=ValueError("malformed request"))
    spare = FakeClient()

    with pytest.raises(ValueError):
        list(_llm([broken, spare])._stream([]))
    assert spare.stream_calls == 0


def test_every_key_exhausted_surfaces_the_last_quota_error():
    clients = [FakeClient(error=QUOTA_ERROR) for _ in range(3)]
    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        list(_llm(clients)._stream([]))
    assert all(c.stream_calls == 1 for c in clients)


class FlakyOnceClient:
    """Fails with a transient (non-failover) error on the first call, then
    succeeds -- simulates a one-off network blip rather than an exhausted key."""

    def __init__(self, chunks=("ok",)):
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


def test_transient_error_gets_one_retry_on_the_same_key_before_failing_over():
    flaky = FlakyOnceClient(chunks=("recovered",))
    spare = FakeClient()

    text = "".join(c.text for c in _llm([flaky, spare])._stream([]))

    assert text == "recovered"
    assert flaky.calls == 2       # failed once, retried, succeeded
    assert spare.stream_calls == 0  # never had to fail over


def test_transient_error_surviving_the_retry_still_raises():
    class AlwaysBroken:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, stop=None, **kwargs):
            self.calls += 1
            raise ConnectionError("still down")

    broken = AlwaysBroken()
    with pytest.raises(ConnectionError):
        list(_llm([broken])._stream([]))
    assert broken.calls == 2  # one original attempt + one retry, then give up
