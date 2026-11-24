import json

import httpx
import pytest

from warstudio import llm, rerank


class R:
    def __init__(self, code, body, headers=None):
        self.status_code, self._b, self.headers = code, body, headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._b


def ok(text='{"scores": [3, 8], "best": 2}', thoughts=40):
    return R(200, {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": text}]}}],
                   "usageMetadata": {"promptTokenCount": 800, "candidatesTokenCount": 20, "thoughtsTokenCount": thoughts}})


BUSY = R(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}})


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "c.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    llm._GEMINI_NO_OPT.clear()
    return str(p)


def test_503_is_retried_then_succeeds(img, monkeypatch):
    seq, sent, slept = [BUSY, BUSY, ok()], [], []
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: (sent.append(json), seq.pop(0))[1])
    res, usage = llm.gemini_pick("k", "gemini-3.8-flash", img, "p", sleep=slept.append)
    assert res["best"] == 2 and usage["_retries"] == 2 and slept == [2.0, 5.0]
    gc = sent[0]["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingLevel": "low"} and gc["mediaResolution"] == "MEDIA_RESOLUTION_MEDIUM"


def test_503_three_times_raises_busy_not_error(img, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: BUSY)
    with pytest.raises(llm.ProviderBusy):
        llm.gemini_pick("k", "m", img, "p", sleep=lambda s: None)


def test_daily_quota_vs_per_minute(img, monkeypatch):
    day = R(429, {"error": {"message": "* Quota exceeded for metric: generativelanguage.googleapis.com/"
                                       "generate_content_free_tier_requests, limit: 20",
                            "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                                {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                 "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}]}})
    monkeypatch.setattr(httpx, "post", lambda *a, **k: day)
    with pytest.raises(llm.QuotaExhausted) as ei:
        llm.gemini_pick("k", "m", img, "p", sleep=lambda s: None)
    assert ei.value.free_tier and ei.value.model == "m" and "FREE tier" in str(ei.value)
    # a free-tier PER-MINUTE limit with a short delay is waited out, not treated as used up
    rpm = R(429, {"error": {"message": "free_tier", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"retryDelay": "12s"}]}})
    seq, slept = [rpm, ok()], []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: seq.pop(0))
    assert llm.gemini_pick("k", "m", img, "p", sleep=slept.append)[0]["best"] == 2 and slept == [12.5]
    minute = R(429, {"error": {"message": "per minute", "details": [{"@type": "RetryInfo", "retryDelay": "7s"}]}})
    seq, slept = [minute, ok()], []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: seq.pop(0))
    res, _ = llm.gemini_pick("k", "m", img, "p", sleep=slept.append)
    assert res["best"] == 2 and slept == [7.5]


def test_rejected_options_are_dropped_and_remembered(img, monkeypatch):
    bad = R(400, {"error": {"message": "Invalid JSON payload: unknown field thinkingLevel in thinking_config"}})
    seq, sent = [bad, ok()], []
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: (sent.append(json), seq.pop(0))[1])
    llm.gemini_pick("k", "gemini-2.5-flash", img, "p", sleep=lambda s: None)
    assert "thinkingConfig" not in sent[1]["generationConfig"] and "mediaResolution" in sent[1]["generationConfig"]
    assert llm._GEMINI_NO_OPT["gemini-2.5-flash"] == {"thinking"}


def test_thought_parts_are_skipped(img, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: ok())
    assert llm.gemini_pick("k", "m", img, "p")[0] == {"scores": [3, 8], "best": 2}


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def make_rr(monkeypatch, behaviour, keys=None, clock=None):
    monkeypatch.setattr(rerank, "resolve_gemini_model", lambda k, w: "gemini-3.8-flash")
    monkeypatch.setattr(rerank, "resolve_groq_model", lambda k, w: "qwen")
    monkeypatch.setattr(rerank, "gemini_fallback_models", lambda k, p: ["gemini-3.5-flash-lite"])
    monkeypatch.setattr(rerank, "gemini_pick", behaviour)
    logs = []
    rr = rerank.Reranker(keys or {"GEMINI_API_KEY": "g"}, {"reranker": "gemini"}, logs.append, clock=clock or Clock())
    return rr, logs


def test_busy_primary_uses_backup_model(monkeypatch, tmp_path):
    calls = []

    def beh(key, model, col, prompt, retry_waits=None):
        calls.append(model)
        if model == "gemini-3.8-flash":
            raise llm.ProviderBusy("503 high demand")
        return {"best": 1}, {"promptTokenCount": 700, "candidatesTokenCount": 10, "thoughtsTokenCount": 30}
    rr, logs = make_rr(monkeypatch, beh)
    res, by = rr.pick(lambda t: str(tmp_path / "c.jpg"), "p")
    assert res == {"best": 1} and by == "gemini" and calls == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    # while the primary rests, the backup goes first (no wasted call on the busy model)
    rr.pick(lambda t: str(tmp_path / "c.jpg"), "p")
    assert calls[-1] == "gemini-3.5-flash-lite" and len(calls) == 3
    assert rr.stats["fallback_model"] == 2 and "Gemini cost ~$" in rr.summary()


def test_all_busy_pauses_then_comes_back(monkeypatch, tmp_path):
    clock = Clock()
    state = {"busy": True}

    def beh(key, model, col, prompt, retry_waits=None):
        if state["busy"]:
            raise llm.ProviderBusy("503")
        return {"best": 1}, {}
    rr, logs = make_rr(monkeypatch, beh, clock=clock)
    for _ in range(5):                       # many busy beats: never dropped, just paused
        assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == (None, None)
    assert rr.chain == ["gemini"] and any("paused for 60s" in l for l in logs)
    state["busy"] = False
    clock.t = 61                             # pause over: Gemini is used again
    assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == ({"best": 1}, "gemini")


def test_busy_gemini_covered_by_groq(monkeypatch, tmp_path):
    def beh(*a, **k):
        raise llm.ProviderBusy("503")
    rr, logs = make_rr(monkeypatch, beh, keys={"GEMINI_API_KEY": "g", "GROQ_API_KEY": "q"})
    monkeypatch.setattr(rerank, "groq_pick", lambda *a, **k: ({"best": 2}, {"prompt_tokens": 5}))
    assert rr.chain == ["gemini", "groq"]
    assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == ({"best": 2}, "groq")
    assert rr.stats["covered_by_other"] == 1


def test_timeout_moves_on_without_retrying(img, monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(llm.ProviderBusy, match="no answer in"):
        llm.gemini_pick("k", "m", img, "p", sleep=lambda s: None)
    assert len(calls) == 1


def test_quota_on_one_model_keeps_the_others(monkeypatch, tmp_path):
    calls = []

    def beh(key, model, col, prompt, retry_waits=None):
        calls.append(model)
        if model == "gemini-3.8-flash":
            raise llm.QuotaExhausted("free quota", model=model, free_tier=True)
        return {"best": 1}, {}
    rr, logs = make_rr(monkeypatch, beh)
    assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == ({"best": 1}, "gemini")
    assert calls == ["gemini-3.8-flash", "gemini-3.5-flash-lite"] and rr.chain == ["gemini"]
    assert any("FREE tier" in l for l in logs)
    rr.pick(lambda t: str(tmp_path / "c.jpg"), "p")          # exhausted model is not asked again
    assert calls[-1] == "gemini-3.5-flash-lite" and len(calls) == 3


def test_all_models_out_of_quota_drops_gemini(monkeypatch, tmp_path):
    def beh(key, model, col, prompt, retry_waits=None):
        raise llm.QuotaExhausted("q", model=model)
    rr, logs = make_rr(monkeypatch, beh)
    assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == (None, None)
    assert rr.chain == [] and any("rerank stopped" in l for l in logs)
