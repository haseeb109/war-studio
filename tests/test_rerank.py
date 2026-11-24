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


def ok(content):
    return R(200, {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 900}})


def test_chain_order():
    k = {"GROQ_API_KEY": "g", "GEMINI_API_KEY": "m"}
    assert rerank.chain_order("auto", k) == ["groq", "gemini"]
    assert rerank.chain_order("gemini", k) == ["gemini", "groq"]
    assert rerank.chain_order("clip", k) == []
    assert rerank.chain_order("auto", {"GEMINI_API_KEY": "m"}) == ["gemini"]
    assert rerank.chain_order("groq", {}) == []


def test_groq_pick_waits_per_minute_and_strips_think(tmp_path, monkeypatch):
    img = tmp_path / "c.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    calls, slept = [], []
    seq = [R(429, {"error": {"message": "rate limit (TPM)"}}, {"retry-after": "7"}),
           ok('<think>hmm</think>```json\n{"scores": [2, 9], "best": 2}\n```')]
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: (calls.append(kw["json"]), seq.pop(0))[1])
    res, usage = llm.groq_pick("k", "qwen/qwen3.6-27b", str(img), "p", sleep=slept.append)
    assert res == {"scores": [2, 9], "best": 2} and slept == [7.5]
    assert calls[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert calls[0]["reasoning_effort"] == "none"


def test_groq_daily_quota_raises(tmp_path, monkeypatch):
    img = tmp_path / "c.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: R(429, {"error": {"message": "Limit tokens per day (TPD)"}},
                                                              {"retry-after": "3600"}))
    with pytest.raises(llm.QuotaExhausted):
        llm.groq_pick("k", "m", str(img), "p", sleep=lambda s: None)


def test_reranker_falls_back_to_gemini_when_groq_quota_gone(monkeypatch, tmp_path):
    monkeypatch.setattr(rerank, "resolve_groq_model", lambda k, w: "qwen/qwen3.6-27b")
    monkeypatch.setattr(rerank, "resolve_gemini_model", lambda k, w: "gemini-flash-latest")
    monkeypatch.setattr(rerank, "gemini_fallback_models", lambda k, p: [])

    def groq(*a, **kw):
        raise llm.QuotaExhausted("TPD")
    monkeypatch.setattr(rerank, "groq_pick", groq)
    monkeypatch.setattr(rerank, "gemini_pick", lambda *a, **kw: ({"scores": [1, 8], "best": 2}, {}))
    logs = []
    rr = rerank.Reranker({"GROQ_API_KEY": "g", "GEMINI_API_KEY": "m"}, {"reranker": "auto"}, logs.append)
    assert rr.chain == ["groq", "gemini"]
    sizes = []
    res, by = rr.pick(lambda t: (sizes.append(t), str(tmp_path / "c.jpg"))[1], "p")
    assert by == "gemini" and res["best"] == 2 and rr.chain == ["gemini"]
    assert sizes == [(384, 216), (512, 288)]
    assert any("groq rerank stopped" in l for l in logs)


def test_reranker_drops_after_five_real_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(rerank, "resolve_groq_model", lambda k, w: "m")
    monkeypatch.setattr(rerank, "groq_pick", lambda *a, **kw: (_ for _ in ()).throw(llm.LLMError("bad")))
    rr = rerank.Reranker({"GROQ_API_KEY": "g"}, {"reranker": "groq"}, lambda m: None)
    for _ in range(4):
        assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == (None, None)
    assert rr.active == "groq"
    assert rr.pick(lambda t: str(tmp_path / "c.jpg"), "p") == (None, None)
    assert rr.active is None
