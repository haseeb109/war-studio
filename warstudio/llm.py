"""Brain (OpenAI-compatible chat: DeepSeek official API by default) and Gemini vision rerank."""
import base64
import json
import re

import httpx


class LLMError(RuntimeError):
    pass


def extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise LLMError("no JSON in reply")
    return json.loads(text[start:])


def chat(messages, brain, max_tokens=16000, json_mode=True, timeout=300):
    """brain: dict from config.brain_config()."""
    key, base_url, model = brain["key"], brain["url"], brain["model"]
    if not key:
        raise LLMError(f"{brain['key_name']} missing (Settings tab)")
    if not base_url or not model:
        raise LLMError(f"{brain['label']}: base URL or model not set (Settings tab)")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(base_url, json=body, timeout=timeout,
                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    if r.status_code >= 400:
        raise LLMError(f"brain HTTP {r.status_code}: {r.text[:400]}")
    d = r.json()
    usage = d.get("usage") or {}
    return d["choices"][0]["message"]["content"], usage


class ModelGone(LLMError):
    """The requested Gemini model does not exist / is not offered to this key (HTTP 404)."""


_GEMINI_MODELS = {}
_GONE = set()          # models that answered 404 on generateContent in this session


def _one_line_error(r):
    try:
        e = r.json().get("error", {})
        return f"HTTP {r.status_code}: {e.get('message') or e}"[:300]
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:200]}".replace("\n", " ")


def gemini_models(key):
    """Names of models this key can call with generateContent (cached per key)."""
    if key not in _GEMINI_MODELS:
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key, "pageSize": 1000}, timeout=30)
        if r.status_code >= 400:
            raise LLMError("gemini " + _one_line_error(r))
        _GEMINI_MODELS[key] = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                               if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    return _GEMINI_MODELS[key]


def pick_flash_model(names):
    """Newest stable vision 'flash' model; aliases like gemini-flash-latest win; lite/tts/image/live/preview last."""
    if "gemini-flash-latest" in names and "gemini-flash-latest" not in _GONE:
        return "gemini-flash-latest"
    bad = ("lite", "tts", "image", "live", "audio", "embedding", "thinking", "native", "computer")

    def ver(n):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return float(m.group(1)) if m else 0.0
    flash = [n for n in names if "flash" in n and not any(b in n for b in bad) and n not in _GONE]
    stable = [n for n in flash if "preview" not in n and "exp" not in n]
    pool = stable or flash
    if not pool:
        return None
    return sorted(pool, key=lambda n: (ver(n), -len(n)), reverse=True)[0]


def resolve_gemini_model(key, wanted="auto"):
    """'auto' (or a model the key can't use) -> best available flash model."""
    names = gemini_models(key)
    if wanted and wanted != "auto" and wanted in names and wanted not in _GONE:
        return wanted
    best = pick_flash_model(names)
    if not best:
        raise LLMError("no Gemini flash model is available for this key")
    return best


class ProviderBusy(LLMError):
    """Temporary trouble on the provider side (503 high demand, 500, timeout). Not our fault, not permanent:
    retry later or use another model/provider for now."""


# per-model request options that the model rejected once (400) - not sent again in this session
_GEMINI_NO_OPT = {}
GEMINI_TIMEOUT = 30     # a rerank answer takes a few seconds; a call this slow means the model is struggling
GEMINI_RETRY_WAITS = (2.0, 5.0)        # 503/500/timeouts: two quick retries, then the caller falls back
GEMINI_MAX_RPM_WAIT = 60               # a per-minute 429 shorter than this is waited out


def _gemini_retry_delay(r):
    """Seconds from a 429's RetryInfo ('17s') or Retry-After header; 0 if none."""
    try:
        for det in (r.json().get("error", {}).get("details") or []):
            d = det.get("retryDelay")
            if d:
                return float(str(d).rstrip("s"))
    except Exception:
        pass
    return _retry_after(r)


def gemini_quota_info(r):
    """What a 429 says: {'ids': [quotaId...], 'metrics': [...], 'free_tier': bool, 'daily': bool, 'delay': s}."""
    ids, metrics = [], []
    try:
        for det in (r.json().get("error", {}).get("details") or []):
            for v in det.get("violations") or []:
                if v.get("quotaId"):
                    ids.append(v["quotaId"])
                if v.get("quotaMetric"):
                    metrics.append(v["quotaMetric"])
    except Exception:
        pass
    text = r.text.lower()
    free = "free_tier" in text or "freetier" in text
    if ids:
        daily = any("perday" in i.lower() for i in ids) and not any("perminute" in i.lower() for i in ids)
    else:
        daily = any(x in text for x in ("perday", "per day", "per_day", "requests_per_day"))
    return {"ids": ids, "metrics": metrics, "free_tier": free, "daily": daily, "delay": _gemini_retry_delay(r)}


def _is_daily_quota(r):
    return gemini_quota_info(r)["daily"]


def gemini_body(img_b64, prompt, opts):
    gc = {"response_mime_type": "application/json", "temperature": 0.1}
    if "thinking" in opts:
        # scoring 5 tiles needs no long reasoning: 'low' works on every Gemini 3.x Flash (3.8 has no 'minimal')
        gc["thinkingConfig"] = {"thinkingLevel": "low"}
    if "media" in opts:
        gc["mediaResolution"] = "MEDIA_RESOLUTION_MEDIUM"     # 560 instead of 1,120 tokens per image
    return {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}, {"text": prompt}]}],
            "generationConfig": gc}


def gemini_pick(key, model, image_path, prompt, timeout=GEMINI_TIMEOUT, retry_waits=GEMINI_RETRY_WAITS, sleep=None):
    """Send one collage image + prompt to Gemini, expect JSON back. Returns (json, usage).

    503 / 500 / timeouts -> quick retries, then ProviderBusy.  429 per minute -> waits the server's delay.
    429 per day -> QuotaExhausted.  404 -> ModelGone.  A 400 about thinking/media options -> sent again
    without that option (remembered)."""
    import time as _t
    sleep = sleep or _t.sleep
    if not key:
        raise LLMError("GEMINI_API_KEY missing")
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    retries, waits = 0, list(retry_waits)
    while True:
        opts = {"thinking", "media"} - _GEMINI_NO_OPT.get(model, set())
        try:
            r = httpx.post(url, json=gemini_body(img, prompt, opts), timeout=timeout)
        except httpx.TimeoutException as e:
            # no retry on the same model: another 30 s wait rarely helps - the caller moves to a backup model
            raise ProviderBusy(f"gemini {model}: no answer in {timeout}s ({type(e).__name__})")
        except httpx.TransportError as e:
            if waits:
                sleep(waits.pop(0))
                retries += 1
                continue
            raise ProviderBusy(f"gemini {model}: network error ({type(e).__name__})")
        if r.status_code == 400 and opts:
            low = r.text.lower()
            bad = {o for o, words in (("thinking", ("thinking",)), ("media", ("media", "resolution"))) if o in opts
                   and any(w in low for w in words)}
            if bad:
                _GEMINI_NO_OPT.setdefault(model, set()).update(bad)
                continue
        if r.status_code == 404:
            _GONE.add(model)
            raise ModelGone(f"gemini model '{model}' " + _one_line_error(r))
        if r.status_code == 429:
            q = gemini_quota_info(r)
            delay = q["delay"]
            if q["daily"] or delay > GEMINI_MAX_RPM_WAIT:
                tier = "FREE tier" if q["free_tier"] else "paid tier"
                which = ", ".join(q["ids"][:2]) or _one_line_error(r)[-120:]
                raise QuotaExhausted(f"gemini {model}: {tier} quota used up ({which})", model=model,
                                     free_tier=q["free_tier"])
            if retries < 4:
                sleep((delay or 10) + 0.5)
                retries += 1
                continue
            raise ProviderBusy("gemini still rate limited: " + _one_line_error(r))
        if r.status_code in (500, 502, 503, 504):
            if waits:
                sleep(waits.pop(0))
                retries += 1
                continue
            raise ProviderBusy(f"gemini {model} " + _one_line_error(r))
        if r.status_code >= 400:
            raise LLMError("gemini " + _one_line_error(r))
        d = r.json()
        try:
            parts = d["candidates"][0]["content"]["parts"]
            text = next(p["text"] for p in parts if "text" in p and not p.get("thought"))
        except (KeyError, IndexError, StopIteration):
            raise LLMError(f"gemini {model}: empty answer ({str(d.get('candidates') or d)[:120]})")
        usage = dict(d.get("usageMetadata") or {}, _retries=retries, _model=model)
        return extract_json(text), usage


def gemini_fallback_models(key, primary, n=2):
    """Other models to use while `primary` is overloaded: the next stable flash, then flash-lite."""
    names = [m for m in gemini_models(key) if m != primary and m not in _GONE]
    bad = ("tts", "image", "live", "audio", "embedding", "native", "computer", "preview", "exp")

    def ver(m):
        x = re.search(r"gemini-(\d+(?:\.\d+)?)", m)
        return float(x.group(1)) if x else 0.0
    ok = [m for m in names if "flash" in m and not any(b in m for b in bad)]
    full = sorted([m for m in ok if "lite" not in m], key=ver, reverse=True)
    lite = sorted([m for m in ok if "lite" in m], key=ver, reverse=True)
    out = full[:1] + lite[:1]
    if "gemini-flash-latest" not in (primary, *out) and "gemini-flash-latest" in names:
        out.insert(0, "gemini-flash-latest")
    return out[:n]


# ---------------------------------------------------------------- Groq (clip rerank only)
GROQ_URL = "https://api.groq.com/openai/v1"
# Vision models on Groq, best first. "auto" takes the first one this key can see.
GROQ_VISION_MODELS = ["qwen/qwen3.6-27b", "meta-llama/llama-4-maverick-17b-128e-instruct",
                      "meta-llama/llama-4-scout-17b-16e-instruct"]
_GROQ_MODELS = {}


class QuotaExhausted(LLMError):
    """Daily / long quota used up: stop calling this model (or provider) for the rest of the run."""

    def __init__(self, msg="", model=None, free_tier=False):
        super().__init__(msg)
        self.model, self.free_tier = model, free_tier


def groq_models(key):
    if key not in _GROQ_MODELS:
        r = httpx.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=25)
        if r.status_code >= 400:
            raise LLMError("groq " + _one_line_error(r))
        _GROQ_MODELS[key] = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
    return _GROQ_MODELS[key]


def resolve_groq_model(key, wanted="auto"):
    have = groq_models(key)
    if wanted and wanted != "auto":
        if wanted in have:
            return wanted
    for m in GROQ_VISION_MODELS:
        if m in have:
            return m
    vis = [m for m in have if any(t in m.lower() for t in ("-vl", "vision", "llama-4", "qwen3.6"))]
    if vis:
        return vis[0]
    raise LLMError("no vision model available on this Groq key")


def _retry_after(r):
    try:
        return float(r.headers.get("retry-after") or 0)
    except ValueError:
        return 0.0


def groq_pick(key, model, image_path, prompt, timeout=90, sleep=None, max_wait=75):
    """Collage + prompt to a Groq vision model (OpenAI format), expect JSON back.
    Per-minute 429s are waited out (Retry-After); a long wait means the daily quota is gone."""
    import time as _t
    sleep = sleep or _t.sleep
    if not key:
        raise LLMError("GROQ_API_KEY missing")
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    body = {"model": model, "temperature": 0.1, "max_completion_tokens": 400,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt + "\nReply with the JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}]}
    if "qwen" in model:
        body["reasoning_effort"] = "none"       # no thinking tokens: faster and cheaper on the token quota
    for attempt in range(5):
        r = httpx.post(f"{GROQ_URL}/chat/completions", json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 400 and "reasoning" in r.text and "reasoning_effort" in body:
            body.pop("reasoning_effort")
            continue
        if r.status_code == 429:
            wait = _retry_after(r) or 20
            if wait > max_wait or "per day" in r.text.lower() or "(tpd)" in r.text.lower() or "(rpd)" in r.text.lower():
                raise QuotaExhausted("groq daily quota used up: " + _one_line_error(r))
            sleep(wait + 0.5)
            continue
        if r.status_code == 404:
            raise ModelGone(f"groq model '{model}' " + _one_line_error(r))
        if r.status_code in (500, 502, 503, 504):
            if attempt < 1:
                sleep(3)
                continue
            raise ProviderBusy(f"groq {model} " + _one_line_error(r))
        if r.status_code >= 400:
            raise LLMError("groq " + _one_line_error(r))
        d = r.json()
        text = d["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return extract_json(text), d.get("usage") or {}
    raise LLMError("groq: still rate limited after 5 tries")
