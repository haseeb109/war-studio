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


def gemini_pick(key, model, image_path, prompt, timeout=90):
    """Send one collage image + prompt to Gemini, expect JSON back."""
    if not key:
        raise LLMError("GEMINI_API_KEY missing")
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img}}, {"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}}
    r = httpx.post(url, json=body, timeout=timeout)
    if r.status_code == 404:
        _GONE.add(model)
        raise ModelGone(f"gemini model '{model}' " + _one_line_error(r))
    if r.status_code >= 400:
        raise LLMError("gemini " + _one_line_error(r))
    d = r.json()
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json(text), d.get("usageMetadata") or {}
