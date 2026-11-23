"""Brain (OpenAI-compatible chat: Hive DeepSeek by default) and Gemini vision rerank."""
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


def chat(messages, key, base_url, model, max_tokens=16000, json_mode=True, timeout=300):
    if not key:
        raise LLMError("HIVE_API_KEY missing (Settings tab)")
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


def gemini_pick(key, model, image_path, prompt, timeout=90):
    """Send one collage image + prompt to Gemini, expect JSON back."""
    if not key:
        raise LLMError("GEMINI_API_KEY missing")
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img}}, {"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}}
    r = httpx.post(url, json=body, timeout=timeout)
    if r.status_code >= 400:
        raise LLMError(f"gemini HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json(text), d.get("usageMetadata") or {}
