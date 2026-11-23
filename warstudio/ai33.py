"""ai33.pro client.

Verified Nov 2026:
  POST https://api.ai33.pro/v3/text-to-speech   header xi-api-key
       body {"text", "voice_id": "elevenlabs_<id>", "model_id"}  -> {"success", "task_id"}
  GET  https://api.ai33.pro/v1/task/<task_id>   -> {"status": "done", "metadata": {"audio_url"}, "credit_cost"}
  voice_id must start with elevenlabs_, minimax_, clone_, edge_, kokoro_, vbee_ or fishaudio_.
The voice-list endpoint is not documented; list_voices() tries several and merges with BUILTIN_VOICES.
"""
import json
import time
import urllib.error
import urllib.request

from .util import UA, download

BASE = "https://api.ai33.pro"

# ElevenLabs premade voices (always usable through ai33 with the elevenlabs_ prefix).
BUILTIN_VOICES = [
    {"voice_id": "elevenlabs_nPczCjzI2devNBz1zQrb", "name": "Brian", "desc": "Deep, resonant narrator (male, US)"},
    {"voice_id": "elevenlabs_onwK4e9ZLuTAKqWW03F9", "name": "Daniel", "desc": "Steady broadcaster (male, UK)"},
    {"voice_id": "elevenlabs_JBFqnCBsd6RMkjVDRZzb", "name": "George", "desc": "Warm storyteller (male, UK)"},
    {"voice_id": "elevenlabs_pNInz6obpgDQGcFmaJgB", "name": "Adam", "desc": "Dominant, firm (male, US)"},
    {"voice_id": "elevenlabs_pqHfZKP75CvOlQylNhV4", "name": "Bill", "desc": "Wise, mature (male, US)"},
    {"voice_id": "elevenlabs_N2lVS1w4EtoT3dr4eOWO", "name": "Callum", "desc": "Husky, gravelly (male, US)"},
    {"voice_id": "elevenlabs_cjVigY5qzO86Huf0OWal", "name": "Eric", "desc": "Smooth, trustworthy (male, US)"},
    {"voice_id": "elevenlabs_IKne3meq5aSn9XLyUdCD", "name": "Charlie", "desc": "Deep, energetic (male, AU)"},
    {"voice_id": "elevenlabs_CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "desc": "Laid-back, resonant (male, US)"},
    {"voice_id": "elevenlabs_SOYHLrjzK2X1ezoPC6cr", "name": "Harry", "desc": "Fierce (male, US)"},
    {"voice_id": "elevenlabs_z9ihBMCAnrOeGUbPDiwi", "name": "Andres", "desc": "Documentary narrator"},
    {"voice_id": "elevenlabs_EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "desc": "Mature, confident (female, US)"},
    {"voice_id": "elevenlabs_XrExE9yKIg1WjnnlVkGX", "name": "Matilda", "desc": "Professional (female, US)"},
    {"voice_id": "elevenlabs_pFZP5JQG7iQjIQuC4Bku", "name": "Lily", "desc": "Velvety (female, UK)"},
]

VOICE_LIST_PATHS = ["/v3/voices", "/v2/voices", "/v1/voices", "/v3/voice", "/v3/elevenlabs/voices",
                    "/v1/elevenlabs/voices", "/v1/shared-voices"]


class Ai33Error(RuntimeError):
    pass


def _req(key, path, data=None, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "User-Agent": UA},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:300]
        raise Ai33Error(f"ai33 {path} HTTP {e.code}: {body}") from None


def normalize_voice_id(v):
    v = v.strip()
    prefixes = ("elevenlabs_", "minimax_", "clone_", "edge_", "kokoro_", "vbee_", "fishaudio_")
    return v if v.startswith(prefixes) else "elevenlabs_" + v


def submit(key, text, voice_id, model_id="eleven_multilingual_v2"):
    r = _req(key, "/v3/text-to-speech", {"text": text, "voice_id": normalize_voice_id(voice_id), "model_id": model_id})
    if not r.get("success") or not r.get("task_id"):
        raise Ai33Error(f"ai33 submit failed: {r}")
    return r["task_id"]


def poll(key, task_id):
    return _req(key, f"/v1/task/{task_id}")


def tts_many(key, items, voice_id, model_id="eleven_multilingual_v2", concurrency=4, on_progress=None, timeout=900):
    """items: list of (out_path, text). Submits `concurrency` tasks at a time, polls, downloads mp3s.
    Returns total credit cost."""
    queue = list(items)
    active = {}
    done = 0
    credits = 0
    t0 = time.time()
    while queue or active:
        while queue and len(active) < concurrency:
            out, text = queue.pop(0)
            active[submit(key, text, voice_id, model_id)] = out
        time.sleep(2.5)
        for tid, out in list(active.items()):
            d = poll(key, tid)
            st = d.get("status")
            if st == "done":
                url = (d.get("metadata") or {}).get("audio_url")
                if not url:
                    raise Ai33Error(f"ai33 task {tid} done without audio_url")
                download(url, out)
                credits += d.get("credit_cost") or 0
                del active[tid]
                done += 1
                if on_progress:
                    on_progress(done, len(items))
            elif st in ("failed", "error", "cancelled"):
                raise Ai33Error(f"ai33 task {tid} {st}: {d.get('error') or d}")
        if time.time() - t0 > timeout:
            raise Ai33Error("ai33 TTS timed out")
    return credits


def _parse_voices(r):
    data = r.get("data", r) if isinstance(r, dict) else r
    if isinstance(data, dict):
        data = data.get("voices") or data.get("items") or data.get("data") or []
    out = []
    for v in data if isinstance(data, list) else []:
        if not isinstance(v, dict):
            continue
        vid = v.get("voice_id") or v.get("id") or v.get("voiceId")
        if not vid:
            continue
        labels = v.get("labels") or {}
        desc = v.get("description") or (" ".join(str(x) for x in labels.values()) if isinstance(labels, dict) else "")
        out.append({"voice_id": normalize_voice_id(str(vid)), "name": v.get("name") or str(vid),
                    "desc": (desc or "")[:120], "preview_url": v.get("preview_url") or v.get("previewUrl") or ""})
    return out


def list_voices(key):
    """Returns (voices, source). Tries ai33 endpoints; always includes the built-in premade list."""
    remote, source = [], "built-in"
    if key:
        for p in VOICE_LIST_PATHS:
            try:
                vs = _parse_voices(_req(key, p, timeout=20))
            except Exception:
                continue
            if vs:
                remote, source = vs, f"ai33 {p}"
                break
    seen = {v["voice_id"] for v in remote}
    merged = remote + [dict(v, preview_url="") for v in BUILTIN_VOICES if v["voice_id"] not in seen]
    return merged, source
