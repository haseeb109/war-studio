"""ai33.pro client.

Verified Nov 2026:
  POST https://api.ai33.pro/v3/text-to-speech   header xi-api-key
       body {"text", "voice_id": "elevenlabs_<id>", "model_id"}  -> {"success", "task_id"}
  GET  https://api.ai33.pro/v1/task/<task_id>   -> {"status": "done", "metadata": {"audio_url"}, "credit_cost"}
  voice_id must start with elevenlabs_, minimax_, clone_, edge_, kokoro_, vbee_ or fishaudio_.
The voice-list endpoint is not documented; list_voices() tries several and merges with BUILTIN_VOICES.
"""
import json
from pathlib import Path
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


class Ai33Busy(Ai33Error):
    """ai33 said 'too many tasks in queue' / rate limited (HTTP 429). Wait and retry; never fatal."""


def _req(key, path, data=None, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "User-Agent": UA},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:300]
        if e.code in (429, 502, 503, 504) or "queue_limit" in body or "server_busy" in body or "busy" in body.lower():
            raise Ai33Busy(f"ai33 busy (HTTP {e.code}): {body}") from None
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


def tts_many(key, items, voice_id, model_id="eleven_multilingual_v2", concurrency=4, on_progress=None, timeout=300,
             state_file=None, log=None, sleep=time.sleep):
    """items: list of (out_path, text). Keeps up to `concurrency` ai33 tasks running, polls, downloads mp3s.

    - HTTP 429 / queue full: waits and retries, and lowers concurrency for the rest of the run (never fatal).
    - Network blips: retried. A task that ai33 reports as failed is re-submitted up to 2 times.
    - state_file: submitted task ids are saved there, so a resumed run picks up tasks that are already
      paid for instead of submitting (and paying for) them again.
    - timeout: seconds WITHOUT any progress before giving up (not a total limit, so long videos are fine).
    Returns total credit cost."""
    import os
    say = log or (lambda m: None)
    state = {}
    if state_file and os.path.exists(state_file):
        try:
            state = json.load(open(state_file))
        except Exception:
            state = {}

    def save_state():
        if state_file:
            tmp = str(state_file) + ".tmp"
            json.dump(state, open(tmp, "w"))
            os.replace(tmp, state_file)

    queue, active = [], {}
    for out, text in items:
        tid = state.get(str(out))
        if tid:
            active[tid] = (out, text, 0)          # already submitted in an earlier run: just poll it
        else:
            queue.append((out, text))
    full_cap = cap = max(1, int(concurrency))
    last_busy = 0.0
    done, credits, total = 0, 0, len(items)
    last_progress = time.time()
    wait = 0.0
    net_errors = 0
    retries = {}

    def net_down(e):
        """No internet / DNS / connection reset: wait (up to 30 min) until ai33 answers again, never fail fast."""
        nonlocal net_errors
        net_errors += 1
        if net_errors == 1:
            say(f"network problem talking to ai33 ({str(e)[:80]}); waiting for the connection, the video continues after")
        t0 = time.time()
        while time.time() - t0 < 1800:
            sleep(min(60, 5 * net_errors))
            try:
                from .util import online
                if online():
                    return
            except Exception:
                return
        raise Ai33Error(f"ai33 unreachable for 30 minutes: {e}")
    while queue or active:
        while queue and len(active) < cap and wait <= 0:
            out, text = queue[0]
            try:
                tid = submit(key, text, voice_id, model_id)
            except Ai33Busy:
                new_cap = max(1, min(cap, len(active)))
                if new_cap != cap:
                    say(f"ai33 queue is full; waiting and lowering parallel voices to {new_cap}")
                cap, wait, last_busy = new_cap, 10.0, time.time()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                net_down(e)
                break
            queue.pop(0)
            active[tid] = (out, text, retries.get(str(out), 0))
            state[str(out)] = tid
            save_state()
        if cap < full_cap and time.time() - last_busy > 45:
            cap += 1                              # queue has been fine for a while: speed back up
            last_busy = time.time()
            say(f"ai33 queue is free again; parallel voices back to {cap}")
        sleep(max(2.5, wait))
        wait = 0.0
        for tid, (out, text, tries) in list(active.items()):
            try:
                d = poll(key, tid)
            except Ai33Busy:
                wait = 10.0                       # ai33 polling busy (503): wait, then poll the same tasks again
                break
            except Ai33Error as e:
                if "HTTP 404" in str(e) or "HTTP 400" in str(e):     # stale task id from an old run
                    del active[tid]
                    state.pop(str(out), None)
                    save_state()
                    queue.insert(0, (out, text))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                net_down(e)
                break
            st = d.get("status")
            if st == "done":
                url = (d.get("metadata") or {}).get("audio_url")
                if not url:
                    raise Ai33Error(f"ai33 task {tid} done without audio_url")
                try:
                    download(url, out)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    Path(out).unlink(missing_ok=True)
                    net_down(e)                   # the task stays active: the same audio is downloaded again
                    break
                net_errors = 0
                credits += d.get("credit_cost") or 0
                del active[tid]
                state.pop(str(out), None)
                save_state()
                done += 1
                last_progress = time.time()
                if on_progress:
                    on_progress(done, total)
            elif st in ("failed", "error", "cancelled"):
                del active[tid]
                state.pop(str(out), None)
                save_state()
                if tries >= 2:
                    raise Ai33Error(f"ai33 task failed 3 times for: {text[:60]}… ({d.get('error') or st})")
                say(f"ai33 task {st}, retrying this sentence")
                retries[str(out)] = tries + 1
                queue.insert(0, (out, text))
        if time.time() - last_progress > timeout:
            raise Ai33Error(f"ai33 made no progress for {timeout}s ({done}/{total} done). Run again to continue.")
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
