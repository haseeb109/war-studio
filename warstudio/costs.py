"""What one video cost: every step that spends money adds its part to <project>/costs.json while it runs.

Parts: voice (ai33 credits -> $ with the price you set: ai33_usd_per_1k_credits), brain (DeepSeek tokens),
rerank (Gemini tokens; Groq's free tier costs nothing), segment_check (vision model, only in vlm mode). Free:
DVIDS, the library (no rerank for library beats), CLIP, Apple Vision, ffmpeg, the
sound design, and library learning (it captions with the beat's own visual line, no Gemini call).
A step that runs again (Re-run from step) adds again - that money was really spent; a skipped step adds nothing.
"""
import threading
from pathlib import Path

from .util import num, read_json, write_json

_LOCK = threading.Lock()
NAMES = {"voice": "voice", "brain": "script planner", "rerank": "clip ranking", "segment_check": "clip check",
         "captions": "library captions"}


def add(proj, part, usd=0.0, **detail):
    """Add what `part` just spent (usd and/or ai33 credits) to the project's costs.json."""
    p = Path(proj) / "costs.json"
    with _LOCK:
        d = read_json(p, {}) or {}
        cur = d.setdefault(part, {"usd": 0.0, "runs": 0})
        cur["usd"] = round(num(cur.get("usd")) + num(usd), 6)
        cur["runs"] = int(cur.get("runs", 0)) + 1
        for k, v in detail.items():
            cur[k] = round(num(cur.get(k)) + num(v), 6) if isinstance(v, (int, float)) else v
        write_json(p, d)
    return d


def summary(proj, settings=None):
    """{"total_usd", "parts": {part: {"usd", ...}}, "voice_priced": bool}. ai33 credits are turned into $ with the
    price from Settings (0 = unknown: the credits are shown instead)."""
    d = read_json(Path(proj) / "costs.json", {}) or {}
    price = num((settings or {}).get("ai33_usd_per_1k_credits"), 0)
    parts = {}
    for k, v in d.items():
        v = dict(v)
        if k == "voice":
            v["usd"] = round(num(v.get("credits")) / 1000 * price, 6) if price else 0.0
        parts[k] = v
    total = round(sum(num(v.get("usd")) for v in parts.values()), 6)
    return {"total_usd": total, "parts": parts, "voice_priced": bool(price) or "voice" not in parts}


def line(s):
    """One log line: 'Cost of this video: $0.081 - voice $0.050 (2155 credits), script planner $0.025, ...'"""
    bits = []
    for k in ("voice", "brain", "rerank", "segment_check", "captions"):
        v = s["parts"].get(k)
        if not v:
            continue
        if k == "voice":
            cr = int(num(v.get("credits")))
            bits.append(f"voice ${v['usd']:.3f} ({cr} ai33 credits)" if s["voice_priced"]
                        else f"voice {cr} ai33 credits (set ai33_usd_per_1k_credits in Settings to see $)")
        else:
            bits.append(f"{NAMES.get(k, k)} ${num(v.get('usd')):.3f}")
    total = f"${s['total_usd']:.3f}" + ("" if s["voice_priced"] else " + voice credits")
    return f"Cost of this video: {total}" + (" - " + ", ".join(bits) if bits else " (nothing paid)")
