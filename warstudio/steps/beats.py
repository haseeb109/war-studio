"""Step 2: sentences -> beats (one shot each).

Rules (Beyond Military): one beat per sentence; in the first `intro_seconds`, sentences longer than
`intro_split_min_seconds` get two beats; anywhere, beats longer than `max_shot_seconds` are split.
Split points prefer commas/clause words; time inside a sentence is shared by character count.
"""
import math
import re
from pathlib import Path

from ..util import write_json, read_json

CLAUSE = re.compile(r"(?<!\d),(?!\d)|;|(?<!\d):(?!\d)| — | - | and | but | as | while | when | because | then ")


def _split_text(text, parts):
    if parts <= 1:
        return [text]
    words = text.split()
    if len(words) < parts * 2:
        parts = max(1, len(words) // 2)
        if parts <= 1:
            return [text]
    targets = [len(text) * k / parts for k in range(1, parts)]
    cuts = []
    candidates = [m.end() for m in CLAUSE.finditer(text)]
    spaces = [m.start() for m in re.finditer(" ", text)] or [len(text) // 2]
    for tg in targets:
        near = [c for c in candidates if abs(c - tg) < len(text) * 0.22 and c not in cuts]
        pool = near or spaces
        c = min(pool, key=lambda x: abs(x - tg))
        cuts.append(c)
    cuts = sorted(set(cuts))
    out, prev = [], 0
    for c in cuts:
        seg = text[prev:c].strip()
        if seg:
            out.append(seg)
        prev = c
    if text[prev:].strip():
        out.append(text[prev:].strip())
    return out


def _even_out(pieces, dur, max_s, min_s):
    """Clause splits can be uneven (one 3 s piece next to a 1 s piece). Split any piece that is still longer
    than max_s at its middle space, as long as both halves stay above min_s and keep 2+ words."""
    for _ in range(8):
        total = sum(len(p) for p in pieces) or 1
        lens = [dur * len(p) / total for p in pieces]
        i = max(range(len(pieces)), key=lambda k: lens[k])
        if lens[i] <= max_s or len(pieces[i].split()) < 4 or lens[i] / 2 < min_s:
            break
        a, b = _split_text(pieces[i], 2) if len(_split_text(pieces[i], 2)) == 2 else (None, None)
        if not a:
            break
        pieces = pieces[:i] + [a, b] + pieces[i + 1:]
    return pieces


def make_beats(sentences, style):
    """Hook zone: every sentence that starts inside the first `hook_chars` characters of the script
    (about the first minute) is cut into shots of at most `hook_max_shot_seconds` - many fast cuts to hook
    the viewer. After the hook, the normal Beyond Military pacing applies."""
    pc = style["pacing"]
    hook_chars = int(pc.get("hook_chars", 0) or 0)
    hook_max = float(pc.get("hook_max_shot_seconds", 2.4))
    hook_min = float(pc.get("hook_min_shot_seconds", 0.9))
    beats = []
    chars = 0
    for s in sentences:
        dur = s["end"] - s["start"]
        hook = chars < hook_chars
        chars += len(s["text"]) + 1
        parts = 1
        if hook:
            parts = max(1, math.ceil(dur / hook_max))
            while parts > 1 and dur / parts < hook_min:
                parts -= 1
        elif s["start"] < pc["intro_seconds"] and dur >= pc["intro_split_min_seconds"]:
            parts = 2
        if pc.get("split_long_beats_everywhere", True) and dur / parts > pc["max_shot_seconds"]:
            parts = math.ceil(dur / (pc["max_shot_seconds"] * 0.85))
        while parts > 1 and dur / parts < pc["min_shot_seconds"]:
            parts -= 1
        pieces = _split_text(s["text"], parts)
        if hook:
            pieces = _even_out(pieces, dur, hook_max, hook_min)
        total_chars = sum(len(p) for p in pieces) or 1
        t = s["start"]
        for i, p in enumerate(pieces):
            end = s["end"] if i == len(pieces) - 1 else t + dur * len(p) / total_chars
            beats.append({"id": len(beats), "sid": s["sid"], "text": p, "sentence": s["text"],
                          "start": round(t, 3), "end": round(end, 3), "hook": hook})
            t = end
    return beats


def run_step(proj, style, log):
    proj = Path(proj)
    out = proj / "beats.json"
    if out.exists():
        log("beats already done, skipping")
        return read_json(out)
    vo = read_json(proj / "vo.json")
    beats = make_beats(vo["sentences"], style)
    write_json(out, beats)
    avg = vo["total"] / max(len(beats), 1)
    hk = [b for b in beats if b.get("hook")]
    if hk:
        hs = hk[-1]["end"] - hk[0]["start"]
        log(f"hook: first {hs:.0f}s cut into {len(hk)} shots (average {hs / len(hk):.2f}s, longest "
            f"{max(b['end'] - b['start'] for b in hk):.2f}s)")
    log(f"{len(beats)} beats, average shot {avg:.2f}s")
    return beats
