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


def make_beats(sentences, style):
    pc = style["pacing"]
    beats = []
    for s in sentences:
        dur = s["end"] - s["start"]
        parts = 1
        if s["start"] < pc["intro_seconds"] and dur >= pc["intro_split_min_seconds"]:
            parts = 2
        if pc.get("split_long_beats_everywhere", True) and dur / parts > pc["max_shot_seconds"]:
            parts = math.ceil(dur / (pc["max_shot_seconds"] * 0.85))
        while parts > 1 and dur / parts < pc["min_shot_seconds"]:
            parts -= 1
        pieces = _split_text(s["text"], parts)
        total_chars = sum(len(p) for p in pieces) or 1
        t = s["start"]
        for i, p in enumerate(pieces):
            end = s["end"] if i == len(pieces) - 1 else t + dur * len(p) / total_chars
            beats.append({"id": len(beats), "sid": s["sid"], "text": p, "sentence": s["text"],
                          "start": round(t, 3), "end": round(end, 3)})
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
    log(f"{len(beats)} beats, average shot {avg:.2f}s")
    return beats
