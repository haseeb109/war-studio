"""Step 2: sentences -> beats (one shot each).

Rules: one beat per sentence or clause; LOCKED: nothing stays on screen longer than `max_shot_seconds` (3 s),
counting the pause before the next sentence; slivers under `min_shot_seconds` are merged.
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


def _times(pieces, start, speech_end, screen_end):
    """Cut times inside a sentence by character share of the spoken part; the last piece runs on to the next
    sentence (the pause before it is on screen too)."""
    total_chars = sum(len(p) for p in pieces) or 1
    dur = speech_end - start
    out, t = [], start
    for i, p in enumerate(pieces):
        end = screen_end if i == len(pieces) - 1 else t + dur * len(p) / total_chars
        out.append((t, end))
        t = end
    return out


def make_beats(sentences, style, total=None):
    """LOCKED RULE: no shot stays on screen longer than `max_shot_seconds` (3 s) - a new shot every 2-3 s.
    On-screen time of a beat runs until the next beat starts, so the pause after a sentence counts too.
    Sentences are split at clause words / spaces; a piece that still runs long (very few words, long pause) is
    split in time with the same words, so the picture still changes."""
    pc = style["pacing"]
    max_s = float(pc["max_shot_seconds"])
    min_s = float(pc["min_shot_seconds"])
    beats = []
    for k, s in enumerate(sentences):
        screen_end = sentences[k + 1]["start"] if k + 1 < len(sentences) else (total or s["end"])
        screen_end = max(screen_end, s["end"])
        screen = screen_end - s["start"]
        parts = max(1, math.ceil(screen / max_s))
        if s["start"] < pc.get("intro_seconds", 0) and screen >= pc.get("intro_split_min_seconds", 99):
            parts = max(parts, 2)
        while parts > 1 and screen / parts < min_s:
            parts -= 1
        pieces = _split_text(s["text"], parts)
        spans = _times(pieces, s["start"], s["end"], screen_end)
        for _ in range(6):                       # uneven clause splits: add a cut until every piece fits
            if max(b - a for a, b in spans) <= max_s + 1e-6:
                break
            more = _split_text(s["text"], len(pieces) + 1)
            if len(more) <= len(pieces):
                break
            pieces = more
            spans = _times(pieces, s["start"], s["end"], screen_end)
        # merge slivers (< min_s) into a neighbour when the result still fits
        i = 0
        while i < len(pieces) and len(pieces) > 1:
            a, b = spans[i]
            if b - a < min_s:
                j = i - 1 if i > 0 else i + 1
                lo, hi = min(i, j), max(i, j)
                if spans[hi][1] - spans[lo][0] <= max_s + 1e-6:
                    pieces[lo:hi + 1] = [pieces[lo] + " " + pieces[hi]]
                    spans[lo:hi + 1] = [(spans[lo][0], spans[hi][1])]
                    continue
            i += 1
        for p, (a, b) in zip(pieces, spans):
            n = max(1, math.ceil((b - a) / max_s - 1e-9))          # still too long: same words, new picture
            for m in range(n):
                t0 = a + (b - a) * m / n
                t1 = a + (b - a) * (m + 1) / n
                beats.append({"id": len(beats), "sid": s["sid"], "text": p, "sentence": s["text"],
                              "start": round(t0, 3), "end": round(t1, 3),
                              **({"cont": True} if m else {})})
    return beats


def run_step(proj, style, log):
    proj = Path(proj)
    out = proj / "beats.json"
    if out.exists():
        log("beats already done, skipping")
        return read_json(out)
    vo = read_json(proj / "vo.json")
    beats = make_beats(vo["sentences"], style, vo.get("total"))
    write_json(out, beats)
    avg = vo["total"] / max(len(beats), 1)
    on = [(beats[i + 1]["start"] if i + 1 < len(beats) else vo["total"]) - b["start"] for i, b in enumerate(beats)]
    log(f"{len(beats)} beats, average shot {avg:.2f}s, longest on screen {max(on or [0]):.2f}s "
        f"(rule: max {style['pacing']['max_shot_seconds']}s)")
    return beats
