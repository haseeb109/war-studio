"""Review editor: change a finished video beat by beat, then re-render only what changed.

Every edit:
  1. snapshots the project's editable JSON (plus any audio it replaces) -> Undo,
  2. changes the project files directly (picks / fetched / overlays / sfx_events / specs / beats / vo),
  3. marks the affected beats "dirty" in review.json.
"Apply" deletes the dirty segments and re-runs fetch -> render -> qc; existing segments are reused,
so only the changed beats are encoded, then the join (stream copy) and the audio mix run again.

Narration edits keep every later segment valid: the new sentence audio is padded to a whole number of
video frames, so all later beat boundaries move by exactly k frames and their frame counts don't change.
"""
import hashlib
import math
import shutil
import time
from pathlib import Path

import numpy as np

from . import ai33, overlays as ovl, sfx
from .config import LIBRARY_DIR, load_keys, load_settings, load_style
from .steps import fetch, render, qc
from .steps.beats import _split_text
from .util import read_json, write_json, run, duration

EDITABLE = ["picks.json", "fetched.json", "overlays.json", "sfx_events.json", "specs.json", "beats.json",
            "vo.json", "review.json", "script.txt"]
MAX_SNAPSHOTS = 40
GRADES = ["real", "flir", "cockpit", "control_room", "map"]


class EditError(Exception):
    pass


# ---------- state ----------

def _review(proj):
    return read_json(proj / "review.json", {"dirty": [], "audio_dirty": False, "state": "unreviewed", "edits": 0}) or {}


def _save_review(proj, rv):
    write_json(proj / "review.json", rv)


def _mark(proj, bids=(), audio=False):
    rv = _review(proj)
    rv["dirty"] = sorted(set(rv.get("dirty", [])) | {int(b) for b in bids})
    rv["audio_dirty"] = bool(rv.get("audio_dirty") or audio)
    rv["edits"] = rv.get("edits", 0) + 1
    rv["state"] = "edited"
    _save_review(proj, rv)


def _check_idle(proj):
    st = read_json(proj / "status.json", {}) or {}
    if st.get("state") in ("running", "queued"):
        raise EditError("this video is being processed right now; wait until it finishes")
    if not (proj / "final.mp4").exists():
        raise EditError("the video is not finished yet")


def snapshot(proj, label, extra_files=()):
    sdir = proj / "review" / f"snap_{time.time():.6f}"
    sdir.mkdir(parents=True, exist_ok=True)
    for f in EDITABLE:
        if (proj / f).exists():
            shutil.copy2(proj / f, sdir / f)
    for f in extra_files:
        f = Path(f)
        if f.exists():
            (sdir / "files").mkdir(exist_ok=True)
            shutil.copy2(f, sdir / "files" / f.name)
    write_json(sdir / "meta.json", {"label": label, "extra": [str(Path(f).relative_to(proj)) for f in extra_files]})
    snaps = sorted((proj / "review").glob("snap_*"))
    for old in snaps[:-MAX_SNAPSHOTS]:
        shutil.rmtree(old, ignore_errors=True)


def undo(proj):
    proj = Path(proj)
    _check_idle(proj)
    snaps = sorted((proj / "review").glob("snap_*"))
    if not snaps:
        raise EditError("nothing to undo")
    s = snaps[-1]
    meta = read_json(s / "meta.json", {})
    cur = _review(proj)
    for f in EDITABLE:
        if (s / f).exists():
            shutil.copy2(s / f, proj / f)
    for rel in meta.get("extra", []):
        src = s / "files" / Path(rel).name
        if src.exists():
            shutil.copy2(src, proj / rel)
    rv = _review(proj)
    # beats that were re-rendered after this snapshot must be redone too
    rv["dirty"] = sorted(set(rv.get("dirty", [])) | set(cur.get("dirty", [])) | set(cur.get("rendered_since", [])))
    rv["audio_dirty"] = True
    _save_review(proj, rv)
    shutil.rmtree(s, ignore_errors=True)
    return meta.get("label", "")


def _load(proj):
    return {"beats": read_json(proj / "beats.json"), "specs": read_json(proj / "specs.json"),
            "picks": read_json(proj / "picks.json", {}) or {}, "fetched": read_json(proj / "fetched.json", {}) or {},
            "overlays": read_json(proj / "overlays.json", {}) or {}, "sfx": read_json(proj / "sfx_events.json", []) or [],
            "vo": read_json(proj / "vo.json")}


def _beat(d, bid):
    for b in d["beats"]:
        if b["id"] == int(bid):
            return b
    raise EditError(f"beat {bid} not found")


def _beat_end(d, b):
    i = d["beats"].index(b)
    return d["vo"]["total"] if i == len(d["beats"]) - 1 else d["beats"][i + 1]["start"]


# ---------- review data for the GUI ----------

def review_data(proj):
    proj = Path(proj)
    d = _load(proj)
    rv = _review(proj)
    spec = {s["id"]: s for s in d["specs"]}
    out = []
    for b in d["beats"]:
        bid = str(b["id"])
        p = d["picks"].get(bid) or {}
        f = d["fetched"].get(bid) or {}
        flags = []
        pending = b["id"] in rv.get("dirty", [])
        if (not f or not Path(f.get("file", "")).exists()) and not (pending and p):
            flags.append("no footage")
        if p.get("gemini_score") is not None and p["gemini_score"] < 6:
            flags.append("low match")
        if spec.get(b["id"], {}).get("fallback") and not (p.get("gemini_score") or 0) >= 6:
            flags.append("generic search")
        alts = p.get("alternates") or []
        out.append({
            "id": b["id"], "sid": b["sid"], "text": b["text"], "start": b["start"], "end": _beat_end(d, b),
            "footage_class": spec.get(b["id"], {}).get("footage_class", "real"),
            "entity": spec.get(b["id"], {}).get("entity", ""),
            "queries": spec.get(b["id"], {}).get("queries", []),
            "pick": {k: p.get(k) for k in ("source", "id", "title", "gemini_score", "clip_score", "page_url", "credit")} if p else None,
            "pick_preview": Path(p["preview_file"]).name if p.get("preview_file") else "",
            "clip": {"start": f.get("start"), "length": f.get("length"), "source_duration": p.get("duration")} if f else None,
            "alternates": [dict({k: a.get(k) for k in ("source", "id", "title", "gemini_score", "clip_score")},
                                preview=Path(a["preview_file"]).name if a.get("preview_file") else "",
                                chosen=(a.get("source"), a.get("id")) == (p.get("source"), p.get("id"))) for a in alts],
            "overlay": d["overlays"].get(bid),
            "flags": flags, "dirty": b["id"] in rv.get("dirty", []),
        })
    events = [dict(e, index=i) for i, e in enumerate(d["sfx"])]
    sentences = d["vo"]["sentences"]
    return {"beats": out, "sfx": events, "sentences": [{"sid": s["sid"], "text": s["text"], "start": s["start"], "end": s["end"]} for s in sentences],
            "total": d["vo"]["total"], "review": rv, "grades": GRADES, "sfx_categories": sfx.CATS,
            "undo_available": bool(list((proj / "review").glob("snap_*"))),
            "flagged": sum(1 for x in out if x["flags"])}


# ---------- edits ----------

def swap_clip(proj, bid, cand):
    proj = Path(proj)
    _check_idle(proj)
    d = _load(proj)
    _beat(d, bid)
    bid = str(bid)
    old = d["picks"].get(bid) or {}
    new = {k: v for k, v in cand.items() if k not in ("preview", "chosen", "index")}
    if not new.get("preview_file") and old.get("alternates"):
        for a in old["alternates"]:
            if (a.get("source"), a.get("id")) == (new.get("source"), new.get("id")):
                new = dict(a)
    if not new.get("source") or not new.get("id"):
        raise EditError("candidate has no source/id")
    if new["source"] != "user" and not (new.get("video_url") or new["source"] == "dvids"):
        raise EditError("candidate has no video URL")
    snapshot(proj, f"clip beat {bid}")
    new["alternates"] = old.get("alternates") or []
    new.pop("forced_start", None)
    d["picks"][bid] = new
    d["fetched"].pop(bid, None)
    write_json(proj / "picks.json", d["picks"])
    write_json(proj / "fetched.json", d["fetched"])
    _mark(proj, [bid])


def set_offset(proj, bid, start):
    proj = Path(proj)
    _check_idle(proj)
    d = _load(proj)
    _beat(d, bid)
    bid = str(bid)
    p = d["picks"].get(bid)
    if not p:
        raise EditError("this beat has no clip")
    if p.get("source") == "user":
        raise EditError("offset is not available for your own clips yet")
    snapshot(proj, f"offset beat {bid}")
    p["forced_start"] = max(0.0, float(start))
    d["fetched"].pop(bid, None)
    write_json(proj / "picks.json", d["picks"])
    write_json(proj / "fetched.json", d["fetched"])
    _mark(proj, [bid])


def use_upload(proj, bid, src, name, style):
    proj = Path(proj)
    _check_idle(proj)
    d = _load(proj)
    _beat(d, bid)
    bid = str(bid)
    h = hashlib.md5(Path(src).read_bytes()[:2_000_000] + name.encode()).hexdigest()[:12]
    out = LIBRARY_DIR / f"user_{h}.mp4"
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    W, H, fps = style["canvas"]["width"], style["canvas"]["height"], style["canvas"]["fps"]
    if not out.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", "120", "-an",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},format=yuv420p",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)], timeout=900)
    prev = proj / "previews" / f"user_{h}.jpg"
    prev.parent.mkdir(exist_ok=True)
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1", "-vf", "scale=512:-1", str(prev)])
    snapshot(proj, f"own clip beat {bid}")
    old = d["picks"].get(bid) or {}
    d["picks"][bid] = {"source": "user", "id": h, "title": name, "file": str(out), "preview_file": str(prev),
                       "duration": duration(out), "alternates": old.get("alternates") or []}
    d["fetched"].pop(bid, None)
    write_json(proj / "picks.json", d["picks"])
    write_json(proj / "fetched.json", d["fetched"])
    _mark(proj, [bid])


def set_grade(proj, bid, fclass):
    proj = Path(proj)
    _check_idle(proj)
    if fclass not in GRADES:
        raise EditError(f"grade must be one of {GRADES}")
    d = _load(proj)
    _beat(d, bid)
    snapshot(proj, f"grade beat {bid}")
    for s in d["specs"]:
        if s["id"] == int(bid):
            s["footage_class"] = fclass
    write_json(proj / "specs.json", d["specs"])
    _mark(proj, [bid])


def _click_near(events, t, tol=0.35):
    return [i for i, e in enumerate(events) if e["cat"] == "ui_click" and abs(e["t"] - t) <= tol]


def set_overlay(proj, bid, kind, line1, line2=""):
    proj = Path(proj)
    _check_idle(proj)
    if kind not in ("time", "measure", "label"):
        raise EditError("kind must be time, measure or label")
    if not line1.strip():
        raise EditError("callout text is empty")
    d = _load(proj)
    b = _beat(d, bid)
    style = load_style(read_json(proj / "project.json")["style"])
    snapshot(proj, f"callout beat {bid}")
    dur = min(_beat_end(d, b) - b["start"], style["overlays"]["max_hold_seconds"])
    d["overlays"][str(bid)] = {"kind": kind, "line1": line1.strip()[:40], "line2": (line2 or "").strip()[:40],
                               "duration": round(max(0.8, dur), 3)}
    write_json(proj / "overlays.json", d["overlays"])
    audio = False
    if not _click_near(d["sfx"], b["start"] + 0.05):
        d["sfx"].append({"t": round(b["start"] + 0.05, 3), "cat": "ui_click", "gain_db": style["sfx"]["gains_db"]["ui_click"]})
        d["sfx"].sort(key=lambda e: e["t"])
        write_json(proj / "sfx_events.json", d["sfx"])
        audio = True
    _mark(proj, [bid], audio=audio)


def remove_overlay(proj, bid):
    proj = Path(proj)
    _check_idle(proj)
    d = _load(proj)
    b = _beat(d, bid)
    if str(bid) not in d["overlays"]:
        raise EditError("this beat has no callout")
    snapshot(proj, f"remove callout beat {bid}")
    d["overlays"].pop(str(bid))
    write_json(proj / "overlays.json", d["overlays"])
    near = _click_near(d["sfx"], b["start"] + 0.05)
    if near:
        d["sfx"] = [e for i, e in enumerate(d["sfx"]) if i not in near]
        write_json(proj / "sfx_events.json", d["sfx"])
    _mark(proj, [bid], audio=bool(near))


def sfx_remove(proj, index):
    proj = Path(proj)
    _check_idle(proj)
    d = _load(proj)
    if not 0 <= int(index) < len(d["sfx"]):
        raise EditError("sound effect not found")
    snapshot(proj, "remove sfx")
    d["sfx"].pop(int(index))
    write_json(proj / "sfx_events.json", d["sfx"])
    _mark(proj, audio=True)


def sfx_add(proj, bid, cat, offset=0.12):
    proj = Path(proj)
    _check_idle(proj)
    if cat not in sfx.CATS:
        raise EditError(f"unknown sound category {cat}")
    d = _load(proj)
    b = _beat(d, bid)
    style = load_style(read_json(proj / "project.json")["style"])
    snapshot(proj, f"add {cat}")
    d["sfx"].append({"t": round(max(0.0, b["start"] + float(offset)), 3), "cat": cat,
                     "gain_db": style["sfx"]["gains_db"].get(cat, -12)})
    d["sfx"].sort(key=lambda e: e["t"])
    write_json(proj / "sfx_events.json", d["sfx"])
    _mark(proj, audio=True)


def _even_split(text, n):
    words = text.split()
    if n <= 1 or len(words) < n:
        return [text] if n <= 1 else (words + [""] * n)[:n]
    k = len(words) / n
    return [" ".join(words[round(i * k):round((i + 1) * k)]) for i in range(n)]


def edit_narration(proj, sid, new_text, keys=None, settings=None):
    """Re-voice one sentence; shift the rest of the timeline by a whole number of frames."""
    proj = Path(proj)
    _check_idle(proj)
    new_text = " ".join(new_text.split())
    if not new_text:
        raise EditError("sentence is empty")
    keys = keys or load_keys()
    settings = settings or load_settings()
    meta = read_json(proj / "project.json")
    style = load_style(meta["style"])
    fps = style["canvas"]["fps"]
    d = _load(proj)
    sents = d["vo"]["sentences"]
    if not 0 <= int(sid) < len(sents):
        raise EditError("sentence not found")
    s = sents[int(sid)]
    if s["text"] == new_text:
        raise EditError("text is unchanged")
    old_file = proj / s["file"]
    tmp = proj / "review" / f"tts_{sid}_{int(time.time())}.mp3"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    credits = 0
    if meta.get("dummy"):
        from .steps.vo import _dummy_tts
        _dummy_tts([(tmp, new_text)])
    else:
        if not keys.get("AI33_API_KEY"):
            raise EditError("AI33_API_KEY missing (Settings tab)")
        credits = ai33.tts_many(keys["AI33_API_KEY"], [(tmp, new_text)], meta["voice_id"], settings.get("tts_model"), 1, timeout=300)

    # frame-align the new sentence length
    old_d = s["end"] - s["start"]
    new_raw = duration(tmp)
    k = math.ceil(round((new_raw - old_d) * fps, 6))
    target = old_d + k / fps
    if target < 0.3:
        raise EditError("the new sentence is too short")
    SR = 48000
    new_audio = sfx.load_audio(tmp, SR)
    n_target = int(round(target * SR))
    new_audio = np.concatenate([new_audio, np.zeros(max(0, n_target - len(new_audio)), np.float32)])[:n_target]
    delta = k / fps

    snapshot(proj, f"narration sentence {sid}", extra_files=[old_file, proj / "vo.wav"])

    # splice vo.wav
    vo_all = sfx.load_audio(proj / "vo.wav", SR)
    a, b_ = int(round(s["start"] * SR)), int(round(s["end"] * SR))
    vo_all = np.concatenate([vo_all[:a], new_audio, vo_all[b_:]])
    sfx.write_wav(proj / "vo.wav", vo_all, SR)
    new_file = proj / "vo" / f"s{int(sid):04d}.wav"
    sfx.write_wav(new_file, new_audio, SR)

    old_start, old_end = s["start"], s["end"]
    s["text"], s["end"], s["file"] = new_text, round(old_start + target, 6), str(new_file.relative_to(proj))
    for t in sents[int(sid) + 1:]:
        t["start"] = round(t["start"] + delta, 6)
        t["end"] = round(t["end"] + delta, 6)
    d["vo"]["total"] = round(d["vo"]["total"] + delta, 6)
    write_json(proj / "vo.json", d["vo"])

    # beats of this sentence: same count, new text, new times; later beats shift
    mine = [b for b in d["beats"] if b["sid"] == int(sid)]
    pieces = _split_text(new_text, len(mine))
    if len(pieces) != len(mine):
        pieces = _even_split(new_text, len(mine))
    tot = sum(max(1, len(p)) for p in pieces)
    t = old_start
    for i, (b, ptxt) in enumerate(zip(mine, pieces)):
        end = s["end"] if i == len(mine) - 1 else t + target * max(1, len(ptxt)) / tot
        b["text"], b["sentence"], b["start"], b["end"] = ptxt, new_text, round(t, 6), round(end, 6)
        t = end
    for b in d["beats"]:
        if b["sid"] > int(sid):
            b["start"] = round(b["start"] + delta, 6)
            b["end"] = round(b["end"] + delta, 6)
    write_json(proj / "beats.json", d["beats"])

    # clips long enough? keep the same part of the same clip
    dirty = [b["id"] for b in mine]
    for b in mine:
        bid = str(b["id"])
        f, p = d["fetched"].get(bid), d["picks"].get(bid)
        need = _beat_end(d, b) - b["start"] + fetch.PAD
        if f and p and p.get("source") != "user" and (f.get("length") or 0) < need - 0.05:
            p["forced_start"] = f.get("start", 0)
            d["fetched"].pop(bid, None)
    write_json(proj / "picks.json", d["picks"])
    write_json(proj / "fetched.json", d["fetched"])

    # callouts follow the new words
    for b in mine:
        bid = str(b["id"])
        found = ovl.detect(b["text"])
        cur = d["overlays"].get(bid)
        dur = round(max(0.8, min(_beat_end(d, b) - b["start"], style["overlays"]["max_hold_seconds"])), 3)
        if found:
            d["overlays"][bid] = dict(found, duration=dur)
        elif cur and cur["kind"] in ("time", "measure"):
            d["overlays"].pop(bid)
        elif cur:
            cur["duration"] = dur
    write_json(proj / "overlays.json", d["overlays"])

    # sound effects: shift later ones, stretch the ones inside the sentence
    for e in d["sfx"]:
        if e["t"] >= old_end - 1e-6:
            e["t"] = round(e["t"] + delta, 3)
        elif e["t"] >= old_start:
            e["t"] = round(old_start + (e["t"] - old_start) * target / max(old_d, 0.01), 3)
    write_json(proj / "sfx_events.json", d["sfx"])

    script = proj / "script.txt"
    try:
        old_vo = read_json(sorted((proj / "review").glob("snap_*"))[-1] / "vo.json")
        old_line = old_vo["sentences"][int(sid)]["text"]
        txt = script.read_text(encoding="utf-8")
        if old_line in txt:
            script.write_text(txt.replace(old_line, new_text, 1), encoding="utf-8")
    except Exception:
        pass
    tmp.unlink(missing_ok=True)
    _mark(proj, dirty, audio=True)
    return {"credits": credits, "shift_seconds": round(delta, 3), "shift_frames": k, "beats": dirty}


def set_reviewed(proj, reviewed=True):
    proj = Path(proj)
    rv = _review(proj)
    rv["state"] = "reviewed" if reviewed else ("edited" if rv.get("edits") else "unreviewed")
    _save_review(proj, rv)
    return rv["state"]


# ---------- apply (runs in the job worker) ----------

def apply(proj, log):
    proj = Path(proj)
    meta = read_json(proj / "project.json")
    style = load_style(meta["style"])
    keys, settings = load_keys(), load_settings()
    rv = _review(proj)
    dirty = rv.get("dirty", [])
    log(f"applying edits: {len(dirty)} segment(s) to re-render{', audio re-mix' if rv.get('audio_dirty') else ''}")
    for bid in dirty:
        (proj / "segments" / f"seg_{int(bid):05d}.mp4").unlink(missing_ok=True)
    t0 = time.time()
    fetch.run_step(proj, keys, style, log)
    render.run_step(proj, settings, style, log)
    qc.run_step(proj, log)
    rv = _review(proj)
    rv["rendered_since"] = sorted(set(rv.get("rendered_since", [])) | set(dirty))
    rv["dirty"], rv["audio_dirty"] = [], False
    _save_review(proj, rv)
    log(f"edits applied in {time.time() - t0:.0f}s")


def search(proj, query, n=12):
    """Free-text footage search for the review screen (uses the normal search cache)."""
    from .steps.search import SOURCES, SourceDown
    keys = load_keys()
    out, errors = [], []
    for src, (kname, fn) in SOURCES.items():
        if not keys.get(kname):
            continue
        try:
            for c in fn(query, keys[kname], n):
                if c.get("preview_url"):
                    out.append(dict(c, query=query))
        except Exception as e:
            errors.append(f"{src}: {str(e)[:100]}")
    if not out and errors:
        raise EditError("; ".join(errors))
    return out
