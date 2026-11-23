"""Step 6: download only the needed seconds of each winning clip, normalized to 1080p30.

ffmpeg reads straight from the clip URL with -ss/-t, so only beat length + padding is pulled.
Files go to library/raw/ and are reused if the same source segment is picked again (phase-2 library data).
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ..config import LIBRARY_DIR
from ..util import read_json, write_json, run, duration, num, UA

PAD = 1.0


def resolve_dvids(pick, key):
    """DVIDS search results carry no file URL; the asset endpoint does."""
    r = httpx.get("https://api.dvidshub.net/asset", params={"api_key": key, "id": pick["id"]},
                  headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    d = r.json().get("results") or r.json()
    files = d.get("files") or []
    if isinstance(files, dict):
        files = list(files.values())
    mp4 = [f for f in files if isinstance(f, dict) and str(f.get("src", "")).split("?")[0].endswith(".mp4")]
    mp4.sort(key=lambda f: (abs(int(f.get("height") or 0) - 1080), -int(f.get("width") or 0)))
    if not mp4:
        raise RuntimeError("DVIDS asset has no mp4 file")
    pick["video_url"] = mp4[0]["src"]
    if not num(pick.get("duration")):
        pick["duration"] = num(d.get("duration"))
    # DVIDS packages open with slates/titles; time_start is where the real footage begins
    pick["min_start"] = num(d.get("time_start"))
    return pick


def choose_start(src_dur, need, used_offsets, min_start=0.0):
    if src_dur <= 0:
        return round(max(0.0, min_start), 2)
    lo = max(0.0, min(min_start, src_dur - need - 0.2))
    span = src_dur - lo
    base = max(lo, min(src_dur - need - 0.2, lo + span * 0.3))
    for k in range(6):
        if span > need:
            cand = lo + ((base - lo) + k * need) % max(span - need, 0.01)
        else:
            cand = lo
        if all(abs(cand - u) > need for u in used_offsets):
            return round(cand, 2)
    return round(base, 2)


def fetch_one(pick, start, need, out, W, H, fps):
    """Download `need` seconds from `start`, normalized. Written to a temp file and only renamed into place
    after it proves playable, so a broken download can never be reused later."""
    vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},format=yuv420p"
    ua = ["-user_agent", UA] if str(pick["video_url"]).startswith("http") else []
    out = Path(out)
    tmp = out.with_name(out.stem + ".part.mp4")
    try:
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", *ua, "-i", pick["video_url"],
             "-t", f"{need:.2f}", "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(tmp)],
            timeout=600)
        got = duration(tmp)
        if got < min(0.5, need * 0.5):
            raise RuntimeError(f"clip too short ({got:.2f}s)")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def run_step(proj, keys, style, log):
    proj = Path(proj)
    out_path = proj / "fetched.json"
    fetched = read_json(out_path, {}) or {}
    beats = read_json(proj / "beats.json")
    picks = read_json(proj / "picks.json")
    vo_total = read_json(proj / "vo.json")["total"]
    W, H, fps = style["canvas"]["width"], style["canvas"]["height"], style["canvas"]["fps"]
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    used = {}
    lock = threading.Lock()
    jobs = []
    for i, b in enumerate(beats):
        bid = str(b["id"])
        p = picks.get(bid)
        if not p or fetched.get(bid):
            continue
        end = vo_total if i == len(beats) - 1 else b["end"]
        need = (end - b["start"]) + PAD
        jobs.append((bid, p, need))

    def try_one(bid, p, need):
        if p["source"] == "user":
            return {"file": p["file"], "start": 0, "length": duration(p["file"]), "source": "user",
                    "id": p["id"], "page_url": "", "credit": "own footage", "title": p.get("title", "")}
        if p["source"] == "dvids" and not p.get("video_url"):
            resolve_dvids(p, keys.get("DVIDS_API_KEY"))
        k = f"{p['source']}:{p['id']}"
        with lock:
            if p.get("forced_start") is not None:
                start = round(num(p["forced_start"]), 2)
            else:
                start = choose_start(num(p.get("duration")), need, used.setdefault(k, []), num(p.get("min_start")))
            used.setdefault(k, []).append(start)
        out = LIBRARY_DIR / f"{p['source']}_{p['id'].replace(':', '_')}_{int(start * 100)}_{int(need * 100)}.mp4"
        if out.exists():
            try:
                duration(out)
            except Exception:
                out.unlink()                      # a broken file from an older version: fetch again
        if not out.exists():
            try:
                fetch_one(p, start, need, out, W, H, fps)
            except Exception:
                if start > 0 and p.get("forced_start") is None:   # some sources can't seek: try from 0
                    start = 0.0
                    out = LIBRARY_DIR / f"{p['source']}_{p['id'].replace(':', '_')}_0_{int(need * 100)}.mp4"
                    fetch_one(p, start, need, out, W, H, fps)
                else:
                    raise
        return {"file": str(out), "start": start, "length": duration(out),
                **{x: p.get(x, "") for x in ("source", "id", "page_url", "credit", "title", "license")}}

    def work(job):
        """The pick first; if it can't be downloaded, the next ranked alternates (up to 3 tries)."""
        bid, p, need = job
        tries = [p] + [a for a in (p.get("alternates") or [])
                       if (a.get("source"), a.get("id")) != (p.get("source"), p.get("id"))]
        errors = []
        for n, cand in enumerate(tries[:3]):
            try:
                res = try_one(bid, cand, need)
                if n:
                    log(f"beat {bid}: first clip failed ({errors[0][:80]}); used ranked alternate #{n + 1}")
                    switched[bid] = dict(cand, alternates=p.get("alternates") or [])
                return bid, res
            except Exception as e:
                errors.append(str(e)[:160])
        log(f"beat {bid}: no clip could be downloaded ({' | '.join(errors)[:240]})")
        return bid, None

    switched = {}
    log(f"downloading {len(jobs)} clip segments")
    with ThreadPoolExecutor(6) as ex:
        for n, (bid, res) in enumerate(ex.map(work, jobs), 1):
            fetched[bid] = res
            if n % 10 == 0 or n == len(jobs):
                write_json(out_path, fetched)
                log(f"fetch {n}/{len(jobs)}", progress=n / max(len(jobs), 1))
    write_json(out_path, fetched)
    if switched:                                  # keep picks.json in line with what was really used
        picks.update(switched)
        write_json(proj / "picks.json", picks)
    missing = [b for b, v in fetched.items() if not v]
    if missing:
        log(f"fetch done; {len(missing)} beat(s) without footage (they repeat the previous clip): {', '.join(missing[:12])}")
    return fetched
