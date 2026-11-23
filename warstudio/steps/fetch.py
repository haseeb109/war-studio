"""Step 6: download only the needed seconds of each winning clip, normalized to 1080p30.

ffmpeg reads straight from the clip URL with -ss/-t, so only beat length + padding is pulled.
Files go to library/raw/ and are reused if the same source segment is picked again (phase-2 library data).
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ..config import LIBRARY_DIR
from ..util import read_json, write_json, run, duration, UA

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
    if not pick.get("duration"):
        pick["duration"] = float(d.get("duration") or 0)
    return pick


def choose_start(src_dur, need, used_offsets):
    if src_dur <= 0:
        return 0.0
    base = max(0.0, min(src_dur - need - 0.2, src_dur * 0.3))
    for k in range(6):
        cand = (base + k * need) % max(src_dur - need, 0.01) if src_dur > need else 0.0
        if all(abs(cand - u) > need for u in used_offsets):
            return round(cand, 2)
    return round(base, 2)


def fetch_one(pick, start, need, out, W, H, fps):
    vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},format=yuv420p"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-user_agent", UA, "-i", pick["video_url"],
         "-t", f"{need:.2f}", "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)],
        timeout=600)
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
        if not p or bid in fetched:
            continue
        end = vo_total if i == len(beats) - 1 else b["end"]
        need = (end - b["start"]) + PAD
        jobs.append((bid, p, need))

    def work(job):
        bid, p, need = job
        try:
            if p["source"] == "dvids" and not p.get("video_url"):
                resolve_dvids(p, keys.get("DVIDS_API_KEY"))
            k = f"{p['source']}:{p['id']}"
            with lock:
                start = choose_start(float(p.get("duration") or 0), need, used.setdefault(k, []))
                used[k].append(start)
            out = LIBRARY_DIR / f"{p['source']}_{p['id'].replace(':', '_')}_{int(start * 100)}_{int(need * 100)}.mp4"
            if not out.exists():
                fetch_one(p, start, need, out, W, H, fps)
            return bid, {"file": str(out), "start": start, "length": duration(out), **{x: p[x] for x in ("source", "id", "page_url", "credit", "title")}}
        except Exception as e:
            log(f"beat {bid}: fetch failed ({str(e)[:120]})")
            return bid, None

    log(f"downloading {len(jobs)} clip segments")
    with ThreadPoolExecutor(4) as ex:
        for n, (bid, res) in enumerate(ex.map(work, jobs), 1):
            fetched[bid] = res
            if n % 10 == 0 or n == len(jobs):
                write_json(out_path, fetched)
                log(f"fetch {n}/{len(jobs)}", progress=n / max(len(jobs), 1))
    write_json(out_path, fetched)
    return fetched
