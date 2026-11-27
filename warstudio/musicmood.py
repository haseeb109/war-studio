"""Listens to every music file in assets/music and decides its mood by itself (no manual sorting needed).

Per track (first ~2.5 min, mono 22 kHz) it measures:
  energy      - loudness (RMS dB) and how much it swells (dynamic range)
  pace        - note/hit onsets per second and a tempo estimate (BPM) from the onset envelope
  brightness  - spectral centroid (dark pads vs bright strings/brass)
  low end     - share of energy under 150 Hz (drones, booms)
  key colour  - major vs minor (chroma vs Krumhansl key profiles)
and scores the five moods: tension / action / somber / triumph / neutral.
Files inside assets/music/<mood>/ keep the mood of their folder (a manual override). Results are cached in
assets/music/.analysis.json (keyed by name + size + mtime), so each file is analysed once.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

MOODS = ["tension", "action", "somber", "triumph", "neutral"]
AUDIO_EXT = (".wav", ".mp3", ".aif", ".aiff", ".flac", ".m4a")
SR = 22050
VERSION = 1
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def decode(path, seconds=150, start=20):
    """Mono float32 at 22 kHz; skips the first 20 s (intros are often quiet) when the track is long enough."""
    def run(ss):
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(ss), "-t", str(seconds), "-i", str(path),
                            "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=180)
        return np.frombuffer(r.stdout, dtype=np.float32)
    x = run(start)
    if len(x) < SR * 20:
        x = run(0)
    return x


def features(x):
    if len(x) < SR * 5:
        raise ValueError("track too short")
    n, hop = 2048, 512
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
    rdb = 20 * np.log10(rms)
    loud = float(np.percentile(rdb, 90))
    dyn = float(np.percentile(rdb, 95) - np.percentile(rdb, 20))
    pw = mag ** 2
    tot = pw.sum(axis=1) + 1e-9
    centroid = float((pw * freqs).sum() / pw.sum())              # energy-weighted over the whole excerpt
    low = float(pw[:, freqs < 150].sum() / pw.sum())
    # onsets: sudden rises (>6 dB in ~23 ms) of the 1.5-8 kHz band - drum hits and plucks have them, pads don't
    band = (freqs > 1500) & (freqs < 8000)
    hf = pw[:, band].sum(axis=1)
    hdb = 10 * np.log10(hf + 1e-12)
    rise = np.diff(hdb, prepend=hdb[0])
    loud_enough = hf > 0.002 * tot.mean()
    peaks = (rise > 6) & loud_enough & (rise >= np.roll(rise, 1)) & (rise >= np.roll(rise, -1))
    flux = np.maximum(0, rise) * loud_enough
    secs = len(flux) * hop / SR
    onsets = float(peaks.sum() / max(secs, 1))
    # tempo: autocorrelation of the onset envelope between 60 and 180 BPM
    env = np.maximum(flux, 0)
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    fps = SR / hop
    lags = np.arange(int(fps * 60 / 180), int(fps * 60 / 60))
    lag = lags[int(np.argmax(ac[lags]))] if len(ac) > lags[-1] else lags[0]
    half = int(round(lag / 2))
    if 60 * fps / lag < 90 and half >= lags[0] and ac[half] > 0.6 * ac[lag]:
        lag = half                                                   # half-time pick -> real tempo
    bpm = float(60 * fps / lag)
    beat_strength = float(ac[lag] / (ac[0] + 1e-9)) if ac[0] > 0 else 0.0
    bpm = bpm if beat_strength > 0.05 else 0.0
    # key colour: chroma from 55-2000 Hz bins
    sel = (freqs > 55) & (freqs < 2000)
    pc = np.round(12 * np.log2(freqs[sel] / 440.0)).astype(int) % 12
    chroma = np.bincount(pc, weights=pw[:, sel].mean(axis=0), minlength=12)
    chroma = chroma / (chroma.sum() + 1e-9)
    corr = lambda prof: max(np.corrcoef(np.roll(prof, s), chroma)[0, 1] for s in range(12))
    major, minor = float(corr(MAJOR)), float(corr(MINOR))
    return {"loud_db": round(loud, 1), "dynamics_db": round(dyn, 1), "centroid_hz": round(centroid),
            "low_share": round(low, 3), "onsets_per_s": round(onsets, 2), "bpm": round(bpm), "beat": round(beat_strength, 2),
            "major": round(major, 2), "minor": round(minor, 2)}


def mood_scores(f):
    """Plain rules on the measured numbers -> a score per mood (highest wins)."""
    pace = min(1.0, f["onsets_per_s"] / 4.0)                      # 0 = held pads, 1 = busy percussion
    beat = min(1.0, f["beat"] / 0.5)
    fast = min(1.0, max(0.0, (f["bpm"] - 85) / 60)) if f["bpm"] else 0.0
    bright = min(1.0, max(0.0, (f["centroid_hz"] - 900) / 2200))
    loud = min(1.0, max(0.0, (f["loud_db"] + 30) / 18))
    low = min(1.0, f["low_share"] / 0.35)
    minor = max(0.0, f["minor"] - f["major"]) * 4
    major = max(0.0, f["major"] - f["minor"]) * 4
    swell = min(1.0, f["dynamics_db"] / 25)
    s = {
        "action": 1.2 * pace + 0.8 * beat + 0.6 * fast + 0.5 * loud,
        "tension": 0.9 * low + 0.7 * minor + 0.5 * (1 - bright) + 0.4 * beat * (1 - pace) + 0.3 * swell,
        "somber": 0.9 * (1 - pace) + 0.7 * minor + 0.6 * (1 - loud) + 0.4 * (1 - beat),
        "triumph": 1.0 * major + 0.7 * bright + 0.5 * loud + 0.5 * swell,
        "neutral": 1.3,
    }
    return {m: round(v, 2) for m, v in s.items()}


def analyse(path):
    f = features(decode(path))
    sc = mood_scores(f)
    return {"mood": max(sc, key=sc.get), "scores": sc, "features": f}


def all_tracks(music_dir):
    d = Path(music_dir)
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and not p.name.startswith("."):
            folder = p.parent.name if p.parent != d else ""
            out.append((p, folder if folder in MOODS else ""))
    return out


def analyse_library(music_dir, log=print):
    """{relative path: {mood, source: folder|auto, scores, features}} for every track; new files analysed now."""
    d = Path(music_dir)
    cache_f = d / ".analysis.json"
    try:
        cache = json.loads(cache_f.read_text())
    except Exception:
        cache = {}
    out, new = {}, 0
    for p, folder in all_tracks(d):
        rel = str(p.relative_to(d))
        st = p.stat()
        key = f"{VERSION}:{st.st_size}:{int(st.st_mtime)}"
        hit = cache.get(rel)
        if not hit or hit.get("key") != key:
            try:
                hit = dict(analyse(p), key=key)
                new += 1
                log(f"music: analysed {p.name} -> {hit['mood']} ({hit['features']['bpm']} bpm)")
            except Exception as e:
                hit = {"key": key, "mood": "neutral", "scores": {}, "features": {}, "error": str(e)[:120]}
        cache[rel] = hit
        out[rel] = dict(hit, mood=folder or hit["mood"], source="folder" if folder else "auto", file=str(p))
    if new or set(cache) - set(out):
        cache = {k: v for k, v in cache.items() if k in out}
        try:
            cache_f.write_text(json.dumps(cache, indent=1))
        except Exception:
            pass
    return out
