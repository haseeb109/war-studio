"""Sound design: built-in synthesized SFX, event planning per beat, and the final audio mix.

Your own SFX files win: drop .wav/.mp3 into assets/sfx/<category>/ (e.g. assets/sfx/explosion/).
If a category folder is empty, synthesized defaults in assets/sfx/_generated/<category>/ are used.
Categories: ui_click whoosh bass_hit riser launch explosion flyby gunfire flares radar radio_static alarm impact
"""
import random
import subprocess
from pathlib import Path

import numpy as np

from .config import SFX_DIR, MUSIC_DIR

SR = 48000
CATS = ["ui_click", "whoosh", "bass_hit", "riser", "launch", "explosion", "flyby", "gunfire", "flares",
        "radar", "radio_static", "alarm", "impact"]
AUDIO_EXT = (".wav", ".mp3", ".aif", ".aiff", ".flac", ".m4a")


# ---------- synthesis ----------

def _t(sec):
    return np.arange(int(sec * SR)) / SR


def _lp(x, k):
    k = max(1, int(k))
    return np.convolve(x, np.ones(k) / k, mode="same")


def _env(n, a, d):
    e = np.ones(n)
    na, nd = int(a * SR), int(d * SR)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nd:
        e[-nd:] *= np.linspace(1, 0, nd) ** 2
    return e


def _norm(x, peak=0.9):
    m = np.max(np.abs(x)) or 1
    return (x / m * peak).astype(np.float32)


def synth(cat, seed=0):
    rng = np.random.default_rng(seed)
    noise = lambda s: rng.standard_normal(int(s * SR))
    if cat == "ui_click":
        t = _t(0.06)
        x = np.sin(2 * np.pi * (1800 + 300 * seed) * t) * np.exp(-t * 70)
    elif cat == "whoosh":
        n = noise(0.9)
        x = np.concatenate([_lp(n[i:i + 4800], 30 - i // 1800) for i in range(0, len(n), 4800)])[:len(n)]
        x *= np.sin(np.linspace(0, np.pi, len(x))) ** 2
    elif cat == "bass_hit":
        t = _t(1.8)
        f = 55 * np.exp(-t * 1.5) + 32
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) + _lp(noise(1.8), 40) * np.exp(-t * 18) * 0.6
    elif cat == "riser":
        t = _t(1.6)
        x = _lp(noise(1.6), 12) * (t / 1.6) ** 2 + np.sin(2 * np.pi * (200 + 600 * (t / 1.6) ** 2) * t) * (t / 1.6) ** 3 * 0.4
    elif cat == "launch":
        t = _t(2.6)
        x = _lp(noise(2.6), 25) * _env(len(t), 0.03, 2.2) + _lp(noise(2.6), 200) * np.exp(-t * 1.2) * 2
    elif cat == "explosion":
        t = _t(3.2)
        x = _lp(noise(3.2), 120) * np.exp(-t * 1.6) * 3 + _lp(noise(3.2), 15) * np.exp(-t * 6) + np.sin(2 * np.pi * 40 * t) * np.exp(-t * 2)
    elif cat == "flyby":
        t = _t(3.0)
        x = _lp(noise(3.0), 18) * np.exp(-((t - 1.4) ** 2) / 0.35)
    elif cat == "gunfire":
        x = np.zeros(int(1.2 * SR))
        for k in range(rng.integers(5, 9)):
            s = int(k * 0.11 * SR + rng.integers(0, 600))
            b = noise(0.08) * np.exp(-_t(0.08) * 60)
            x[s:s + len(b)] += b[:len(x) - s]
    elif cat == "flares":
        x = np.zeros(int(1.6 * SR))
        for k in range(6):
            s = int(k * 0.22 * SR)
            b = _lp(noise(0.25), 6) * np.exp(-_t(0.25) * 20)
            x[s:s + len(b)] += b[:len(x) - s]
    elif cat == "radar":
        x = np.zeros(int(0.9 * SR))
        for k in range(3):
            s = int(k * 0.28 * SR)
            b = np.sin(2 * np.pi * (1000 + 200 * seed) * _t(0.12)) * _env(int(0.12 * SR), 0.005, 0.03)
            x[s:s + len(b)] += b
    elif cat == "radio_static":
        t = _t(0.7)
        x = _lp(noise(0.7), 3) * 0.5 * _env(len(t), 0.01, 0.1)
        x[:400] += np.sin(2 * np.pi * 2500 * t[:400])
    elif cat == "alarm":
        t = _t(1.6)
        x = np.sign(np.sin(2 * np.pi * np.where((t * 2.5).astype(int) % 2 == 0, 700, 900) * t)) * 0.4
    elif cat == "impact":
        t = _t(0.8)
        x = np.sin(2 * np.pi * 420 * t) * np.exp(-t * 9) + noise(0.8) * np.exp(-t * 30)
    else:
        raise ValueError(cat)
    return _norm(x)


def write_wav(path, x, sr=SR):
    import wave
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def ensure_defaults():
    for c in CATS:
        d = SFX_DIR / "_generated" / c
        for v in range(3):
            p = d / f"{c}_{v}.wav"
            if not p.exists():
                write_wav(p, synth(c, v))


def sfx_files(cat):
    user = sorted(p for p in (SFX_DIR / cat).glob("*") if p.suffix.lower() in AUDIO_EXT) if (SFX_DIR / cat).exists() else []
    return user or sorted((SFX_DIR / "_generated" / cat).glob("*.wav"))


# ---------- decode / mix ----------

def load_audio(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def db(x):
    return 10 ** (x / 20)


def keyword_hit(word, text):
    """Whole-word match; a trailing * allows any ending (detonat* -> detonated, detonation)."""
    import re
    stem = word.rstrip("*").lower()
    return re.search(r"\b" + re.escape(stem) + ("" if word.endswith("*") else r"\b"), text) is not None


def plan_events(beats, specs, overlays, style):
    """Decide which SFX play where. Returns list of {t, cat, gain_db}."""
    sc = style["sfx"]
    gains = sc["gains_db"]
    spec = {s["id"]: s for s in specs}
    ev, last = [], {}
    first_beat_of_sentence = set()
    seen_sid = set()
    for b in beats:
        if b["sid"] not in seen_sid:
            seen_sid.add(b["sid"])
            first_beat_of_sentence.add(b["id"])

    def add(t, cat, cooldown=2.5, offset_db=0.0):
        if t < 0 or t - last.get(cat, -99) < cooldown:
            return
        last[cat] = t
        ev.append({"t": round(t, 3), "cat": cat, "gain_db": gains.get(cat, -12) + offset_db})

    for b in beats:
        s = spec.get(b["id"], {})
        t0 = b["start"]
        low = b["text"].lower()
        if b["id"] in overlays:
            add(t0 + 0.05, "ui_click", cooldown=1.0)
        if s.get("footage_class") == "map":
            add(t0 - 0.35, "whoosh", cooldown=3)
            add(t0, "bass_hit", cooldown=6, offset_db=-4)
        if b["id"] in first_beat_of_sentence and any(low.startswith(w) for w in sc["pivot_words"]):
            add(t0 - 1.5, "riser", cooldown=10)
            add(t0, "bass_hit", cooldown=6)
        actions = list(s.get("sfx") or [])
        for cat, words in sc["keywords"].items():
            if cat not in actions and any(keyword_hit(w, low) for w in words):
                actions.append(cat)
        for cat in actions[:1]:
            add(t0 + 0.12, cat, cooldown=3.0)
    return ev


def mix(vo_wav, events, total, style, out_wav, music_file=""):
    sc = style["sfx"]
    vo = load_audio(vo_wav)
    n = int(total * SR) + SR
    out = np.zeros(n, dtype=np.float32)
    out[:len(vo)] += vo[:n] * db(style["audio"].get("vo_gain_db", 0))
    rnd = random.Random(7)
    cache = {}
    explosions = []
    for e in events:
        files = sfx_files(e["cat"])
        if not files:
            continue
        f = rnd.choice(files)
        if f not in cache:
            cache[f] = load_audio(f)
        x = cache[f] * db(e["gain_db"])
        s = int(e["t"] * SR)
        end = min(n, s + len(x))
        if end > s:
            out[s:end] += x[:end - s]
        if e["cat"] == "explosion":
            explosions.append(e["t"])
    mfile = music_file or next((str(p) for p in sorted(MUSIC_DIR.glob("*")) if p.suffix.lower() in AUDIO_EXT), "")
    if mfile and Path(mfile).exists():
        m = load_audio(mfile)
        if len(m):
            m = np.tile(m, int(np.ceil(n / len(m))))[:n]
            env = np.abs(np.concatenate([vo, np.zeros(max(0, n - len(vo)), dtype=np.float32)])[:n])
            win = int(0.25 * SR)
            env = np.convolve(env, np.ones(win) / win, mode="same")
            talking = np.clip(env / (np.percentile(env, 90) * 0.25 + 1e-6), 0, 1)
            gain = db(sc["music_base_db"]) * (1 - talking) + db(sc["music_base_db"] + sc["music_duck_db"]) * talking
            for t in explosions:
                a, b2 = int(t * SR), int((t + sc["explosion_music_dropout_seconds"]) * SR)
                gain[a:b2] *= 0.05
            out += m * gain.astype(np.float32)
    peak = np.max(np.abs(out)) or 1
    if peak > 0.98:
        out *= 0.98 / peak
    write_wav(out_wav, out)
    return out_wav
