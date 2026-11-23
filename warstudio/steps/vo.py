"""Step 1: script -> sentences -> ai33 voice per sentence -> timeline (vo.json) + vo.wav.

Generating one audio file per sentence gives exact sentence start/end times without a speech
recognizer: the timeline is just the running sum of measured mp3 durations.
"""
import re
import subprocess
from pathlib import Path

from .. import ai33
from ..util import duration, run, write_json, read_json

ABBREV = ["U.S.", "U.K.", "Mr.", "Mrs.", "Dr.", "St.", "No.", "Mk.", "Lt.", "Col.", "Gen.", "Sgt.", "Capt.", "Adm.",
          "Cmdr.", "vs.", "e.g.", "i.e.", "approx.", "Jr.", "Sr.", "Inc.", "Co.", "a.m.", "p.m."]


def split_sentences(script):
    text = re.sub(r"\s+", " ", script.replace("’", "'")).strip()
    protected = text
    for i, a in enumerate(ABBREV):
        protected = protected.replace(a, a.replace(".", f"\u0000{i}\u0000"))
    parts = re.split(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])", protected)
    out = []
    for p in parts:
        for i, a in enumerate(ABBREV):
            p = p.replace(a.replace(".", f"\u0000{i}\u0000"), a)
        p = p.strip()
        if p:
            out.append(p)
    return out


def _dummy_tts(items):
    """Silent audio sized like speech (~2.6 words/s). Used for tests and dry runs without credits."""
    for out, text in items:
        secs = max(0.8, len(text.split()) / 2.6)
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", f"{secs:.2f}", "-c:a", "libmp3lame", "-q:a", "6", str(out)])
    return 0


def run_step(proj, keys, settings, voice_id, log, dummy=False):
    proj = Path(proj)
    out_json = proj / "vo.json"
    if out_json.exists() and (proj / "vo.wav").exists():
        log("VO already done, skipping")
        return read_json(out_json)
    script = (proj / "script.txt").read_text(encoding="utf-8")
    sents = split_sentences(script)
    if not sents:
        raise ValueError("script is empty")
    vdir = proj / "vo"
    vdir.mkdir(exist_ok=True)
    items = [(vdir / f"s{i:04d}.mp3", s) for i, s in enumerate(sents) if not (vdir / f"s{i:04d}.mp3").exists()]
    chars = sum(len(s) for s in sents)
    log(f"{len(sents)} sentences, {chars} characters; generating {len(items)} voice files")
    credits = 0
    if items:
        if dummy:
            credits = _dummy_tts(items)
        else:
            if not keys.get("AI33_API_KEY"):
                raise RuntimeError("AI33_API_KEY missing (Settings tab)")
            credits = ai33.tts_many(keys["AI33_API_KEY"], items, voice_id, settings.get("tts_model"),
                                    int(settings.get("tts_concurrency", 4)),
                                    on_progress=lambda d, n: log(f"voice {d}/{n}", progress=d / max(n, 1)))
    timeline, t = [], 0.0
    for i, s in enumerate(sents):
        f = vdir / f"s{i:04d}.mp3"
        d = duration(f)
        timeline.append({"sid": i, "text": s, "start": round(t, 3), "end": round(t + d, 3), "file": str(f.relative_to(proj))})
        t += d
    lst = proj / "vo_concat.txt"
    lst.write_text("".join(f"file '{(proj / x['file']).resolve()}'\n" for x in timeline))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-ar", "48000", "-ac", "1", str(proj / "vo.wav")])
    data = {"sentences": timeline, "total": round(t, 3), "chars": chars, "credits": credits}
    write_json(out_json, data)
    log(f"VO done: {t:.1f}s, ai33 credits {credits}")
    return data
