"""Step 7: build the video.

Per beat: take the fetched clip, apply slow drift + the footage-class grade, bake that beat's overlay in,
encode exactly the beat's frame count (frame counts come from the cumulative timeline, so audio/video
never drift). Segments are joined with stream copy, then the sound mix (sfx.py: on-screen SFX, footage sound,
intensity-shaped music bed) is muxed and loudness-normalized.
"""
import json
from pathlib import Path

from .. import overlays as ovl
from .. import sfx
from ..util import read_json, write_json, run, video_encoder, encoder_args

FALLBACK_COLOR = "0x101418"


def segment_cmd(src, frames, fps, style, fclass, ov, out, enc, crop_bottom=0.0):
    W, H = style["canvas"]["width"], style["canvas"]["height"]
    dur = frames / fps
    grade = style["grade"].get(fclass, style["grade"]["real"])
    s = style["footage"]["motion"]["drift_scale"] if style["footage"]["motion"].get("drift") else 1.0
    if fclass == "map_clip":                      # a rendered map already moves and has its locked colours
        grade, s = "null", 1.0
    y = "(ih-oh)/2"
    if crop_bottom and src:
        # a lower-third sits in the bottom `crop_bottom` of the clip: zoom in, anchored to the top, so it leaves the frame
        s = max(s, 1.0 / (1.0 - min(float(crop_bottom), 0.2)))
        y = "0"
    sw, sh = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    chain = (f"[0:v]scale={sw}:{sh},crop={W}:{H}:x='(iw-ow)*t/{max(dur, 0.1):.3f}':y='{y}',"
             f"{grade},tpad=stop_mode=clone:stop_duration={dur:.2f},trim=end_frame={frames},setpts=PTS-STARTPTS,fps={fps}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if src:
        cmd += ["-i", str(src)]
    else:
        cmd += ["-f", "lavfi", "-i", f"color=c={FALLBACK_COLOR}:s={W}x{H}:r={fps}:d={dur + 0.5:.2f}"]
    if ov:
        if ov["type"] == "webm":
            cmd += ["-c:v", "libvpx-vp9", "-i", ov["path"]]
            ovchain = "[1:v]format=rgba,setpts=PTS-STARTPTS[o]"
        else:
            fade_out = max(0.3, ov["duration"] - 0.2)
            cmd += ["-loop", "1", "-t", f"{ov['duration']:.2f}", "-i", ov["path"]]
            ovchain = (f"[1:v]format=rgba,fade=t=in:st=0:d=0.18:alpha=1,"
                       f"fade=t=out:st={fade_out:.2f}:d=0.2:alpha=1,setpts=PTS-STARTPTS[o]")
        fg = f"{chain}[b];{ovchain};[b][o]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[v]"
    else:
        fg = f"{chain},format=yuv420p[v]"
    cmd += ["-filter_complex", fg, "-map", "[v]", "-frames:v", str(frames), "-an", "-r", str(fps)]
    cmd += encoder_args(enc) + ["-g", str(fps * 2), str(out)]
    return cmd


def loudnorm_filter(wav, lufs):
    """Two-pass loudnorm: measure first, then apply linear normalization to hit the target."""
    import subprocess
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af",
                        f"loudnorm=I={lufs}:TP=-1.0:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    try:
        m = json.loads(p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1])
        if float(m["input_i"]) < -70:              # (near) silent mix: nothing to normalise
            return "anull"
        return (f"loudnorm=I={lufs}:TP=-1.0:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    except Exception:
        return f"loudnorm=I={lufs}:TP=-1.0:LRA=11"


def run_step(proj, settings, style, log, use_hf=True):
    proj = Path(proj)
    beats = read_json(proj / "beats.json")
    specs = read_json(proj / "specs.json")
    fetched = read_json(proj / "fetched.json", {}) or {}
    vo = read_json(proj / "vo.json")
    fps = style["canvas"]["fps"]
    total = vo["total"]
    enc = video_encoder(settings.get("encoder", "auto"))
    spec = {s["id"]: s for s in specs}

    plan = read_json(proj / "overlays.json")
    if plan is None:
        plan = {str(k): v for k, v in ovl.plan(beats, specs, style).items()}
        write_json(proj / "overlays.json", plan)
    use_hf = use_hf and str(settings.get("hyperframes", True)).lower() not in ("false", "0", "no")
    rendered = ovl.render_all({k: v for k, v in plan.items()}, style, proj / "overlays", log, use_hf=use_hf)

    segdir = proj / "segments"
    segdir.mkdir(exist_ok=True)
    seg_files = []
    jobs = []
    last_src = None
    for i, b in enumerate(beats):
        bid = str(b["id"])
        start_f = round(b["start"] * fps)
        end_f = round((total if i == len(beats) - 1 else beats[i + 1]["start"]) * fps)
        frames = max(1, end_f - start_f)
        f = fetched.get(bid)
        src = f["file"] if f and Path(f["file"]).exists() else last_src
        if f and Path(f["file"]).exists():
            last_src = f["file"]
        out = segdir / f"seg_{int(bid):05d}.mp4"
        seg_files.append(out)
        if out.exists():
            continue
        crop = (f or {}).get("crop_bottom", 0.0) if f and Path(f["file"]).exists() else 0.0
        jobs.append((bid, out, segment_cmd(src, frames, fps, style, ("map_clip" if (f or {}).get("source") == "map" and src == (f or {}).get("file")
                                            else spec.get(b["id"], {}).get("footage_class", "real")),
                                           rendered.get(bid), out, enc, crop)))

    # several segments at once: the Mac's media engine and CPU cores have room for 4
    import threading
    from concurrent.futures import ThreadPoolExecutor
    conc = max(1, int(settings.get("render_concurrency", 4)))
    lock, done, failed = threading.Lock(), [0], []

    def one(job):
        bid, out, cmd = job
        if failed:
            return
        try:
            run(cmd, timeout=600)
        except RuntimeError as e:
            Path(out).unlink(missing_ok=True)          # never leave a half-written segment behind
            with lock:
                failed.append((bid, e))
            return
        with lock:
            done[0] += 1
            n = done[0]
        if n % 10 == 0 or n == len(jobs):
            log(f"render segment {n}/{len(jobs)} ({enc}, {conc} at a time)", progress=n / max(len(jobs), 1))

    if jobs:
        with ThreadPoolExecutor(conc) as ex:
            list(ex.map(one, jobs))
    if failed:
        bid, e = failed[0]
        if enc != "libx264":
            log(f"segment {bid}: {enc} failed ({str(e)[-160:]}); switching to libx264 and re-rendering all segments")
            from ..config import save_settings
            save_settings({"encoder": "libx264"})
            for p in segdir.glob("seg_*.mp4"):
                p.unlink()
            return run_step(proj, dict(settings, encoder="libx264"), style, log, use_hf=use_hf)
        raise e

    lst = proj / "segments.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_files))
    video = proj / "video_only.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)])

    log("mixing audio (voice + clip sound + chosen music)" if sfx.clip_audio_only(style) else "mixing audio (voice + SFX + footage sound + music)")
    sfx.ensure_defaults()
    picks = read_json(proj / "picks.json", {}) or {}
    events = read_json(proj / "sfx_events.json")
    if sfx.needs_replan(events):
        log("sfx: this project still has sound events from the old rules - planning them again with the new sound design")
        events = None
    if sfx.clip_audio_only(style):
        events = []                               # user's choice: no sound effects, no music
    elif events is None:
        events = sfx.plan_events(beats, specs, plan, style, picks)
        write_json(proj / "sfx_events.json", events)
    why = {}
    for e in events:
        k = str(e.get("rule") or "manual").split(":")[0]
        why[k] = why.get(k, 0) + 1
    mins = max(total / 60, 0.01)
    log(f"sfx: {len(events)} sounds ({len(events) / mins:.1f}/min) - " + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
    levels = sfx.plan_music(beats, specs, plan, style, picks)
    footage = sfx.plan_footage_audio(beats, specs, picks, fetched, style, total, settings)
    write_json(proj / "sound_plan.json", {"music_levels": levels, "footage_audio": footage})
    mixwav = proj / "mix.wav"
    bgm = (read_json(proj / "project.json", {}) or {}).get("bgm") or ""
    music = str(sfx.MUSIC_DIR / bgm) if bgm and sfx.clip_audio_only(style) else settings.get("music_file", "")
    sfx.mix(proj / "vo.wav", events, total, style, mixwav, music,
            levels=levels, footage=footage, seed=proj.name, log=log)

    final = proj / "final.mp4"
    lufs = style["audio"]["target_lufs"]
    af = loudnorm_filter(mixwav, lufs)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(mixwav),
         "-af", af, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
         "-movflags", "+faststart", str(final)], timeout=3600)
    log(f"final video written: {final.name}")
    return final
