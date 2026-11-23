"""Step 7: build the video.

Per beat: take the fetched clip, apply slow drift + the footage-class grade, bake that beat's overlay in,
encode exactly the beat's frame count (frame counts come from the cumulative timeline, so audio/video
never drift). Segments are joined with stream copy, then the SFX/music mix is muxed and loudness-normalized.
"""
import json
from pathlib import Path

from .. import overlays as ovl
from .. import sfx
from ..util import read_json, write_json, run, video_encoder, encoder_args

FALLBACK_COLOR = "0x101418"


def segment_cmd(src, frames, fps, style, fclass, ov, out, enc):
    W, H = style["canvas"]["width"], style["canvas"]["height"]
    dur = frames / fps
    grade = style["grade"].get(fclass, style["grade"]["real"])
    s = style["footage"]["motion"]["drift_scale"] if style["footage"]["motion"].get("drift") else 1.0
    sw, sh = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    chain = (f"[0:v]scale={sw}:{sh},crop={W}:{H}:x='(iw-ow)*t/{max(dur, 0.1):.3f}':y='(ih-oh)/2',"
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
    rendered = ovl.render_all({k: v for k, v in plan.items()}, style, proj / "overlays", log, use_hf=use_hf)

    segdir = proj / "segments"
    segdir.mkdir(exist_ok=True)
    seg_files = []
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
        cmd = segment_cmd(src, frames, fps, style, spec.get(b["id"], {}).get("footage_class", "real"),
                          rendered.get(bid), out, enc)
        try:
            run(cmd, timeout=600)
        except RuntimeError as e:
            if enc != "libx264":
                log(f"segment {bid}: {enc} failed, retrying with libx264")
                enc = "libx264"
                run(segment_cmd(src, frames, fps, style, spec.get(b["id"], {}).get("footage_class", "real"),
                                rendered.get(bid), out, enc), timeout=600)
            else:
                raise
        if i % 10 == 0 or i == len(beats) - 1:
            log(f"render segment {i + 1}/{len(beats)}", progress=(i + 1) / len(beats))

    lst = proj / "segments.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_files))
    video = proj / "video_only.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)])

    log("mixing audio (voice + SFX + music)")
    sfx.ensure_defaults()
    events = sfx.plan_events(beats, specs, {int(k) for k in plan}, style)
    write_json(proj / "sfx_events.json", events)
    mixwav = proj / "mix.wav"
    sfx.mix(proj / "vo.wav", events, total, style, mixwav, settings.get("music_file", ""))

    final = proj / "final.mp4"
    lufs = style["audio"]["target_lufs"]
    af = loudnorm_filter(mixwav, lufs)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(mixwav),
         "-af", af, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
         "-movflags", "+faststart", str(final)], timeout=3600)
    log(f"final video written: {final.name}")
    return final
