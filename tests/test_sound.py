"""Sound design v2: on-screen SFX, density, moments, music intensity curve, footage sound."""
import subprocess

import numpy as np

from warstudio import sfx
from warstudio.config import load_style
from warstudio.steps import specs as specs_step

STYLE = load_style("beyond_military")
SR = sfx.SR


def B(i, start, text="the jets hold", end=None, **kw):
    return dict(id=i, sid=i, text=text, start=start, end=end if end is not None else start + 2.5, **kw)


def S(i, visual="fighter jet parked on the ramp", **kw):
    return dict(id=i, visual=visual, footage_class=kw.pop("footage_class", "real"), scene=kw.pop("scene", "s"), **kw)


def cats(ev, bid=None):
    return [e["cat"] for e in ev if bid is None or e["bid"] == bid]


# ---------------------------------------------------------------- what is on screen

def test_sfx_follow_the_shot_not_the_words():
    beats = [B(0, 100, "and then it explodes"), B(1, 110, "the crew waits"), B(2, 120, "nothing happens yet")]
    sp = [S(0, "two fighter jets parked at dusk"), S(1, "huge explosion and fireball on the ground", scene="b"),
          S(2, "a missile leaves the rail of a fighter jet", scene="c")]
    ev = sfx.plan_events(beats, sp, {}, STYLE)
    assert "explosion" not in cats(ev, 0)                       # the word "explodes" no longer triggers anything
    assert {"explosion", "bass_hit"} <= set(cats(ev, 1))        # the fireball on screen does
    assert "launch" in cats(ev, 2)


def test_library_action_tag_wins_over_visual_line():
    beats = [B(0, 100)]
    sp = [S(0, "fighter jet parked on the ramp")]
    picks = {"0": {"source": "library", "action": "takeoff", "title": "F-15E takes off with afterburner"}}
    ev = sfx.plan_events(beats, sp, {}, STYLE, picks)
    assert cats(ev, 0) == ["jet_roar"]
    assert ev[0]["rule"] == "action:takeoff"


def test_helicopters_rotor_drones_silent_big_guns_impact():
    assert sfx.sounds_for("flying", "black hawk helicopter flying low") == [("rotor", 0.0)]
    assert sfx.sounds_for("flying", "mq-9 reaper drone in flight") == []
    assert [c for c, _ in sfx.sounds_for("firing", "m777 howitzer firing")] == ["impact", "bass_hit"]
    assert [c for c, _ in sfx.sounds_for("firing", "himars firing rockets at night")] == ["launch"]
    assert [c for c, _ in sfx.sounds_for("firing", "marines firing rifles")] == ["gunfire"]
    assert ("flares", 0.1) in sfx.sounds_for("flying", "c-130 releasing flares")


def test_overlay_blip_map_whoosh_comms_and_warning():
    beats = [B(0, 100, "at 28,000 feet"), B(1, 110, "the map shows the strait"), B(2, 120, "his wingman calls it"),
             B(3, 130, "the radar warning receiver screams")]
    sp = [S(0), S(1, "map of the strait of hormuz", footage_class="map", scene="m"), S(2, scene="w"), S(3, scene="x")]
    ov = {"0": {"kind": "measure"}, "2": {"kind": "label"}}
    ev = sfx.plan_events(beats, sp, ov, STYLE)
    assert cats(ev, 0) == ["ui_click"]                          # a number on screen: only the blip
    assert {"whoosh", "bass_hit"} <= set(cats(ev, 1))
    w = [e for e in ev if e["bid"] == 1 and e["cat"] == "whoosh"][0]
    assert w["t"] < 110                                         # the whoosh peaks on the cut
    assert "radio_static" in cats(ev, 2) and "ui_click" in cats(ev, 2)
    assert "radar" in cats(ev, 3)


def test_moments_riser_ends_on_the_hit_and_cliffhanger_chord():
    beats = [B(0, 100, "but unknown to them"), B(1, 120, "the second missile hits"), B(2, 160, "and that was only the start")]
    sp = [S(0), S(1, scene="a", moment="attack"), S(2, scene="b", moment="cliffhanger")]
    ev = sfx.plan_events(beats, sp, {}, STYLE)
    assert "riser" not in cats(ev, 0)                           # pivot words alone: no riser any more
    r = [e for e in ev if e["cat"] == "riser"][0]
    assert r["bid"] == 1 and r["hit"] == 120 and r["align"] == "end"
    assert "bass_hit" in cats(ev, 1)
    assert "drone_chord" in cats(ev, 2)


def test_quiet_number_beats_and_intensity_one():
    beats = [B(0, 100, "it costs 80 million dollars"), B(1, 110, "the range is long")]
    sp = [S(0, "huge explosion fireball"), S(1, "huge explosion fireball", scene="z", intensity=1)]
    ev = sfx.plan_events(beats, sp, {}, STYLE)
    assert ev == []


def test_density_hook_busier_than_body_and_capped():
    beats, sp, t = [], [], 0.0
    for i in range(240):                                        # 10 minutes of 2.5 s cuts, a new scene every 4
        beats.append(B(i, t, "the jets push on"))
        sp.append(S(i, "two fighter jets flying in formation" if i % 2 else "radar operators at consoles",
                    scene=f"s{i // 4}"))
        t += 2.5
    ev = sfx.plan_events(beats, sp, {}, STYLE)
    hook = len({e["bid"] for e in ev if e["t"] < 60})
    for m in range(1, 10):
        n = len({e["bid"] for e in ev if m * 60 <= e["t"] < (m + 1) * 60})
        assert n <= STYLE["sfx"]["max_per_minute"]["body"] + 2
        assert n < hook
    assert hook >= 12
    assert all(e["rule"] for e in ev)


def test_same_category_cooldown():
    beats = [B(i, 100 + 2.5 * i, "the jets push on") for i in range(4)]
    sp = [S(i, "fighter jet flying low", scene="one") for i in range(4)]
    ev = sfx.plan_events(beats, sp, {}, STYLE)
    t = [e["t"] for e in ev if e["cat"] == "flyby"]
    assert all(b - a >= STYLE["sfx"]["cooldown_seconds"]["flyby"] for a, b in zip(t, t[1:]))


def test_replan_beat_after_clip_swap_and_old_events():
    beats = [B(0, 100), B(1, 110)]
    sp = [S(0, "fighter jet flying"), S(1)]
    ev = sfx.plan_events(beats, sp, {"0": {"kind": "label"}}, STYLE)
    assert "flyby" in cats(ev, 0)
    picks = {"0": {"source": "library", "action": "explosion"}}
    ev2 = sfx.replan_beat(ev, beats, sp, STYLE, picks, 0)
    assert "flyby" not in cats(ev2, 0) and "explosion" in cats(ev2, 0) and "ui_click" in cats(ev2, 0)
    assert sfx.needs_replan([{"t": 1, "cat": "ui_click", "gain_db": -16}])
    assert not sfx.needs_replan(ev) and not sfx.needs_replan([])


# ---------------------------------------------------------------- music

def test_music_levels():
    beats = [B(0, 0, "it begins"), B(1, 70, "it weighs 30 tons"), B(2, 80, "the strike lands"), B(3, 90, "they wait"),
             B(4, 100, "and wait")]
    sp = [S(0, intensity=1), S(1), S(2, moment="attack"), S(3, intensity=2), S(4, intensity=2)]
    lv = sfx.plan_music(beats, sp, {}, STYLE)
    assert lv == [[0.0, 2], [70.0, 1], [80.0, 3], [90.0, 2]]   # hook >= 2, numbers 1, attack 3


def test_music_does_not_pump_from_shot_to_shot():
    beats = [B(i, 100 + 2.5 * i) for i in range(40)]
    sp = [S(i, "huge explosion fireball" if i % 2 else "fighter jet parked", intensity=None) for i in range(40)]
    sp[20]["moment"] = "attack"
    lv = sfx.plan_music(beats, sp, {}, STYLE)
    starts = [t for t, _ in lv] + [beats[-1]["end"]]
    assert all(b - a >= STYLE["music"]["min_hold_seconds"] for a, b in zip(starts[1:], starts[2:]))
    at = 100 + 2.5 * 20
    assert [l for t, l in lv if t <= at][-1] == 3               # the attack is loud


def _tone(path, sec, f=220.0):
    t = np.arange(int(sec * SR)) / SR
    sfx.write_wav(path, (np.sin(2 * np.pi * f * t) * 0.5).astype(np.float32))


def _rms(x, a, b):
    seg = x[int(a * SR):int(b * SR)].astype(np.float64)
    return float(np.sqrt(np.mean(seg ** 2)))


def test_music_ducks_under_voice_and_follows_intensity(tmp_path):
    vo = np.zeros(60 * SR, dtype=np.float32)
    t = np.arange(10 * SR) / SR
    vo[5 * SR:15 * SR] = np.sin(2 * np.pi * 150 * t) * 0.3       # voice 5-15 s, silence elsewhere
    sfx.write_wav(tmp_path / "vo.wav", vo)
    _tone(tmp_path / "m.wav", 20.0, 440.0)                       # 20 s track -> chained to fill 60 s
    out = tmp_path / "mix.wav"
    sfx.mix(tmp_path / "vo.wav", [], 60.0, STYLE, out, music_file=str(tmp_path / "m.wav"),
            levels=[[0.0, 2], [30.0, 1], [42.0, 3]])
    x = sfx.load_audio(out)
    music = x.copy()
    music[:len(vo)] -= vo[:len(music)]
    gap2, talk2 = _rms(music, 20, 28), _rms(music, 8, 12)
    calm, action = _rms(music, 34, 40), _rms(music, 46, 52)
    assert talk2 < gap2 * 0.7                                    # ducked under the voice
    assert calm < gap2 * 0.75 and action > gap2 * 1.15           # intensity curve
    assert _rms(music, 58.5, 59.5) < gap2 * 0.5                  # fade-out at the end
    assert _rms(music, 0, 60) > 0.005                            # no hole where the track loops


def test_generated_bed_when_no_tracks_and_off_switch(tmp_path, monkeypatch):
    sfx.write_wav(tmp_path / "vo.wav", np.zeros(10 * SR, dtype=np.float32))
    monkeypatch.setattr(sfx, "MUSIC_DIR", tmp_path / "nomusic")
    sfx.mix(tmp_path / "vo.wav", [], 10.0, STYLE, tmp_path / "a.wav")
    assert _rms(sfx.load_audio(tmp_path / "a.wav"), 2, 6) > 0.01
    st = dict(STYLE, music=dict(STYLE["music"], fallback="none"))
    sfx.mix(tmp_path / "vo.wav", [], 10.0, st, tmp_path / "b.wav")
    assert _rms(sfx.load_audio(tmp_path / "b.wav"), 2, 6) < 1e-4


def test_riser_ends_exactly_on_the_hit(tmp_path, monkeypatch):
    sfx.ensure_defaults()
    sfx.write_wav(tmp_path / "vo.wav", np.zeros(8 * SR, dtype=np.float32))
    st = dict(STYLE, music=dict(STYLE["music"], fallback="none"))
    sfx.mix(tmp_path / "vo.wav", [{"t": 3.0, "cat": "riser", "gain_db": -6, "hit": 5.0, "align": "end"}], 8.0, st,
            tmp_path / "r.wav")
    x = np.abs(sfx.load_audio(tmp_path / "r.wav"))
    nz = np.nonzero(x > 1e-4)[0]
    assert abs(nz[-1] / SR - 5.0) < 0.05 and nz[0] / SR > 2.5


def test_synth_all_categories_and_bed_loops_cleanly():
    for c in sfx.CATS:
        x = sfx.synth(c)
        assert len(x) > 1000 and np.abs(x).max() <= 0.91
    full, dark, pulse = sfx.synth_bed()
    assert len(full) == int(sfx.BED_LOOP * SR)
    jump = abs(float(full[-1]) - float(full[0]))
    assert jump < 0.05                                           # whole cycles: no click at the loop point


# ---------------------------------------------------------------- footage sound

def _speech(sec, seed=0):
    r = np.random.default_rng(seed)
    n, x, t = int(sec * SR), np.zeros(int(sec * SR)), 0.0
    while t < sec:
        d = r.uniform(0.12, 0.3)
        s, L = int(t * SR), int(d * SR)
        tt = np.arange(L) / SR
        f0 = r.uniform(110, 200)
        syl = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 15)) * np.sin(np.pi * tt / d) ** 2
        x[s:s + L] += syl[:n - s]
        t += d + r.uniform(0.02, 0.12)
    return x


def test_speech_detector():
    assert sfx.speech_like(_speech(3.0).astype(np.float32))
    for c in ("jet_roar", "flyby", "explosion", "gunfire", "rotor", "launch"):
        x = sfx.synth(c).astype(np.float64)
        assert not sfx.speech_like(np.pad(x, (0, max(0, 3 * SR - len(x)))).astype(np.float32)), c


def _clip(path, audio=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=30:d=3"]
    if audio is not None:
        wav = path.with_suffix(".wav")
        sfx.write_wav(wav, audio.astype(np.float32))
        cmd += ["-i", str(wav), "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", str(path)], check=True)
    return str(path)


def test_plan_footage_audio_filters():
    beats = [B(0, 0, end=3), B(1, 3, end=6), B(2, 6, end=9), B(3, 9, end=12, text="it costs 9 million")]
    sp = [S(0, "fighter jet takes off"), S(1, "radar operators at consoles"), S(2, "fighter jet takes off"),
          S(3, "fighter jet takes off")]
    fetched = {str(i): {"file": __file__, "title": "F-15E B-roll", "source": "dvids"} for i in range(4)}
    fetched["2"]["title"] = "Interview with the pilot"
    got = sfx.plan_footage_audio(beats, sp, {}, fetched, STYLE, 12.0)
    # every shot keeps its sound: takeoff = action, console + numbers = quiet ambience, interview = left out
    assert [(g["bid"], g["kind"]) for g in got] == [(0, "action"), (1, "ambient"), (3, "ambient")]
    assert got[0]["dur"] == 3.0
    only_action = dict(STYLE, footage_audio=dict(STYLE["footage_audio"], all_shots=False))
    assert [g["bid"] for g in sfx.plan_footage_audio(beats, sp, {}, fetched, only_action, 12.0)] == [0]
    assert sfx.plan_footage_audio(beats, sp, {}, fetched, STYLE, 12.0, {"footage_audio": False}) == []


def test_footage_sound_mixed_and_speech_skipped(tmp_path):
    rng = np.random.default_rng(0)
    engine = sfx._lp(rng.standard_normal(3 * SR), 6) * 0.3
    files = [_clip(tmp_path / "engine.mp4", engine), _clip(tmp_path / "talk.mp4", _speech(3.0) * 0.3),
             _clip(tmp_path / "mute.mp4")]
    sfx.write_wav(tmp_path / "vo.wav", np.zeros(10 * SR, dtype=np.float32))
    st = dict(STYLE, music=dict(STYLE["music"], fallback="none"))
    items = [{"bid": 0, "file": files[0], "t": 0.0, "dur": 3.0}, {"bid": 1, "file": files[1], "t": 3.5, "dur": 3.0},
             {"bid": 2, "file": files[2], "t": 7.0, "dur": 2.0}]
    logs = []
    sfx.mix(tmp_path / "vo.wav", [], 10.0, st, tmp_path / "f.wav", footage=items, log=logs.append)
    x = sfx.load_audio(tmp_path / "f.wav")
    assert _rms(x, 0.5, 2.5) > 0.005                            # engine sound is in the mix
    assert _rms(x, 4.0, 6.0) < 1e-4                             # talking clip skipped
    assert any("1 shot(s) keep their own sound (1 action, 0 ambient)" in m and "speech" in m and "no sound" in m
               for m in logs)


def test_ambient_shot_sound_sits_under_the_voice(tmp_path):
    rng = np.random.default_rng(1)
    noise = sfx._lp(rng.standard_normal(6 * SR), 6) * 0.3
    clip = _clip(tmp_path / "amb.mp4", noise)
    vo = np.zeros(12 * SR, dtype=np.float32)
    vo[:3 * SR] = (np.sin(2 * np.pi * 150 * np.arange(3 * SR) / SR) * 0.3).astype(np.float32)   # voice 0-3 s
    sfx.write_wav(tmp_path / "vo.wav", vo)
    st = dict(STYLE, music=dict(STYLE["music"], fallback="none"))
    res = {}
    for kind in ("action", "ambient"):
        items = [{"bid": 0, "file": clip, "t": 0.0, "dur": 3.0, "kind": kind},
                 {"bid": 1, "file": clip, "t": 6.0, "dur": 3.0, "kind": kind}]
        sfx.mix(tmp_path / "vo.wav", [], 12.0, st, tmp_path / f"{kind}.wav", footage=items)
        x = sfx.load_audio(tmp_path / f"{kind}.wav")
        x[:len(vo)] -= vo[:len(x)]
        res[kind] = (_rms(x, 0.5, 2.5), _rms(x, 6.5, 8.5))        # (under the voice, in a pause)
    assert res["ambient"][0] < res["action"][0] * 0.5 and res["ambient"][1] < res["action"][1]
    assert res["ambient"][0] < res["ambient"][1] * 0.2        # far under the narrator, audible in the pause
    vo_rms = 0.3 / np.sqrt(2)
    assert res["ambient"][0] < vo_rms * 10 ** (-28 / 20)       # the voice is never covered


def test_master_prompt_has_sound_cues_not_sfx():
    txt = (specs_step.ROOT / "prompts" / "master_prompt.md").read_text()
    assert '"intensity"' in txt and '"moment"' in txt and '"sfx"' not in txt
