import json
import subprocess
from pathlib import Path

from warstudio import maps

BEATS = [{"id": 1, "start": 0.0, "end": 2.0, "text": "The ships sailed."},
         {"id": 2, "start": 2.0, "end": 4.5, "text": "Thirty miles off Djibouti,"},
         {"id": 3, "start": 4.5, "end": 7.0, "text": "near Bab-el-Mandeb."},
         {"id": 4, "start": 7.0, "end": 9.0, "text": "The jets launched."},
         {"id": 5, "start": 9.0, "end": 11.0, "text": "South of Vovchansk."}]
SPECS = [{"id": 1, "footage_class": "real", "scene": "a"}, {"id": 2, "footage_class": "map", "scene": "red sea"},
         {"id": 3, "footage_class": "map", "scene": "red sea"}, {"id": 4, "footage_class": "real", "scene": "b"},
         {"id": 5, "footage_class": "map", "scene": "ukraine"}]
COLORS = {"enemy": "#b3261e", "ally": "#3a5a80", "neutral": "#4b5a6b"}


def test_runs_group_consecutive_same_scene():
    runs = maps.find_runs(BEATS, SPECS, 12.0)
    assert [r["beats"] for r in runs] == [["2", "3"], ["5"]]
    assert runs[0]["start"] == 2.0 and runs[0]["end"] == 7.0 and runs[1]["end"] == 12.0


def test_plan_validation_drops_bad_values():
    p = {"title": "red <sea>", "from": {"center": [42, 18], "zoom": 4}, "to": {"center": [43.3, 13.2], "zoom": 5.6},
         "highlight": [{"iso": "YEM", "side": "enemy"}, {"iso": "XXX", "side": "ally"}, {"iso": "DJI", "side": "?"}],
         "points": [{"name": "Hodeidah", "lon": 42.9, "lat": 14.8}, {"name": "bad", "lon": 500, "lat": 0}],
         "arrows": [{"from": [1, 1], "to": [1, 1]}, {"from": [42.9, 14.8], "to": [43.3, 12.8]}],
         "rings": [{"lon": 42.9, "lat": 14.8, "km": 180}, {"lon": 0, "lat": 0, "km": -3}]}
    v = maps.validate_plan(p, {"YEM", "DJI"}, COLORS)
    assert v["title"] == "RED SEA"
    assert v["highlight"] == [{"iso": "YEM", "color": "#b3261e"}, {"iso": "DJI", "color": "#4b5a6b"}]
    assert len(v["points"]) == 1 and len(v["arrows"]) == 1 and len(v["rings"]) == 1
    assert maps.validate_plan({"to": {"center": [0, 0], "zoom": 40}}, set(), COLORS) is None
    assert maps.validate_plan({"to": {"center": [10, 10], "zoom": 5}}, set(), COLORS)["from"]["zoom"] == 3.8


def test_iso_codes_come_from_the_vendored_map():
    isos = maps._iso_set()
    assert {"YEM", "UKR", "USA"} <= isos


def test_run_step_cuts_one_clip_per_beat(tmp_path, monkeypatch):
    plan = maps.validate_plan({"to": {"center": [43, 13], "zoom": 5}}, set(), COLORS)
    monkeypatch.setattr(maps, "plan_runs", lambda runs, *a, **k: ({i: plan for i in range(len(runs))}, 0.0))
    monkeypatch.setattr(maps, "hf_cmd", lambda: ["true"])
    calls = []

    def fake_render(work, out, fps, timeout=0):
        calls.append(json.loads(Path(work, "index.html").read_text().split("const P = ")[1].split(";")[0])["duration"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=navy:s=320x180:r=30:d=7",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
    monkeypatch.setattr(maps, "render_one", fake_render)
    monkeypatch.setattr(maps, "CACHE_DIR", tmp_path / "cache")
    style = json.loads(Path("styles/beyond_military/style.json").read_text())
    fetched = {}
    n = maps.run_step(tmp_path, BEATS, SPECS, 12.0, {"maps_per_video": 2}, {}, style, lambda *a, **k: None, fetched)
    assert n == 4 and {"2", "5"} <= set(fetched)           # each map moment stretched to whole beats inside 4-5 s
    assert all(5.0 <= c <= 6.0 for c in calls)               # 4-5 s map + 1 s pad
    assert all(v["source"] == "map" and Path(v["file"]).exists() for v in fetched.values())
    assert all(v["length"] > 1.5 for v in fetched.values())
    # a second run reuses the clips and the saved plans
    monkeypatch.setattr(maps, "plan_runs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replanned")))
    assert maps.run_step(tmp_path, BEATS, SPECS, 12.0, {"maps_per_video": 2}, {}, style, lambda *a, **k: None, {}) == 4


def test_maps_off_or_no_renderer_changes_nothing(tmp_path, monkeypatch):
    style = json.loads(Path("styles/beyond_military/style.json").read_text())
    assert maps.run_step(tmp_path, BEATS, SPECS, 12.0, {"maps_enabled": False}, {}, style, print, {}) == 0
    monkeypatch.setattr(maps, "hf_cmd", lambda: None)
    assert maps.run_step(tmp_path, BEATS, SPECS, 12.0, {}, {}, style, print, {}) == 0


def test_plan_survives_empty_json_mode_reply(monkeypatch):
    import warstudio.llm as llm
    import warstudio.config as cfg
    calls = []

    def fake_chat(messages, brain, max_tokens=0, json_mode=True, timeout=0):
        calls.append(json_mode)
        if json_mode:
            return "", {}
        ids = [json.loads(l)["run"] for l in messages[1]["content"].split("RUNS:\n")[1].splitlines()]
        return "```json\n" + json.dumps({"runs": [{"run": i, "to": {"center": [43, 13], "zoom": 5}} for i in ids]}) + "\n```", {}
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(cfg, "brain_config", lambda *a, **k: {"key": "x"})
    runs = [{"text": f"run {i}"} for i in range(11)]
    plans, _ = maps.plan_runs(runs, "t", {}, {}, lambda *a, **k: None)
    assert sorted(plans) == list(range(11))
    assert calls.count(True) == 2 and calls.count(False) == 2      # 2 chunks, each: empty JSON mode -> plain retry



def test_select_runs_rule_count_length_first_minute():
    # 25 min video, a map moment every ~20 s, none in the first minute
    beats, specs, t, i = [], [], 0.0, 0
    while t < 1500:
        beats.append({"id": i, "start": t, "end": t + 2.5, "text": f"b{i}"})
        specs.append({"id": i, "footage_class": "map" if (i % 8 == 0 and t > 70) else "real", "scene": f"s{i}"})
        t += 2.5
        i += 1
    runs = maps.find_runs(beats, specs, t)
    got = maps.select_runs(runs, beats, specs, t, {})
    assert 5 <= len(got) <= 6
    assert got[0]["start"] < 60                                    # one map in the first minute (forced)
    assert all(4.0 <= r["end"] - r["start"] <= 5.0 for r in got)       # LOCKED: 4-5 s, never outside
    gaps = [b["start"] - a["start"] for a, b in zip(got, got[1:])]
    assert min(gaps) > 120                                         # spread over the video


def test_every_map_gets_a_different_camera_move():
    moves = [maps.camera(maps.SAMPLE, k, seed="video-1")["camera"] for k in range(6)]
    assert len(set(moves)) == 6
    other = [maps.camera(maps.SAMPLE, k, seed="video-2")["camera"] for k in range(6)]
    assert moves != other                                  # another video opens differently


def test_strict_window_skips_runs_that_cannot_fit():
    beats = [{"id": 0, "start": 0.0, "end": 6.0, "text": "a"}, {"id": 1, "start": 6.0, "end": 9.0, "text": "b"},
             {"id": 2, "start": 9.0, "end": 11.0, "text": "c"}, {"id": 3, "start": 11.0, "end": 13.5, "text": "d"}]
    specs = [{"id": i, "footage_class": "map", "scene": str(i)} for i in range(4)]
    got = maps.select_runs(maps.find_runs(beats, specs, 13.5), beats, specs, 13.5, {"maps_per_video": 3})
    assert got and all(4.0 <= r["end"] - r["start"] <= 5.0 for r in got)
