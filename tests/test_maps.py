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
    style = json.loads(Path("styles/beyond_military/style.json").read_text())
    fetched = {}
    n = maps.run_step(tmp_path, BEATS, SPECS, 12.0, {}, {}, style, lambda *a, **k: None, fetched)
    assert n == 3 and sorted(fetched) == ["2", "3", "5"]
    assert sorted(calls) == [4.0, 6.0]                       # run length + pad
    assert all(v["source"] == "map" and Path(v["file"]).exists() for v in fetched.values())
    assert 3.0 < fetched["3"]["length"] < 3.8                # 2.5 s beat + pad
    # a second run reuses the clips and the saved plans
    monkeypatch.setattr(maps, "plan_runs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replanned")))
    assert maps.run_step(tmp_path, BEATS, SPECS, 12.0, {}, {}, style, lambda *a, **k: None, {}) == 3


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


def test_map_look_satellite_only_with_key():
    style = json.loads(Path("styles/beyond_military/style.json").read_text())
    th, extra = maps.map_look(style, {}, {})
    assert extra == {} and th == style["maps"]["theme"]                   # no key: locked Tactical dark
    th, extra = maps.map_look(style, {}, {"ARCGIS_API_KEY": "k"})
    assert th["satellite"] and extra["provider"] == "esri" and extra["tile_url"].startswith("https://ibasemaps")
    th, extra = maps.map_look(style, {"map_satellite": False}, {"ARCGIS_API_KEY": "k"})
    assert extra == {}
    plan = maps.tilt({"to": {"center": [1, 2], "zoom": 5}}, style["maps"]["satellite_theme"])
    assert plan["to"]["pitch"] == 55
