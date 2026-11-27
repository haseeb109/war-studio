import subprocess
from pathlib import Path

import pytest

from warstudio import events
from warstudio.util import duration, write_json


def test_plan_pattern():
    s = events.plan_slots(1260, 1, 2)
    kinds = [k for _t, k, _i, _s in s]
    assert kinds == ["video", "image", "video", "image", "video", "video"]
    assert s[0][0] <= 20 and sum(k == "video" for k in kinds) == 4
    assert [st for _t, k, _i, st in s if k == "video"] == ["hook", "replay", "framed", "montage"]
    assert s[0][0] == 0.0
    times = [t for t, *_ in s]
    assert times == sorted(times) and 55 <= times[1] <= 95 and times[-1] < 1260 * 0.9
    assert events.plan_slots(600, 0, 0) == []
    assert [k for _t, k, *_ in events.plan_slots(600, 0, 2)] == ["image", "image"]


def test_assign_beats_skips_taken():
    beats = [{"id": i, "start": i * 3.0, "text": "x"} for i in range(400)]
    plan = events.assign_beats(events.plan_slots(1200, 1, 2), beats, 1200, {"2"}, {"video": 6, "image": 5})
    assert len(plan) == 6 and "2" not in plan[0]["beats"]
    assert plan[0]["start"] <= 20
    allb = [b for s in plan for b in s["beats"]]
    assert len(allb) == len(set(allb))


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode, reason="no ffmpeg")
def test_run_step_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_DIR", tmp_path / "ev")
    pack = events.create_pack("Test strike", ["IRN", "XXX"])
    assert pack["countries"] == ["IRN"]
    d = events.pack_dir(pack["slug"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=720x1280:d=6:r=30",
                    "-f", "lavfi", "-i", "sine=d=6", "-shortest", str(d / "videos" / "clip.mp4")], check=True)
    from PIL import Image
    Image.new("RGB", (1200, 800), (240, 240, 240)).save(d / "images" / "news.png")
    proj = tmp_path / "p"
    proj.mkdir()
    write_json(proj / "project.json", {"event": pack["slug"]})
    beats = [{"id": i, "start": i * 4.0, "text": "missile strike base"} for i in range(60)]
    fetched, logs = {}, []
    n = events.run_step(proj, beats, 240.0, {}, {"canvas": {"fps": 30}}, logs.append, fetched)
    assert n > 0, logs
    ev = [b for b, v in fetched.items() if v["source"] == "event"]
    hero = [v for v in fetched.values() if ":clip.mp4:" in v["id"]]
    assert fetched["0"]["id"] == hero[0]["id"]                            # second 0: straight on the real footage
    assert abs(duration(fetched["0"]["file"]) - 5.0) < 0.5
    assert len({v["id"] for v in fetched.values()}) == 4 + 1   # 4 hero uses + 1 report card


def test_overlays_caps_typing_and_clean_hook():
    import json
    from warstudio import overlays
    st = json.load(open("styles/beyond_military/style.json"))
    beats = [{"id": i, "start": i * 3.0, "end": i * 3.0 + 3, "text": "ships in the Strait of Hormuz at 0400"} for i in range(60)]
    pl = overlays.plan(beats, [{"id": b["id"]} for b in beats], st)
    assert pl and all(beats[int(k)]["start"] >= 30 for k in pl)
    assert all(v["line1"] == v["line1"].upper() for v in pl.values())
    tt = overlays.type_times({"line1": "AB C", "line2": ""})
    assert len(tt) == 3 and tt[0] == overlays.TYPE_START and tt == sorted(tt)


def test_dates_times_and_places_everywhere():
    from warstudio import overlays as o
    get = lambda t: (o.detect_datetime(t) or o.place_label(t, False) or {}).get("line1")
    assert get("On 2 April 1982, a small force") == "APR 2, 1982"
    assert get("at 4:30 a.m. they launched") == "4:30 AM"
    assert get("struck Al Udeid Air Base in Qatar") == "AL UDEID AIR BASE"
    assert get("the Port of Bandar Abbas burned") == "PORT OF BANDAR ABBAS"
    assert get("troops moved into Kharkiv") == "KHARKIV"
    assert get("in American service") is None and get("The President spoke") is None
