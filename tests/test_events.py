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
    assert [st for _t, k, _i, st in s if k == "video"] == events.STYLES
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
    assert int(min(ev, key=int)) * 4 <= 20
    assert abs(duration(fetched[ev[0]]["file"]) - 5.0) < 0.5
    assert len({v["id"] for v in fetched.values()}) == 4 + 1   # 4 hero uses + 1 report card
