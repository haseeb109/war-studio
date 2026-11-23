"""End-to-end review-editor test on a tiny real project (local test clips, dummy voice)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warstudio import edits, pipeline  # noqa: E402
from warstudio.config import load_settings, load_style  # noqa: E402
from warstudio.steps import vo, beats, specs, render, qc, fetch  # noqa: E402
from warstudio.util import duration, run, read_json, write_json  # noqa: E402

SCRIPT = ("At 17:14 local time, an MQ-9 Reaper is 32 miles east of the island. "
          "Behind it, two F-15E Strike Eagles hold at 28,000 feet. "
          "The missile bursts 500 feet behind the tail. "
          "On the radar screen, one dot is dragging a trail.")


def frames(p):
    out = run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)])
    return int(out.strip())


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    pipeline.PROJECTS_DIR = root / "projects"
    import warstudio.config as cfg
    import warstudio.steps.fetch as f
    import warstudio.edits as e
    f.LIBRARY_DIR = e.LIBRARY_DIR = cfg.LIBRARY_DIR = root / "lib"
    p = pipeline.create_project("Edit test", SCRIPT, "elevenlabs_x", "beyond_military", dummy=True)
    log = lambda *a, **k: None
    style, settings = load_style("beyond_military"), dict(load_settings(), encoder="libx264", hyperframes=False)
    import warstudio.config as cfg2
    cfg2.save_settings({"encoder": "libx264", "hyperframes": False})
    vo.run_step(p, {}, settings, "x", log, dummy=True)
    bs = beats.run_step(p, style, log)
    specs.run_step(p, {}, settings, log, dummy=True)
    clips = root / "clips"
    clips.mkdir()
    picks, fetched = {}, {}
    alts = []
    for i, color in enumerate(["red", "green", "blue"]):
        src = clips / f"{color}.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=25:d=30",
             "-c:v", "libx264", "-preset", "ultrafast", str(src)])
        prev = clips / f"{color}.jpg"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-frames:v", "1", str(prev)])
        alts.append({"source": "test", "id": color, "title": color, "video_url": str(src), "duration": 30,
                     "preview_file": str(prev), "page_url": "", "credit": "", "clip_score": 0.3 - i * 0.1})
    for b in bs:
        picks[str(b["id"])] = dict(alts[0], alternates=alts, gemini_score=8)
    write_json(p / "picks.json", picks)
    write_json(p / "fetched.json", {})
    fetch.run_step(p, {}, style, log)
    render.run_step(p, settings, style, log, use_hf=False)
    qc.run_step(p, log)
    p.joinpath("status.json").write_text('{"state": "done", "log": []}')
    return p


def test_review_data(proj):
    d = edits.review_data(proj)
    assert d["beats"] and len(d["beats"][0]["alternates"]) == 3
    assert d["beats"][0]["overlay"]["line1"] == "17:14"


def test_edits_apply_and_partial_render(proj):
    log = lambda *a, **k: None
    segs = sorted((proj / "segments").glob("seg_*.mp4"))
    before = {s.name: s.stat().st_mtime for s in segs}
    d = edits.review_data(proj)
    b1 = d["beats"][1]["id"]
    edits.swap_clip(proj, b1, d["beats"][1]["alternates"][2])          # blue clip
    edits.set_overlay(proj, b1, "label", "TEST LABEL")
    edits.set_grade(proj, b1, "flir")
    edits.sfx_add(proj, b1, "explosion")
    rv = read_json(proj / "review.json")
    assert rv["dirty"] == [b1] and rv["audio_dirty"]
    edits.apply(proj, log)
    after = {s.name: s.stat().st_mtime for s in sorted((proj / "segments").glob("seg_*.mp4"))}
    changed = [k for k in after if after[k] != before.get(k)]
    assert changed == [f"seg_{b1:05d}.mp4"]
    total = read_json(proj / "vo.json")["total"]
    assert abs(duration(proj / "final.mp4") - total) < 0.05
    assert read_json(proj / "fetched.json")[str(b1)]["id"] == "blue"


def test_narration_edit_keeps_later_segments(proj):
    log = lambda *a, **k: None
    before = {s.name: (s.stat().st_mtime, frames(s)) for s in sorted((proj / "segments").glob("seg_*.mp4"))}
    old_total = read_json(proj / "vo.json")["total"]
    res = edits.edit_narration(proj, 1, "Behind it, a pair of F-15E Strike Eagles from Lakenheath hold at 30,000 feet over the water.")
    assert res["shift_seconds"] > 0
    assert isinstance(res["shift_frames"], int) and res["shift_frames"] > 0
    new_total = read_json(proj / "vo.json")["total"]
    assert abs(new_total - old_total - res["shift_frames"] / 30) < 1e-5
    ovs = read_json(proj / "overlays.json")
    assert any(o["line1"] == "30,000 ft" for o in ovs.values())
    edits.apply(proj, log)
    after = {s.name: (s.stat().st_mtime, frames(s)) for s in sorted((proj / "segments").glob("seg_*.mp4"))}
    redone = {k for k in after if after[k][0] != before[k][0]}
    mine = {f"seg_{b:05d}.mp4" for b in res["beats"]}
    assert redone == mine
    assert abs(duration(proj / "final.mp4") - new_total) < 0.05
    assert abs(duration(proj / "vo.wav") - new_total) < 0.01


def test_undo_restores(proj):
    log = lambda *a, **k: None
    total = read_json(proj / "vo.json")["total"]
    edits.remove_overlay(proj, 0)
    assert "0" not in read_json(proj / "overlays.json")
    edits.undo(proj)
    assert "0" in read_json(proj / "overlays.json")
    label = edits.undo(proj)          # undo the narration edit
    assert label.startswith("narration")
    assert read_json(proj / "vo.json")["total"] < total
    edits.apply(proj, log)
    assert abs(duration(proj / "final.mp4") - read_json(proj / "vo.json")["total"]) < 0.05
    with pytest.raises(edits.EditError):
        edits.set_overlay(proj, 0, "bogus", "x")


def test_recover_interrupted(tmp_path, monkeypatch):
    from warstudio import server
    monkeypatch.setattr(server, "PROJECTS_DIR", tmp_path)
    (tmp_path / "a").mkdir()
    write_json(tmp_path / "a" / "status.json", {"state": "running", "log": []})
    assert server.recover_interrupted() == 1
    assert read_json(tmp_path / "a" / "status.json")["state"] == "error"
