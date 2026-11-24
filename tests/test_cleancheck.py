import subprocess

import numpy as np
from PIL import Image

from warstudio import cleancheck
from warstudio.steps import fetch, rank, render


def test_classify_text_rules():
    # fuselage marking: small, mid-frame -> fine
    assert cleancheck.classify_text([("U.S. AIR FORCE", 0.9, (0.4, 0.5, 0.12, 0.03))]) == ([], 0.0)
    # name strip bottom-left -> lower_third, crop to its top edge
    p, c = cleancheck.classify_text([("SGT. JOHN DOE, 1ST BATTALION", 0.9, (0.05, 0.08, 0.4, 0.05))])
    assert p == ["lower_third"] and c == 0.13
    # centred wide bottom text -> subtitles
    p, _ = cleancheck.classify_text([("we were ready for anything that night", 0.9, (0.2, 0.06, 0.6, 0.05))])
    assert p == ["subtitles"]
    # lots of text -> title card
    p, _ = cleancheck.classify_text([("OPERATION INHERENT RESOLVE", 0.9, (0.1, 0.45, 0.8, 0.1))])
    assert "title_card" in p
    # low-confidence junk ignored
    assert cleancheck.classify_text([("xx", 0.2, (0.1, 0.1, 0.5, 0.1))]) == ([], 0.0)


def test_pixel_checks(tmp_path):
    blank = tmp_path / "b.jpg"
    Image.new("RGB", (640, 360), (5, 5, 5)).save(blank)
    assert cleancheck.pixel_problems(blank) == ["blank_card"]
    bars = np.zeros((360, 630, 3), np.uint8)
    cols = [(192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0), (192, 0, 192), (192, 0, 0), (0, 0, 192)]
    for i, c in enumerate(cols):
        bars[:, i * 90:(i + 1) * 90] = c
    Image.fromarray(bars).save(tmp_path / "bars.png")
    assert cleancheck.pixel_problems(tmp_path / "bars.png") == ["color_bars"]
    noise = (np.random.default_rng(1).random((360, 640, 3)) * 255).astype(np.uint8)
    Image.fromarray(noise).save(tmp_path / "n.png")
    assert cleancheck.pixel_problems(tmp_path / "n.png") == []


def test_check_segment_black_and_cache(tmp_path):
    v = tmp_path / "black.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=2",
                    "-c:v", "libx264", str(v)], check=True)
    r = cleancheck.check_segment(v, 2.0, "auto")
    assert "black" in r["problems"]
    assert (tmp_path / "black.mp4.check.json").exists()
    assert cleancheck.check_segment(v, 2.0, "off")["problems"] == []


def test_lower_third_becomes_crop(monkeypatch, tmp_path):
    v = tmp_path / "n.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=2",
                    "-c:v", "libx264", str(v)], check=True)
    monkeypatch.setattr(cleancheck, "vision_available", lambda: True)
    monkeypatch.setattr(cleancheck, "vision_text", lambda f: [("SGT. JOHN DOE, 1ST BATTALION", 0.9, (0.05, 0.05, 0.4, 0.05))])
    monkeypatch.setattr(cleancheck, "vision_faces", lambda f: [])
    r = cleancheck.check_segment(v, 2.0, "auto")
    assert r["problems"] == [] and r["crop_bottom"] == 0.12


def test_talking_head(monkeypatch, tmp_path):
    v = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=2",
                    "-c:v", "libx264", str(v)], check=True)
    monkeypatch.setattr(cleancheck, "vision_available", lambda: True)
    monkeypatch.setattr(cleancheck, "vision_text", lambda f: [])
    monkeypatch.setattr(cleancheck, "vision_faces", lambda f: [0.08])
    assert "talking_head" in cleancheck.check_segment(v, 2.0, "auto")["problems"]


def test_title_adjust():
    assert rank.title_adjust("F-35A Lightning II B-Roll") > 0
    assert rank.title_adjust("Airman interview: life on the flightline") < 0
    assert rank.title_adjust("Pentagon Press Briefing") < 0
    assert rank.title_adjust("F-16 takeoff at dusk") == 0


def test_choose_start_keeps_out_of_tail():
    for used in ([], [10.0], [10.0, 20.0]):
        s = fetch.choose_start(300, 5, used)
        assert s + 5 <= 300 - 8


def test_render_crop_anchors_top():
    style = {"canvas": {"width": 1920, "height": 1080}, "grade": {"real": "null"},
             "footage": {"motion": {"drift": True, "drift_scale": 1.06}}}
    cmd = render.segment_cmd("a.mp4", 60, 30, style, "real", None, "o.mp4", "libx264", crop_bottom=0.15)
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert "y='0'" in fg and "scale=2258:1270" in fg
    cmd = render.segment_cmd("a.mp4", 60, 30, style, "real", None, "o.mp4", "libx264")
    assert "y='(ih-oh)/2'" in cmd[cmd.index("-filter_complex") + 1]
