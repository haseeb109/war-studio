"""Map animation test: MapLibre GL JS + HyperFrames -> a 4 s Red Sea clip.

Run on the Mac (from the war-studio folder):   uv run python tools/map_test.py
Output: map_test.mp4 in the war-studio folder, plus the render time.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "styles" / "beyond_military" / "maps"
HF = ROOT / "styles" / "beyond_military" / "hf"

PLAN = {
    "duration": 4.0,
    "title": "RED SEA",
    "from": {"center": [42.0, 18.0], "zoom": 4.2},
    "to": {"center": [43.3, 13.2], "zoom": 5.6},
    "highlight": [{"iso": "YEM", "color": "#b3261e"}, {"iso": "ERI", "color": "#3a5a80"}, {"iso": "DJI", "color": "#3a5a80"}],
    "countries": [{"name": "YEMEN", "at": [47.5, 15.6]}, {"name": "SAUDI ARABIA", "at": [44.5, 21.5]},
                  {"name": "ERITREA", "at": [38.6, 15.4]}],
    "points": [{"name": "Bab-el-Mandeb", "lon": 43.4, "lat": 12.6}, {"name": "Hodeidah", "lon": 42.95, "lat": 14.8}],
    "arrows": [{"from": [42.95, 14.8], "to": [43.3, 12.8]}],
    "rings": [{"lon": 42.95, "lat": 14.8, "km": 180}],
}


def main():
    work = ROOT / "map_test_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    for f in ("maplibre-gl.js", "maplibre-gl.css", "countries.geojson"):
        shutil.copy(MAPS / f, work / f)
    shutil.copy(HF / "gsap.min.js", work / "gsap.min.js")
    style = json.loads((ROOT / "styles" / "beyond_military" / "style.json").read_text())
    PLAN["theme"] = style["maps"]["theme"]               # the locked map style
    page = (MAPS / "map_template.html").read_text().replace("__DUR__", f"{PLAN['duration']:.3f}") \
        .replace("__PLAN__", json.dumps(PLAN))
    (work / "index.html").write_text(page)
    out = ROOT / "map_test.mp4"
    cmd = (["hyperframes"] if shutil.which("hyperframes") else ["npx", "-y", "hyperframes"]) + \
        ["render", str(work), "--format", "mp4", "-o", str(out), "--fps", "30", "--workers", "1"]
    print("rendering a 4 s map clip with HyperFrames ...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    secs = time.time() - t0
    if r.returncode != 0 or not out.exists():
        print("FAILED after %.0f s\n%s\n%s" % (secs, r.stdout[-2000:], r.stderr[-2000:]))
        sys.exit(1)
    print("OK: %s  (render took %.0f s for 4 s of video)" % (out, secs))
    if sys.platform == "darwin":
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
