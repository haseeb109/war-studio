"""Satellite map style test: 4 styles on real Mapbox satellite imagery, deep zoom from space to the port.

Run on the Mac (from the war-studio folder):   uv run python tools/map_sat_test.py
Needs MAPBOX_TOKEN in Settings (keys), or pass it:  uv run python tools/map_sat_test.py pk.xxxx Output: map_sat_<style>.mp4 for each style + map_sat_all.mp4 (all 4 in a row).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from warstudio.config import load_keys          # noqa: E402

MAPS = ROOT / "styles" / "beyond_military" / "maps"
HF = ROOT / "styles" / "beyond_military" / "hf"

PLAN = {
    "duration": 6.0, "title": "RED SEA",
    "from": {"center": [44.0, 17.0], "zoom": 3.2},          # from high above the region
    "to": {"center": [42.96, 14.80], "zoom": 9.2},           # down to Hodeidah port
    "highlight": [{"iso": "YEM", "color": "#b3261e"}, {"iso": "ERI", "color": "#3a5a80"}],
    "countries": [{"name": "YEMEN", "at": [44.2, 15.3]}, {"name": "ERITREA", "at": [39.0, 15.2]}],
    "points": [{"name": "Hodeidah", "lon": 42.95, "lat": 14.80}],
    "arrows": [{"from": [42.95, 14.80], "to": [42.55, 14.55]}],
    "rings": [{"lon": 42.95, "lat": 14.80, "km": 25}],
}
BASE = {"font": "\"Helvetica Neue\",Arial,sans-serif", "satellite": True, "hl": "#ffffff", "border": "#ffffff"}
STYLES = {
    # 1: Real - satellite as it is (Google Earth look), thin white borders, small clean labels
    "real": dict(BASE, sat={"raster-contrast": 0.05}, land_opacity=0, hl_opacity=0.0, border="rgba(255,255,255,.55)",
                 arrow="#ff3b30", ring="#ffffff", accent="#ffffff", label="#ffffff", vignette=0.25, grain=0.06),
    # 2: Real graded - same imagery, colour-graded like the archive footage (a bit warm, soft contrast, film grain)
    "graded": dict(BASE, sat={"raster-saturation": -0.25, "raster-contrast": -0.05, "raster-brightness-max": 0.88},
                   land_opacity=0, hl_opacity=0.18, arrow="#ff3b30", ring="#FFC400", accent="#FFC400", label="#ffffff",
                   border="rgba(255,255,255,.6)", tint="rgba(40,30,10,.10)", vignette=0.4, grain=0.10),
    # 3: Real 3D - graded imagery and the camera tilts down while it dives in (drone / Google Earth fly-in)
    "real3d": dict(BASE, sat={"raster-saturation": -0.2, "raster-brightness-max": 0.9}, land_opacity=0, hl_opacity=0.15,
                   arrow="#ff3b30", ring="#FFC400", accent="#FFC400", label="#ffffff", border="rgba(255,255,255,.6)",
                   vignette=0.4, grain=0.08, pitch=True),
    # 4: Real + enemy/ally tint - real imagery, the sides shown with a soft colour wash
    "sides": dict(BASE, sat={"raster-saturation": -0.15, "raster-brightness-max": 0.85}, land_opacity=0, hl_opacity=0.32,
                  arrow="#ff3b30", ring="#FFC400", accent="#FFC400", label="#ffffff", border="#ffffff",
                  vignette=0.4, grain=0.08),
}


def render(name, theme, token):
    work = ROOT / "map_sat_work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in ("maplibre-gl.js", "maplibre-gl.css", "countries.geojson"):
        shutil.copy(MAPS / f, work / f)
    shutil.copy(HF / "gsap.min.js", work / "gsap.min.js")
    plan = dict(PLAN, theme=theme, token=token)
    if theme.get("pitch"):
        plan["from"] = dict(PLAN["from"], pitch=0, bearing=0)
        plan["to"] = dict(PLAN["to"], pitch=55, bearing=-20)
    page = (MAPS / "map_template.html").read_text().replace("__DUR__", f"{PLAN['duration']:.3f}") \
        .replace("__PLAN__", json.dumps(plan))
    (work / "index.html").write_text(page)
    out = ROOT / f"map_sat_{name}.mp4"
    cmd = (["hyperframes"] if shutil.which("hyperframes") else ["npx", "-y", "hyperframes"]) + \
        ["render", str(work), "--format", "mp4", "-o", str(out), "--fps", "30", "--workers", "1"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f"{name}: FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return None
    print(f"{name}: OK ({time.time() - t0:.0f} s)")
    return out


def main():
    import os
    token = (sys.argv[1] if len(sys.argv) > 1 else "") or os.environ.get("MAPBOX_TOKEN", "") or \
        load_keys().get("MAPBOX_TOKEN", "")
    token = token.strip().strip("'\"")
    if not token.startswith("pk."):
        print("MAPBOX_TOKEN missing: add it in Settings (it starts with pk.)")
        sys.exit(1)
    outs = [o for n, th in STYLES.items() if (o := render(n, th, token))]
    if outs:
        lst = ROOT / "map_sat_work" / "list.txt"
        lst.write_text("".join(f"file '{o}'\n" for o in outs))
        allv = ROOT / "map_sat_all.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(allv)])
        print(f"all styles in one video: {allv}  (order: {', '.join(STYLES)})")
        if sys.platform == "darwin":
            subprocess.run(["open", str(allv)])


if __name__ == "__main__":
    main()
