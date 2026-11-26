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
    # 1: Tactical satellite - dark, cool, low colour; the locked yellow/red overlays on top
    "tactical": dict(BASE, sat={"raster-saturation": -0.55, "raster-brightness-max": 0.62, "raster-contrast": 0.15},
                     land_opacity=0, hl_opacity=0.30, arrow="#ff3b30", ring="#FFC400", accent="#FFC400",
                     label="#ffffff", border="#cfd8e3", tint="rgba(8,19,33,.25)", vignette=0.55),
    # 2: Natural - Google-Earth-like real colours, clean white labels
    "natural": dict(BASE, sat={"raster-saturation": 0.05, "raster-contrast": 0.05}, land_opacity=0, hl_opacity=0.22,
                    arrow="#ff2d2d", ring="#ffffff", accent="#ffffff", label="#ffffff", border="#ffffff", vignette=0.35),
    # 3: Night ops - almost black and white, green tint, grid + scanlines
    "nightops": dict(BASE, sat={"raster-saturation": -1, "raster-brightness-max": 0.55, "raster-contrast": 0.3},
                     land_opacity=0, hl_opacity=0.28, arrow="#ff3b30", ring="#39ff88", accent="#39ff88",
                     label="#d8ffe6", border="#39ff88", tint="rgba(0,60,25,.30)", grid="rgba(57,255,136,.10)",
                     scan=True, vignette=0.6),
    # 4: Hybrid - real colours but darker, strong enemy/ally fills (newsroom look)
    "hybrid": dict(BASE, sat={"raster-saturation": -0.2, "raster-brightness-max": 0.75}, land_opacity=0, hl_opacity=0.45,
                   arrow="#ff3b30", ring="#FFC400", accent="#FFC400", label="#ffffff", border="#ffffff",
                   tint="rgba(0,0,0,.15)", vignette=0.45),
}


def render(name, theme, token):
    work = ROOT / "map_sat_work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in ("maplibre-gl.js", "maplibre-gl.css", "countries.geojson"):
        shutil.copy(MAPS / f, work / f)
    shutil.copy(HF / "gsap.min.js", work / "gsap.min.js")
    plan = dict(PLAN, theme=theme, token=token)
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
