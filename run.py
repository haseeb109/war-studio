"""WarStudio command line.

  uv run python run.py gui                      # open the GUI (http://127.0.0.1:8765)
  uv run python run.py new "Title" script.txt --voice elevenlabs_nPczCjzI2devNBz1zQrb
  uv run python run.py resume projects/<name> [--from search]
  uv run python run.py doctor [--full]          # system check report
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="warstudio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gui")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    n = sub.add_parser("new")
    n.add_argument("title")
    n.add_argument("script_file")
    n.add_argument("--voice", default="elevenlabs_nPczCjzI2devNBz1zQrb")
    n.add_argument("--style", default="beyond_military")
    n.add_argument("--dummy", action="store_true", help="silent voice + fallback specs (no TTS/brain credits)")
    d = sub.add_parser("doctor", help="system check; prints the report to send to Claude")
    d.add_argument("--full", action="store_true", help="also test CLIP, HyperFrames, live APIs end to end (uses a few credits)")
    r = sub.add_parser("resume")
    r.add_argument("project")
    r.add_argument("--from", dest="from_step", default=None)
    a = ap.parse_args()

    if a.cmd == "gui":
        from warstudio.server import serve
        serve(a.port, open_browser=not a.no_browser)
    elif a.cmd == "new":
        from warstudio.pipeline import create_project, run_project
        proj = create_project(a.title, Path(a.script_file).read_text(encoding="utf-8"), a.voice, a.style, a.dummy)
        print(f"project: {proj}")
        run_project(proj)
    elif a.cmd == "doctor":
        from warstudio.diagnostics import run_all
        _, text = run_all(a.full, on_update=lambda r: print(f"  {r[-1]['status']:>7}  {r[-1]['name']}", flush=True) if r[-1]["status"] != "running" else None)
        print("\n" + text)
        print("Saved: data/diagnostics/latest.txt")
    elif a.cmd == "resume":
        from warstudio.pipeline import run_project
        run_project(Path(a.project), a.from_step)


if __name__ == "__main__":
    sys.exit(main())
