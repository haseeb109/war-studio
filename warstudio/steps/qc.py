"""Step 8: quality checks + review files (contact sheet, footage log, qc.json)."""
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..util import duration, read_json, write_json, run


def contact_sheet(proj, beats, picks, out, cols=6):
    tw, th = 320, 180
    rows = (len(beats) + cols - 1) // cols
    im = Image.new("RGB", (cols * tw, rows * (th + 46)), (14, 16, 20))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except Exception:
        f = ImageFont.load_default()
    for i, b in enumerate(beats):
        x, y = (i % cols) * tw, (i // cols) * (th + 46)
        p = picks.get(str(b["id"])) or {}
        try:
            t = Image.open(p.get("preview_file", "")).convert("RGB").resize((tw - 6, th - 6))
            im.paste(t, (x + 3, y + 3))
        except Exception:
            d.rectangle([x + 3, y + 3, x + tw - 3, y + th - 3], outline=(200, 60, 60), width=3)
            d.text((x + 12, y + 80), "NO FOOTAGE", font=f, fill=(220, 80, 80))
        d.text((x + 6, y + th + 2), f"#{b['id']} {b['start']:.1f}s  {p.get('source', '')}", font=f, fill=(255, 196, 0))
        d.text((x + 6, y + th + 22), b["text"][:40], font=f, fill=(220, 220, 220))
    im.save(out, quality=80)


def write_credits(proj, beats, fetched):
    """credits.txt for the YouTube description. CC BY clips (Wikimedia Commons) must be attributed;
    DVIDS asks for a courtesy credit. Pexels/Pixabay need none but are listed for completeness."""
    seen, lines = set(), []
    for b in beats:
        f = fetched.get(str(b["id"])) or {}
        k = (f.get("source"), f.get("id"))
        if not f or k in seen or f.get("source") == "user":
            continue
        seen.add(k)
        lic = f.get("license") or {"pexels": "Pexels License", "pixabay": "Pixabay Content License"}.get(f.get("source"), "")
        lines.append(f"- {f.get('title') or f.get('id')} | {f.get('credit', '')} | {lic} | {f.get('page_url', '')}")
    must = [l for l in lines if "cc by" in l.lower() or "cc-by" in l.lower()]
    txt = "FOOTAGE CREDITS\n"
    if must:
        txt += "\nRequired attribution (CC BY / CC BY-SA):\n" + "\n".join(must) + "\n"
    txt += "\nAll sources:\n" + "\n".join(lines) + "\n"
    (Path(proj) / "credits.txt").write_text(txt)


def run_step(proj, log):
    proj = Path(proj)
    beats = read_json(proj / "beats.json")
    picks = read_json(proj / "picks.json", {}) or {}
    fetched = read_json(proj / "fetched.json", {}) or {}
    vo = read_json(proj / "vo.json")
    contact_sheet(proj, beats, picks, proj / "contact_sheet.jpg")
    with open(proj / "footage_log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beat", "start", "end", "text", "source", "id", "source_offset", "page_url", "credit", "license"])
        for b in beats:
            f = fetched.get(str(b["id"])) or {}
            w.writerow([b["id"], b["start"], b["end"], b["text"], f.get("source", ""), f.get("id", ""),
                        f.get("start", ""), f.get("page_url", ""), f.get("credit", ""), f.get("license", "")])
    write_credits(proj, beats, fetched)
    final = proj / "final.mp4"
    from collections import Counter
    qc = {"beats": len(beats), "vo_seconds": vo["total"],
          "footage_by_source": dict(Counter((fetched.get(str(b["id"])) or {}).get("source", "none") for b in beats)),
          "beats_without_footage": [b["id"] for b in beats if not fetched.get(str(b["id"]))]}
    if final.exists():
        qc["video_seconds"] = round(duration(final), 3)
        qc["av_drift_seconds"] = round(abs(qc["video_seconds"] - vo["total"]), 3)
        import subprocess
        p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(final), "-af", "ebur128", "-f", "null", "-"],
                           capture_output=True, text=True)
        m = re.findall(r"I:\s+(-?\d+\.\d+) LUFS", p.stderr)
        if m:
            qc["integrated_lufs"] = float(m[-1])
    write_json(proj / "qc.json", qc)
    log(f"QC: {qc.get('video_seconds', 0):.1f}s video, {len(qc['beats_without_footage'])} beats without footage, "
        f"loudness {qc.get('integrated_lufs', 'n/a')} LUFS")
    return qc
