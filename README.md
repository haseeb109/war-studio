# WarStudio

Script in, finished military documentary video out. Built for Apple Silicon Macs (tested design target: M1 Pro, 32 GB).

**Input:** title + script + voice + style. **Output:** `final.mp4` (1080p30, -14 LUFS), `contact_sheet.jpg`, `footage_log.csv`, `qc.json`.

## Start

```bash
git clone https://github.com/haseeb109/war-studio.git
cd war-studio
bash setup_mac.sh          # once: ffmpeg, node, uv, Python packages, CLIP model, SFX
```

Then double-click **`WarStudio.command`** (or `uv run python run.py gui`). The GUI opens at http://127.0.0.1:8765.

1. **Settings** tab: paste your API keys (stored only in `~/.config/warstudio/keys.env`) and press **Test** next to each one; it checks the key with a tiny real request before you save (DeepSeek also shows your balance).
2. **Voices** tab: *Load voices from ai33*, preview, star your favorites (favorites show first everywhere).
3. **Create** tab: title, script, voice, style → *Start production*.
4. **Videos** tab: your video library. Open a video to watch it, download it, see the contact sheet and footage log, or *Mark as reviewed*.
5. **Review timeline** (button under every finished video): the whole edit beat by beat. Flagged beats (low match, generic search, no footage) are coloured; jump between them with *Flagged ‹ ›*. Click a beat to:
   - swap its clip (the ranked top 5, a new search, or your own file), pick which part of the clip plays,
   - change the look (real / FLIR / cockpit / control room / map),
   - add, edit or remove the callout,
   - add or remove sound effects,
   - rewrite the narration sentence and re-voice it (the rest of the timeline shifts by whole frames, so nothing else re-renders).
   *Apply changes* re-renders only the changed beats (seconds, not minutes). *Undo* steps back through every edit.
6. **Check for update** (top right): pulls the latest version from GitHub and restarts. One click.
7. **System Check** tab: tests every installed piece one by one and gives a report. **Quick** (~1 min, free) or **Full** (~3-5 min, a few cents: HyperFrames, CLIP on the Apple GPU, Gemini, an ai33 sample, real clip downloads, a mini dry-run video). When something breaks, press **Copy report** and paste it to Claude. A failed video also has a **Copy log** button.

Tip: tick **Dry run** for a first test: silent voice and no AI credits, but real footage search, ranking and render.


## Master prompt (locked)

`prompts/master_prompt.md` is the rulebook the brain follows for every beat, and the later steps obey what it writes:

1. **Visual line** - a concrete description of the frame; CLIP and the reranker judge clips against it, not against the narration.
2. **Three search tiers** - exact system -> look-alike substitute -> broad category. All tiers are searched, so a beat falls back to a correct-looking clip instead of a random one. Rank prefers exact > substitute > category.
3. **Scene continuity** - beats in the same situation share a `scene` label; inside a scene a strong clip may be reused (max 3 times) from a different moment, like a human editor cutting several shots from one clip. The cooldown only blocks repeats across different scenes.
4. **Single shots over reels** - source videos longer than 90 s (multi-shot reels with interviews/slates) are ranked lower unless they are the exact system.
5. **Cinematic stock only** - Pixabay is searched with `video_type=film` and `min_width=1920` (no 3D renders, no low-res).

The log shows `match level: exact N, substitute N, category N, reuse N`; the Review editor flags `category clip` where a named system only got its broad class.

## API keys

| Key | Used for | Needed? |
|---|---|---|
| `AI33_API_KEY` | Voiceover (ai33.pro, ElevenLabs voices) | Yes (not for dry run) |
| `DEEPSEEK_API_KEY` | Brain: beat -> shot spec. Default provider: DeepSeek official API (`https://api.deepseek.com`, model `deepseek-flash`). Settings → *Script planner* can switch to Hive or any OpenAI-compatible API (`HIVE_API_KEY` / `CUSTOM_BRAIN_API_KEY`) | Recommended (fallback specs otherwise) |
| `GROQ_API_KEY` | **Clip rerank only** (never the script brain): a Groq vision model picks the best of the top-5 clips from one collage image. Free key at console.groq.com/keys | Recommended |
| `GEMINI_API_KEY` | Rerank: picks the best of the top-5 clips from one collage image | Optional backup for rerank |

Rerank order is the `reranker` setting: `auto` (Groq, then Gemini, then CLIP only), `groq`, `gemini` or `clip`. A provider that runs out of daily quota is dropped for the rest of the run and the next one takes over; the log line `reranked by: ...` shows who ranked how many beats.
| `DVIDS_API_KEY` | Real US military footage (public domain) — the most important source | At least one footage key |
| `PEXELS_API_KEY` | Stock footage | |
| `PIXABAY_API_KEY` | Stock footage | |

One key per service. Searches are cached in `data/search_cache`, so repeated queries cost no requests; a source that hits its rate limit is skipped and the others carry on.

## Pipeline (8 steps, each resumable)

| # | Step | What happens | Output |
|---|---|---|---|
| 1 | voice | Script split into sentences; one ai33 voice file per sentence; measured durations give exact sentence timing | `vo/`, `vo.wav`, `vo.json` |
| 2 | beats | One shot per sentence; first 3 minutes: sentences >= 3.2 s get 2 shots; any shot > 6 s is split | `beats.json` |
| 3 | specs | Brain writes per beat: 3 search queries (literal first), footage class, exact entity, wrong matches to avoid, ID label, SFX events | `specs.json` |
| 4 | search | DVIDS -> Wikimedia Commons (no key) -> Pexels -> Pixabay; metadata + 640px previews only | `candidates.json` |
| 5 | rank | CLIP (local, MPS) scores previews vs narration -> top 5 -> numbered collage -> Gemini picks the winner. Same source clip not reused within 90 s | `picks.json`, `previews/` |
| 6 | fetch | Downloads only beat length + 1 s of each winner, normalized to 1080p30, into `library/raw/` (kept for the future library phase) | `fetched.json` |
| 7 | render | Per beat: slow drift, footage-class grade (FLIR = mono/contrast/noise), overlay baked in, exact frame count from the timeline. Stream-copy join. Voice + SFX + music mix, two-pass loudnorm | `final.mp4` |
| 8 | qc | Contact sheet, footage log (source URL + offset per shot), A/V drift and loudness check | `qc.json`, `contact_sheet.jpg`, `footage_log.csv` |

Same check from Terminal: `uv run python run.py doctor` (add `--full` for the deep version). Report: `data/diagnostics/latest.txt`.

CLI: `uv run python run.py new "Title" script.txt --voice elevenlabs_nPczCjzI2devNBz1zQrb [--dummy]` and `uv run python run.py resume projects/<name> --from rank`.

## Styles

Each style is a folder in `styles/<id>/` with `style.json` (pacing, grades, overlay look, SFX rules) and `hf/overlay_template.html` (HyperFrames / GSAP animation). Copy `styles/beyond_military` to make a new channel style; it appears in the GUI automatically.

**Beyond Military** (default) was locked from frame-by-frame analysis of 5 channel videos: hard cuts on sentence starts or the named noun, average shot 2.2-3.5 s, literal matches for every named system, yellow imperial+metric callouts, black ID labels, FLIR grade for sensor shots, UI click on every callout, whoosh + bass hit on map scenes, riser into a bass hit on narrative pivots ("Unknown to...", "But..."), one-shot SFX on action words, music ducked under voice and dropped out on explosions.

## Sound effects and music

Built-in synthesized SFX are generated on first run (`assets/sfx/_generated/`). To use real sounds, put `.wav`/`.mp3` files in `assets/sfx/<category>/`; your files always win over the built-in ones. Categories: `ui_click whoosh bass_hit riser launch explosion flyby gunfire flares radar radio_static alarm impact`.

Background music: put a track in `assets/music/` (or set `music_file` in Settings). It loops, ducks under the voice, and drops out on explosions.

## Overlays (HyperFrames)

If `npx hyperframes` works and ffmpeg has `libvpx-vp9`, callouts are rendered as animated transparent WebM from `styles/<id>/hf/overlay_template.html`. Otherwise a static PNG with fade in/out is used, so a render never blocks on HyperFrames.

## Not in this MVP (next phases)

- Shared clip library with index, dedupe, usage ledger, rerank cache (clips are already being saved to `library/raw/`).
- Animated 3D maps (map beats currently use aerial/satellite stock footage).
- Licensing: DVIDS footage is public domain; Wikimedia Commons clips are limited to public domain / CC0 / CC BY (CC BY-SA only if `commons_allow_sharealike` is true); Pexels/Pixabay follow their licenses. `footage_log.csv` records source and license of every shot, and `credits.txt` holds the attribution lines to paste into the YouTube description (required for CC BY clips).

## Clean-footage check (DVIDS and other long reels)

After every cut, WarStudio checks the actual segment (not the search thumbnail):

1. **B-Roll first** - titles with "B-Roll" rank higher; "Package", "Interview", "Briefing", "Press Conference", "Podcast" rank lower.
2. **Segment check** (`segment_check` setting): `auto` = ffmpeg black detection + blank-card / colour-bar detection + on Mac, Apple Vision text and face detection (built into macOS, free). `vlm` also asks the vision reranker. `off` disables it.
   Burned-in text, subtitles, title cards, banners or a talking head -> another moment of the same long video is tried (2 more), then the next ranked alternate (up to 4 clips). If nothing is clean, the least-bad clip is used and the Review editor flags it `unclean: ...`.
3. **Slates, black, end credits** - start after the DVIDS `time_start`, never in the last 5% (max 8 s) of long videos, black/blank/colour-bar cuts rejected.
4. **Lower-third crop** - a name strip in the bottom ~16% alone is not a rejection: render zooms in from the top so it leaves the frame.

## Gemini rerank: overload handling and cost

- **503 "high demand" / 500 / timeouts** are temporary: 2 quick retries (2 s, 5 s), then the same collage goes to a
  backup model (the next stable Flash, then Flash-Lite). If every model is busy, Gemini is **paused for 60 s**
  (Groq or CLIP covers those beats) and then used again - it is never switched off for the whole run because of 503s.
- **429 per minute**: the server's retry delay is waited out. **429 per day** (free quota): Gemini stops for the run.
- Real errors (bad request) stop it only after 5 in a row.
- Requests use `thinkingLevel: low` and medium image resolution (560 instead of 1,120 image tokens). If a model
  rejects either option it is sent again without it.
- The log ends the rank step with real usage: calls, retries, backup-model uses, tokens and the Gemini cost
  (`gemini_price_in_per_m` / `gemini_price_out_per_m` in Settings).

## Speed (v0.6.6)

- **Rank in two phases**: (A) all previews downloaded 24 at a time (max 24 per beat, every search tier keeps seats),
  CLIP for every beat, then the vision rerank for **8 beats at a time** (`rerank_concurrency`); (B) a quick walk over
  the timeline that applies cooldown and in-scene reuse (max 3 uses of a clip per scene) with the scores from (A).
- Gemini thinking `minimal` (falls to `low` on models without it) - fewer billed thinking tokens, faster answers.
- Render: 4 segments at a time (`render_concurrency`), 5 HyperFrames overlays at a time.
- Search 6 beats at a time, specs 6 chunks at a time, fetch 8 clips at a time.

## Shot library (v0.8.0)

**Library tab** -> pick categories, total shots (e.g. 1000), shots per DVIDS video (4) -> **Build / top up**.
Leave it running overnight; it resumes where it stopped (Stop button, crash or restart).

Per DVIDS video: small preview file (360-480p) -> ffmpeg scene cuts -> 3-8 s candidate shots (no opening slate,
no end credits) -> clean check on the preview (black, blank card, colour bars, Apple Vision text and faces) ->
the clean ones are cut **once** in 1080p from the full-quality file -> keyframe -> CLIP vector -> one caption
from the vision model (~$0.002 per shot; `library_captions: off` for none). Stored in `library/shots/`
(`library_dir`), index in `index.db`. Stops at 15 GB free disk or `library_max_gb` (200).

Every video ("My Library (first)" on the Create page):
- each beat is matched against the library: CLIP(visual line) vs shot vectors, **a beat that names a system only
  gets shots of that system**, caption words add a little, shots used in other videos this week score lower;
- a match >= `library_min_clip` (0.24) = library clip: no online search, **no Gemini call**;
- the rest go online (DVIDS etc.) with the Gemini rerank as before;
- uses are counted per shot; online clips Gemini scored 7+ that passed the clean check join the library
  (`library_learn`), so the library grows with every video at no extra cost.

### Library v0.8.1: no duplicates, keeps growing
- **Duplicates**: a shot is skipped when its keyframe hash is within 6 bits of an existing shot, or CLIP >= 0.965 with
  a loosely similar hash (the same B-roll re-posted in another DVIDS package), or it overlaps half of an existing
  shot from the same source video. Skipped files are deleted; the log counts them.
- **Learning**: every clean online clip of a finished video joins the library (DVIDS only by default:
  `library_learn_sources`; `library_learn_min_score` = 0 keeps all, quality = its Gemini score).
- **Growing past 1,000s of shots**: the builder walks DVIDS result pages per category (50 per page, up to 40 pages)
  and remembers where it stopped. "Stop at library size (GB)" on the Library tab (default 100).
- Matching prefers shots with a higher quality score.

### Library v0.8.2
- **Browse shots** (Library tab): search, category, sort (newest / most used / least used / quality), favourites,
  click to play, star = favourite (small matching bonus), bin = delete (files removed; the picture is remembered
  and never added again).
- **Review editor search** shows your library shots first.
- **3-frame vectors**: each shot's CLIP vector is the average of frames at 25/50/75 %; older shots are upgraded at
  the start of the next build.
- **Action tags** (takeoff, landing, refueling, launch, explosion, firing, console, hover, taxi, sailing, patrol,
  loading, formation, flying) from the caption; a beat whose visual line describes an action prefers shots with it.
- **Rotation**: a shot used in one of the last 5 videos rests (`library_rotation_videos`); it is used only when
  nothing else fits that beat, with a penalty.
- **Captions on Flash-Lite** (`library_caption_model: lite`, about half the price).
- **Daily backup** of `index.db` in `library/shots/backups/` (last 7 days).
- **720p minimum**: DVIDS videos below 720p are skipped; learned clips from sub-720p sources too.
