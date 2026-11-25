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
| 3 | specs | Brain writes per beat: 3 search queries (literal first), footage class, exact entity, wrong matches to avoid, ID label, sound cues (intensity 1-3, attack / reveal / cliffhanger moment) | `specs.json` |
| 4 | search | DVIDS -> Wikimedia Commons (no key) -> Pexels -> Pixabay; metadata + 640px previews only | `candidates.json` |
| 5 | rank | CLIP (local, MPS) scores previews vs narration -> top 5 -> numbered collage -> Gemini picks the winner. Same source clip not reused within 90 s | `picks.json`, `previews/` |
| 6 | fetch | Downloads only beat length + 1 s of each winner, normalized to 1080p30, into `library/raw/` (kept for the future library phase) | `fetched.json` |
| 7 | render | Per beat: slow drift, footage-class grade (FLIR = mono/contrast/noise), overlay baked in, exact frame count from the timeline. Stream-copy join. Sound mix (on-screen SFX, footage sound, intensity-shaped music), two-pass loudnorm | `final.mp4`, `sfx_events.json`, `sound_plan.json` |
| 8 | qc | Contact sheet, footage log (source URL + offset per shot), A/V drift and loudness check | `qc.json`, `contact_sheet.jpg`, `footage_log.csv` |

Same check from Terminal: `uv run python run.py doctor` (add `--full` for the deep version). Report: `data/diagnostics/latest.txt`.

CLI: `uv run python run.py new "Title" script.txt --voice elevenlabs_nPczCjzI2devNBz1zQrb [--dummy]` and `uv run python run.py resume projects/<name> --from rank`.

## Styles

Each style is a folder in `styles/<id>/` with `style.json` (pacing, grades, overlay look, sound design) and `hf/overlay_template.html` (HyperFrames / GSAP animation). Copy `styles/beyond_military` to make a new channel style; it appears in the GUI automatically.

**Beyond Military** (default) was locked from frame-by-frame analysis of 5 channel videos: hard cuts on sentence starts or the named noun, average shot 2.2-3.5 s, literal matches for every named system, yellow imperial+metric callouts, black ID labels, FLIR grade for sensor shots, and the sound design below (from a NexLev study of the channel's latest videos).

## Library index safety (v0.9.4)
`library/shots/index.db` was by mistake in git from v0.8.0 to v0.9.3, so the Update button stashed the real index
away and put an empty one back (the shot files stayed on disk). Now the whole `library/shots/` folder is ignored by
git, the index is backed up before every update (`library/shots/backups/index-preupdate-*.db`), and at every start
WarStudio checks index vs. shot files: if most files are missing from the index it merges every saved copy (git
stashes + backups) back in, keeping only shots whose file is on disk. Diagnostics shows it under Library / shot index.

## Footage mix (v0.9.3)
At the end of every run the log shows where the footage came from, by share of screen time, e.g.
`Footage mix (screen time): My Library 74% (96 shots, 2m 41s) · DVIDS 26% (35 shots, 0m 57s)`. The video page
shows the same as a coloured bar (also saved in `qc.json` -> `source_mix`).
**Library only** (only "My Library" ticked): there is no online fallback, so every beat takes the best library
shot - weaker matches (down to `library_only_min_clip`, 0.15), maps (an aerial shot) and systems the library does
not have (closest shot of the same kind) instead of holding the previous clip past the 3 s rule.

## Sound design (v0.9.0)

All local, no API cost. The old keyword / pivot-word / explosion-dropout rules are gone; sound now follows what the viewer sees and how the story moves.

**Sound effects follow the shot on screen** (the library shot's action tag, else the beat's visual line): takeoff/landing -> `jet_roar`, flying/formation -> `flyby`, helicopters -> `rotor`, launch and rocket artillery -> `launch`, explosion -> `explosion` + `bass_hit`, small arms -> `gunfire`, big guns -> `impact` + `bass_hit`, consoles/radar screens -> `radar`, flares/chaff -> `flares`; drones get nothing (they are quiet).
Plus: a UI blip on every label / timestamp / number box, whoosh + low bass hit on map beats, cut accents (`whoosh` / `metal_hit`) on most cuts in the first 60 s and on scene changes after that, `radio_static` on comms words (wingman, radio, tower...), a radar beep on lock / warning words.
**Moments** from the brain: `attack` / `reveal` -> a riser that ends exactly on the beat + bass hit; `cliffhanger` -> a sustained dark chord.
**Density**: up to 16 sound moments a minute in the hook, about 11 after it; number / explanation beats only get the UI blip. Every event in `sfx_events.json` says why it is there (`rule`).

**Music**: one continuous bed. Tracks in `assets/music/` are chained with crossfades (a 30-min video does not loop one track); without tracks a generated tension bed is used. An intensity curve (brain `intensity` + moments) shapes it: calm beats quieter and darker, action louder, a 2 s swell into every attack/reveal, a dip under cliffhanger chords, ducked to -20 dB under the voice and lifted to -14 dB in pauses, fade-out at the end. Levels hold at least 6 s, so the music never pumps from shot to shot. See `sound_plan.json`.

**Footage sound (v0.9.5: every shot)**: every shot keeps its own sound, ducked under the voice. Action shots (engines, guns, blasts) sit a little up (-24 dB under the voice, -12 dB in pauses); every other shot is quiet ambience (-32 dB under the voice, -18 dB in pauses), so the voiceover is never covered. Quiet clips are never boosted more than 10 dB (no hiss). Clips that sound like speech or come from packages / interviews / briefings are skipped. `all_shots: false` in the style = action shots only; `footage_audio: false` in settings = off.
**Old library shots**: shots cut before v0.9.0 have no sound. Library tab -> **Add sound to old shots** cuts the same seconds, audio only, from the smallest DVIDS file of each source video and muxes it in (the picture is copied untouched; ~1-2 MB download per shot, ~1-1.5 h for 5000 shots). Sources without sound are remembered; Stop / Retry / auto-continue work as for the build. The Library stats show how many shots have their own sound.

Your own SFX files always win over the built-in ones: put `.wav`/`.mp3` files in `assets/sfx/<category>/`. Categories: `ui_click whoosh metal_hit bass_hit riser drone_chord launch explosion flyby jet_roar rotor gunfire impact flares radar radio_static alarm`. Tuning lives in `styles/beyond_military/style.json` (`sfx`, `music`, `footage_audio`).

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

## Pacing rule (locked, v0.8.3)
No shot stays on screen longer than **3 s** (`max_shot_seconds`), counting the pause before the next sentence:
sentences are cut at clause words / spaces, slivers under 1 s are merged, and a piece that still runs long
(few words, long pause) is split in time with the same words so the picture changes. Average ~2 s.

## Internet drops (v0.8.4)
- **Library build**: when the internet goes, the build waits and checks every **5 s**, then continues by itself
  (the Library tab shows "waiting for internet"). "Back" means a named host resolves and answers (v0.9.1: a raw
  IP is not enough, because the router's DNS can lag a minute behind the connection). DNS errors get ~60 s of extra
  tries before they count. One outage is logged once, not once per worker. A video interrupted by the network
  goes back in the queue (up to 3 times, then it is marked failed so it can never loop). Preview files resume from the bytes already saved (HTTP Range). ffmpeg reconnects
  on its own after a drop and gives up on a stalled read after 30 s (then retried).
- **Pause + Retry (v0.9.2)**: 2 network / DNS failures in a row pause the WHOLE build - no next file and no next
  search is tried. The Library tab shows "PAUSED". **Retry** continues from the very same file; with
  "Auto-continue when the internet is really back" ticked (default) it also continues by itself once a named host
  answers (checked every 5 s, backing off to 5 min if the line keeps dropping). Untick it to always wait for Retry.
  A stopped / crashed build: **Retry / continue** starts it again with the same choices and carries on where it
  stopped. One file that keeps failing while the internet works gives up after 6 tries (never blocks the build).
- **Mac sleep** is blocked while the library builds (`caffeinate`; keep the lid open on battery Macs).
- **App closed / Mac restarted mid-build**: the build continues automatically when WarStudio starts
  (`library_auto_resume`).
- **Video runs**: footage searches, DVIDS lookups, clip downloads and previews retry every 5 s and wait out an outage.
