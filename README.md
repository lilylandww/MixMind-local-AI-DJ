# MixMind

MixMind turns a queue of songs into a single, continuously beat‑mixed DJ set. For each pair of adjacent tracks it analyses the music (tempo, key, beat grid, song structure, vocals) and uses an LLM to design a musical transition — beatmatched crossfades, key‑safe blends, filter sweeps, vinyl stops, EQ kills — then renders the whole queue into one seamless audio file you can play in the browser.

It's a local‑first, single‑user web app: everything runs on your own machine.

> Full design details live in [mixmind‑spec.md](mixmind-spec.md); per‑feature write‑ups in [docs/feature-designs.md](docs/feature-designs.md).

## What it does

1. **Get music in** — search YouTube, upload files from disk, or point MixMind at a folder on the server. Drop a `.lrc` lyric file alongside a track and it's picked up automatically.
2. **Analyse each track** — detect BPM / key / beat grid / sections / energy, split into 4 stems (vocals, drums, bass, other), and work out *when each word is sung*.
3. **Design transitions** — for every adjacent pair, an LLM plans a transition as a sequence of mixing "tool calls," guided by both songs' structure and the regions where vocals are/aren't present (so cuts never chop a word).
4. **Render & stitch** — a deterministic mixer interprets those tool calls (time‑stretch to beatmatch, downbeat‑aligned per‑stem crossfades, pitch handling, effects) and stitches all transitions and songs into one continuous M4A (256 kbps AAC).
5. **Play** — the player streams the continuous mix with a live indicator of the current track and the transition in progress, plus a per‑song hard‑cut fallback.

## Getting music in

Three sources, all landing in the same library and running the same pipeline.

| Source | Where | Notes |
|---|---|---|
| **YouTube** | search box on the home page | Audio is pulled with `yt‑dlp`. On a datacenter IP set `YT_DLP_COOKIES_FILE` (see below). |
| **File upload** | *Import from disk* (drag & drop) | `mp3 wav flac m4a aac ogg opus aiff wma`. Sends an optional `.lrc` alongside; `MUSIC_MAX_UPLOAD_MB` caps the size. |
| **Server folder** | *Folder import* | Requires `IMPORT_ROOT`. Bind‑mount your library into the backend container and import a whole tree at once. |

Imports are idempotent on content hash — re‑importing an unchanged file or folder returns what's already there instead of duplicating it, which is also how you attach a sidecar to a track you added earlier.

### Lyric sidecars (`.lrc`)

If a `.lrc` sits next to an audio file with the same stem (`track.mp3` → `track.lrc`, any case), it becomes that song's **authoritative lyric source** and Genius is skipped entirely. That's a deliberate precedence: a sidecar belongs to this exact recording, it carries timestamps, it works for tracks Genius has never heard of, and it needs no network round trip. A later Genius scrape never overwrites it.

What gets read:

- **Line stamps** — `[00:12.00] first line`, including several per line (`[00:21][01:45] chorus`) for repeated sections, and `[hh:mm:ss.frac]` extended stamps.
- **Enhanced word stamps** — `<00:12.00>first <00:12.40>line`, which give per‑word timings directly.
- **`[offset:...]`**, plus metadata tags (`ti`/`ar`/`al`/`by`).
- **Encodings** beyond UTF‑8: GB18030, Big5, Shift_JIS, EUC‑KR. Sidecars from CJK sources are frequently not UTF‑8.
- **Unsynced `.lrc`** (text, no timestamps) still works — the text is aligned to Whisper instead.

Bare three‑part stamps like `[03:45:00]` are genuinely ambiguous (`mm:ss:ff` vs `hh:mm:ss`) and are resolved per file by whether the file uses fractional stamps anywhere.

## How it works (high level)

```
Search / upload / folder ─▶ Queue ─▶ per-song pipeline ─▶ per-pair LLM plan ─▶ render ─▶ stitch ─▶ Player
                                (analyze →                  (tool-call list)    (numpy +    (one M4A)
                                 stem-separate →                                rubberband)
                                 transcribe →
                                 lyrics align)
```

- **Audio analysis** uses `librosa` for tempo / beats / downbeats / key / sections / energy. Optionally, installing the `allin1` extra (`pip install -e backend[pro-analysis]`) and setting `SECTION_DETECTOR=allin1` switches beats / downbeats / tempo / sections to the All‑In‑One neural model, which also produces *real* section labels (intro / verse / chorus / …) that flow into the planner; any failure falls back to librosa.
- **Stem separation** uses Demucs — `htdemucs` on the local path, `htdemucs_ft` on the Modal path. Both it and transcription are GPU‑heavy and run on a native worker (Apple Silicon MPS, or CUDA on Linux) or on Modal.
- **Vocal‑safe regions** are derived from the vocal stem's energy envelope cross‑referenced with word timings, so the planner only places hard cuts where there are no vocals. See [Lyrics and word timing](#lyrics-and-word-timing) below.
- **Transition planning (planner v2)** goes through a pluggable `LLMProvider` (Gemini / Groq / OpenAI‑compatible), but the LLM no longer does any timestamp math. The pipeline pre‑computes a small menu of *valid‑by‑construction* seam candidates for each song (downbeat‑snapped, headroom‑checked, vocal‑safety‑annotated), and the model — seeing each song's **title and artist** plus its section/energy structure — picks an exit point, an entry point, and one of eight **transition archetypes** (`smooth_blend`, `drop_swap`, `drum_bridge`, `wash_out`, `stutter_buildup`, `vinyl_stop`, plus two acapella-layering styles: `acapella_out` rides A's vocals over B's incoming instrumental before handing off to B's vocals at a vocal-safe phrase boundary, and `acapella_in` teases B's vocals over A's still-playing backing before the beat swaps — both require compatible effective keys and are auto-downgraded to a blend on a clash unless the user pins them) with a couple of bounded knobs. A deterministic expander turns that decision into the exact tool‑call list (`crossfade_stem`, `set_tempo_ramp`, `temporary_pitch_shift`, `filter_sweep`, `apply_reverb`, `turntable_stop`, `volume_fade`, …), so the executor's invariants hold by construction. A **set‑level pass** runs once per locked queue and assigns each pair a suggested style so the whole mix has an arc. Malformed LLM output is *repaired* where mechanically possible instead of silently discarded, and every plan records its provenance (`plan_source`) — LLM, repaired, or fallback — surfaced in the UI. Each transition can be **re‑rolled** (with an optional pinned style) from the player; re‑rolls bump a nonce in the LLM cache key so they are genuinely fresh samples. The v1 free‑form prompt remains available via `PLANNER_VERSION=legacy`.
- **Whole‑song key handling (no more pitch glides).** Mid‑song key glides — holding the incoming track in the outgoing track's key, then bending it back after the crossfade — are the most audible artifact a mixer can produce, so by default (`PITCH_MODE=whole_song`) MixMind never does them. Instead, a deterministic set‑level resolver walks the locked queue once and assigns each song a constant semitone offset (capped at ±2, reset to 0 whenever neighboring keys are already compatible) applied to its **entire** play — the professional "Pitch 'n Time" approach; a constant ±1–2 semitone shift is essentially inaudible while a bend is always audible. Pair renders pre‑shift the song's stems and master (cached per song+offset in storage), so the two renders sharing a song agree exactly and stitch junctions stay seamless. Clashes beyond the cap are *accepted* and the planner routes them to short or washy transition styles where keys barely overlap. `PITCH_MODE=temporary` restores the v1 glide behavior; `PITCH_MODE=off` never shifts. Tempo is unaffected — beatmatching still uses a (now longer, gentler) post‑crossfade ramp, since tempo glides don't change key.
- **Gain staging.** Every transition includes automatic **seam loudness matching**: the incoming song's perceived level (from the analysis energy curve) is aligned to the outgoing one at the seam — boost capped at +4 dB, cut at −6 dB, gaps under ~1.25 dB untouched — then glides back to its native level after the crossfade, returning to exactly unity before the next stitch junction. Disable with `LOUDNESS_MATCH=false`. The acapella styles additionally bring the backing track in **staged**, the way a DJ would: drums crossfade first (the groove morphs), the bass quick‑swaps in one bar (two basslines overlapping is instant mud), and the melodic stem arrives **ducked** under the riding vocal, restoring to full only when the vocal hands over.
- **The mixer executor** turns that tool list into audio with `numpy` / `soundfile` / `pyrubberband`: it time‑stretches the incoming track to the outgoing track's tempo, aligns downbeats at the seam, and applies per‑stem crossfades and effects. The stitcher equal‑power crossfades the per‑pair renders into one file.
- **A DJ host voice** can open the set and drop the occasional mic moment over the music (`TTS_PROVIDER=kokoro|openai|off`, `HOST_FREQUENCY=off|intro_only|sparse|chatty`). The script is written from the queue, synthesised, and ducked under the music at stitch time. Kokoro‑82M runs locally and free; `off` disables the feature entirely.
- **Storage** is behind a `StorageBackend` protocol (local filesystem by default; S3‑compatible swap is a config change). Generated artifacts live under `cache/` and are evicted LRU once they exceed a size budget.

## Lyrics and word timing

This is the part that decides where you can and can't cut, so it's worth being precise about. **What the pipeline needs is word *timing*, not word *text*** — the text is only ever a label.

```
             ┌── .lrc sidecar ──────┐   text + line/word stamps (best)
known text ──┤                       ├──▶ aligner ─▶ aligned_words ─▶ vocal_safe_regions
             └── Genius scrape ─────┘   text only                     └▶ lyrics_preview → LLM
                                       └▶ find_hook (acapella tease)
audio ─▶ Demucs vocals ─┬─▶ Whisper (recognise)  ─┐
                        └─▶ MMS_FA (align)   ─────┘  word timestamps
```

**Three word-timing sources, best available per word:**

1. **Forced alignment** (`FORCED_ALIGN_ENABLED=true`) — when we already hold the true text, *recognition is the wrong problem*. Meta's MMS_FA aligns the known text to the audio in one forward pass: no autoregressive decode, so it's much faster than Whisper, and the text **cannot be misheard**. It covers 1000+ languages with no per‑language setup (this is why it's preferred over WhisperX's per‑language checkpoints or the Montreal Forced Aligner, which needs a pronunciation lexicon per language). It ships with the normal `--group worker` install via `torchaudio`.
2. **Whisper** (`large-v3`) — recognises the Demucs vocal stem. `large-v3-turbo` was A/B‑tested twice and lost on sparse‑vocal tracks, so the choice is deliberately locked to `large-v3`.
3. **The `.lrc` itself** — an enhanced `<time>` word stamp is the author's own timing and beats any guess; a plain line stamp at least bounds the line.

Every word ends up with a timestamp from whichever of those could place it; nothing is left untimed just because one source went quiet.

**Multilingual.** The decode language is *auto‑detected per song* by default. Pinning it globally (the old behaviour) transcribes a Japanese vocal as English gibberish, so `WHISPER_LANGUAGE` exists only for known‑monolingual libraries. A song whose `.lrc` is in kana / Hangul / bare ideographs gets a per‑song hint, which is far steadier than detection on a reverbed Demucs vocal stem. Lyric matching is script‑aware too: unspaced scripts (Japanese, Chinese, Korean, Thai, …) are aligned character‑by‑character — whitespace tokenising a Japanese line yields one giant "word" and makes timing impossible — while accented Latin folds for fuzzy matching (`café` ≡ `cafe`) without mangling anything else.

**Line-timed alignment and drift.** Given `.lrc` line stamps, the aligner doesn't do one free global match — that's what breaks on a chorus sung three times, where a diff happily binds the second occurrence's text to the third occurrence's audio. Instead the transcript is **partitioned strictly** at the midpoints between consecutive line stamps and matched per line. Because hand‑made `.lrc` files are almost always shifted by a second or two (and sometimes stretched, having come off a different master), a robust Theil‑Sen fit recovers `audio = offset + slope × lrc_time` and corrects for it before matching. The fitted warp is persisted so re‑runs stay stable.

**Why any of this matters:** `vocal_safe_regions` cross‑references those word boundaries against the vocal stem's 10 Hz RMS/peak envelope to find the intervals where a hard cut, stem swap or drop swap won't chop a syllable. The planner only ever places seams there. Whisper‑backed and `.lrc`‑backed words are trusted outright; estimated ones must show real vocal energy in the envelope before they're believed.

## Tech stack

- **Backend:** FastAPI, Celery + Redis, PostgreSQL, SQLAlchemy + Alembic (Python 3.11)
- **Audio/ML:** librosa, Demucs, Whisper (mlx‑whisper on Apple Silicon) + MMS_FA forced alignment (torchaudio), pyrubberband (`rubberband-cli`), numpy / scipy / soundfile, yt‑dlp
- **LLM:** pluggable provider (Gemini / Groq / DigitalOcean / OpenAI‑compatible)
- **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind, TanStack Query, wavesurfer.js, dnd‑kit; Playwright for e2e
- **Infra:** Docker Compose (Postgres, Redis, backend, frontend); the Celery worker runs natively for GPU access, or delegates GPU work to Modal

## Repository layout

```
backend/    FastAPI app, Celery workers, Alembic migrations
frontend/   Next.js app (App Router)
cache/      Generated audio, stems, analyses, mixes (gitignored except .gitkeep)
docs/       Feature designs and agent work logs
deploy/     Droplet deployment script
```

## Prerequisites

```bash
brew install uv fnm docker ffmpeg rubberband
fnm install 20
```

- [`uv`](https://docs.astral.sh/uv/) manages Python 3.11 + deps; [`fnm`](https://github.com/Schniz/fnm) manages Node 20.
- `ffmpeg` is used by yt‑dlp for audio extraction **and** by the file importer for normalisation; `rubberband` (rubberband‑cli) powers time‑stretch / pitch‑shift.
- Stem separation and transcription want a GPU: Apple Silicon (MPS) out of the box, or CUDA on Linux (or Modal — see below).

### Why a native worker?

On macOS, Docker containers can't see the Apple MPS GPU. So Postgres, Redis, the FastAPI backend, and the Next.js frontend run in Docker, while the **Celery worker runs natively on the host** (it needs MPS for Demucs/Whisper and `rubberband-cli` for time‑stretching). `start-dev.sh` runs it with `--group worker`, which pulls torch + demucs + torchaudio; the Docker images deliberately skip that ~2 GB install. On a Linux/CUDA host the worker can move into Compose with no other changes.

## Configuration

```bash
cp .env.example .env
```

`.env` is gitignored — never commit real keys.

Every variable below is read from the environment by `backend/app/core/config.py`.
`.env.example` carries the subset you're most likely to edit; the others take the
defaults shown and can be set whenever you want to override one.

### Core

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` / `REDIS_URL` | local dev values | Overridden to container hostnames inside Compose |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `LOCAL_STORAGE_PATH` | `./cache` | Where generated audio lives |
| `CACHE_MAX_SIZE_GB` | `50` | LRU eviction budget for generated artifacts |
| `LOG_LEVEL` | `INFO` | |

### LLM planning

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | | `gemini` \| `groq` \| `digitalocean` |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `DO_INFERENCE_API_KEY` | | One of these, matching the provider |
| `GROQ_MODEL` / `DO_INFERENCE_MODEL` | see `.env.example` | Per‑provider model id |
| `USE_LLM_PLANNER` | `True` | `False` falls back to the deterministic planner |
| `PLANNER_VERSION` | `v2` | `legacy` restores the v1 free‑form prompt |

### Importing music

| Variable | Default | Purpose |
|---|---|---|
| `MUSIC_MAX_UPLOAD_MB` | `300` | Cap on a single uploaded file |
| `IMPORT_ROOT` | *(empty)* | Server‑side folder import root. Must be a path **inside the backend container** — bind‑mount your library in `docker-compose.yml` (e.g. `- ~/Music:/music:ro`). Empty disables folder import. |
| `YT_DLP_COOKIES_FILE` | *(empty)* | Netscape `cookies.txt` for YouTube. Required on datacenter IPs; leave empty on a residential connection. |

### Lyrics and speech‑to‑text

| Variable | Default | Purpose |
|---|---|---|
| `GENIUS_ACCESS_TOKEN` | *(empty)* | Genius lyric fetch. Optional — `.lrc` sidecars don't need it, and the pipeline degrades gracefully without either. |
| `WHISPER_MODEL` | `large-v3` | Whisper weights. `large-v3-turbo` is faster but measurably worse on sparse vocals. |
| `WHISPER_MODEL_REPO` | *(empty)* | Pin a specific HuggingFace MLX repo; empty derives `mlx-community/whisper-<WHISPER_MODEL>-mlx`. |
| `WHISPER_LANGUAGE` | *(empty)* | **Empty = auto‑detect per song**, which is what a multilingual queue needs. Pin it only for a library you know is monolingual. |
| `FORCED_ALIGN_ENABLED` | `false` | Reuse known lyric text via MMS_FA instead of recognising it. Faster, and the text can't be misheard. Falls back to Whisper automatically when there's no text or the script isn't covered. |
| `FORCED_ALIGN_BACKEND` | `mms_fa` | |

### Transcription offload (Modal) and mix behaviour

| Variable | Default | Purpose |
|---|---|---|
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | *(empty)* | Run stem separation + transcription on Modal GPU workers instead of locally |
| `SECTION_DETECTOR` | `librosa_laplacian` | `allin1` for neural section labels (needs the `pro-analysis` extra) |
| `PITCH_MODE` | `whole_song` | `whole_song` \| `temporary` \| `off` |
| `LOUDNESS_MATCH` / `BASS_SWAP` / `TEMPO_MEET_IN_MIDDLE` | `true` | Mix behaviour toggles |

> **`DEMUCS_MODEL` and `DEMUCS_DEVICE` are currently ignored.** They appear in
> `.env.example` but nothing reads them: the Demucs model is hardcoded per path
> (`htdemucs` locally, `htdemucs_ft` on Modal) and the device is chosen by torch at
> runtime. They're listed here so nobody spends time tuning a knob that does nothing.

### Host voice

| Variable | Default | Purpose |
|---|---|---|
| `TTS_PROVIDER` | `kokoro` | `kokoro` (local Kokoro‑82M, free) \| `openai` (needs `OPENAI_API_KEY`) \| `off` |
| `TTS_VOICE` | `af_heart` | Provider‑specific voice id (`af_heart` / `am_puck`, or `onyx` / `nova` for OpenAI) |
| `HOST_FREQUENCY` | `intro_only` | `off` \| `intro_only` \| `sparse` \| `chatty` |

## Run it

From the repo root:

```bash
./start-dev.sh    # brings up the Docker stack, then runs the native worker in the foreground
```

Then open the app and check health:

```bash
open http://localhost:3000          # the app
curl localhost:8000/health          # -> {"status":"ok","db":"ok","redis":"ok"}
```

Stop everything:

```bash
# Ctrl‑C the worker, then:
./stop-dev.sh
docker compose down -v   # optional: fully wipe past data from the containers
```

## Using the app

1. **Get music in.** Search YouTube and add tracks, or use *Import from disk* to drag in files (drop a `.lrc` next to a track and it's used as its lyrics), or import a whole server folder. Add a few and **drag to reorder**.
2. Click **Done** to lock the queue. The **Processing** view shows each song moving through the pipeline and each transition rendering; failures surface inline with a **Retry**.
3. When the continuous mix is ready it advances to the **Player**, which streams the stitched set and shows the active transition. You can re‑roll any transition, download the mix as an M4A, or switch to per‑song "Queue Mode."
4. Per‑song detail — BPM/key, sections, stems, the transcription and how its lyrics aligned — is under `/songs/<id>/debug`, which is the place to look when a cut lands somewhere unexpected.

## Tests

```bash
# backend (needs Postgres + Redis up: docker compose up -d postgres redis)
cd backend && uv run pytest -q

# frontend
cd frontend && npx tsc --noEmit        # type-check
cd frontend && npm run lint
cd frontend && npm run build
cd frontend && npm run test:e2e        # Playwright (stubbed backend)
```

Linting: `cd backend && uv run ruff check app`.

## Scope

Single‑user, local‑first, no authentication. Not intended for multi‑tenant or public deployment as‑is.
