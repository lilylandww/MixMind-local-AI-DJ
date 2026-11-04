# Agent session — 2026-11-04 — README rewrite

**Branch:** `chore/readme-refresh` (on top of `feat/lrc-multilingual-stt` @ `109ad79`)
**Prompt:** "write the README"

## Goal

Rewrite `README.md` so it matches the code as it now stands. The old one predates
local-file/folder import, `.lrc` sidecars, the multilingual work and forced
alignment, and it documented config that no longer means what it says.

## What changed

- **New: "Getting music in"** — the three sources (YouTube / file upload / server
  folder) as a table, plus a subsection on `.lrc` sidecars: what's parsed (line
  stamps, enhanced `<time>` word stamps, multi-stamp repeats, `[offset:]`, the
  `[mm:ss:ff]` ambiguity) and the "`.lrc` beats Genius, and is never overwritten"
  precedence rule.
- **New: "Lyrics and word timing"** — the section that explains *why* the pipeline
  needs word timing rather than word text, the three timing sources (MMS_FA forced
  alignment → Whisper → the `.lrc` itself), the line-partitioned alignment and
  Theil-Sen drift correction, and how that feeds `vocal_safe_regions`.
- **Configuration** went from 3 bullets to grouped tables covering every
  `Settings` field, with defaults and the non-obvious ones explained.
- **New: "Host voice"** — the TTS host feature was entirely absent from the old
  README despite being real, shipped code (`services/host/`).
- Updated tech stack (MMS_FA/torchaudio), repo layout (`docs/`, `deploy/`),
  prerequisites (ffmpeg also serves the importer), and the usage walkthrough
  (imports, re-rolling, `/songs/<id>/debug`).

## Correctness checks run against the code

Two claims failed verification and were corrected rather than copied forward:

1. **`DEMUCS_MODEL` / `DEMUCS_DEVICE` are dead config** — nothing reads them, same
   bug `WHISPER_MODEL` had before this branch fixed it. The README now says so
   explicitly instead of implying they're tunable.
2. **The Demucs model differs per path** — `htdemucs` locally
   (`services/stems/service.py:106`) vs `htdemucs_ft` on Modal
   (`workers/modal_stubs.py:143`), while `.env.example` advertises `htdemucs_ft`.
   The README states the actual behaviour.

Also fixed a duplicate heading anchor (`Getting music in` appeared twice).

Verified: both relative links resolve; the in-page anchor resolves; no duplicate
heading slugs; every documented env var maps to a real `Settings` field.

## Commands run

```
grep/docker-compose service-name check   # postgres, redis exist -> test cmd valid
grep htdemucs|DEMUCS_ app/               # found the dead-config + model mismatch
python3 anchor/link slug check           # 0 duplicates, anchor OK
grep config-field cross-check            # every documented var is a Settings field
```

No code changed, so tests/linters/build were not re-run for this commit.

## Follow-ups for the codebase (not fixed here)

- **Wire up or delete `DEMUCS_MODEL` / `DEMUCS_DEVICE`**, and decide whether the
  local path should also use `htdemucs_ft` — right now local and Modal separations
  are not reproducible from one another.
- **`.env.example` is missing** `PLANNER_VERSION`, `SECTION_DETECTOR`, `PITCH_MODE`,
  `LOUDNESS_MATCH`, `BASS_SWAP`, `TEMPO_MEET_IN_MIDDLE`, `TTS_PROVIDER`,
  `TTS_VOICE`, `HOST_FREQUENCY`. They all work via the environment; the README says
  so, but they'd be easier to discover in the example file.
- **`backend/app/core/config.py` declares `tts_provider`, `tts_voice`,
  `openai_api_key` and `host_frequency` twice** (lines ~104-113 and ~117-125). The
  duplicates are identical so it's harmless today, but it's a copy-paste artefact.
