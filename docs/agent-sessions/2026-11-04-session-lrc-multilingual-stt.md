# Agent session — 2026-11-04 — Multilingual STT + `.lrc` lyric sidecars

**Branch:** `feat/lrc-multilingual-stt` (on top of `feat/local-file-import` @ `e02b36f`)
**Prompt:** "this project use whisper for what? can we swap to other model for better
performance on multi-language playlist? Since many files actually have corresponding
`.lrc` files, can we use them to improve the STT performance?"

## Goal

1. Explain Whisper's role, and whether a different model helps a multilingual queue.
2. Fix what actually breaks on a multilingual playlist (three bugs, not the model).
3. Use `.lrc` sidecars as an authoritative, *timed* lyric source — improving word
   boundaries (what the DJ cut-safety needs) rather than transcript wording.

**Answer given:** Whisper transcribes the Demucs vocal stem into timestamped words;
its *timing* matters far more than its *text* (text already comes from Genius). So the
win is not "swap the model" — large-v3 is already strong and turbo was A/B-rejected
twice. The win is better word timestamps and correct language handling.

## What was broken (all fixed)

| # | Bug | Where |
|---|---|---|
| a | Local path forced `language: "en"` while Modal auto-detected → non-English vocals decoded as English gibberish | `services/transcription/service.py:80` |
| b | Aligner's `_clean_word` was `[^a-z0-9]` → every CJK/Hangul/Arabic word normalised to `""` (all "equal", garbage diff) and accented Latin was mangled (`café`→`caf`) | `services/lyrics_alignment/aligner.py:9` |
| c | `WHISPER_MODEL` in `.env` was dead config, never read | `core/config.py` |

Also found: NFKD decomposes Hangul syllables into jamo, so a naive Unicode fix still
breaks Korean. Solved with NFKD → drop combining marks → **NFC** → casefold.

## Design decisions

- **`.lrc` beats Genius** as a lyric source: local, timed, correct for the exact
  recording, works where Genius has nothing. `LyricsSource.lrc` is never overwritten by
  a later Genius scrape.
- **Line-timed constrained alignment** (the core work): a free global diff is what
  breaks on a chorus sung 3× — it binds occurrence 2's text to occurrence 3's Whisper
  words. The transcript is now **partitioned** at midpoints between consecutive warped
  line stamps (strict, non-overlapping), then matched per line.
- **Drift correction**: hand-made `.lrc` files are shifted 0.5–2 s and sometimes
  stretched. A Theil-Sen fit (median of all pairwise slopes — robust to the bogus
  anchors a fuzzy match produces) recovers `audio = offset + slope * lrc_time`, applied
  before partitioning. Persisted to `Lyrics.alignment_offset/alignment_slope`.
- **Timing fallback chain** per token: Whisper per-word DTW → enhanced `<time>` word tag
  → the line's own stamp. Nothing is ever left untimed on the `.lrc` path.
- **Model swap evaluated and declined**, except: `faster-whisper` would help the CUDA
  path (documented, not implemented); WhisperX/MFA lose to MMS_FA on setup cost.
- **Forced alignment (MMS_FA)** behind `FORCED_ALIGN_ENABLED`: given known text, align
  rather than recognise. Faster, and the text cannot be misheard.

## Files changed

**new**
- `backend/app/services/lyrics/tokenizer.py` — script-aware tokenise/normalise; per-char
  tokens for unspaced scripts (CJK/Thai/…); `script_hint()` (kana⇒ja priority rule)
- `backend/app/services/lyrics/lrc.py` — `.lrc` parser: standard, enhanced word-level
  `<time>`, multi-stamp lines, `[offset:]`, `[hh:mm:ss]` + the `[mm:ss:ff]` ambiguity,
  GB18030/Big5/Shift-JIS decoding
- `backend/app/services/lyrics/ingest.py` — `attach_lrc()` upsert
- `backend/app/services/imports/lrc.py` — sidecar discovery + read
- `backend/app/services/lyrics_alignment/forced_align.py` — MMS_FA adapter; pure
  frame-span→word-span mapping + `segments_from_words()`; degrades to Whisper rather
  than emit bad timing
- `backend/alembic/versions/b8f4c2d1a903_add_lrc_source_and_timed_lyrics.py`
- tests: `test_lyrics_tokenizer.py`, `test_lrc_parser.py`, `test_lyrics_aligner_lrc.py`,
  `test_forced_align.py`, `test_lrc_ingest.py`

**changed**
- `services/lyrics_alignment/aligner.py` — Unicode-safe matching; `.lrc` path
  (partition + drift fit + fallback chain); `align_lyrics(..., line_times=)`
- `services/transcription/service.py` — auto-detect language (overridable per song /
  `WHISPER_LANGUAGE`); `WHISPER_MODEL`/`WHISPER_MODEL_REPO` wired up; imports without
  mlx_whisper so its tests run on Linux
- `services/vocal_safety/safety.py` — trust `lrc_word`, corroborate `lrc_line`;
  null-safe confidences (forced-align output has none)
- `workers/transcribe.py` — forced-align path (works for Modal too); `.lrc` script hint
  pins the Whisper decode
- `workers/align_lyrics.py`, `workers/fetch_lyrics.py`, `workers/modal_stubs.py`
- `api/imports.py` — `lrc_file` upload param; folder import picks up sidecars
- `models/lyrics.py`, `models/__init__.py`, `core/config.py`, `.env.example`
- `frontend/lib/api.ts`, `frontend/components/LocalImport.tsx` — send/pair `.lrc`
- `tests/conftest.py`, `tests/test_transcription_service.py` (+10→14 tests, now runs on
  Linux)

## Commands run

```
uv run pytest app/tests/test_{lrc_parser,lyrics_tokenizer,lyrics_aligner,lyrics_aligner_lrc,forced_align,lrc_ingest,transcription_service,lyrics_workers,vocal_safety,lyrics_and_safety_api,imports_api,genius_service}.py -q
  -> 166 passed
uv run pytest app/tests/test_{transcribe_task,lyrics_and_safety_api,songs_api}.py -q -> 60 passed
uv run alembic upgrade head && uv run alembic downgrade -1 && uv run alembic upgrade head  # round-trips
uv run ruff check app alembic          # 2 pre-existing findings, none in new code
npx tsc --noEmit && npm run lint && npm run build   # clean (1 pre-existing api.ts error)
```

## Publish

Repo published to **https://github.com/lilylandww/MixMind-local-AI-DJ** (public), per
user instruction to "create a repo, then push the code". `main` and
`feat/lrc-multilingual-stt` were pushed; `feat/local-file-import` stays local (its
commits are in today's branch history anyway). The pre-existing `origin`
(`Intellligent15/MixMind--AI-DJ`, public, last pushed 2026-07-04) was renamed to
`upstream-public` so nothing is lost and `git push` targets the new repo.

**Nothing large or secret was published.** `.gitignore` already keeps `cache/`,
`node_modules/`, `.venv/`, `.next/`, `cookies.txt` and `.env` local; the largest
tracked file is `backend/uv.lock` at 366 KB and total history is 3.4 MB. Only
`.env.example` is tracked, so no keys left the machine.

## Not done / follow-ups

- **MMS_FA needs a real smoke test** on a machine with `torchaudio` — only its pure
  mapping helpers were testable here (no torch in this env). It covers Latin-script
  languages out of the box; CJK needs `bundle.get_to_phone()` (present in some
  torchaudio builds) and otherwise falls back to Whisper + `.lrc`-constrained timing.
- `faster-whisper` for the Modal/CUDA path (same weights, ~4× faster) — not implemented.
- Embedded lyrics (ID3 `USLT` / FLAC `LYRICS`) — `LyricsSource.embedded` exists but is
  not populated yet.
- Debug UI doesn't yet show `Lyrics.source`, line timestamps, or the fitted drift.
- Pre-existing lint noise left untouched: `services/lyrics/genius.py:38` E402,
  `workers/modal_stubs.py:131` unused `os`, `frontend/lib/api.ts:366` `request<any>`.
