# Agent session — 2026-11-05 — Security review remediation

**Branch:** `fix/security-review-hardening`
**Prompt:** a pasted 9-finding security review ("some review for the codebase").
**Decisions taken:** retire the droplet deployment · do **not** rewrite git history ·
do all mechanical fixes, defer the stitch-memory refactor.

## Residual risk and ownership

**The credential is the upstream maintainer's, not this repository owner's.**
`deploy/` was inherited from `Intellligent15/MixMind--AI-DJ`. So there is no
rotation for the owner of this fork to perform — but three things are still true:

1. **It is still a live credential in public history.** Deleting `deploy/` stops
   *future* leaks; it does not revoke an already-published password. The upstream
   repo has carried it since ~2026-07 and this fork republished it. The useful
   action is to **notify the upstream maintainer**, who is the only party that can
   rotate it or destroy the host. Whoever owns that droplet should also assume
   `backend/.env` was readable — i.e. the Modal, Gemini, Groq, DO, Genius and
   OpenAI keys that lived there are burned, plus the YouTube cookie session.
2. **It should not have been in a script at all.** Deploy credentials belong in the
   operator's environment or SSH config, never as a `${VAR:-default}` fallback in a
   committed file. That is why the script is deleted rather than patched — a fork
   that keeps the pattern will keep re-leaking it.
3. **No secrets belonging to this repo's owner are tracked.** Full-history scan
   found no `.env`, `.pem`, `.key`, `cookies.txt` or credential file; only
   `.env.example` is committed. The generated local-stack passwords from this
   session live in the gitignored `.env`.

## Findings and disposition

| # | Finding | Sev | Status |
|---|---|---|---|
| 1 | Root SSH password committed | crit | **Mitigated in tree** — `deploy/` deleted; history deliberately retained; **rotation still required** |
| 2 | Redis published, no password | crit | **Fixed** — loopback bind + `requirepass` from `.env` |
| 3 | Every route unauthenticated | high | **Accepted by design + deployment retired.** No public interface remains; README now states the consequence plainly |
| 4 | Search passes query to yt-dlp | high | **Fixed** — explicit `ytsearch{N}:` term + URL rejection (400) |
| 5 | Postgres public default password | high | **Fixed** — loopback bind + env-driven password, no default |
| 6 | YouTube cookies on every download | med | **Contained** — no public surface left; documented as a credential |
| 7 | Video ids unconstrained | med | **Fixed** — `^[A-Za-z0-9_-]{11}$` |
| 8 | Full mix buffered in memory | med | **Deferred** (see below) |
| 9 | TTS settings declared twice | low | **Fixed** — duplicate block deleted |

Note on #3: the honest reading of the review is that findings **2, 3, 5, 6 are one
root cause** — a local-first app with no auth deployed to a public IP. Retiring the
deployment closes the exploitable surface for all four at once. That is why adding
an API-key layer was not done: it would be auth guarding a host that shouldn't be
reachable, and it is easy to miss one route.

## What changed

**Deleted**
- `deploy/deploy_to_droplet.sh` — contained the live password, used `sshpass`,
  disabled host-key checking, and rsynced to `/root` as root.

**`docker-compose.yml`**
- Every `ports:` entry now binds `127.0.0.1` (postgres, redis, backend, frontend).
  Previously `0.0.0.0`, i.e. reachable from the LAN — reproduced live before the fix.
- Redis: `--requirepass` from `${REDIS_PASSWORD}`, healthcheck authenticates.
- Postgres: `POSTGRES_PASSWORD` from `${POSTGRES_PASSWORD}`.
- Both use `${VAR:?message}` so Compose **fails fast** instead of defaulting to a
  well-known password. Verified: `docker compose config` exits 1 without them.
- Removed droplet-specific comments; kept the `prod` worker profile for Linux hosts.

**`start-dev.sh`**
- Generates `POSTGRES_PASSWORD` / `REDIS_PASSWORD` with `openssl rand -hex 24` into
  the gitignored `.env` on first run (idempotent — verified twice-run is a no-op).
- Derives `DATABASE_URL` / `REDIS_URL` from them and exports both, overriding stale
  values in `backend/.env`.
- Runs `ALTER USER aidj WITH PASSWORD ...` after Postgres is healthy, because
  `POSTGRES_PASSWORD` only applies on first volume init — without this an existing
  volume keeps its old role password and the app can't connect.
- Syncs the two URLs into `backend/.env` so `cd backend && uv run pytest` agrees.

**`main.py`** — dropped the droplet origin from CORS; added `127.0.0.1:3000`.
Comment now states CORS is not authentication.

**`services/youtube/service.py`** — `search()` now builds `ytsearch{limit}:{query}`
explicitly instead of relying on `default_search`, which only applies when the input
is *not* URL-shaped, and rejects `scheme://` / `www.` queries via a new
`SearchQueryRejected`. Deliberately not a bare-domain heuristic: "Mr. Brightside"
is a song. `api/search.py` maps the rejection to 400.

**`schemas/song.py`** — `youtube_video_id` constrained to `^[A-Za-z0-9_-]{11}$`
(the id is interpolated into the download URL and storage keys). Local imports use
their own `local-<hash>` keys and bypass this schema.

**`core/config.py`** — deleted the second, shadowed TTS block (`tts_provider`,
`tts_voice`, `openai_api_key`, `host_frequency` were declared twice; the later one
won, so edits to the first silently did nothing). Reworded the cookies comment.

**`.env.example` / `README.md`** — documented the two generated secrets, removed the
`deploy/` entry, and replaced the one-line "Scope" with a real security section
covering loopback-only access, the worker-as-crown-jewel risk, CORS, search
hardening and cookie handling.

**Tests** — updated `test_songs_api.py` ids to the real 11-char shape; added
`test_youtube_service.py` cases for the explicit search term, six URL-shaped
rejections (incl. `169.254.169.254` and `gopher://`), empty query, and the
"Mr. Brightside" non-regression; added a 400 assertion to `test_search_api.py`.

## Verification

```
uv run pytest -q                      -> 607 passed
uv run ruff check <changed files>     -> all checks passed
docker compose config (no creds)      -> exit 1, fail-fast message
docker compose config (creds set)     -> parses
bash -n start-dev.sh                  -> OK
secret generation logic               -> 48-char hex, distinct, idempotent
```

Not verified: `start-dev.sh` end-to-end (would rebuild the stack). The
`ALTER USER` / `.env` sync paths are untested against a live volume.

## Deferred

- **Finding 8, the in-memory mix.** `stitch_queue.py` does
  `np.concatenate(stitched)` → `sf.write` → ffmpeg → `f.read()` → upload, holding
  the whole set at 44.1 kHz twice. Needs streaming the encode to storage. Deferred
  as a separate change with its own testing.
- **Rotate the credential / destroy the droplet** — operator action, above.
