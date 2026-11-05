#!/usr/bin/env bash
# MixMind — bring up the full dev stack.
#
# - Docker: postgres, redis, backend, frontend
# - Native (host): Celery worker (needs MPS access on macOS)
#
# The worker runs in the foreground; Ctrl-C stops it. Run ./stop-dev.sh to tear
# down the Docker stack afterwards.
#
# Everything binds to 127.0.0.1. This app has no authentication by design
# (see README "Scope"), so never publish these ports on a public interface.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# yt-dlp uses ffmpeg for the WAV extraction postprocessor on the host worker.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH. Install with: brew install ffmpeg" >&2
  exit 1
fi

# --- Local stack credentials -------------------------------------------
# Both live in the gitignored root .env and are generated here on first
# run. There is deliberately no default in docker-compose.yml: a committed
# well-known password is precisely how this project leaked a host login.
ENV_FILE="$REPO_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found." >&2
  echo "       cp .env.example .env  # then fill in your API keys" >&2
  exit 1
fi

ensure_secret() {
  local var="$1" val
  grep -qE "^${var}=.+" "$ENV_FILE" 2>/dev/null && return 0
  val="$(openssl rand -hex 24)"
  printf '%s=%s\n' "$var" "$val" >> "$ENV_FILE"
  echo "==> generated ${var} (written to .env)"
}

ensure_secret POSTGRES_PASSWORD
ensure_secret REDIS_PASSWORD

# Read back the last definition of each (append-only, so last wins).
read_env() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2-; }
POSTGRES_PASSWORD="$(read_env POSTGRES_PASSWORD)"
REDIS_PASSWORD="$(read_env REDIS_PASSWORD)"
export POSTGRES_PASSWORD REDIS_PASSWORD

# Host-side URLs for the native worker and alembic. Compose derives its own
# container URLs from the same two passwords, so they cannot drift.
export DATABASE_URL="postgresql+psycopg://aidj:${POSTGRES_PASSWORD}@localhost:5432/aidj"
export REDIS_URL="redis://:${REDIS_PASSWORD}@localhost:6379/0"

# Keep backend/.env's URLs in step for tooling that only reads the file
# (`cd backend && uv run pytest`). Exported env vars already win for
# alembic and the worker; this just stops the file from lying.
if command -v python3 >/dev/null 2>&1 && [[ -f "$REPO_ROOT/backend/.env" ]]; then
  python3 - "$REPO_ROOT/backend/.env" "$DATABASE_URL" "$REDIS_URL" <<'PY'
import pathlib, sys

path, db_url, redis_url = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
lines = p.read_text().splitlines()


def upsert(lines, key, value):
    out, seen = [], False
    for line in lines:
        if line.startswith(key + "="):
            out.append(f"{key}={value}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"{key}={value}")
    return out


lines = upsert(lines, "DATABASE_URL", db_url)
lines = upsert(lines, "REDIS_URL", redis_url)
p.write_text("\n".join(lines) + "\n")
PY
fi

echo "==> Starting Docker services (postgres, redis, backend, frontend)..."
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

echo "==> Waiting for postgres + redis to be healthy..."
for i in {1..30}; do
  pg_status=$(docker compose ps --format json postgres | grep -o '"Health":"[^"]*"' | head -1 || true)
  redis_status=$(docker compose ps --format json redis | grep -o '"Health":"[^"]*"' | head -1 || true)
  if [[ "$pg_status" == '"Health":"healthy"' && "$redis_status" == '"Health":"healthy"' ]]; then
    echo "    postgres + redis healthy."
    break
  fi
  sleep 1
done

# POSTGRES_PASSWORD only takes effect when the data directory is first
# initialised. An existing volume therefore keeps its old role password,
# which would leave the app unable to connect after a credential change.
docker compose exec -T postgres \
  psql -U aidj -d aidj \
  -c "ALTER USER aidj WITH PASSWORD '${POSTGRES_PASSWORD}'" >/dev/null 2>&1 \
  || echo "    warning: could not sync the postgres role password"

cd backend

# Make the host worker share the same cache directory the dockerized backend
# mounts (./cache at the repo root). Without this, the worker's CWD changes
# to backend/ and a relative LOCAL_STORAGE_PATH would land in backend/cache/
# — invisible to the container that serves the audio.
export LOCAL_STORAGE_PATH="${LOCAL_STORAGE_PATH:-$REPO_ROOT/cache}"

# Demucs downloads the htdemucs_ft weights on first run via urllib. macOS
# Python doesn't trust the system root CAs by default, so point urllib +
# requests at certifi's bundle (which is already on the dep tree).
CERTIFI_PEM=$(uv run python -c "import certifi; print(certifi.where())" 2>/dev/null || true)
if [[ -n "$CERTIFI_PEM" ]]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERTIFI_PEM}"
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERTIFI_PEM}"
fi

echo "==> Applying database migrations..."
uv run alembic upgrade head

echo "==> Starting native Celery worker (Ctrl-C to stop)..."
echo "    Backend:  http://localhost:8000/health"
echo "    Frontend: http://localhost:3000"
echo "    Stop docker stack with: ./stop-dev.sh"
echo

# --group worker: pulls torch + demucs + torchaudio (the optional dep
# group defined in pyproject.toml). The dockerized backend skips this
# group entirely so its image stays small; only the native worker needs
# the ML stack.
#
# --pool=threads --concurrency=4: lets the worker run up to 4 tasks
# concurrently in OS threads. Safe now that heavy ML runs on Modal —
# the local worker mostly orchestrates I/O (yt-dlp download, S3
# transfers, blocking modal.Function.remote() calls), all of which
# release the GIL or are network-bound. We avoid --pool=prefork
# specifically because torch + MPS crashes (SIGABRT) under fork on
# macOS; threads sidestep that since they share the parent's memory.
# Bump --concurrency higher only if you want more songs in-flight than
# Modal's free tier wants to spin up at once (each parallel separation
# is its own GPU container).
exec uv run --group worker celery -A app.workers worker --loglevel=info --pool=threads --concurrency=8
