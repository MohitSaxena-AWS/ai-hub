#!/usr/bin/env bash
#
# Full build pipeline (assignment sections 6 & 7):
#   1. dependency installation
#   2. build / lint
#   3. unit & integration tests (offline, deterministic)
#   4. package an executable Docker image
#   5. end-to-end test against the packaged stack (docker-compose)
#
# Run from the repository root:  ./build.sh
# Requires: python3, docker (with the compose plugin), curl.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

echo "==> [1/5] Dependency installation"
"$PYTHON" -m venv .venv
# Activate the venv on both Unix (bin) and Git Bash on Windows (Scripts).
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  source .venv/Scripts/activate
fi
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "==> [2/5] Build / lint"
ruff check app tests e2e eval

echo "==> [3/5] Unit & integration tests (offline)"
pytest -q

echo "==> [4/5] Package Docker image (ai-hub:latest)"
# Pin the deterministic, offline backends so the E2E run is hermetic regardless
# of any ANTHROPIC_API_KEY present in the environment, and build the lean
# (ML-free) image so the build stays fast. SIMILARITY_BACKEND is a single knob:
# it both selects the lean (lexical) build AND is baked in as the image's
# runtime default, so the image packaged here and the stack started in step 5
# can't drift. The semantic backend is covered by the unit tests; the E2E
# exercises the exact-match duplicate layer end to end. Build the embedding
# image instead with SIMILARITY_BACKEND=embedding.
export USE_MOCK_LLM=true
export SIMILARITY_BACKEND=lexical
docker compose build

echo "==> [5/5] End-to-end test against the packaged stack"
docker compose up -d
cleanup() { docker compose down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "Waiting for the app to become healthy..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/healthz >/dev/null 2>&1; then
    echo "App is healthy."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: app did not become healthy in time." >&2
    docker compose logs app >&2 || true
    exit 1
  fi
  sleep 2
done

E2E_BASE_URL=http://localhost:8000 pytest e2e -q

echo "==> Build complete. Executable image: ai-hub:latest"
