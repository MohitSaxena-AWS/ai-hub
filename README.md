# AI Hub — Engineering Service Desk

A configuration-driven backend that exposes a conversational AI assistant over a
REST API. The assistant guides an engineer through submitting an engineering
request, gathers the required information, detects possible duplicates without
exposing personal data, and persists the result to MongoDB.

The assistant's behaviour is **driven entirely by a prompt** loaded at startup
(`config/prompt.yaml`). Changing the prompt changes which fields are collected
and the shape of the stored data — and historical records remain intact.

## Quick start

```bash
# Bring up the API + MongoDB. With no API key it runs the deterministic mock
# assistant; set ANTHROPIC_API_KEY to use the real Claude API.
docker compose up --build

# The API is now on http://localhost:8000  (Swagger UI at /docs)
curl http://localhost:8000/healthz
```

Then drive a conversation (see [API](#api)) with `curl` or Postman.

## Build pipeline

`build.sh` runs the full series of build steps required by the assignment —
dependency installation, lint/build, tests, packaging into a Docker image, and
an end-to-end test against the packaged stack:

```bash
./build.sh
```

> On Windows, run it from Git Bash or WSL. Requires `python3`, `docker` (with the
> compose plugin) and `curl`.

## Running tests on their own

```bash
python -m venv .venv && source .venv/bin/activate   # .venv/Scripts/activate on Windows
pip install -e ".[dev]"

pytest            # fast, offline unit & integration tests (mock LLM + in-memory Mongo)
```

The end-to-end test runs against a live stack and is executed by `build.sh`; to
run it manually after `docker compose up`:

```bash
E2E_BASE_URL=http://localhost:8000 pytest e2e -q
```

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/sessions` | Start a conversation; returns `session_id` and the opening message. |
| `POST` | `/sessions/{id}/messages` | Send a user message; returns the assistant's reply. |
| `GET`  | `/sessions/{id}` | Inspect session state (status, history, collected fields). |
| `GET`  | `/healthz` | Liveness/readiness probe; pings MongoDB and returns HTTP 503 (`"status":"degraded"`) if it is unreachable. |

Example:

```bash
# Start a session
SID=$(curl -s -X POST http://localhost:8000/sessions | jq -r .session_id)

# Answer the assistant's questions one message at a time
curl -s -X POST http://localhost:8000/sessions/$SID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message": "infrastructure-provisioning"}'

curl -s -X POST http://localhost:8000/sessions/$SID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message": "production"}'
# ...continue until the response status is "completed".
```

When all fields are collected the service either stores a new request and closes
the session, or — if a similar request already exists — asks you to confirm
whether to update the existing one (answer `yes` / `no`).

### Authentication

The application is **secure by default** (`AUTH_ENABLED=true`): every `/sessions`
endpoint requires a JWT bearer token, and a session can only be read or
continued by the principal (token `sub`) that created it (others get HTTP 403).
`/healthz` stays public.

The bundled `docker compose` turns auth **off** so the curl/Postman demo works
with zero setup. To run with auth enabled (verifying HS256 tokens against a
shared secret):

```bash
export AUTH_SECRET=$(openssl rand -hex 32)
AUTH_ENABLED=true docker compose up --build

# Mint a token for the demo and call the API with it:
TOKEN=$(python -c "import jwt,os;print(jwt.encode({'sub':'alice'},os.environ['AUTH_SECRET'],algorithm='HS256'))")
curl -s -X POST http://localhost:8000/sessions -H "Authorization: Bearer $TOKEN"
```

In production, verify tokens against your corporate IdP's public keys
(RS256/JWKS) instead of a shared secret — that swap is isolated to
`app/api/auth.py`. See [`SECURITY.md`](SECURITY.md).

## Configuration

Runtime configuration is read from environment variables (see `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string. |
| `MONGO_DB` | `ai_hub` | Database name. |
| `USE_MOCK_LLM` | `false` | Use the deterministic mock assistant (no API key/network). |
| `ANTHROPIC_API_KEY` | — | Required when `USE_MOCK_LLM=false`. |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Claude model id. |
| `LLM_MAX_TOKENS` | `1024` | Max tokens generated per assistant turn (bounds reply size / per-turn cost). |
| `SIMILARITY_BACKEND` | `embedding` | `embedding` (local semantic) or `lexical` (no ML). One knob: it selects the Docker build variant *and* is baked into the image as the runtime default, so image and runtime never drift. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-embedding model. |
| `DUPLICATE_SIMILARITY_THRESHOLD` | _(backend default)_ | Optional override; unset uses each backend's tuned cut-off (lexical ≈ 0.35, embedding ≈ 0.90 — calibrated in `eval/`). |
| `LLM_HISTORY_WINDOW` | `20` | Max recent transcript messages sent to the LLM per turn (token-cost cap; full history is still stored). |
| `MAX_USER_MESSAGE_CHARS` | `4000` | Reject longer user messages before they reach the LLM (cost/abuse guard). |
| `MAX_SESSION_TURNS` | `50` | Max user turns per session; beyond it `POST .../messages` returns HTTP 409 (cost/abuse guard). |
| `SESSION_TTL_SECONDS` | `604800` | Auto-expire stale sessions after this long (MongoDB TTL). Stored requests are kept. A session reaped mid-conversation makes the next `POST .../messages` return HTTP 410 (vs 409 for a concurrent-update conflict). |
| `MONGO_SERVER_SELECTION_TIMEOUT_MS` | `5000` | Fail fast (don't hang) when MongoDB is unreachable. |
| `LOG_LEVEL` | `INFO` | Root log level. |
| `LOG_FORMAT` | `json` | `json` (structured, one object per line) or `text` (human-readable). |

The assistant's behaviour (system prompt, fields, enums, completion message)
lives in `config/prompt.yaml`.

### Observability

Logs are structured JSON by default (`LOG_FORMAT=json`). Every request carries a
correlation id: the app honours an inbound `X-Request-ID` header or mints one,
stamps it on every log line for that request, and echoes it back in the
response. Each request create/update emits a **PII-free audit event** on the
`ai_hub.audit` logger (`event`, `actor`, `request_id`, `session_id`,
`schema_version`, `prompt_fingerprint`) — never the requester name, employee id,
or justification.

### Using the real Claude API

Provide a key (the app uses the real API automatically when one is present):

```bash
ANTHROPIC_API_KEY=sk-... docker compose up --build
```

### Duplicate-detection backend

By default the image installs the local **embedding** backend and bakes the
model weights in, so semantic (paraphrase) duplicate detection works out of the
box and no text — nor any model download — leaves the container at runtime.

For a smaller, fully offline image without the ML stack, build the lean variant
with the dependency-free lexical backend (catches re-worded requests that share
vocabulary, but not pure synonyms). A single knob, `SIMILARITY_BACKEND`, selects
both what gets installed and what runs:

```bash
SIMILARITY_BACKEND=lexical docker compose up --build
```

> Note: `build.sh` packages the **lean** image (mock LLM + lexical backend) for a
> fast, hermetic build. Because `SIMILARITY_BACKEND` is baked into the image as
> its runtime default, a plain `docker compose up` against that image keeps
> running lexical — image and runtime can't drift. For real semantic detection,
> rebuild the embedding image with `docker compose up --build` (embedding is the
> default when `SIMILARITY_BACKEND` is unset).

## Security

This is a take-home scope, but the design makes a banking-appropriate posture
explicit (privacy-preserving duplicate detection, in-process embeddings, PII
separated at rest, non-root container, no DB port exposed). See
[`SECURITY.md`](SECURITY.md) for the full threat model, what is mitigated, and
the gaps that must be closed before production (authentication first).

## Documentation

See [`DESIGN.md`](DESIGN.md) for the architecture and the rationale behind the
key design decisions, and [`SECURITY.md`](SECURITY.md) for the security review.
