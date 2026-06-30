# Design

This document explains the architecture and the reasoning behind the main decisions.

## Overview

```
Client (Postman / curl)
        │  REST/HTTP
        ▼
   FastAPI app  ──────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      ▼
 ConversationEngine ──► LLMClient (Claude tool-use | mock)   /healthz
        │
        ├──► DuplicateService ──► SimilarityProvider (local embeddings | lexical)
        │
        ├──► SessionsRepository ──► MongoDB (sessions)
        └──► RequestsRepository ──► MongoDB (requests)
```

Layers are kept thin and separated by small interfaces (`LLMClient`,
`SimilarityProvider`), so external concerns (Anthropic, embedding models) never
leak into the core logic and are trivially mockable in tests.

## Key decisions

### Configuration-driven behaviour
The system prompt and the list of fields to collect live in `config/prompt.yaml`
and are loaded once at startup. The engine uses the field definitions both to
steer the LLM and to know when the request is complete. Running with a different
prompt yields a different data structure with no code changes.

The **user-facing wording** is configured too, not hard-coded in the engine: the
greeting (`opening_message`), the duplicate-confirmation question
(`duplicate_prompt_message`), its clarification re-ask (`duplicate_clarify_message`)
and the post-update line (`duplicate_updated_message`) all come from the prompt
config alongside `completion_message`. So swapping `prompt.yaml` for a different
domain changes the whole conversation, not only the fields — and the
duplicate-flow text stays subject to the privacy rule (it must reveal nothing
about the other request/requester). `tests/test_config_driven.py` proves the
engine emits the configured text.

### Persistence & historical compatibility
Storage is schema-less (MongoDB) and **append-only**: requests are never
migrated destructively. Every stored request stamps the `schema_version` and a
`prompt_fingerprint` of the prompt that produced it, so records written under an
earlier prompt remain intact and queryable after the prompt — and thus the data
shape — changes. This directly satisfies "preserve historical data even when the
prompt changes".

### Conversation engine — one forced tool call per turn
Each turn the model is asked to call a single `respond` tool that returns both
the natural-language reply **and** the structured fields collected so far plus an
`is_complete` flag. Forcing a tool call gives reliable structured output
alongside the chat reply — better than parsing free text, cheaper than a second
extraction call. Enum constraints from the config are injected into the tool
schema, so the model is constrained to valid values at the API level.

Completion is **validated on our side** (`_all_required_present`) rather than
trusting the model's `is_complete`, guarding against premature completion. Enum
values reported by the model are also re-checked against the configured allowed
set (`_sanitize`): the tool schema already constrains them at the API level, but
an out-of-vocabulary value is dropped server-side so it can never be persisted.

### Privacy-preserving duplicate detection
Two layers, **neither of which reads PII** (name, employee ID, contact details):

1. **Exact** — a SHA-256 `business_hash` over the normalised non-PII business
   fields (request type + environment + justification). Catches identical
   requests instantly.
2. **Semantic** — among existing requests sharing the same *categorical* fields
   (request type + environment), the *free-text* justification is compared with
   a similarity provider. Catches the same request worded differently.

This is a small hybrid of exact keyword matching and semantic matching — the
standard production approach to near-duplicate detection — scaled down to a
service-desk-sized dataset (a linear scan over the narrow candidate set is
plenty; no ANN/LSH needed). When a duplicate is found the requester is asked
whether to **update the existing request**; the prompt never reveals anything
about the other request or requester.

### Local embeddings for the semantic layer
Embeddings are computed **in-process** (sentence-transformers) so request text
never leaves the trust boundary — important for a bank, where the free-text
justification may be sensitive. This was preferred over a cloud embedding API
(which would send text to a third party and add a network/key dependency). The
embedding backend is the **default**: the model weights are baked into the
Docker image at build time, so paraphrase detection works out of the box with
nothing fetched at runtime. A dependency-free **lexical** backend (Jaccard)
sits behind the same interface for offline tests and as an explicit lean,
ML-free option (`SIMILARITY_BACKEND=lexical`). That one variable is the single
source of truth: in the Docker build it both selects what is installed *and* is
baked into the image as the runtime default, so a packaged image always runs the
backend it was built with — no silent build/runtime drift.

Because lexical overlap and embedding cosine live on **different score scales**,
each backend declares its own tuned `default_threshold` (lexical ≈ 0.35,
embedding ≈ 0.90) rather than sharing one number; `DUPLICATE_SIMILARITY_THRESHOLD`
overrides it only when explicitly set. If `embedding` is requested but its
dependencies are missing, the service logs a **warning** before falling back to
lexical, so the degradation is visible rather than silent.

The embedding threshold is **calibrated against a labelled set**, not guessed:
`eval/` holds ~28 hard, same-category justification pairs and a sweep script
(`python -m eval.calibrate --backend embedding`). That measurement is what moved
the default from the naive 0.80 (only ~0.54 precision on same-category pairs — it
would flag half of legitimately distinct requests as duplicates) to **0.90**,
which maximises F1 (P≈0.78, R≈1.00) on the set. The same measurement also showed
the **lexical** backend has *negative* separation on these hard negatives (shared
boilerplate dominates token overlap), confirming it is a dependency-free fallback
only — never a backend to trust for precision. `tests/test_dedup_quality.py`
turns both findings into CI gates; `eval/README.md` documents the seed-set
limitations and the path to a production benchmark.

### Authentication & per-session ownership
A session holds the requester's PII, so access is controlled, not just obscured
by an unguessable id. Every `/sessions` endpoint requires a JWT bearer token; the
token's `sub` is recorded as the session `owner` at creation, and reads/writes
are authorized against it (HTTP 403 on mismatch). It is **secure by default**
(`AUTH_ENABLED=true`) and fails fast at startup if enabled without a secret. The
token-verification logic is isolated to `app/api/auth.py`, so swapping the demo's
shared-secret HS256 for IdP public-key (RS256/JWKS) verification in production
touches nothing else. (`/healthz` stays public; the local demo compose disables
auth for a zero-config curl/Postman walkthrough.) See SECURITY.md.

### Error resilience & cost control
The LLM is the one unreliable, metered dependency, so it is handled defensively:

- **Retries with backoff.** `AnthropicLLMClient` retries transient failures
  (connection drops, rate limits, 5xx) with exponential backoff and does *not*
  retry deterministic 4xx errors. On exhaustion it raises a typed `LLMError`.
- **Graceful degradation.** The engine catches `LLMError` and returns a polite
  "please retry" message **without mutating session state**, so an upstream
  outage never corrupts a conversation — the requester just resends.
- **Bounded prompt window.** Only the last `LLM_HISTORY_WINDOW` transcript
  messages are sent to the model each turn (already-collected fields are re-sent
  separately), capping per-turn token cost and latency on long conversations.
  The full transcript is still persisted for the audit trail.
- **Input size cap.** Messages over `MAX_USER_MESSAGE_CHARS` are rejected at the
  API boundary (HTTP 413) before any LLM call — an abuse / runaway-cost guard.
- **Turn cap.** A session accepts at most `MAX_SESSION_TURNS` user turns; beyond
  that the API returns HTTP 409. The size cap bounds one turn, this bounds the
  count, so an unbounded conversation can't rack up LLM cost indefinitely.
- **Readiness probe.** `/healthz` pings MongoDB and returns 503 when it is
  unreachable, so orchestrators stop routing to a backend that can't serve. The
  Mongo client uses a bounded `serverSelectionTimeoutMS` so the probe (and the
  first request) fail fast instead of hanging when the database is down.
- **Session TTL.** A MongoDB TTL index auto-expires stale sessions after
  `SESSION_TTL_SECONDS`; finalized requests live in a separate collection and
  are never reaped. If a session is reaped *mid-conversation*, the next write
  distinguishes "gone" (HTTP 410) from a concurrent-update conflict (HTTP 409),
  so the caller gets an accurate signal instead of a misleading "retry".

### Observability & audit
- **Structured logging.** Logs are JSON by default (`LOG_FORMAT`), one object
  per line, suitable for a banking log pipeline; `text` is available for local
  debugging.
- **Correlation id.** A middleware honours an inbound `X-Request-ID` (or mints
  one), stamps it on every log line for the request via a `contextvar`, and
  echoes it back in the response — so a single request is traceable end to end.
- **Audit trail.** Each request create/update emits a structured event on the
  dedicated `ai_hub.audit` logger (actor, action, request id, provenance). It is
  deliberately **PII-free** — no requester name, employee id, or justification —
  so the audit stream can be retained/forwarded without leaking personal data.
  The actor is the authenticated principal, threaded from the API into the
  engine purely for the audit (never used for business logic).
- **Lean health probe.** `/healthz` is public, so it returns only
  liveness/readiness; configuration detail (prompt fingerprint, schema version,
  effective backend) is logged at startup instead of exposed unauthenticated.

### Why FastAPI + MongoDB
FastAPI gives async I/O (a good fit for Mongo via `motor` and Claude API calls),
Pydantic validation that maps cleanly onto structured LLM output, and
auto-generated OpenAPI docs for easy Postman/`curl` demos. MongoDB's schema-less
documents are the natural fit for the prompt-driven, evolving data shape.

### Testing strategy
- **Unit / integration tests** run fully offline: an in-memory Mongo
  (`mongomock-motor`), the deterministic mock LLM, and the lexical similarity
  backend. Fast inner loop, no network or API key.
- **One end-to-end test** drives the actual Docker image + a real MongoDB over
  HTTP (via docker-compose) in a single thorough scenario: collect all fields →
  persist → trigger duplicate detection → confirm update. The LLM and similarity
  backends are pinned to their deterministic variants so the build stays
  hermetic. See `e2e/test_e2e.py` and `build.sh`.
- **Embedding integration tests** (`tests/test_embedding_similarity.py`, marked
  `slow`) load the *real* configured sentence-transformers model and validate the
  production-default semantic path end to end at the backend's own threshold —
  paraphrase ranks above unrelated and clears the cut-off, unrelated stays below.
  They are auto-skipped when the `embeddings` extra isn't installed, so the lean
  dev loop stays fast; they run inside the embedding image (weights baked in) and
  after `pip install -e ".[embeddings]"`. This covers the otherwise test-blind gap
  between the lexical/stubbed unit tests and the embedding backend that ships by
  default.

## Deliberate simplifications

These are intentional, to keep the code lean and explainable:

- **Yes/no confirmation is keyword-based**, not full NLU. If the answer is
  ambiguous (both affirmative and negative words present, or neither) the
  assistant re-asks rather than guessing, so it never takes the wrong action.
  To keep the conversation from looping forever on an unparseable reply, the
  re-ask is **bounded** (`_MAX_DUPLICATE_CLARIFY`): after a few ambiguous
  answers the engine takes the safe, non-destructive default — create a *new*
  request rather than risk overwriting the existing one — so the session always
  terminates.
- **Updating a duplicate restamps `schema_version` / `prompt_fingerprint`** to
  the prompt that produced the new payload, so a record's provenance always
  describes the data it currently holds (important for the audit trail when the
  prompt changed between the original request and the update). `created_at` is
  preserved so the record's age is unchanged.
- **Duplicate detection scans only a narrow candidate set.** The exact-hash
  lookup and the categorical candidate filter are both indexed (created at
  startup from the prompt config); the semantic comparison is then a linear scan
  over that small same-category set, appropriate for the expected data volume.
  At larger scale this would move to an ANN vector index.
- **Sessions use optimistic concurrency** (a per-session `version`), not locking.
  Two messages racing on one session is unusual; the loser gets HTTP 409 and
  retries rather than paying for pessimistic locks on the common path.
- **Request persistence and the session save are not one atomic transaction.**
  On the final turn the engine writes the request (and emits the audit event),
  then the API saves the session. If that session save loses the optimistic
  check (concurrent turn → 409) or finds the session gone (TTL-reaped → 410),
  the request is already stored while the client sees an error. The window is
  tiny — both failures require a second write against the *same* session within
  a single turn — and the create path is self-healing: a retry re-runs duplicate
  detection, finds the just-written record by its exact hash, and asks to confirm
  instead of silently double-writing. The update path would re-apply the same
  overwrite idempotently. For strict atomicity in production, wrap both writes in
  a MongoDB multi-document transaction (replica set) or use a transactional
  outbox so the request and session commit together. Kept simple here
  deliberately, with the failure modes understood rather than hidden.

## Extending the system

- **Collect different information:** edit `config/prompt.yaml` (bump
  `schema_version`); no code change. Old records remain valid.
- **Swap the LLM or embedding model:** implement `LLMClient` /
  `SimilarityProvider`, or change the model id via environment variables.
- **New endpoints / queries:** the repositories already expose typed read
  methods; add to the API router as needed.
