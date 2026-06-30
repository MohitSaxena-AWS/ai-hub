# Security review

A focused threat assessment for the banking context. It states what the design
**already mitigates**, then the **gaps that must be closed before production**,
prioritised. The take-home scope intentionally stops at the application boundary;
items below marked _(prod)_ are platform concerns that the deploying team owns.

## Assets & trust boundary

- **Sensitive data:** requester PII (name, employee ID) and the free-text
  business justification (may contain confidential project information).
- **Secrets:** `ANTHROPIC_API_KEY`, MongoDB credentials _(prod)_.
- **Trust boundary:** the application process + MongoDB. The duplicate-detection
  embeddings run **in-process**, so request text never crosses this boundary for
  duplicate detection. The only outbound dependency is the Anthropic API (over
  TLS). Be precise about what that call carries: the conversation turns **and the
  collected fields** are sent to the model each turn — and because the assignment
  requires the assistant to collect the requester's name and employee ID, that
  PII *is* sent to Anthropic once provided (it has to be, for the model to confirm
  it back), as is the free-text justification. What local embeddings buy is that
  the *duplicate-detection* path adds **no further egress** — they do not make the
  conversation itself PII-free. Residual risk and mitigations: Anthropic is a
  third-party processor, so this must be covered by a DPA/zero-retention
  agreement (_prod_); a deployment that cannot send PII off-site at all should run
  the model self-hosted behind the same ``LLMClient`` interface.

## What is already mitigated

| Area | Control |
|------|---------|
| **Authentication & ownership** | JWT bearer auth on every `/sessions` endpoint (`app/api/auth.py`); a session is accessible only to the principal (token `sub`) that created it — others get HTTP 403. Secure by default (`AUTH_ENABLED=true`); startup fails fast if enabled without a secret. |
| **PII in duplicate detection** | Keys are a SHA-256 over **non-PII business fields only** (`hashing.py`); name/employee ID are never read by `DuplicateService`. The "duplicate found" prompt reveals nothing about the other request or requester. |
| **PII at rest** | Stored split: `pii` vs `data`; `business_hash` excludes PII (`models/request.py`). |
| **Data exfiltration** | Embeddings computed locally (`LocalEmbeddingSimilarity`); model weights baked into the image, so no text or model download leaves the container at runtime. |
| **Output integrity** | LLM output is constrained by a forced tool schema with enums, **re-validated server-side** (`_sanitize`), and completion is decided by the server (`_all_required_present`), not the model — limits prompt-injection blast radius (a user can't force premature persistence or out-of-vocabulary values). |
| **Input validation** | Pydantic models; message length bounded (`MAX_USER_MESSAGE_CHARS`, HTTP 413) before any LLM call. |
| **Cost / abuse** | Bounded LLM history window, input size cap, max output tokens, and a per-session turn cap (`MAX_SESSION_TURNS` → HTTP 409). |
| **Lost updates** | Optimistic concurrency on sessions (`version` check → HTTP 409). |
| **Error handling** | LLM failures degrade gracefully without state corruption; internal errors aren't leaked to clients; failure logs carry session id only, not PII. |
| **Audit trail** | Every request create/update emits a structured, **PII-free** audit event (`ai_hub.audit`: actor, action, request id, provenance — no name/employee id/justification). Logs are JSON with a per-request `X-Request-ID` correlation id. The public `/healthz` probe exposes only liveness/readiness, not configuration detail. |
| **Data retention** | TTL index auto-expires stale sessions. |
| **Container** | Runs as a non-root user; MongoDB is **not** published to the host (only reachable on the compose network). |
| **Secret hygiene** | `.env` is git-ignored; `.env.example` ships with a blank key; no secrets in code. |

## Gaps to close before production (prioritised)

1. **Harden the authentication that now exists.** Bearer auth + per-session
   ownership are implemented, but the demo verifies HS256 against a shared
   secret. For production, verify tokens against the corporate IdP's public keys
   (RS256 / JWKS), enforce `exp`/`aud`/`iss`, and source the secret from a
   secrets manager. The verification seam is isolated to `app/api/auth.py`. Also
   enable it in the deployment (`AUTH_ENABLED=true`) — it is off only in the
   local demo compose. _(prod)_
2. **Transport security.** Serve only behind a TLS-terminating gateway; enforce
   HTTPS/HSTS. (Outbound Anthropic traffic is already HTTPS.) _(prod)_
3. **MongoDB hardening.** Enable SCRAM authentication, TLS in transit, and
   encryption at rest; restrict by network policy. Supply DB credentials via a
   secrets manager, not plaintext compose env. _(prod)_
4. **Secrets management.** Move `ANTHROPIC_API_KEY` and DB creds to a secrets
   manager (Vault / cloud KMS); rotate regularly; never log them. _(prod)_
5. **PII protection in depth.** Consider field-level encryption / tokenization
   for `requester_name` and `employee_id`, an access/audit log of who reads
   requests, and a data-subject deletion path (GDPR). Residual risk: a requester
   may type PII into the free-text justification — mitigated against exfiltration
   by local embeddings, but it is stored verbatim, so apply the same protection
   and sanitize on display.
6. **Rate limiting / DoS.** Per-session turn count and message size are already
   bounded in-app; add per-client throttling at the gateway for IP/principal-level
   abuse. _(prod)_
7. **Supply chain.** Pin exact dependency versions and the embedding-model
   revision; scan dependencies (`pip-audit`) and the image (Trivy/Grype) in CI;
   produce an SBOM. Verify the model artifact's integrity at build time.
8. **Container hardening (further).** Read-only root filesystem,
   `no-new-privileges`, dropped Linux capabilities, image scanning. _(prod)_
9. **Audit logging.** _Implemented_ — request create/update emits a structured,
   PII-free audit event (`ai_hub.audit`: actor, action, request id, provenance)
   correlated by `X-Request-ID`. Remaining _(prod)_: ship these to a
   tamper-evident / append-only store with retention, and alert on anomalies.

## Prompt-injection note

User free text reaches the LLM, so prompt injection is in scope. It is contained
rather than eliminated: the model can only act through a typed tool whose enum
values and completion are re-checked on the server, so injection cannot persist
invalid data, skip required fields, or change control flow. The remaining effect
is conversational (e.g. odd replies); the justification is stored verbatim, so
treat it as untrusted on any future display surface.
