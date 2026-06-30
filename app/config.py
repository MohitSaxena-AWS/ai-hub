"""Application settings and the configuration-driven prompt.

Two distinct concerns live here:

* ``Settings`` — runtime/infrastructure configuration (Mongo URI, API keys),
  sourced from environment variables / ``.env``.
* ``PromptConfig`` — the assignment's "configuration-driven behaviour": the
  prompt and the field definitions loaded from ``config/prompt.yaml`` at
  startup. Swapping this file changes what the assistant collects and the shape
  of the persisted data, while historical records remain valid thanks to the
  per-record ``schema_version`` / ``prompt_fingerprint`` (see section 4).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (the parent of the ``app`` package). Used to resolve a
# relative ``prompt_config_path`` independently of the process working
# directory, so the config loads the same whether the app is started from the
# repo root, from ``/app`` in the container, or by the test runner.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Infrastructure configuration, read from the environment / ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "ai_hub"
    # Cap how long a Mongo operation waits to find a reachable server, so the
    # readiness probe / first request fail fast (HTTP 503) instead of hanging
    # when the database is down or misconfigured.
    mongo_server_selection_timeout_ms: int = 5000

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # Upper bound on tokens generated per assistant turn. One turn is a short
    # question or a summary, so this caps reply size (and per-turn cost).
    llm_max_tokens: int = 1024

    # Duplicate detection (privacy-preserving). The semantic backend runs
    # locally so request text never leaves the trust boundary.
    similarity_backend: str = "embedding"  # "embedding" | "lexical"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Optional override for the minimum similarity over the free-text business
    # fields above which two same-category requests are treated as duplicates.
    # When unset, each backend uses its own tuned default (see
    # SimilarityProvider.default_threshold), because lexical-overlap and
    # embedding-cosine scores are not on the same scale.
    duplicate_similarity_threshold: float | None = None

    # --- Authentication ---
    # When enabled, every session endpoint requires a valid JWT bearer token and
    # a session may only be accessed by the principal (token ``sub``) that
    # created it. Secure by default; turn off only for a fully offline demo.
    auth_enabled: bool = True
    # Shared secret for verifying HS256 tokens. Required when auth is enabled
    # (startup fails fast otherwise). In production, prefer verifying against a
    # corporate IdP's public keys (RS256/JWKS) — see SECURITY.md.
    auth_secret: str = ""
    auth_algorithm: str = "HS256"
    # Optional expected audience/issuer; verified only when set.
    auth_audience: str | None = None
    auth_issuer: str | None = None

    # Path to the configuration-driven prompt (assignment section 3).
    prompt_config_path: Path = Path("config/prompt.yaml")

    # When true the engine uses a deterministic mock LLM instead of calling the
    # real Claude API. Tests and offline builds rely on this.
    use_mock_llm: bool = False

    # Cost control: only the last N transcript messages are forwarded to the LLM
    # each turn (the full history is still persisted). Caps token spend/latency
    # on long conversations.
    llm_history_window: int = 20

    # Stale conversation sessions are auto-expired by a MongoDB TTL index this
    # many seconds after their last update. Completed requests live in a separate
    # collection and are never affected. Default: 7 days.
    session_ttl_seconds: int = 7 * 24 * 60 * 60

    # Reject user messages longer than this many characters (abuse / runaway
    # token-cost guard) before they ever reach the LLM.
    max_user_message_chars: int = 4000

    # Cap the number of user turns in a single session. Every turn is an LLM
    # call, so an unbounded conversation is a cost/abuse vector; once the cap is
    # reached the session is rejected (HTTP 409) and the requester starts a new
    # one. The message-size cap above bounds one turn; this bounds the count.
    max_session_turns: int = 50

    # --- Observability ---
    # Structured JSON logging by default (machine-parseable for a banking log
    # pipeline); set LOG_FORMAT=text for human-readable local debugging.
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"


class FieldDef(BaseModel):
    """One piece of information the assistant must collect."""

    name: str
    description: str = ""
    # Optional natural-language question used when the engine has to ask for this
    # field directly (the opening message, and the mock assistant). Kept in the
    # config so a different-domain prompt reads naturally instead of relying on a
    # generated phrase derived from ``description``. Falls back to a generated
    # question when empty.
    question: str = ""
    enum: list[str] | None = None
    # Non-PII fields that participate in privacy-preserving duplicate detection.
    business: bool = False
    # Personal data that must never be used as a duplicate-detection key.
    pii: bool = False


class PromptConfig(BaseModel):
    """The parsed ``prompt.yaml`` plus a fingerprint of its raw contents."""

    schema_version: int = 1
    system_prompt: str
    fields: list[FieldDef] = Field(default_factory=list)
    completion_message: str = "Your request has been recorded. This session is now closed."

    # Domain-agnostic conversational text. Kept in the config (not hard-coded in
    # the engine) so that swapping prompt.yaml for a different domain changes the
    # whole user-facing experience, not just the fields collected (assignment
    # section 3). ``opening_message`` is the assistant's first turn; the engine
    # appends a question for the first field when one exists. The duplicate
    # messages must never reference the other request/requester (section 2.3).
    opening_message: str = (
        "Hello! I'm the assistant. I'll help you submit a new request."
    )
    duplicate_prompt_message: str = (
        "It looks like a similar request already exists. Would you like to update "
        "the existing request instead of creating a new one? (yes/no)"
    )
    duplicate_clarify_message: str = (
        "Please answer 'yes' to update the existing request, or 'no' to create a "
        "new one."
    )
    duplicate_updated_message: str = "Your existing request has been updated."

    # Stable hash of the loaded config, stamped onto every persisted request so
    # records can always be traced back to the prompt that produced them.
    fingerprint: str = ""

    @property
    def business_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if f.business]

    @property
    def categorical_business_fields(self) -> list[FieldDef]:
        """Business fields with a fixed value set (e.g. request_type)."""

        return [f for f in self.business_fields if f.enum]

    @property
    def freetext_business_fields(self) -> list[FieldDef]:
        """Free-text business fields (e.g. justification) compared semantically."""

        return [f for f in self.business_fields if not f.enum]

    @property
    def pii_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if f.pii]

    @property
    def required_field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def enum_choices(self) -> dict[str, set[str]]:
        """Map of field name -> allowed values for every enum-constrained field."""

        return {f.name: set(f.enum) for f in self.fields if f.enum}


def load_prompt_config(path: Path) -> PromptConfig:
    """Load and validate the prompt configuration from ``path``."""

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    config = PromptConfig(**data)
    # Fingerprint the normalised parsed content (not raw bytes) so that
    # cosmetic edits like comments/whitespace don't bump the fingerprint.
    canonical = json.dumps(config.model_dump(exclude={"fingerprint"}), sort_keys=True)
    config.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return config


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_prompt_config() -> PromptConfig:
    path = get_settings().prompt_config_path
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return load_prompt_config(path)
