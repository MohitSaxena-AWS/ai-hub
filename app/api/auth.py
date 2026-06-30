"""JWT bearer authentication and per-session ownership.

A session holds the requester's PII, so it must not be readable by anyone who
guesses its id. Two controls enforce that:

* **Authentication** — every session endpoint requires a valid JWT bearer token.
  The token's ``sub`` claim identifies the calling principal.
* **Ownership** — a session records the principal that created it; only that
  same principal may read it or post to it (otherwise HTTP 403).

Verification uses a shared HS256 secret here, which is self-contained and easy to
demo/test. In production this seam is where you would instead validate tokens
against a corporate IdP's public keys (RS256 + JWKS); the rest of the app is
unaffected. See SECURITY.md.

Authentication can be disabled (``AUTH_ENABLED=false``) for a fully offline demo;
then ``get_principal`` returns ``None`` and ownership checks are skipped.
"""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import Settings, get_settings

# auto_error=False so we can return a 401 with a WWW-Authenticate header (and
# allow the auth-disabled path) instead of FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid authentication token.",
    headers={"WWW-Authenticate": "Bearer"},
)


class Principal(BaseModel):
    """The authenticated caller, identified by the token's ``sub`` claim."""

    subject: str


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal | None:
    """Validate the bearer token and return the calling principal.

    Returns ``None`` when authentication is disabled. Raises HTTP 401 when a
    token is required but missing, malformed, expired, or otherwise invalid.
    """

    settings: Settings = get_settings()
    if not settings.auth_enabled:
        return None

    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.auth_secret,
            algorithms=[settings.auth_algorithm],
            audience=settings.auth_audience,
            issuer=settings.auth_issuer,
            # Verify aud/iss only when configured.
            options={
                "require": ["sub"],
                "verify_aud": settings.auth_audience is not None,
                "verify_iss": settings.auth_issuer is not None,
            },
        )
    except jwt.PyJWTError:
        raise _UNAUTHENTICATED from None

    subject = payload.get("sub")
    if not subject:
        raise _UNAUTHENTICATED
    return Principal(subject=subject)


def authorize_session_owner(session_owner: str | None, principal: Principal | None) -> None:
    """Ensure ``principal`` owns the session; raise HTTP 403 otherwise.

    A no-op when auth is disabled (``principal is None``). With auth enabled,
    every session created under it carries a non-null owner, so a mismatch — or a
    legacy ownerless session — is denied.
    """

    if principal is None:
        return
    if session_owner != principal.subject:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session.",
        )
