"""Supabase Auth: verify the caller's JWT instead of trusting a body field.

Supports both Supabase key styles:
- legacy shared secret (HS256) via ``SUPABASE_JWT_SECRET``
- asymmetric signing keys (RS256/ES256) via the project JWKS endpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Supabase issues access tokens with this audience for signed-in users,
# including anonymous ones.
_AUDIENCE = "authenticated"
_ASYMMETRIC_ALGORITHMS = ["RS256", "ES256"]


class AuthNotConfiguredError(RuntimeError):
    """No way to verify signatures: refuse rather than trust the token."""


@dataclass(frozen=True)
class AuthUser:
    id: UUID
    is_anonymous: bool = False
    email: str | None = None


@lru_cache
def _jwk_client(url: str) -> jwt.PyJWKClient:
    # Caches keys in-process, so only the first request pays the fetch.
    return jwt.PyJWKClient(url)


def _jwks_url(settings: Settings) -> str:
    if settings.supabase_jwks_url:
        return settings.supabase_jwks_url
    if settings.supabase_url:
        base = settings.supabase_url.rstrip("/")
        return f"{base}/auth/v1/.well-known/jwks.json"
    return ""


def decode_token(token: str, settings: Settings) -> dict:
    algorithm = jwt.get_unverified_header(token).get("alg", "")

    if algorithm == "HS256":
        if not settings.supabase_jwt_secret:
            raise AuthNotConfiguredError(
                "Token is HS256 but SUPABASE_JWT_SECRET is not set"
            )
        return jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=_AUDIENCE,
        )

    url = _jwks_url(settings)
    if not url:
        raise AuthNotConfiguredError(
            "Set SUPABASE_URL (or SUPABASE_JWKS_URL) to verify asymmetric tokens"
        )
    signing_key = _jwk_client(url).get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=_ASYMMETRIC_ALGORITHMS,
        audience=_AUDIENCE,
    )


def user_from_token(token: str, settings: Settings) -> AuthUser:
    """Verify a Supabase access token, raising HTTP errors for bad input."""
    try:
        claims = decode_token(token, settings)
    except AuthNotConfiguredError as exc:
        logger.error("Auth is not configured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth is not configured on the server",
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    subject = claims.get("sub")
    try:
        user_id = UUID(str(subject))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has no usable subject",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return AuthUser(
        id=user_id,
        is_anonymous=bool(claims.get("is_anonymous", False)),
        email=claims.get("email") or None,
    )


# auto_error=False so anonymous-but-tokenless callers reach the endpoint and get
# a domain-specific answer instead of a blanket 403 from the security scheme.
_bearer = HTTPBearer(auto_error=False)


def optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> AuthUser | None:
    """The caller, or None when no Authorization header was sent."""
    if credentials is None or not credentials.credentials:
        return None
    return user_from_token(credentials.credentials, settings)


def require_user(user: AuthUser | None = Depends(optional_user)) -> AuthUser:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
