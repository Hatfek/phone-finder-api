import hashlib
import hmac
import logging

from fastapi import Header, HTTPException, Request

from app.config import get_settings

log = logging.getLogger(__name__)


def configured_keys() -> tuple[str, ...]:
    raw = get_settings().api_keys
    return tuple(key.strip() for key in raw.split(",") if key.strip())


def fingerprint(key: str) -> str:
    """A stable, non-reversible tag for logs. The key itself is never logged."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def client_ip(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


def _presented(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None),
) -> str:
    """Rejects any request without a key from API_KEYS.

    An empty API_KEYS fails closed with 503 rather than disabling the check, so a
    misconfigured deployment refuses traffic instead of serving every route unguarded.
    """
    keys = configured_keys()
    if not keys:
        log.error("API_KEYS is empty; refusing every authenticated request")
        raise HTTPException(status_code=503, detail="authentication is not configured")

    candidate = _presented(x_api_key, authorization)
    if candidate is None:
        log.warning(
            "auth failure: no key presented ip=%s method=%s path=%s",
            client_ip(request),
            request.method,
            request.url.path,
        )
        raise HTTPException(
            status_code=401,
            detail="missing api key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not any(hmac.compare_digest(candidate, key) for key in keys):
        log.warning(
            "auth failure: unknown key ip=%s method=%s path=%s fingerprint=%s",
            client_ip(request),
            request.method,
            request.url.path,
            fingerprint(candidate),
        )
        raise HTTPException(
            status_code=401,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return candidate
