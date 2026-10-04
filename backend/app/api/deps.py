from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, Request

from app.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_api_key(
    x_api_key: str | None = Header(default=None, description="Required when the server sets API_KEY."),
    c: Container = Depends(get_container),
) -> None:
    """Guard for write endpoints. A no-op when API_KEY is unset (local demo mode)."""
    expected = c.settings.api_key
    if not expected:
        return
    # compare_digest: constant-time, so response timing does not leak the key prefix.
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.",
                            headers={"WWW-Authenticate": "ApiKey"})


def resolve_campaign(campaign_id: str | None, c: Container) -> str:
    """Validated campaign id for a request; the server default when the client names none."""
    from app.ingestion.service import normalize_campaign

    try:
        return normalize_campaign(campaign_id or c.settings.default_campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
