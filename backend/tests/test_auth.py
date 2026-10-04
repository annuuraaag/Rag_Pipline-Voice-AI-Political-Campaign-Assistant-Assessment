import logging

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import sample

KEY = "s3cret-key"


@pytest.fixture
def keyed(settings, container):
    settings.api_key = KEY
    with TestClient(create_app(settings, container)) as tc:
        yield tc


def _upload(client, headers=None):
    return client.post("/upload", files={"file": ("faq.md", sample("faq/campaign_faq.md"))}, headers=headers or {})


def test_write_endpoints_require_key(keyed):
    assert _upload(keyed).status_code == 401
    assert _upload(keyed, {"X-API-Key": "wrong"}).status_code == 401
    r = _upload(keyed, {"X-API-Key": KEY})
    assert r.status_code == 200
    doc_id = r.json()["document"]["document_id"]
    assert keyed.delete(f"/documents/{doc_id}").status_code == 401
    assert keyed.delete(f"/documents/{doc_id}", headers={"X-API-Key": KEY}).status_code == 200


def test_read_endpoints_stay_open_with_key(keyed):
    _upload(keyed, {"X-API-Key": KEY})
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/documents").status_code == 200
    assert keyed.post("/retrieve", json={"query": "volunteer"}).status_code == 200
    assert keyed.post("/query", json={"query": "volunteer"}).status_code == 200


def test_no_key_configured_means_open_writes(client):
    assert _upload(client).status_code == 200


def test_cors_wildcard_only_without_key(settings, caplog):
    assert settings.cors_allow_origins() == ["*"]
    with caplog.at_level(logging.WARNING):
        create_app(settings)
    assert "CORS_ORIGINS=* lets any website" in caplog.text

    settings.api_key = KEY
    assert settings.cors_allow_origins() == []
    settings.cors_origins = "*, http://ui.example"
    assert settings.cors_allow_origins() == ["http://ui.example"]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        create_app(settings)
    assert "ignored because API_KEY is set" in caplog.text


def test_cors_preflight_with_key_and_wildcard_is_not_allowed(settings, container):
    settings.api_key = KEY
    with TestClient(create_app(settings, container)) as tc:
        r = tc.options("/upload", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in r.headers
