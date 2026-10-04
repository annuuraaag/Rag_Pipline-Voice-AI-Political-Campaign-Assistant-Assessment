"""Re-indexing must never replace an unrelated document or lose the old version."""
import pytest

from app.ingestion.service import UploadRejected

PLAN = b"# Plan\n\n## Hospitals\n\nThe {d} plan adds a 100-bed hospital block and twelve new clinics for residents.\n"


def _upload(client, name: str, data: bytes, **form):
    return client.post("/upload", files={"file": (name, data)}, data=form)


def _plan(district: str, extra: str = "") -> bytes:
    return PLAN.replace(b"{d}", district.encode()) + extra.encode()


def test_same_filename_different_district_is_a_new_document(client):
    a = _upload(client, "plan.md", _plan("Guntur"), district="guntur").json()
    b = _upload(client, "plan.md", _plan("Vijayawada"), district="vijayawada").json()
    assert a["status"] == b["status"] == "indexed"
    assert b["replaced_document_id"] is None
    assert {d["metadata"]["district"] for d in client.get("/documents").json()} == {"guntur", "vijayawada"}


def test_same_filename_same_district_is_reindexed(client):
    a = _upload(client, "plan.md", _plan("Guntur"), district="guntur").json()
    b = _upload(client, "plan.md", _plan("Guntur", "\nAlso a new bus depot.\n"), district="guntur").json()
    assert b["status"] == "reindexed" and b["replaced_document_id"] == a["document"]["document_id"]
    assert [d["document_id"] for d in client.get("/documents").json()] == [b["document"]["document_id"]]


def test_explicit_replace_document_id(client):
    old = _upload(client, "draft.md", _plan("Guntur"), district="guntur").json()["document"]
    r = _upload(client, "final.md", _plan("Guntur", "\nFinal version.\n"), replace_document_id=old["document_id"])
    assert r.json()["status"] == "reindexed" and r.json()["replaced_document_id"] == old["document_id"]
    assert [d["filename"] for d in client.get("/documents").json()] == ["final.md"]
    assert _upload(client, "x.md", _plan("Guntur", "x"), replace_document_id="nope").status_code == 404


def _chunk_doc_ids(container) -> set[str]:
    return {c.metadata.document_id for c in container.store.iter_chunks()}


def test_failed_upsert_keeps_old_version_and_leaves_no_partial_chunks(client, container, monkeypatch):
    old = _upload(client, "plan.md", _plan("Guntur"), district="guntur").json()["document"]

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(container.store, "upsert", boom)
    with pytest.raises(UploadRejected) as err:
        container.ingestion.ingest(_plan("Guntur", "\nv2\n"), "plan.md", {"district": "guntur"})
    assert err.value.status_code == 500
    indexed = [d.document_id for d in container.registry.all() if d.status == "indexed"]
    assert indexed == [old["document_id"]]
    failed = [d for d in container.registry.all() if d.status == "failed"]
    assert len(failed) == 1 and "disk full" in failed[0].error   # the failed attempt stays visible
    assert _chunk_doc_ids(container) == {old["document_id"]}


def test_failed_old_delete_keeps_both_versions_and_warns(client, container, monkeypatch):
    old = _upload(client, "plan.md", _plan("Guntur"), district="guntur").json()["document"]
    real_delete = container.store.delete_document

    def fail_for_old(document_id):
        if document_id == old["document_id"]:
            raise RuntimeError("store timeout")
        real_delete(document_id)

    monkeypatch.setattr(container.store, "delete_document", fail_for_old)
    body = _upload(client, "plan.md", _plan("Guntur", "\nv2\n"), district="guntur").json()
    assert body["status"] == "indexed" and body["replaced_document_id"] is None
    assert body["warnings"] and old["document_id"] in body["warnings"][0]
    ids = {d.document_id for d in container.registry.all()}
    assert ids == {old["document_id"], body["document"]["document_id"]} == _chunk_doc_ids(container)
