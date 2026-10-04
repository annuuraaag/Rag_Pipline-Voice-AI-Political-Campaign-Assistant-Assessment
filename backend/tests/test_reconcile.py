import logging

from app.container import _reconcile
from app.ingestion.service import IngestionService
from tests.conftest import sample


def test_reconcile_deletes_orphan_chunks_and_empty_registry_rows(container, caplog):
    ing: IngestionService = container.ingestion
    kept = ing.ingest(sample("faq/campaign_faq.md"), "faq.md").document
    orphan = ing.ingest(sample("districts/guntur_district_plan.md"), "guntur.md").document
    ghost = ing.ingest(sample("schemes/education_schemes.txt"), "edu.txt").document

    container.registry.remove(orphan.document_id)        # chunks with no registry row
    container.store.delete_document(ghost.document_id)   # registry row with no chunks

    with caplog.at_level(logging.WARNING):
        report = _reconcile(container.store, container.registry)

    assert report == {"deleted_orphan_documents": [orphan.document_id], "removed_registry_rows": [ghost.document_id],
                      "interrupted_uploads": []}
    assert container.store.document_ids() == {kept.document_id}
    assert [r.document_id for r in container.registry.all()] == [kept.document_id]
    assert orphan.document_id in caplog.text and ghost.document_id in caplog.text


def test_reconcile_is_a_noop_when_consistent(container):
    container.ingestion.ingest(sample("faq/campaign_faq.md"), "faq.md")
    assert _reconcile(container.store, container.registry) == {
        "deleted_orphan_documents": [], "removed_registry_rows": [], "interrupted_uploads": []}


def test_reconcile_marks_interrupted_uploads_failed_and_keeps_failed_rows(container):
    doc = container.ingestion.ingest(sample("faq/campaign_faq.md"), "faq.md").document
    container.registry.update(doc.document_id, status="embedding")     # as if the server died mid-upload
    container.store.delete_document(doc.document_id)
    report = _reconcile(container.store, container.registry)
    assert report["interrupted_uploads"] == [doc.document_id]
    rec = container.registry.get(doc.document_id)
    assert rec.status == "failed" and "restart" in rec.error
    assert _reconcile(container.store, container.registry)["removed_registry_rows"] == []  # failed rows stay


def test_reconcile_with_empty_store_clears_registry(container):
    doc = container.ingestion.ingest(sample("faq/campaign_faq.md"), "faq.md").document
    container.store.delete_document(doc.document_id)
    _reconcile(container.store, container.registry)
    assert len(container.registry) == 0
