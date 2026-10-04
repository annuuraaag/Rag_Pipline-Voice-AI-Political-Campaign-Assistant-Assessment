"""Campaign isolation, document lifecycle, model tracking, token budgets, tables and OCR."""
import pytest

from app.domain import ParsedBlock, ParsedDocument
from app.ingestion.chunker import SectionChunker
from app.ingestion.ocr import OcrEngine
from app.ingestion.parsers import parse_document
from tests.conftest import sample


def upload(client, path: str, campaign: str | None = None, **form):
    name = path.rsplit("/", 1)[-1]
    data = {**form, **({"campaign_id": campaign} if campaign else {})}
    return client.post("/upload", files={"file": (name, sample(path))}, data=data)


# ── 1. campaign isolation ────────────────────────────────────────────────
def test_campaigns_never_see_each_others_documents(client):
    assert upload(client, "districts/guntur_district_plan.md", campaign="alpha").status_code == 200
    assert upload(client, "faq/campaign_faq.md", campaign="beta").status_code == 200

    a = client.post("/retrieve", json={"query": "chilli cold storage volunteer", "campaign_id": "alpha"}).json()
    b = client.post("/retrieve", json={"query": "chilli cold storage volunteer", "campaign_id": "beta"}).json()
    assert a["campaign_id"] == "alpha" and a["candidates"]
    assert {c["document_name"] for c in a["candidates"]} == {"guntur_district_plan.md"}
    assert {c["document_name"] for c in b["candidates"]} == {"campaign_faq.md"}
    assert client.get("/documents", params={"campaign_id": "alpha"}).json()[0]["campaign_id"] == "alpha"
    assert client.post("/retrieve", json={"query": "chilli", "campaign_id": "nobody"}).json()["candidates"] == []


def test_same_file_in_two_campaigns_is_two_documents_and_campaign_delete(client):
    d1 = upload(client, "faq/campaign_faq.md", campaign="alpha").json()["document"]
    d2 = upload(client, "faq/campaign_faq.md", campaign="beta").json()
    assert d2["status"] == "indexed" and d2["document"]["document_id"] != d1["document_id"]
    assert upload(client, "faq/campaign_faq.md", campaign="beta").json()["status"] == "duplicate"

    r = client.delete("/campaigns/alpha").json()
    assert r == {"campaign_id": "alpha", "deleted_documents": 1}
    assert client.get("/documents", params={"campaign_id": "alpha"}).json() == []
    assert client.post("/retrieve", json={"query": "volunteer", "campaign_id": "beta"}).json()["candidates"]


def test_invalid_campaign_id_is_rejected(client):
    assert client.post("/retrieve", json={"query": "x", "campaign_id": "Bad Name!"}).status_code == 422
    assert upload(client, "faq/campaign_faq.md", campaign="../etc").status_code == 422


def test_retriever_refuses_to_search_without_a_campaign(container):
    with pytest.raises(ValueError, match="campaign_id"):
        container.retriever.retrieve("anything")


# ── 2. document lifecycle ────────────────────────────────────────────────
def test_failed_upload_is_recorded_and_can_be_retried(client):
    r = client.post("/upload", files={"file": ("broken.pdf", b"%PDF-1.4 garbage")})
    assert r.status_code == 422
    docs = client.get("/documents").json()
    assert len(docs) == 1 and docs[0]["status"] == "failed" and "Malformed PDF" in docs[0]["error"]
    assert client.get("/health").json()["components"]["documents"]["failed"] == 1

    ok = upload(client, "faq/campaign_faq.md").json()["document"]
    assert ok["status"] == "indexed" and ok["chunk_count"] > 0 and ok["token_count"] > 0


def test_status_moves_through_every_stage(container, monkeypatch):
    seen = []
    real = container.registry.update

    def spy(doc_id, **changes):
        if "status" in changes:
            seen.append(changes["status"])
        return real(doc_id, **changes)

    monkeypatch.setattr(container.registry, "update", spy)
    container.ingestion.ingest(sample("faq/campaign_faq.md"), "faq.md")
    assert seen == ["parsing", "chunking", "embedding", "indexed"]


# ── 3. embedding model on every chunk ────────────────────────────────────
def test_chunks_record_their_embedding_model_and_stale_ones_are_counted(client, container):
    upload(client, "faq/campaign_faq.md")
    chunks = container.store.iter_chunks()
    assert {c.metadata.embedding_model for c in chunks} == {container.embedder.model_name}
    assert container.store.count_stale(container.embedder.model_name) == 0
    assert container.store.count_stale("some-other-model") == len(chunks)
    health = client.get("/health").json()["components"]["index_consistency"]
    assert health["chunks_from_other_embedding_models"] == 0


# ── 4. token budget ──────────────────────────────────────────────────────
def test_section_chunker_budget_uses_the_given_tokenizer():
    def chars(text: str) -> int:          # stand-in tokenizer: 1 token per non-space character
        return len(text.replace(" ", ""))

    doc = ParsedDocument(title="T", file_type="txt", blocks=[
        ParsedBlock(" ".join(f"Sentence {i} is here." for i in range(30)), section_path=["S"])])
    chunks = SectionChunker(max_size=120, min_size=5, measure=chars).chunk(doc)
    assert len(chunks) > 3 and all(chars(c.text) <= 120 for c in chunks)


def test_chunk_token_counts_are_stored(client, container):
    upload(client, "districts/guntur_district_plan.md")
    chunks = container.store.iter_chunks()
    assert all(c.metadata.token_count == container.embedder.count_tokens(c.text) for c in chunks)


# ── 5. tables ────────────────────────────────────────────────────────────
def test_table_rows_repeat_headers_and_live_in_their_own_chunks(client, container):
    doc = parse_document(sample("employment/employment_plan.md"), "employment_plan.md")
    rows = [b.text for b in doc.blocks if b.content_type == "table"]
    assert rows[1] == "District: Guntur | Flagship project: Food processing park | Expected jobs: 6,500"

    upload(client, "employment/employment_plan.md")
    tables = [c for c in container.store.iter_chunks() if c.metadata.content_type == "table"]
    assert tables and all(line.startswith("District:") for c in tables for line in c.text.splitlines())
    body = client.post("/retrieve", json={"query": "jobs Guntur food processing",
                                          "filters": {"content_type": "table"}}).json()
    assert body["results"] and all(r["content_type"] == "table" for r in body["candidates"])


# ── 6. OCR ───────────────────────────────────────────────────────────────
ocr_available = OcrEngine().available


@pytest.mark.skipif(not ocr_available, reason="rapidocr / pypdfium2 not installed")
def test_scanned_pdf_is_ocrd_automatically(client, container):
    container.ingestion.ocr = OcrEngine()
    r = upload(client, "extra/kurnool_water_plan_scanned.pdf", district="kurnool")
    assert r.status_code == 200, r.text
    doc = r.json()["document"]
    assert doc["ocr_pages"] == [1] and doc["content_types"] == {"ocr": doc["chunk_count"]}
    assert any("OCR used" in w for w in r.json()["warnings"])
    text = " ".join(c.text for c in container.store.iter_chunks())
    assert "220 villages" in text and "drip irrigation" in text
    assert "FICTIONAL SAMPLE DOCUMENT" not in text            # boilerplate filter applies to OCR text too
    assert "Page 1" not in text or "Kurnool - Page" not in text
    hit = client.post("/retrieve", json={"query": "pipeline villages Kurnool drinking water"}).json()
    assert hit["results"] and hit["results"][0]["content_type"] == "ocr" and hit["results"][0]["page"] == 1


@pytest.mark.skipif(not ocr_available, reason="rapidocr / pypdfium2 not installed")
def test_image_upload_is_ocrd(client, container):
    container.ingestion.ocr = OcrEngine()
    r = upload(client, "extra/kurnool_water_plan_photo.png")
    assert r.status_code == 200, r.text
    assert r.json()["document"]["file_type"] == "image" and r.json()["document"]["ocr_pages"] == [0]


@pytest.mark.skipif(not ocr_available, reason="rapidocr / pypdfium2 not installed")
def test_scan_with_a_small_text_layer_is_still_ocrd(client, container):
    """Real scans often carry a stamp like 'Scanned with CamScanner': more than 25 characters of
    text, but the content is in the image. A big image + sparse text still triggers OCR."""
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.drawImage(ImageReader(io.BytesIO(sample("extra/kurnool_water_plan_photo.png"))), 20, 60,
                  width=A4[0] - 40, height=A4[1] - 120, preserveAspectRatio=True)
    pdf.drawString(20, 20, "Scanned with CamScanner - Page 1 of 1")
    pdf.save()

    container.ingestion.ocr = OcrEngine()
    r = client.post("/upload", files={"file": ("camscan.pdf", buf.getvalue())})
    assert r.status_code == 200, r.text
    assert r.json()["document"]["ocr_pages"] == [1]
    assert "220 villages" in " ".join(c.text for c in container.store.iter_chunks())


def test_ocr_self_test_reports_status():
    engine = OcrEngine()
    status = engine.self_test()
    assert status == "ready" if ocr_available else status.startswith("error")


def test_scanned_pdf_without_ocr_fails_with_a_clear_reason(client, container):
    container.ingestion.ocr = None
    r = upload(client, "extra/kurnool_water_plan_scanned.pdf")
    assert r.status_code == 422 and "OCR is not available" in r.json()["detail"]
    assert client.get("/documents").json()[0]["status"] == "failed"
