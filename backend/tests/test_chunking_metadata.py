from app.domain import ParsedBlock, ParsedDocument
from app.ingestion.chunker import FixedChunker, SectionChunker, split_sentences
from app.ingestion.metadata import enrich_chunks, resolve_document_metadata
from app.ingestion.parsers import parse_document
from tests.conftest import sample


def test_sentence_splitter_protects_abbreviations():
    s = split_sentences("Dr. Anitha Rao is a doctor. She founded Chinnari Care in 2012. Rs. 500 is paid.")
    assert s == ["Dr. Anitha Rao is a doctor.", "She founded Chinnari Care in 2012.", "Rs. 500 is paid."]


def test_section_chunker_never_crosses_headings_and_respects_budget():
    long_para = " ".join(f"Sentence number {i} talks about hospitals." for i in range(60))
    doc = ParsedDocument(
        title="T", file_type="txt",
        blocks=[
            ParsedBlock(long_para, page=1, section_path=["Healthcare"]),
            ParsedBlock("Schools get labs and libraries in every district. " * 6, page=2, section_path=["Education"]),
        ],
    )
    chunks = SectionChunker(max_size=60, min_size=5).chunk(doc)
    assert all(c.word_count <= 60 for c in chunks)
    assert {tuple(c.section_path) for c in chunks} == {("Healthcare",), ("Education",)}
    assert not any("hospitals" in c.text and "Schools" in c.text for c in chunks)
    health = [c for c in chunks if c.section_path == ["Healthcare"]]
    assert len(health) > 1
    # one-sentence overlap between consecutive pieces of a split section
    assert health[0].text.split(". ")[-1].rstrip(".") in health[1].text


def test_tiny_sections_merge_forward_within_parent():
    doc = ParsedDocument(
        title="T", file_type="md",
        blocks=[
            ParsedBlock("Short intro.", section_path=["Agriculture", "Storage"]),
            ParsedBlock("Fishermen receive an allowance of 10,000 rupees during the ban period each year.",
                        section_path=["Agriculture", "Fisheries"]),
        ],
    )
    chunks = SectionChunker(max_size=100, min_size=10).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].section == "Agriculture › Storage; Fisheries"


def test_fixed_chunker_overlaps():
    doc = ParsedDocument(title="T", file_type="txt", blocks=[ParsedBlock(" ".join(f"w{i}" for i in range(250)))])
    chunks = FixedChunker(max_words=100, overlap_words=20).chunk(doc)
    assert [c.word_count for c in chunks] == [100, 100, 90]
    assert chunks[1].text.startswith("w80")


def test_metadata_precedence_form_over_front_matter_over_manifest():
    doc = parse_document(sample("districts/guntur_district_plan.md"), "guntur_district_plan.md")
    meta = resolve_document_metadata(doc, "guntur_district_plan.md", manifest_entry={"district": "kurnool"})
    assert meta.district == "guntur"  # front matter beats manifest
    meta = resolve_document_metadata(doc, "guntur_district_plan.md", overrides={"district": "Vizag"})
    assert meta.district == "visakhapatnam"  # form override wins, aliases normalised


def test_auto_metadata_without_hints():
    doc = parse_document(sample("districts/vijayawada_district_plan.pdf"), "vijayawada_district_plan.pdf")
    meta = resolve_document_metadata(doc, "vijayawada_district_plan.pdf")
    assert meta.district == "vijayawada"
    assert meta.category == "district_profile"  # district detected + "plan" in filename


def test_auto_category_district_plan_vs_statewide_plan():
    from app.ingestion.metadata import _auto_category

    assert _auto_category("guntur_district_plan.md", "guntur") == "district_profile"
    assert _auto_category("plan.md", "guntur") == "district_profile"
    assert _auto_category("employment_plan.md", "statewide") == "policy"
    assert _auto_category("guntur_manifesto.pdf", "guntur") == "manifesto"
    assert _auto_category("notes.txt", "statewide") == "other"


def test_guntur_plan_without_hints_is_district_profile():
    doc = parse_document(sample("districts/guntur_district_plan.md"), "guntur_district_plan.md")
    doc.front_matter.clear()  # force auto-detection
    meta = resolve_document_metadata(doc, "guntur_district_plan.md")
    assert (meta.district, meta.category) == ("guntur", "district_profile")


def test_statewide_chunk_gets_district_from_its_section_heading():
    doc = parse_document(sample("manifesto/sca_manifesto_2026.pdf"), "m.pdf")
    meta = resolve_document_metadata(doc, "m.pdf", manifest_entry={"district": "statewide", "category": "manifesto"})
    chunks = enrich_chunks(SectionChunker().chunk(doc), doc, meta, document_id="m", document_name="m.pdf",
                           strategy="section", uploaded_at="now")
    vij = next(c for c in chunks if c.metadata.section == "District Commitments › Vijayawada")
    assert vij.metadata.district == "vijayawada"
    assert vij.metadata.page == 6
    assert "[district: Vijayawada" in vij.embed_text and vij.embed_text.endswith(vij.text)
    shield = next(c for c in chunks if "Health Shield" in c.metadata.section)
    assert shield.metadata.district == "statewide" and shield.metadata.topic == "healthcare"


def test_oversized_single_sentence_is_split_on_words():
    run_on = " ".join(f"word{i}" for i in range(450))  # one "sentence": no terminal punctuation
    doc = ParsedDocument(
        title="T", file_type="pdf",
        blocks=[ParsedBlock(run_on, page=3, section_path=["Annexure"]),
                ParsedBlock("• " + " ".join(f"item{i}" for i in range(200)), page=4, section_path=["Annexure"],
                            kind="list_item")],
    )
    chunks = SectionChunker(max_size=180, min_size=25).chunk(doc)
    assert max(c.word_count for c in chunks) <= 180
    assert " ".join(c.text for c in chunks).count("word") == 450  # nothing lost or duplicated
    assert all(c.section_path == ["Annexure"] for c in chunks)
    assert {c.page for c in chunks} == {3, 4}


def test_boilerplate_disclaimer_is_not_indexed(container):
    import re

    from app.ingestion.cleaning import DEFAULT_BOILERPLATE, strip_boilerplate

    doc = parse_document(sample("schemes/education_schemes.txt"), "education_schemes.txt")
    assert "FICTIONAL SAMPLE DOCUMENT" in doc.full_text
    blocks = strip_boilerplate(doc.blocks, [re.compile(p, re.DOTALL) for p in DEFAULT_BOILERPLATE])
    text = " ".join(b.text for b in blocks)
    assert "FICTIONAL SAMPLE DOCUMENT" not in text and "This guide explains" in text
    # end to end through ingestion (settings default patterns)
    container.ingestion.ingest(sample("candidate/candidate_profile.docx"), "candidate_profile.docx")
    assert not any("FICTIONAL" in c.text for c in container.store.iter_chunks())
