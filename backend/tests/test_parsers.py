import pytest

from app.ingestion.parsers import DocumentParseError, parse_document
from tests.conftest import sample


def test_pdf_keeps_pages_and_sections_and_drops_running_footer():
    doc = parse_document(sample("manifesto/sca_manifesto_2026.pdf"), "sca_manifesto_2026.pdf")
    assert doc.file_type == "pdf"
    assert doc.page_count == 6
    assert doc.title.startswith("Sunrise Coast Alliance")
    shield = next(b for b in doc.blocks if "Health Shield is a cashless" in b.text)
    assert shield.page == 2
    assert shield.section_path == ["Healthcare", "SCA Health Shield"]
    # "Sunrise Coast Alliance · People's Manifesto 2026 · Page N" repeats on every page → removed
    assert not any("Page" in b.text and "Manifesto 2026 ·" in b.text for b in doc.blocks)
    vij = next(b for b in doc.blocks if "Budameru canal, building" in b.text)
    assert vij.section_path == ["District Commitments", "Vijayawada"] and vij.page == 6


def test_docx_uses_heading_styles_and_tables():
    doc = parse_document(sample("districts/visakhapatnam_district_plan.docx"), "visakhapatnam_district_plan.docx")
    assert doc.title == "Visakhapatnam District Development Plan"
    sections = {tuple(b.section_path) for b in doc.blocks}
    assert ("Jobs and Industry in Visakhapatnam", "IT and Electronics Corridor") in sections
    rows = [b for b in doc.blocks if b.kind == "table_row"]
    assert any("10,000 rupees per family" in r.text for r in rows)


def test_docx_bullets_are_list_items():
    doc = parse_document(sample("candidate/candidate_profile.docx"), "candidate_profile.docx")
    items = [b for b in doc.blocks if b.kind == "list_item"]
    assert len(items) == 4
    assert all(b.section_path == ["Healthcare Commitments for Vijayawada Central"] for b in items)


def test_markdown_front_matter_and_heading_levels():
    doc = parse_document(sample("districts/guntur_district_plan.md"), "guntur_district_plan.md")
    assert doc.front_matter["district"] == "guntur"
    assert doc.title == "Guntur District Development Plan"
    b = next(b for b in doc.blocks if "1 lakh tonne chilli" in b.text)
    assert b.section_path == ["Agriculture in Guntur", "Chilli Cold Storage"]


def test_txt_detects_numbered_headings():
    doc = parse_document(sample("schemes/education_schemes.txt"), "education_schemes.txt")
    b = next(b for b in doc.blocks if "35,000 rupees" in b.text)
    assert b.section_path == ["Pratibha Scholarship"]
    assert b.page is None


def test_malformed_pdf_is_rejected():
    with pytest.raises(DocumentParseError):
        parse_document(b"%PDF-1.4 not really a pdf", "broken.pdf")


def test_empty_and_unsupported_files_are_rejected():
    with pytest.raises(DocumentParseError):
        parse_document(b"", "empty.txt")
    with pytest.raises(DocumentParseError):
        parse_document(b"hello", "slides.pptx")
    with pytest.raises(DocumentParseError):
        parse_document(b"\n\n   \n", "blank.md")


def test_broken_emoji_from_pdf_extraction_is_cleaned():
    from app.ingestion.parsers import fix_surrogates
    assert fix_surrogates("\ud83d VISUAL UNDERSTANDING") == " VISUAL UNDERSTANDING"   # lone half dropped
    assert fix_surrogates("\ud83d\udcca chart") == "\U0001f4ca chart"                 # valid pair rejoined
    assert fix_surrogates("plain") == "plain"
