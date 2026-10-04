"""Claim-level citation verification: every answer sentence against the passages it cites."""
import pytest

from app.domain import Chunk, ChunkMetadata, ScoredChunk
from app.generation.prompt import REFUSAL_TEXT
from app.generation.verify import ClaimVerifier, citation_status, numbers, split_claims
from tests.conftest import FakeLLM, sample


def _sc(text: str, district: str = "statewide", title: str = "Healthcare Schemes", section: str = "SCA Health Shield",
        mentioned: list[str] | None = None) -> ScoredChunk:
    meta = ChunkMetadata(document_id=title, document_name=f"{title}.md", document_title=title, file_type="md",
                         chunk_id=f"{title}:{abs(hash(text)) % 1000}", chunk_index=0, section=section,
                         district=district, districts_mentioned=mentioned or [])
    return ScoredChunk(chunk=Chunk(text=text, embed_text=f"{title} > {section}\n{text}", metadata=meta), score=0.8)


SOURCES = [
    _sc("Cashless treatment worth up to 15 lakh rupees per family per year for 2,400 procedures. "
        "Who is eligible: families with an annual income below 8 lakh rupees."),
    _sc("Government General Hospital, Vijayawada will receive a 200-bed mother and child block and a new cath lab.",
        district="vijayawada", title="Vijayawada District Plan", section="Healthcare"),
]
V = ClaimVerifier()


def verdicts(answer: str, **kw) -> list[str]:
    return [c.verdict for c in V.verify(answer, SOURCES, **kw)[1].claims]


def test_supported_claims_pass_including_paraphrase():
    assert verdicts("The Health Shield offers cashless treatment worth up to 15 lakh rupees per family per year [S1].") == ["supported"]
    assert verdicts("Families earning under 8 lakh a year are eligible [S1].") == ["supported"]


@pytest.mark.parametrize("answer, issue", [
    ("The Health Shield covers up to 25 lakh rupees per family [S1].", "figure 25"),                 # wrong number
    ("Government General Hospital in Guntur will get a 200-bed block [S2].", "Guntur not covered"),  # wrong district
    ("It also covers free dental implants for all senior citizens [S1].", "content words"),          # not in source
])
def test_unsupported_claims_are_flagged_with_a_reason(answer, issue):
    _, report = V.verify(answer, SOURCES)
    c = report.claims[0]
    assert c.verdict == "unsupported" and report.unsupported == 1
    assert any(issue in i for i in c.issues)


def test_misattributed_and_missing_citations_are_corrected():
    answer, report = V.verify("A 200-bed mother and child block is planned at the hospital [S1]. "
                              "The Health Shield covers 2,400 procedures.", SOURCES)
    assert [c.verdict for c in report.claims] == ["corrected", "corrected"]
    assert answer == ("A 200-bed mother and child block is planned at the hospital [S2]. "
                      "The Health Shield covers 2,400 procedures [S1].")


def test_meta_sentences_and_markers_after_the_full_stop():
    claims = split_claims("The documents do not mention dental care. The Shield covers 2,400 procedures. [S1]")
    assert claims == [("The documents do not mention dental care.", []), ("The Shield covers 2,400 procedures.", [1])]
    assert verdicts("The documents do not mention dental care. The Shield covers 2,400 procedures. [S1]") == \
        ["no_claim", "supported"]


def test_strict_policy_drops_unsupported_sentences_or_refuses():
    answer, report = V.verify("The Health Shield covers 2,400 procedures [S1]. It also covers dental implants [S1].",
                              SOURCES, policy="strict")
    assert answer == "The Health Shield covers 2,400 procedures [S1]." and report.removed == 1
    answer, report = V.verify("It covers dental implants for everyone [S1].", SOURCES, policy="strict")
    assert answer == REFUSAL_TEXT


def test_unchanged_answers_keep_their_exact_text_and_citation_status():
    text = "Vijayawada will get a new cath lab [S2]."
    answer, report = V.verify(text, SOURCES)
    assert answer == text and citation_status(report, 2) is True and citation_status(report, 1) is None


def test_numbers_normalise_separators_and_words():
    assert numbers("₹15,000 for 2,400 procedures, 6.5 km") == {"15000", "2400", "6.5"}
    assert "2" in numbers("within two years", words=True)


def test_cross_encoder_rescues_paraphrases_the_word_check_misses():
    class Fake:
        model_name = "fake"

        def score(self, query, texts):
            return [0.9 for _ in texts]
    claim = "Low-income households qualify when they earn below 8 lakh [S1]."
    assert verdicts(claim) == ["unsupported"]
    assert [c.verdict for c in ClaimVerifier(reranker=Fake()).verify(claim, SOURCES)[1].claims] == ["supported"]


# ── through the API ──────────────────────────────────────────────────────
def test_query_reports_verification_and_flags_an_invented_figure(client, container, fake_llm):
    container.ingestion.ingest(sample("schemes/healthcare_schemes.md"), "healthcare_schemes.md")
    fake_llm.reply = ("The SCA Health Shield covers up to 15 lakh rupees per family per year [S1]. "
                      "It also pays 50,000 rupees for every outpatient visit [S1].")
    body = client.post("/query", json={"query": "How much does the SCA Health Shield cover?"}).json()
    v = body["verification"]
    first, second = v["claims"]
    # The 15 lakh sentence is kept: on the passage it cites, or re-attributed to the one that says it.
    assert v["policy"] == "flag" and first["verdict"] in ("supported", "corrected")
    assert "15 lakh" in body["retrieval"]["results"][first["sources"][0] - 1]["text"]
    assert second["verdict"] == "unsupported" and "figure 50000" in second["issues"][0]
    assert body["answerable"] and "50,000" in body["answer"]               # flag: shown, but marked
    by_number = {c["source_id"]: c for c in body["citations"]}
    assert by_number[f"S{second['sources'][0]}"]["verified"] is False     # a claim citing it failed
    assert "verify" in body["latency_ms"]


def test_strict_policy_removes_the_invented_sentence_in_the_stream(client, container, fake_llm):
    import json
    container.ingestion.ingest(sample("schemes/healthcare_schemes.md"), "healthcare_schemes.md")
    container.rag.verify_policy = "strict"
    fake_llm.reply = "It pays 50,000 rupees for every outpatient visit [S1]."
    with client.stream("POST", "/query", json={"query": "How much does the SCA Health Shield cover?", "stream": True}) as r:
        events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    done = events[-1]
    assert done["type"] == "done" and done["answer"] == REFUSAL_TEXT
    assert done["refusal_reason"] == "unverified" and done["verification"]["removed"] == 1


def test_verification_can_be_switched_off(container):
    container.rag.verify_policy = "off"
    container.ingestion.ingest(sample("schemes/healthcare_schemes.md"), "healthcare_schemes.md")
    container.rag.llm = FakeLLM("It pays 50,000 rupees per visit [S1].")
    import asyncio
    r = asyncio.run(container.rag.answer("How much does the SCA Health Shield cover?"))
    assert r.verification is None and "50,000" in r.answer
