from app.domain import Chunk, ChunkMetadata, MetadataFilter, ScoredChunk
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.context_builder import build_context
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.pipeline import Retriever
from app.retrieval.reranker import rerank
from tests.conftest import sample

CAMPAIGN = MetadataFilter(campaign_id="default")


def _chunk(cid: str, text: str, doc: str = "d1", district: str = "statewide", words: int | None = None,
           mentioned: list[str] | None = None) -> Chunk:
    meta = ChunkMetadata(document_id=doc, document_name=f"{doc}.md", document_title=doc, file_type="md",
                         chunk_id=cid, chunk_index=0, district=district, districts_mentioned=mentioned or [],
                         content_hash=cid, word_count=words or len(text.split()))
    return Chunk(text=text, embed_text=text, metadata=meta)


def _sc(cid: str, rank: int, doc: str = "d1", score: float = 0.5, name: str = "dense") -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid, f"text {cid}", doc), score=score, rank=rank, scores={name: score})


def test_tokenizer_folds_plurals_and_drops_stopwords():
    assert tokenize("What are the healthcare Schemes in Guntur?") == ["healthcare", "scheme", "guntur"]


def test_bm25_ranks_exact_terms_and_applies_district_filter():
    chunks = [
        _chunk("a", "Udayam Breakfast gives a free hot breakfast to students", district="statewide"),
        _chunk("b", "Guntur chilli farmers get cold storage", district="guntur"),
        _chunk("c", "Vijayawada flood canal works", district="vijayawada"),
    ]
    idx = BM25Index(lambda: chunks)
    assert [r.chunk.metadata.chunk_id for r in idx.search("udayam breakfast", 5)] == ["a"]
    hits = idx.search("guntur vijayawada", 5, MetadataFilter(district="guntur"))
    assert [h.chunk.metadata.chunk_id for h in hits] == ["b"]


def test_bm25_rebuilds_after_invalidate_and_handles_empty_corpus():
    corpus: list[Chunk] = []
    idx = BM25Index(lambda: list(corpus))
    assert idx.search("anything", 5) == []
    # (BM25 idf is ≤ 0 for a term in every document, so use a few documents, as in real corpora.)
    corpus += [_chunk("x", "dialysis centre at Gollapudi"), _chunk("y", "school labs"), _chunk("z", "bus passes")]
    assert idx.search("dialysis", 5) == []       # stale until invalidated
    idx.invalidate()
    assert idx.search("dialysis", 5)[0].chunk.metadata.chunk_id == "x"


def test_rrf_rewards_agreement_and_keeps_per_retriever_scores():
    dense = [_sc("a", 1), _sc("b", 2), _sc("c", 3)]
    bm25 = [_sc("c", 1, name="bm25"), _sc("a", 2, name="bm25"), _sc("d", 3, name="bm25")]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25})
    assert [f.chunk.metadata.chunk_id for f in fused][:2] == ["a", "c"]   # found by both
    top = fused[0]
    assert top.scores["dense_rank"] == 1 and top.scores["bm25_rank"] == 2 and "rrf" in top.scores
    assert {f.chunk.metadata.chunk_id for f in fused} == {"a", "b", "c", "d"}


def test_context_builder_caps_chunks_per_document():
    ranked = [_sc("a1", 1, "A"), _sc("a2", 2, "A"), _sc("a3", 3, "A"), _sc("b1", 4, "B"), _sc("c1", 5, "C")]
    ctx = build_context(ranked, top_k=4, max_per_doc=2)
    assert [c.chunk.metadata.chunk_id for c in ctx] == ["a1", "a2", "b1", "c1"]
    # with only one document, the cap gives way so the context is still filled
    ctx = build_context(ranked[:3], top_k=3, max_per_doc=2)
    assert len(ctx) == 3


class FakeReranker:
    model_name = "fake-ce"

    def score(self, query, texts):
        terms = [w for w in query.lower().split() if len(w) > 3]
        return [0.9 if any(w in t.lower() for w in terms) else 0.01 for t in texts]


def test_rerank_reorders_and_records_scores():
    cands = [ScoredChunk(chunk=_chunk("x", "unrelated text"), score=0.03, rank=1, scores={"rrf": 0.03}),
             ScoredChunk(chunk=_chunk("y", "chilli cold storage"), score=0.02, rank=2, scores={"rrf": 0.02})]
    out = rerank(FakeReranker(), "chilli storage", cands)
    assert out[0].chunk.metadata.chunk_id == "y" and out[0].scores["rerank"] == 0.9 and out[0].scores["rrf"] == 0.02


def test_pipeline_with_reranker_gates_per_chunk(container):
    container.ingestion.ingest(sample("districts/guntur_district_plan.md"), "guntur_district_plan.md")
    r = Retriever(container.store, container.embedder, reranker=FakeReranker(), rerank_threshold=0.5)
    r.refresh_count()
    res = r.retrieve("chilli cold storage in Guntur", CAMPAIGN)
    assert res.gate == "rerank" and res.answerable
    assert all(c.scores["rerank"] >= 0.5 for c in res.results)
    assert "bm25" in res.counts and res.counts["reranked"] > 0
    assert not r.retrieve("lunar mining", CAMPAIGN).answerable


def test_district_filter_is_relaxed_when_it_leaves_too_few_candidates(container):
    container.ingestion.ingest(sample("districts/guntur_district_plan.md"), "guntur_district_plan.md")
    r = Retriever(container.store, container.embedder, threshold=0.0, min_filtered=3)
    r.refresh_count()
    res = r.retrieve("chilli cold storage", MetadataFilter(campaign_id="default", district="kurnool"))
    assert res.filter_relaxed and res.results
