"""Claim-check evaluation on synthetic claims built from the sample corpus.

    python eval/run_verifier_eval.py

Every factual sentence of the corpus (one with a figure, place or name) becomes a set of claims
with a known right answer, checked by app/generation/verify.py exactly as it checks answers:

  supported expected      verbatim       the sentence, citing its own passage
                          framed         "According to the campaign documents, <sentence>"
                          trimmed        the sentence cut at its first comma (a shorter paraphrase)
  corrected expected      wrong source   cites another document's passage; its own passage is S2
  unsupported expected    changed figure one number altered (15 lakh → 30 lakh)
                          swapped place  a district plan's district replaced by another district
                          absent         cites a passage from another document, and its own
                                         passage is not in the context

Synthetic claims test whether each check catches the error it targets and how often a correct
sentence is flagged; they do not measure how an LLM paraphrases. The answer evaluation
(run_answer_eval.py, with GROQ_API_KEY and --judge) measures the verifier on real answers.
Needs no embedding model or API key (chunks come from the real chunker; the test embedder only
fills the index). Results: eval/results/verifier_eval.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.container import build_container, seed_sample_data  # noqa: E402
from app.domain import ScoredChunk  # noqa: E402
from app.generation.verify import ClaimVerifier, names, numbers, split_claims  # noqa: E402
from app.ingestion.chunker import split_sentences  # noqa: E402
from app.ingestion.registry import DocumentRegistry  # noqa: E402
from app.lexicon import DISTRICT_ALIASES, DISTRICT_DISPLAY, detect_districts  # noqa: E402
from app.retrieval.embedder import HashingEmbedder  # noqa: E402
from app.retrieval.vector_store import QdrantStore  # noqa: E402

EXPECTED = {"verbatim": "supported", "framed": "supported", "trimmed": "supported", "wrong source": "corrected",
            "changed figure": "unsupported", "swapped place": "unsupported", "absent": "unsupported"}
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)(?!\w)")


def corpus_chunks() -> list:
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(data_dir=Path(tmp), llm_provider="extractive", seed_sample_data=False, log_level="WARNING",
                     _env_file=None)
        c = build_container(s, embedder=HashingEmbedder(), store=QdrantStore("verify-eval", in_memory=True),
                            registry=DocumentRegistry(None), reranker=None)
        seed_sample_data(c, ROOT / "sample_data")
        chunks = c.store.iter_chunks()
        c.store.close()
    return chunks


def facts(chunk) -> list[str]:
    out = []
    for line in chunk.text.splitlines():
        for sent in split_sentences(line.lstrip("•*- ").strip()):
            words = sent.split()
            if 6 <= len(words) <= 45 and (numbers(sent) or detect_districts(sent) or names(sent)) and sent[-1] in ".!?":
                out.append(sent)
    return out


def change_figure(sentence: str) -> str | None:
    m = _NUM.search(sentence)
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    new = n * 2 if n >= 3 else n + 7
    text = f"{int(new):,}" if new == int(new) else f"{new:g}"
    return sentence[: m.start()] + text + sentence[m.end():]


def swap_place(sentence: str, chunk) -> str | None:
    if chunk.metadata.district == "statewide":
        return None
    own = chunk.metadata.district
    others = [d for d in ("guntur", "vijayawada", "visakhapatnam", "kurnool") if d != own and d not in
              chunk.metadata.districts_mentioned]
    for alias in DISTRICT_ALIASES[own]:
        m = re.search(rf"\b{re.escape(alias)}\b", sentence, re.IGNORECASE)
        if m and others:
            return sentence[: m.start()] + DISTRICT_DISPLAY[others[0]] + sentence[m.end():]
    return None


def cases(chunks, rng: random.Random) -> list[dict]:
    by_doc = defaultdict(list)
    for ch in chunks:
        by_doc[ch.metadata.document_id].append(ch)
    out = []
    for ch in chunks:
        others = [o for d, cs in by_doc.items() if d != ch.metadata.document_id for o in cs]
        for sent in facts(ch):
            own_nums = numbers(sent)
            # An "other" passage that happens to state the same figures would make "absent" ambiguous.
            unrelated = [o for o in others if not (own_nums & numbers(o.embed_text, words=True))]
            other = rng.choice(unrelated) if unrelated else None
            out.append({"kind": "verbatim", "claim": f"{sent[:-1]} [S1]{sent[-1]}", "ctx": [ch]})
            out.append({"kind": "framed", "claim": f"According to the campaign documents, {sent[0].lower()}{sent[1:-1]} [S1]{sent[-1]}",
                        "ctx": [ch]})
            if ", " in sent[20:]:
                cut = sent[: sent.index(", ", 20)]
                if len(cut.split()) >= 5:
                    out.append({"kind": "trimmed", "claim": f"{cut} [S1].", "ctx": [ch]})
            changed = change_figure(sent)
            if changed:
                out.append({"kind": "changed figure", "claim": f"{changed[:-1]} [S1]{changed[-1]}", "ctx": [ch]})
            swapped = swap_place(sent, ch)
            if swapped:
                out.append({"kind": "swapped place", "claim": f"{swapped[:-1]} [S1]{swapped[-1]}", "ctx": [ch]})
            if other is not None:
                out.append({"kind": "wrong source", "claim": f"{sent[:-1]} [S1]{sent[-1]}", "ctx": [other, ch]})
                out.append({"kind": "absent", "claim": f"{sent[:-1]} [S1]{sent[-1]}", "ctx": [other]})
    return out


def run(all_cases: list[dict], verifier: ClaimVerifier) -> dict:
    per = defaultdict(lambda: {"n": 0, "correct": 0, "verdicts": defaultdict(int)})
    errors = []
    for case in all_cases:
        ctx = [ScoredChunk(chunk=c, score=1.0) for c in case["ctx"]]
        _, report = verifier.verify(case["claim"], ctx)
        claims = [c for c in report.claims if c.verdict != "no_claim"]
        verdict = claims[0].verdict if claims else "no_claim"
        if case["kind"] == "wrong source" and verdict == "corrected" and claims[0].sources != [2]:
            verdict = "corrected (wrong target)"
        p = per[case["kind"]]
        p["n"] += 1
        p["verdicts"][verdict] += 1
        ok = verdict == EXPECTED[case["kind"]]
        p["correct"] += ok
        if not ok and len(errors) < 40:
            errors.append({"kind": case["kind"], "verdict": verdict, "claim": split_claims(case["claim"])[0][0],
                           "issues": claims[0].issues if claims else []})
    positives = [k for k, v in EXPECTED.items() if v != "unsupported"]
    negatives = [k for k, v in EXPECTED.items() if v == "unsupported"]
    true_ok = sum(per[k]["n"] for k in positives)
    flagged_ok = sum(per[k]["verdicts"]["unsupported"] for k in positives)
    caught = sum(per[k]["verdicts"]["unsupported"] for k in negatives)
    bad = sum(per[k]["n"] for k in negatives)
    return {
        "method": verifier.method,
        "cases": len(all_cases),
        "per_kind": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 3) if v["n"] else None,
                         "verdicts": {x: n for x, n in v["verdicts"].items() if n}} for k, v in per.items()},
        "false_flag_rate": round(flagged_ok / true_ok, 3) if true_ok else None,   # correct claims marked unsupported
        "catch_rate": round(caught / bad, 3) if bad else None,                    # wrong claims marked unsupported
        "errors": errors,
    }


def markdown(r: dict) -> str:
    lines = [
        "# Claim check on synthetic claims", "",
        f"Generated by `eval/run_verifier_eval.py`; do not edit by hand. {r['cases']} claims built from every "
        f"factual sentence of the sample corpus; verifier: **{r['method']}**.", "",
        f"- **Wrong claims caught:** {r['catch_rate']:.1%} (changed figure, swapped place, absent from the context)",
        f"- **Correct claims wrongly flagged:** {r['false_flag_rate']:.1%} (verbatim, framed, trimmed, wrong source)", "",
        "| Claim | expected | n | accuracy | verdicts |", "|---|---|---|---|---|",
    ]
    for kind in EXPECTED:
        v = r["per_kind"].get(kind)
        if v:
            lines.append(f"| {kind} | {EXPECTED[kind]} | {v['n']} | {v['accuracy']:.1%} | {v['verdicts']} |")
    lines += ["", "These claims are built from the corpus, so they test that each check catches the error it "
              "targets; they say little about LLM paraphrases. Answer-level numbers come from "
              "`run_answer_eval.py` with a real LLM and `--judge`.", "", "## Misses (first 40)", "",
              "| kind | verdict | claim | issues |", "|---|---|---|---|"]
    for e in r["errors"]:
        lines.append(f"| {e['kind']} | {e['verdict']} | {e['claim'][:140]} | {'; '.join(e['issues'])[:120]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    all_cases = cases(corpus_chunks(), random.Random(args.seed))
    report = run(all_cases, ClaimVerifier())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verifier_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(report)
    (out / "verifier_eval.md").write_text(md, encoding="utf-8")
    print(md.split("## Misses")[0])
    print(f"wrote {out / 'verifier_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
