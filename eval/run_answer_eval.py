"""Answer-level evaluation: does the assistant answer when it should, refuse when it should,
cite the right passages, and stay faithful to them?

    python eval/run_answer_eval.py [--rerank] [--judge]
    docker compose exec backend python eval/run_answer_eval.py --rerank --judge

Every query in eval/queries.jsonl goes through RAGService.answer() (the POST /query code path)
in a fresh conversation, with its `history` turns replayed first. Metrics:

  Decision accuracy   answerable → answered, unanswerable → declined (gate + the model's own refusal)
  Cited               answered questions whose answer carries at least one [S#] citation
  Citation accuracy   answered questions where a cited passage contains gold evidence
  Citation precision  cited passages that contain any gold evidence (strict: other relevant
                      passages count as misses, so this is a lower bound)
  Key-fact recall     answers containing a number from the gold evidence ("15 lakh", "18,000")
  Number faithfulness answers in which every number also appears in the passages given to the
                      model (catches invented figures, the most damaging hallucination here)
  Lexical support     share of the answer's content words found in the cited passages
  Claim check         the runtime verifier (app/generation/verify.py): sentences supported by the
                      passage they cite, supported after moving the citation, or unsupported
  Judge (optional)    --judge asks the configured LLM to count supported vs unsupported claims
                      against the passages; reported separately as it is model-graded. With both,
                      the report also shows how often the verifier and the judge agree per answer.

The LLM comes from the environment (GROQ_API_KEY → Groq); without a key the extractive fallback
answers, which is faithful by construction, so the report labels which one produced the numbers.
Results: eval/results/answer_eval.{json,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import tempfile
import uuid
from pathlib import Path

from harness import CAMPAIGN, ROOT, build_eval_container, describe

from app.domain import MetadataFilter

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[a-z][a-z'-]{3,}")
STOP = set("""about above after again against also among another because been before being below between both
could does doing down during each from further have having here into itself just more most must other over
same should some such than that their theirs them then there these they this those through under until very
were what when where which while will with would your campaign documents document according sources source
proposes propose proposed plans plan says states mentions""".split())


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def numbers(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(s)}


def strip_markers(s: str) -> str:
    return re.sub(r"\s*\[S\d+\]", "", s)


JUDGE_PROMPT = """You check an assistant's answer against the source passages it was given.
Split the ANSWER into its factual claims. For each claim decide if the PASSAGES state it.
Reply with JSON only: {{"supported": <int>, "unsupported": <int>, "unsupported_claims": [<short strings>]}}

PASSAGES:
{passages}

ANSWER:
{answer}"""


async def judge(llm, answer: str, passages: list[str]) -> dict | None:
    msgs = [{"role": "user", "content": JUDGE_PROMPT.format(passages="\n\n".join(passages), answer=answer)}]
    try:
        raw = await llm.complete(msgs, max_tokens=300)
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return {"supported": int(data.get("supported", 0)), "unsupported": int(data.get("unsupported", 0)),
                "unsupported_claims": data.get("unsupported_claims", [])[:5]}
    except Exception as exc:  # noqa: BLE001 - a judge failure is reported, not fatal
        return {"error": str(exc)[:120]}


async def evaluate(c, queries: list[dict], use_judge: bool) -> list[dict]:
    rows = []
    for q in queries:
        session = uuid.uuid4().hex
        for h in q.get("history", []):
            await c.rag.answer(h, session_id=session, campaign_id=CAMPAIGN)
        r = await c.rag.answer(q["query"], session_id=session, campaign_id=CAMPAIGN)
        context = {x.chunk_id: x.text for x in r.retrieval.results}
        cited = [x for x in r.citations if x.cited]
        cited_text = [context.get(x.chunk_id, "") for x in cited]
        gold_ev = [norm(g["evidence"]) for g in q.get("gold", [])]
        row = {
            "id": q["id"], "type": q["type"], "query": q["query"], "answerable": q["answerable"],
            "answered": r.answerable, "refusal_reason": r.refusal_reason, "answer": r.answer,
            "llm": f"{r.llm.provider}/{r.llm.model}", "cited": [x.source_id for x in cited],
        }
        row["decision_correct"] = r.answerable == q["answerable"]
        if r.answerable:
            answer = strip_markers(r.answer)
            ctx_all = " ".join(context.values())
            row["has_citation"] = bool(cited)
            row["answer_numbers_grounded"] = numbers(answer) <= numbers(ctx_all)
            words = [w for w in WORD.findall(answer.lower()) if w not in STOP]
            support_pool = set(WORD.findall(" ".join(cited_text).lower())) if cited_text else set()
            row["lexical_support"] = round(sum(w in support_pool for w in words) / len(words), 3) if words else None
            if gold_ev:
                hits = [any(ev in norm(t) for ev in gold_ev) for t in cited_text]
                row["citation_accurate"] = any(hits)
                row["citation_precision"] = round(sum(hits) / len(hits), 3) if hits else 0.0
                gold_nums = set().union(*(numbers(g["evidence"]) for g in q["gold"]))
                if gold_nums:
                    row["key_fact"] = bool(numbers(answer) & gold_nums)
            if r.verification is not None:
                v = r.verification
                row["claims"] = {"supported": v.supported, "corrected": v.corrected, "unsupported": v.unsupported}
                row["unsupported_claims"] = [x.text for x in v.claims if x.verdict == "unsupported"][:3]
            if use_judge and not c.llm.is_fallback:
                row["judge"] = await judge(c.llm, answer, list(context.values()))
        rows.append(row)
    return rows


def rate(rows: list[dict], key: str) -> tuple[float | None, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (round(sum(map(float, vals)) / len(vals), 3) if vals else None), len(vals)


def summarise(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    s = {
        "decision_accuracy": round(sum(r["decision_correct"] for r in rows) / len(rows), 3),
        "answered_answerable": f"{sum(r['answered'] for r in pos)}/{len(pos)}",
        "declined_unanswerable": f"{sum(not r['answered'] for r in neg)}/{len(neg)}",
        "false_answers": [r["id"] for r in neg if r["answered"]],
        "missed_answers": [r["id"] for r in pos if not r["answered"]],
    }
    for key in ("has_citation", "citation_accurate", "citation_precision", "key_fact", "answer_numbers_grounded"):
        s[key], s[f"{key}_n"] = rate(answered, key)
    lex = [r["lexical_support"] for r in answered if r.get("lexical_support") is not None]
    s["lexical_support_mean"] = round(statistics.fmean(lex), 3) if lex else None
    checked = [r["claims"] for r in answered if r.get("claims")]
    if checked:
        tot = {k: sum(x[k] for x in checked) for k in ("supported", "corrected", "unsupported")}
        n_claims = sum(tot.values())
        s["claims_total"] = n_claims
        s["claims_supported"] = round(tot["supported"] / n_claims, 3) if n_claims else None
        s["claims_corrected"] = round(tot["corrected"] / n_claims, 3) if n_claims else None
        s["claims_unsupported"] = round(tot["unsupported"] / n_claims, 3) if n_claims else None
        s["answers_fully_verified"] = round(sum(x["unsupported"] == 0 for x in checked) / len(checked), 3)
    both = [r for r in answered if r.get("claims") and isinstance(r.get("judge"), dict) and "supported" in r["judge"]]
    if both:
        # Per answer: does the verifier's "all claims verified" agree with the judge's "no unsupported claim"?
        s["verifier_judge_agreement"] = round(
            sum((r["claims"]["unsupported"] == 0) == (r["judge"]["unsupported"] == 0) for r in both) / len(both), 3)
        s["verifier_judge_n"] = len(both)
    judged = [r["judge"] for r in answered if isinstance(r.get("judge"), dict) and "supported" in r["judge"]]
    if judged:
        sup = sum(j["supported"] for j in judged)
        uns = sum(j["unsupported"] for j in judged)
        s["judge_supported_claims"] = round(sup / (sup + uns), 3) if sup + uns else None
        s["judge_answers_fully_supported"] = round(sum(j["unsupported"] == 0 for j in judged) / len(judged), 3)
        s["judge_n"] = len(judged)
    return s


def markdown(report: dict) -> str:
    s, cfg = report["summary"], report["config"]

    def pctf(v):
        return "–" if v is None else f"{v:.0%}"
    lines = [
        "# Answer-level evaluation", "",
        f"Generated by `eval/run_answer_eval.py`; do not edit by hand. {report['queries']} queries "
        f"(`eval/queries.jsonl`). LLM: **{cfg['llm']}** · retrieval: {cfg['retrieval']} · gate: {cfg['gate']}.", "",
        "| Metric | Value | n |", "|---|---|---|",
        f"| Decision accuracy (answer vs decline) | {pctf(s['decision_accuracy'])} | {report['queries']} |",
        f"| Answerable questions answered | {s['answered_answerable']} | |",
        f"| Unanswerable questions declined | {s['declined_unanswerable']} | |",
        f"| Answers with a citation | {pctf(s['has_citation'])} | {s['has_citation_n']} |",
        f"| Citation accuracy (a cited passage holds gold evidence) | {pctf(s['citation_accurate'])} | {s['citation_accurate_n']} |",
        f"| Citation precision, strict | {pctf(s['citation_precision'])} | {s['citation_precision_n']} |",
        f"| Key-fact recall (gold number in the answer) | {pctf(s['key_fact'])} | {s['key_fact_n']} |",
        f"| Number faithfulness (no invented figures) | {pctf(s['answer_numbers_grounded'])} | {s['answer_numbers_grounded_n']} |",
        f"| Lexical support by cited passages (mean) | {pctf(s['lexical_support_mean'])} | |",
    ]
    if "claims_total" in s:
        lines += [f"| Claim check: claims supported by the cited passage | {pctf(s['claims_supported'])} | {s['claims_total']} |",
                  f"| Claim check: supported after correcting the citation | {pctf(s['claims_corrected'])} | {s['claims_total']} |",
                  f"| Claim check: claims not found in any passage | {pctf(s['claims_unsupported'])} | {s['claims_total']} |",
                  f"| Claim check: answers with every claim verified | {pctf(s['answers_fully_verified'])} | |"]
    if "verifier_judge_n" in s:
        lines.append(f"| Claim check agrees with the judge (per answer) | {pctf(s['verifier_judge_agreement'])} | {s['verifier_judge_n']} |")
    if "judge_n" in s:
        lines += [f"| Judge: claims supported by passages | {pctf(s['judge_supported_claims'])} | {s['judge_n']} |",
                  f"| Judge: answers with no unsupported claim | {pctf(s['judge_answers_fully_supported'])} | {s['judge_n']} |"]
    lines += ["", f"Wrongly answered: {', '.join(s['false_answers']) or 'none'} · wrongly declined: {', '.join(s['missed_answers']) or 'none'}.", ""]
    if report["config"]["llm"].startswith("extractive"):
        lines.append("The extractive fallback copies sentences from the passages, so faithfulness is high by "
                     "construction here; run with GROQ_API_KEY set to measure the real LLM.")
    lines += ["", "## Per query", "", "| id | type | expected | decision | cited | accurate | key fact | numbers ok |",
              "|---|---|---|---|---|---|---|---|"]
    yn = {True: "yes", False: "no", None: "–"}
    for r in report["rows"]:
        lines.append(f"| {r['id']} | {r['type']} | {'answer' if r['answerable'] else 'decline'} | "
                     f"{'answered' if r['answered'] else 'declined'}{'' if r['decision_correct'] else ' ✗'} | "
                     f"{' '.join(r['cited']) or '–'} | {yn[r.get('citation_accurate')]} | {yn[r.get('key_fact')]} | "
                     f"{yn[r.get('answer_numbers_grounded')]} |")
    return "\n".join(lines) + "\n"


async def main_async(args) -> dict:
    queries = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        c = build_eval_container(Path(tmp), args.rerank)
        c.retriever.retrieve("warm up", MetadataFilter(campaign_id=CAMPAIGN))
        verify = getattr(c.llm, "verify_model", None)
        if verify:
            await verify()
        rows = await evaluate(c, queries, args.judge)
        report = {"config": describe(c), "queries": len(queries), "summary": summarise(rows), "rows": rows}
        aclose = getattr(c.llm, "aclose", None)
        if aclose:
            await aclose()
        c.store.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--judge", action="store_true", help="also grade faithfulness with the configured LLM")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()
    report = asyncio.run(main_async(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(report)
    (out / "answer_eval.md").write_text(md, encoding="utf-8")
    print(md.split("## Per query")[0])
    print(f"wrote {out / 'answer_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
