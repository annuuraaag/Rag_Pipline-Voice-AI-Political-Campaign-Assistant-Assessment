import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, BookOpen, ChevronDown, FileText, Gauge, GitBranch, Image, Table2, X, Zap } from "lucide-react";
import { Badge, EmptyState, IconButton, Meter, Tabs } from "./ui.jsx";
import { cx, ms, pages, titleCase } from "../lib/format.js";

const TABS = [
  { id: "sources", label: "Sources", icon: BookOpen },
  { id: "retrieval", label: "Retrieval", icon: GitBranch },
  { id: "latency", label: "Latency", icon: Gauge },
];

export default function Inspector({ message, tab, setTab, focusSource, onClose }) {
  return (
    <aside className="inspector" aria-label="Answer details">
      <div className="inspector-head">
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        <IconButton label="Close details" icon={X} onClick={onClose} />
      </div>
      <div className="inspector-body">
        {!message ? (
          <EmptyState icon={BookOpen} title="Nothing to inspect yet">
            Ask a question to see its sources, how they were retrieved, and where the time went.
          </EmptyState>
        ) : tab === "sources" ? (
          <Sources message={message} focus={focusSource} />
        ) : tab === "retrieval" ? (
          <Retrieval message={message} />
        ) : (
          <Latency message={message} />
        )}
      </div>
    </aside>
  );
}

// ── sources ────────────────────────────────────────────────────────────
const typeIcon = (c) => (c.content_type === "table" ? Table2 : c.content_type === "ocr" ? Image : FileText);

function Sources({ message, focus }) {
  const trace = message.trace;
  const byChunk = useMemo(() => Object.fromEntries((trace?.results || []).map((r) => [r.chunk_id, r])), [trace]);
  const items = message.citations?.length
    ? [...message.citations].sort((a, b) => Number(b.cited) - Number(a.cited))
    : (trace?.results || []).map((r, i) => ({ ...r, source_id: `S${i + 1}`, cited: false, snippet: r.text }));
  const refs = useRef({});
  useEffect(() => {
    if (focus) refs.current[focus]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [focus, message.id]);

  if (trace?.strategy?.startsWith("skipped")) {
    return <EmptyState icon={BookOpen} title="No search needed">This was conversation (a greeting or an introduction), so no documents were consulted.</EmptyState>;
  }
  if (!items.length) {
    return message.status === "retrieving"
      ? <EmptyState icon={BookOpen} title="Searching…" />
      : <EmptyState icon={BookOpen} title="No sources passed the relevance check">The assistant declined to answer rather than guess.</EmptyState>;
  }
  return (
    <div className="sources">
      {items.map((c) => (
        <SourceCard key={c.chunk_id} c={c} full={byChunk[c.chunk_id]?.text} scores={byChunk[c.chunk_id]?.scores}
                    focused={focus === c.source_id} ref={(el) => { refs.current[c.source_id] = el; }} />
      ))}
    </div>
  );
}

function SourceCard({ c, full, scores, focused, ref }) {
  const [open, setOpen] = useState(false);
  const Icon = typeIcon(c);
  const text = open && full ? full : c.snippet;
  // Fusion (RRF) scores are rank-based and tiny; show the calibrated cross-encoder score when the
  // reranker ran, otherwise the semantic (cosine) similarity.
  const s = scores || c.scores || {};
  const reranked = s.rerank !== undefined;
  const score = reranked ? s.rerank : s.dense ?? 0;
  const scoreLabel = reranked ? "Cross-encoder relevance (0–1)" : "Semantic similarity (cosine)";
  return (
    <article ref={ref} className={cx("source", focused && "focused", !c.cited && "uncited")}>
      <div className="source-head">
        <span className="source-num">{c.source_id?.replace("S", "")}</span>
        <div className="min0">
          <div className="source-doc"><Icon size={14} aria-hidden /> <span className="truncate">{c.document_name}</span></div>
          <div className="source-loc">
            {[pages(c), c.section].filter(Boolean).join(" · ") || "Whole document"}
          </div>
        </div>
      </div>
      <p className="source-text">{text}</p>
      <div className="source-foot">
        <Badge>{titleCase(c.district)}</Badge>
        <Badge>{titleCase(c.category)}</Badge>
        {c.content_type && c.content_type !== "text" && <Badge tone="info">{c.content_type === "ocr" ? "OCR" : "Table"}</Badge>}
        <span className="grow" />
        <span className="source-score" data-tip={scoreLabel}>
          <Meter value={score} label={`${scoreLabel}: ${score.toFixed(2)}`} /> {score.toFixed(2)}
        </span>
      </div>
      <div className="source-actions">
        <span className={cx("cited-label", c.cited ? "yes" : "no")}>{c.cited ? "Cited in the answer" : "Retrieved, not cited"}</span>
        {full && full.length > (c.snippet?.length || 0) && (
          <button className="link-btn" onClick={() => setOpen(!open)}>
            {open ? "Show less" : "Show full passage"} <ChevronDown size={13} className={cx(open && "rot")} aria-hidden />
          </button>
        )}
      </div>
    </article>
  );
}

// ── retrieval trace ────────────────────────────────────────────────────
function Retrieval({ message }) {
  const t = message.trace;
  if (!t) return <EmptyState icon={GitBranch} title="Retrieving…" />;
  if (t.strategy?.startsWith("skipped")) {
    return <EmptyState icon={GitBranch} title="Search skipped">{t.strategy.replace("skipped ", "").replace(/[()]/g, "")}: answered without searching or calling the language model.</EmptyState>;
  }
  const c = t.counts || {};
  const inContext = new Set(t.results.map((r) => r.chunk_id));
  const rows = (t.candidates?.length ? t.candidates : t.results).slice(0, 12);
  const passed = t.results.length > 0;
  const steps = [
    { label: "Dense", value: c.dense, hint: "Semantic vector search (bge-small)" },
    { label: "BM25", value: c.bm25, hint: "Keyword search" },
    { label: "Fused", value: c.fused, hint: "Reciprocal rank fusion" },
    ...(c.reranked ? [{ label: "Reranked", value: c.reranked, hint: "Cross-encoder" }] : []),
    { label: "Passed", value: c.passed, hint: "Above the relevance threshold" },
    { label: "Context", value: c.final ?? t.results.length, hint: "Sent to the language model (≤2 per document)" },
  ];
  return (
    <div className="trace">
      <section className="panel">
        <div className="panel-label">Query</div>
        <div className="q-orig">{t.original_query}</div>
        {t.rewritten_query !== t.original_query && (
          <div className="q-rewrite"><ArrowRight size={13} aria-hidden /> <span>{t.rewritten_query}</span></div>
        )}
        {t.rewrite_reasons?.length > 0 && (
          <div className="chips">{t.rewrite_reasons.map((r) => <span className="chip" key={r}>{r}</span>)}</div>
        )}
        <div className="chips">
          {Object.entries(t.filters || {}).map(([k, v]) => <span className="chip chip-filter" key={k}>{k}: {String(v)}</span>)}
          {t.filter_relaxed && <span className="chip chip-warn">district filter relaxed (too few matches)</span>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-label">Pipeline</div>
        <div className="funnel">
          {steps.map((s, i) => (
            <div className="funnel-step" key={s.label} data-tip={s.hint}>
              <span className="funnel-n">{s.value ?? "–"}</span>
              <span className="funnel-l">{s.label}</span>
              {i < steps.length - 1 && <ArrowRight size={12} className="funnel-arrow" aria-hidden />}
            </div>
          ))}
        </div>
        <div className={cx("gate", passed ? "gate-pass" : "gate-fail")}>
          <span>{t.gate === "rerank" ? "Cross-encoder score" : t.gate === "dense" ? "Best semantic similarity" : "Ranking only"}</span>
          <span className="gate-math">{Number(t.top_score).toFixed(3)} {passed ? "≥" : "<"} {t.threshold}</span>
          <Badge tone={passed ? "ok" : "warn"}>{passed ? "Answer" : "Decline"}</Badge>
        </div>
      </section>

      <section className="panel">
        <div className="panel-label">Candidates <span className="muted">· top {rows.length}</span></div>
        <div className="table-wrap">
          <table className="cand">
            <thead>
              <tr><th>#</th><th>Passage</th><th className="num">Dense</th><th className="num">BM25</th><th className="num">RRF</th>{c.reranked ? <th className="num">Rerank</th> : null}</tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.chunk_id} className={cx(inContext.has(r.chunk_id) && "in-ctx")}>
                  <td className="num">{i + 1}</td>
                  <td>
                    <div className="cand-doc truncate" title={r.text}>{r.document_name}</div>
                    <div className="cand-sec truncate">{r.section || "–"}{r.page ? ` · p. ${r.page}` : ""}</div>
                  </td>
                  <td className="num">{fmt(r.scores?.dense)}</td>
                  <td className="num">{r.scores?.bm25_rank ?? "–"}</td>
                  <td className="num">{fmt(r.scores?.rrf, 4)}</td>
                  {c.reranked ? <td className="num">{fmt(r.scores?.rerank)}</td> : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="panel-note">Highlighted rows were sent to the language model.</p>
      </section>
    </div>
  );
}

const fmt = (v, d = 3) => (v === undefined || v === null ? "–" : Number(v).toFixed(d));

// ── latency ────────────────────────────────────────────────────────────
const STAGES = [
  ["rewrite", "Understand & rewrite"], ["embed", "Embed query"], ["dense", "Vector search"], ["bm25", "Keyword search"],
  ["fusion", "Fuse"], ["rerank", "Rerank"], ["threshold", "Relevance gate"], ["context", "Build context"],
  ["retrieval_total", "Retrieval total"], ["first_token", "First token"], ["llm", "Language model"],
];

function Latency({ message }) {
  const l = message.latency;
  const v = message.voice || {};
  if (!l) return <EmptyState icon={Gauge} title="Measuring…" />;
  const rows = STAGES.filter(([k]) => l[k] !== undefined).map(([k, label]) => ({ k, label, v: l[k] }));
  const max = Math.max(...rows.map((r) => r.v), 1);
  // Last word → first audio: the wait to be sure the question was over, then the answer's first sound.
  const e2e = v.first_audio_ms !== undefined && v.endpoint_wait_ms != null ? v.endpoint_wait_ms + v.first_audio_ms : undefined;
  const hero = e2e ?? v.first_audio_ms ?? l.first_token;
  const heroLabel = e2e !== undefined ? "Last word → first spoken word"
    : v.first_audio_ms !== undefined ? "End of question → first spoken word" : "Time to first word";
  return (
    <div className="latency">
      <section className="panel hero-panel">
        <div className="panel-label">{heroLabel}</div>
        <div className="hero-num">{ms(hero)}</div>
        <div className="hero-sub">
          {e2e !== undefined ? `End of question detected after ${ms(v.endpoint_wait_ms)} · first audio ${ms(v.first_audio_ms)} later`
            : `Total ${ms(l.total)} · measured on this request`}
        </div>
      </section>
      {(v.cache || v.final_to_first_token_ms !== undefined || v.endpoint_wait_ms !== undefined) && (
        <section className="panel">
          <div className="panel-label">Voice pipeline</div>
          <dl className="kv">
            <dt>Speculative retrieval</dt>
            <dd>{v.cache === "hit" ? <Badge tone="accent" icon={Zap}>Reused in full</Badge>
              : v.cache === "stage1" ? <Badge tone="accent" icon={Zap}>Candidates reused</Badge>
              : v.cache === "miss" ? <Badge>Not reusable</Badge> : "–"}</dd>
            {v.retrieval_saved_ms > 0 && (<><dt>Retrieval time saved</dt><dd>{ms(v.retrieval_saved_ms)}</dd></>)}
            {v.endpoint_wait_ms != null && (<><dt>Last word → end of question</dt><dd>{ms(v.endpoint_wait_ms)}</dd></>)}
            {v.final_to_first_token_ms != null && (<><dt>Final transcript → first token</dt><dd>{ms(v.final_to_first_token_ms)}</dd></>)}
            {v.first_audio_ms !== undefined && (<><dt>→ first spoken word</dt><dd>{ms(v.first_audio_ms)}</dd></>)}
            {v.partials !== undefined && (<><dt>Partial transcripts</dt><dd>{v.partials} received · {v.speculative_searches} searches · {v.speculative_refines} reranks</dd></>)}
          </dl>
        </section>
      )}
      <section className="panel">
        <div className="panel-label">Server stages (ms)</div>
        <div className="bars" role="table" aria-label="Latency per stage">
          {rows.map((r) => (
            <div className={cx("bar-row", (r.k === "retrieval_total" || r.k === "first_token") && "bar-sum")} key={r.k} role="row"
                 data-tip={`${r.label}: ${r.v.toFixed(1)} ms`}>
              <span className="bar-label" role="cell">{r.label}</span>
              <span className="bar-track" role="cell"><span className="bar" style={{ width: `${Math.max(0.6, (r.v / max) * 100)}%` }} /></span>
              <span className="bar-val" role="cell">{r.v < 10 ? r.v.toFixed(1) : Math.round(r.v)}</span>
            </div>
          ))}
        </div>
        <p className="panel-note">Dense and keyword search run in parallel; "First token" is measured from request start.</p>
      </section>
    </div>
  );
}
