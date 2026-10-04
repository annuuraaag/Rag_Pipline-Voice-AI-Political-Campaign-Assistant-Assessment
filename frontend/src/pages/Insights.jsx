import { useCallback, useEffect, useState } from "react";
import {
  Activity, AudioLines, Boxes, BrainCircuit, Database, FileSearch, Layers, RefreshCw, ScanText, SearchCheck, Sparkles,
} from "lucide-react";
import { api } from "../api.js";
import { Badge, Dot, EmptyState, IconButton } from "../components/ui.jsx";
import { ms, titleCase } from "../lib/format.js";

function components(h) {
  const c = h?.components || {};
  return [
    { key: "vector_store", icon: Database, title: "Vector store", ok: c.vector_store?.ok,
      lines: [`Qdrant (${c.vector_store?.mode || "–"})`, `${(c.vector_store?.chunks ?? 0).toLocaleString()} passages`] },
    { key: "embedder", icon: Layers, title: "Embeddings", ok: c.embedder?.ok,
      lines: [c.embedder?.model, `${c.embedder?.dim ?? "–"} dimensions`] },
    { key: "bm25", icon: FileSearch, title: "Keyword index", ok: c.bm25?.ok, lines: ["BM25, in memory", `${c.bm25?.indexed_chunks ?? 0} passages`] },
    { key: "reranker", icon: SearchCheck, title: "Reranker", ok: c.reranker?.ok, warnOnly: true,
      lines: [c.reranker?.ok ? "Cross-encoder active" : "Not loaded: dense-score gate in use", c.reranker?.status] },
    { key: "llm", icon: BrainCircuit, title: "Language model", ok: c.llm?.ok,
      lines: [`${c.llm?.provider || "–"} · ${c.llm?.model || "–"}`, c.llm?.note || c.llm?.model_check] },
    { key: "voice", icon: AudioLines, title: "Voice", ok: c.voice?.ok,
      lines: ["WebSocket streaming + speculative search", c.voice?.server_transcription ? `Server speech-to-text: ${c.voice.server_transcription}` : "Speech-to-text in the browser"] },
    { key: "ocr", icon: ScanText, title: "OCR", ok: c.ocr?.ok, warnOnly: true, lines: [c.ocr?.engine || "Disabled", c.ocr?.status] },
    { key: "documents", icon: Boxes, title: "Documents", ok: !(c.documents?.failed > 0), warnOnly: true,
      lines: [`${c.documents?.count ?? 0} indexed across ${Object.keys(c.documents?.campaigns || {}).length} campaign(s)`,
        c.documents?.failed ? `${c.documents.failed} failed upload(s)` : "No failed uploads"] },
  ];
}

const SHOWN = [
  ["query_stream", "Typed questions (streaming)"], ["voice", "Voice turns"], ["query", "Typed questions (JSON)"], ["retrieve", "Retrieval only"],
];
const KEY_STAGES = ["first_token", "retrieval_total", "rerank", "llm", "total", "final_to_first_token", "retrieval_saved"];
const STAGE_LABEL = {
  first_token: "First token", retrieval_total: "Retrieval", rerank: "Rerank", llm: "Language model", total: "Total",
  final_to_first_token: "Final transcript → first token", retrieval_saved: "Retrieval saved by speculation",
};

export default function Insights({ health, refreshHealth }) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(() => {
    refreshHealth();
    api.metrics().then((m) => { setMetrics(m); setError(null); }).catch((e) => setError(e.message));
  }, [refreshHealth]);
  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  const cfg = health?.config || {};
  return (
    <div className="page">
      <header className="topbar">
        <div className="topbar-title">
          <h1>Insights</h1>
          <span className="topbar-sub">Live system health and measured latency · refreshes every 10 s</span>
        </div>
        <div className="topbar-actions"><IconButton label="Refresh" icon={RefreshCw} onClick={load} /></div>
      </header>
      <div className="page-body">
        <section className="health-grid">
          {components(health).map((c) => {
            const tone = c.ok ? "ok" : c.warnOnly ? "warn" : "err";
            return (
              <article key={c.key} className="health-card">
                <div className="health-head">
                  <span className="health-icon"><c.icon size={16} aria-hidden /></span>
                  <span className="health-title">{c.title}</span>
                  <Badge tone={tone}><Dot tone={tone} /> {c.ok ? "Healthy" : c.warnOnly ? "Limited" : "Down"}</Badge>
                </div>
                {c.lines.filter(Boolean).map((l, i) => <div key={i} className={i ? "health-line muted" : "health-line"}>{l}</div>)}
              </article>
            );
          })}
        </section>

        <section className="card">
          <div className="card-head">
            <h2 className="card-title"><Activity size={16} aria-hidden /> Latency percentiles</h2>
            <span className="muted small">Rolling window of recent requests on this server</span>
          </div>
          {error && <div className="callout callout-error">{error}</div>}
          {metrics && !SHOWN.some(([k]) => metrics[k]) && (
            <EmptyState icon={Activity} title="No requests yet">Ask a few questions; percentiles appear here as traffic comes in.</EmptyState>
          )}
          {metrics && SHOWN.filter(([k]) => metrics[k]).map(([k, label]) => (
            <div className="pct-block" key={k}>
              <div className="pct-title">{label} <span className="muted">· {metrics[k].requests} requests</span></div>
              <div className="table-wrap">
                <table className="pct">
                  <thead><tr><th>Stage</th><th className="num">p50</th><th className="num">p90</th><th className="num">p95</th><th className="num">max</th></tr></thead>
                  <tbody>
                    {KEY_STAGES.filter((s) => metrics[k].stages[s]).map((s) => {
                      const v = metrics[k].stages[s];
                      return (
                        <tr key={s}>
                          <td>{STAGE_LABEL[s] || titleCase(s)}</td>
                          <td className="num">{ms(v.p50)}</td><td className="num">{ms(v.p90)}</td>
                          <td className="num">{ms(v.p95)}</td><td className="num muted">{ms(v.max)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </section>

        <section className="card">
          <div className="card-head"><h2 className="card-title"><Sparkles size={16} aria-hidden /> Retrieval configuration</h2></div>
          <dl className="kv kv-grid">
            <dt>Strategy</dt><dd>{cfg.retrieval_strategy || "–"}</dd>
            <dt>Chunking</dt><dd>{cfg.chunk_strategy} · ≤{cfg.chunk_max_tokens} tokens</dd>
            <dt>Candidates</dt><dd>{cfg.candidate_k} per retriever → rerank {cfg.rerank_k} → context {cfg.top_k}</dd>
            <dt>Relevance gate</dt><dd>{cfg.gate === "rerank" ? `cross-encoder ≥ ${cfg.rerank_threshold}` : `dense cosine ≥ ${cfg.similarity_threshold}`}</dd>
            <dt>Per-document cap</dt><dd>{cfg.max_chunks_per_doc} passages</dd>
            <dt>Query rewriting</dt><dd>{cfg.query_rewriter}</dd>
          </dl>
        </section>
      </div>
    </div>
  );
}
