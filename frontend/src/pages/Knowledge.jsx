import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle, CheckCircle2, ChevronDown, Copy as Duplicate, FileImage, FileText, FileType2, Library, Loader2,
  RefreshCw, ScanText, Search, Table2, Trash2, UploadCloud, XCircle,
} from "lucide-react";
import { api } from "../api.js";
import { Badge, Button, Drawer, EmptyState, Field, IconButton, Select, Skeleton } from "../components/ui.jsx";
import { useToast } from "../lib/context.jsx";
import { bytes, cx, ms, timeAgo, titleCase } from "../lib/format.js";
import { DISTRICTS, TOPICS } from "./Assistant.jsx";

const ACCEPT = ".pdf,.docx,.md,.markdown,.txt,.png,.jpg,.jpeg";
const CATEGORIES = ["manifesto", "district_profile", "candidate_profile", "scheme", "policy", "faq", "other"];
const fileIcon = (t) => (t === "image" ? FileImage : t === "pdf" ? FileType2 : FileText);

export default function Knowledge({ campaign, documents, loading, reload, health }) {
  const toast = useToast();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [open, setOpen] = useState(null);
  const [confirm, setConfirm] = useState(null);

  const filtered = useMemo(() => documents.filter((d) => {
    const hay = `${d.filename} ${d.title} ${d.metadata?.district} ${d.metadata?.topic}`.toLowerCase();
    return (!query || hay.includes(query.toLowerCase())) && (!category || d.metadata?.category === category);
  }), [documents, query, category]);
  const chunks = documents.reduce((n, d) => n + (d.chunk_count || 0), 0);

  async function remove(d) {
    try {
      await api.deleteDocument(d.document_id);
      toast(`${d.filename} was removed from the knowledge base.`, { tone: "success", title: "Document deleted" });
      reload();
    } catch (err) {
      toast(err.status === 401 ? "This server requires an API key. Add it in Settings." : err.message, { tone: "error", title: "Couldn't delete" });
    } finally {
      setConfirm(null);
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="topbar-title">
          <h1>Knowledge base</h1>
          <span className="topbar-sub">Campaign <b>{campaign}</b> · {documents.length} documents · {chunks.toLocaleString()} passages</span>
        </div>
        <div className="topbar-actions">
          <IconButton label="Refresh" icon={RefreshCw} onClick={reload} />
        </div>
      </header>

      <div className="page-body">
        <Uploader campaign={campaign} onUploaded={reload} ocr={health?.components?.ocr?.ok} />

        <section className="card">
          <div className="card-head">
            <h2 className="card-title">Documents</h2>
            <div className="card-tools">
              <label className="search">
                <Search size={14} aria-hidden />
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search documents" aria-label="Search documents" />
              </label>
              <Select value={category} onChange={setCategory} aria-label="Category"
                      options={[{ value: "", label: "All categories" }, ...CATEGORIES.map((c) => ({ value: c, label: titleCase(c) }))]} />
            </div>
          </div>
          {loading && !documents.length ? (
            <div className="table-skeleton">{[0, 1, 2, 3].map((i) => <Skeleton key={i} h={40} />)}</div>
          ) : !documents.length ? (
            <EmptyState icon={Library} title="No documents yet">Upload manifestos, district plans, FAQs or scheme documents above. They become searchable within seconds.</EmptyState>
          ) : !filtered.length ? (
            <EmptyState icon={Search} title="No matches">Try a different search or category.</EmptyState>
          ) : (
            <div className="table-wrap">
              <table className="docs">
                <thead>
                  <tr><th>Document</th><th>District</th><th>Category</th><th className="num">Passages</th><th>Status</th><th>Added</th><th aria-label="Actions" /></tr>
                </thead>
                <tbody>
                  {filtered.map((d) => {
                    const Icon = fileIcon(d.file_type);
                    return (
                      <tr key={d.document_id} onClick={() => d.status === "indexed" && setOpen(d)} className={cx(d.status === "indexed" && "clickable")}>
                        <td>
                          <div className="doc-cell">
                            <span className={cx("file-icon", `ft-${d.file_type}`)}><Icon size={16} aria-hidden /></span>
                            <div className="min0">
                              <div className="doc-name truncate" title={d.filename}>{d.filename}</div>
                              <div className="doc-sub truncate">
                                {d.file_type.toUpperCase()} · {bytes(d.size_bytes)}{d.page_count ? ` · ${d.page_count} pages` : ""}
                                {d.ocr_pages?.length > 0 && <span className="inline-tag"><ScanText size={11} aria-hidden /> OCR</span>}
                                {d.content_types?.table > 0 && <span className="inline-tag"><Table2 size={11} aria-hidden /> Tables</span>}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td>{titleCase(d.metadata?.district)}</td>
                        <td>{titleCase(d.metadata?.category)}</td>
                        <td className="num">{d.chunk_count}</td>
                        <td><StatusBadge d={d} /></td>
                        <td className="muted nowrap">{timeAgo(d.uploaded_at)}</td>
                        <td className="actions">
                          <IconButton size="sm" label="Delete" icon={Trash2} onClick={(e) => { e.stopPropagation(); setConfirm(d); }} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      <ChunkDrawer doc={open} onClose={() => setOpen(null)} />
      {confirm && (
        <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && setConfirm(null)}>
          <div className="dialog confirm" role="alertdialog" aria-modal="true" aria-label="Delete document">
            <h2 className="dialog-title">Delete “{confirm.filename}”?</h2>
            <p className="dialog-desc">Its {confirm.chunk_count} passages will no longer be used to answer questions. This can't be undone.</p>
            <div className="dialog-foot">
              <Button variant="ghost" onClick={() => setConfirm(null)}>Cancel</Button>
              <Button variant="danger" icon={Trash2} onClick={() => remove(confirm)}>Delete</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ d }) {
  if (d.status === "indexed") return <Badge tone="ok" icon={CheckCircle2}>Indexed</Badge>;
  if (d.status === "failed") return <Badge tone="err" icon={XCircle} data-tip={d.error || ""}>Failed</Badge>;
  return <Badge tone="info" icon={Loader2}>{titleCase(d.status)}</Badge>;
}

// ── uploader ───────────────────────────────────────────────────────────
function Uploader({ campaign, onUploaded, ocr }) {
  const toast = useToast();
  const input = useRef(null);
  const [drag, setDrag] = useState(false);
  const [queue, setQueue] = useState([]);
  const [meta, setMeta] = useState({ district: "", category: "", topic: "", source: "" });
  const [showMeta, setShowMeta] = useState(false);
  const busy = useRef(false);

  const patch = (id, p) => setQueue((q) => q.map((x) => (x.id === id ? { ...x, ...p } : x)));

  const run = useCallback(async (items) => {
    if (busy.current) return;
    busy.current = true;
    for (const item of items) {
      patch(item.id, { state: "uploading" });
      try {
        const r = await api.upload(item.file, { ...meta, campaign_id: campaign });
        patch(item.id, {
          state: r.status === "duplicate" ? "duplicate" : "done",
          detail: r.status === "duplicate"
            ? "Already in the knowledge base"
            : `${r.status === "reindexed" ? "Replaced the previous version · " : ""}${r.document.chunk_count} passages · ${ms(r.timings_ms.total)}${r.document.ocr_pages?.length ? " · read with OCR" : ""}`,
          warnings: r.warnings,
        });
        onUploaded();
      } catch (err) {
        const message = err.status === 401 ? "This server requires an API key. Add it in Settings." : err.message;
        patch(item.id, { state: "error", detail: message });
        toast(message, { tone: "error", title: `Couldn't index ${item.file.name}` });
      }
    }
    busy.current = false;
  }, [campaign, meta, onUploaded, toast]);

  const add = (files) => {
    const items = [...files].map((file) => ({ id: `${file.name}-${file.size}-${Math.random()}`, file, state: "queued" }));
    if (!items.length) return;
    setQueue((q) => [...items, ...q].slice(0, 20));
    run(items);
  };

  useEffect(() => {
    // Drag-and-drop anywhere on the page.
    const over = (e) => { if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); setDrag(true); } };
    const leave = (e) => { if (!e.relatedTarget) setDrag(false); };
    const drop = (e) => { if (e.dataTransfer?.files?.length) { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); } };
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => { window.removeEventListener("dragover", over); window.removeEventListener("dragleave", leave); window.removeEventListener("drop", drop); };
  });

  const set = (k) => (v) => setMeta((m) => ({ ...m, [k]: typeof v === "string" ? v : v.target.value }));
  return (
    <section className="card">
      <div className={cx("dropzone", drag && "drag")} onClick={() => input.current?.click()} role="button" tabIndex={0}
           onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}>
        <span className="dz-icon"><UploadCloud size={22} aria-hidden /></span>
        <div className="dz-title">Drop files here or <span className="link">browse</span></div>
        <div className="dz-sub">PDF, Word, Markdown or text{ocr ? " · scanned PDFs and photos are read with OCR" : ""} · up to 20 MB each</div>
        <input ref={input} type="file" multiple accept={ACCEPT} hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
      </div>

      <button className="disclosure" onClick={() => setShowMeta(!showMeta)} aria-expanded={showMeta}>
        <ChevronDown size={14} className={cx(showMeta && "rot")} aria-hidden /> Metadata <span className="muted">(optional; detected automatically)</span>
      </button>
      {showMeta && (
        <div className="meta-grid">
          <Field label="District">
            <Select value={meta.district} onChange={set("district")} options={[{ value: "", label: "Detect automatically" }, { value: "statewide", label: "Statewide" }, ...DISTRICTS.map((d) => ({ value: d, label: titleCase(d) }))]} />
          </Field>
          <Field label="Category">
            <Select value={meta.category} onChange={set("category")} options={[{ value: "", label: "Detect automatically" }, ...CATEGORIES.map((c) => ({ value: c, label: titleCase(c) }))]} />
          </Field>
          <Field label="Topic">
            <Select value={meta.topic} onChange={set("topic")} options={[{ value: "", label: "Detect automatically" }, ...TOPICS.map((t) => ({ value: t, label: titleCase(t) }))]} />
          </Field>
          <Field label="Source label"><input className="input" value={meta.source} onChange={set("source")} placeholder="e.g. Manifesto 2026" /></Field>
        </div>
      )}

      {queue.length > 0 && (
        <ul className="queue">
          {queue.map((q) => (
            <li key={q.id} className={cx("queue-item", `q-${q.state}`)}>
              <span className="q-icon">
                {q.state === "uploading" || q.state === "queued" ? <Loader2 size={16} className={cx(q.state === "uploading" && "spin")} aria-hidden />
                  : q.state === "done" ? <CheckCircle2 size={16} aria-hidden />
                  : q.state === "duplicate" ? <Duplicate size={16} aria-hidden />
                  : <XCircle size={16} aria-hidden />}
              </span>
              <div className="min0 grow">
                <div className="q-name truncate">{q.file.name}</div>
                <div className="q-detail">
                  {q.state === "queued" ? "Waiting…" : q.state === "uploading" ? "Reading, chunking and indexing…" : q.detail}
                </div>
                {q.warnings?.length > 0 && (
                  <div className="q-warn"><AlertTriangle size={12} aria-hidden /> {q.warnings.join(" · ")}</div>
                )}
                {q.state === "uploading" && <div className="progress"><span /></div>}
              </div>
              <span className="muted small nowrap">{bytes(q.file.size)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ── chunk viewer ───────────────────────────────────────────────────────
function ChunkDrawer({ doc, onClose }) {
  const [chunks, setChunks] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!doc) return;
    setChunks(null);
    setError(null);
    api.chunks(doc.document_id).then(setChunks).catch((e) => setError(e.message));
  }, [doc]);
  return (
    <Drawer open={Boolean(doc)} onClose={onClose} title={doc?.title || doc?.filename}
            subtitle={doc && `${doc.filename} · ${doc.chunk_count} passages · ${titleCase(doc.metadata?.district)} · ${titleCase(doc.metadata?.category)}`}>
      {error && <div className="callout callout-error">{error}</div>}
      {!chunks && !error && <div className="table-skeleton">{[0, 1, 2].map((i) => <Skeleton key={i} h={80} />)}</div>}
      {chunks && (
        <ol className="chunk-list">
          {chunks.map((c) => (
            <li key={c.chunk_id} className="chunk">
              <div className="chunk-meta">
                <span className="chunk-idx">#{c.chunk_index + 1}</span>
                <span className="truncate">{c.section || "Introduction"}</span>
                {c.page && <span>p. {c.page}</span>}
                <span className="grow" />
                {c.content_type !== "text" && <Badge tone="info">{c.content_type === "ocr" ? "OCR" : "Table"}</Badge>}
                <span className="muted">{c.token_count} tokens</span>
              </div>
              <p className="chunk-text">{c.text}</p>
            </li>
          ))}
        </ol>
      )}
    </Drawer>
  );
}
