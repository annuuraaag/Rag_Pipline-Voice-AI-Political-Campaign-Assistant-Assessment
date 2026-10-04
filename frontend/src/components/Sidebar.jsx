import { useEffect, useRef, useState } from "react";
import { Activity, AudioLines, Check, ChevronsUpDown, Library, Moon, Plus, Settings, Sun, X } from "lucide-react";
import { Dot } from "./ui.jsx";
import { useSettings } from "../lib/context.jsx";
import { cx } from "../lib/format.js";

export const NAV = [
  { id: "assistant", label: "Assistant", icon: AudioLines },
  { id: "knowledge", label: "Knowledge base", icon: Library },
  { id: "insights", label: "Insights", icon: Activity },
];

const CAMPAIGN_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export function systemStatus(health) {
  if (!health) return { tone: "err", label: "Server unreachable", detail: "Check that the backend is running." };
  const c = health.components;
  if (health.status === "down") return { tone: "err", label: "Vector store down", detail: "Search is unavailable." };
  if (!c.llm.ok) return { tone: "warn", label: "Offline answers", detail: c.llm.note };
  if (!c.vector_store.chunks) return { tone: "warn", label: "No documents", detail: "Upload documents to start." };
  return { tone: "ok", label: "All systems normal", detail: `${c.llm.provider} · ${c.llm.model}` };
}

export function Brand() {
  return (
    <div className="brand">
      <span className="brand-mark" aria-hidden><span /></span>
      <span className="brand-text">
        <span className="brand-name">Campaign Voice</span>
        <span className="brand-sub">Grounded RAG assistant</span>
      </span>
    </div>
  );
}

export default function Sidebar({ route, onNavigate, campaign, onCampaign, health, docCount, onSettings, open, onClose }) {
  const { settings, update } = useSettings();
  const status = systemStatus(health);
  const dark = settings.theme === "dark" ||
    (settings.theme === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  return (
    <>
      {open && <div className="nav-scrim" onClick={onClose} aria-hidden />}
      <nav className={cx("sidebar", open && "open")} aria-label="Main">
        <div className="sidebar-top">
          <Brand />
          <button className="icon-btn md mobile-only" onClick={onClose} aria-label="Close menu"><X size={17} /></button>
        </div>
        <CampaignSwitcher campaign={campaign} onChange={onCampaign} campaigns={health?.components?.documents?.campaigns || {}} />
        <ul className="nav">
          {NAV.map((n) => (
            <li key={n.id}>
              <a href={`#/${n.id}`} className={cx("nav-item", route === n.id && "active")}
                 aria-current={route === n.id ? "page" : undefined} onClick={() => onNavigate(n.id)}>
                <n.icon size={17} aria-hidden />
                <span>{n.label}</span>
                {n.id === "knowledge" && docCount > 0 && <span className="nav-count">{docCount}</span>}
              </a>
            </li>
          ))}
        </ul>
        <div className="grow" />
        <div className="sidebar-foot">
          <div className="status-row" data-tip={status.detail}>
            <Dot tone={status.tone} pulse={status.tone === "ok"} />
            <div className="min0">
              <div className="status-label">{status.label}</div>
              {status.detail && <div className="status-detail truncate">{status.detail}</div>}
            </div>
          </div>
          <div className="foot-actions">
            <button className="nav-item" onClick={onSettings}><Settings size={17} aria-hidden /><span>Settings</span></button>
            <button className="icon-btn md" aria-label={dark ? "Light theme" : "Dark theme"} data-tip={dark ? "Light theme" : "Dark theme"}
                    onClick={() => update({ theme: dark ? "light" : "dark" })}>
              {dark ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          </div>
          <p className="fineprint">Fictional sample data. Informational only.</p>
        </div>
      </nav>
    </>
  );
}

function CampaignSwitcher({ campaign, onChange, campaigns }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => !ref.current?.contains(e.target) && setOpen(false);
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);
  const names = Array.from(new Set([campaign, ...Object.keys(campaigns)])).sort();
  const valid = CAMPAIGN_RE.test(draft);
  const pick = (c) => { onChange(c); setOpen(false); setDraft(""); };
  return (
    <div className="campaign" ref={ref}>
      <button className="campaign-btn" onClick={() => setOpen(!open)} aria-expanded={open} aria-haspopup="listbox">
        <span className="campaign-avatar" aria-hidden>{campaign.slice(0, 1).toUpperCase()}</span>
        <span className="min0 grow">
          <span className="campaign-label">Campaign</span>
          <span className="campaign-name truncate">{campaign}</span>
        </span>
        <ChevronsUpDown size={15} aria-hidden />
      </button>
      {open && (
        <div className="popover campaign-pop" role="listbox">
          <div className="popover-title">Switch campaign</div>
          {names.map((c) => (
            <button key={c} role="option" aria-selected={c === campaign} className="campaign-opt" onClick={() => pick(c)}>
              <span className="campaign-avatar sm" aria-hidden>{c.slice(0, 1).toUpperCase()}</span>
              <span className="grow truncate">{c}</span>
              <span className="muted small">{campaigns[c] ?? 0} docs</span>
              {c === campaign && <Check size={14} aria-hidden />}
            </button>
          ))}
          <form className="campaign-new" onSubmit={(e) => { e.preventDefault(); if (valid) pick(draft); }}>
            <input className="input" value={draft} onChange={(e) => setDraft(e.target.value.toLowerCase())}
                   placeholder="new-campaign-id" aria-label="New campaign id" />
            <button className="icon-btn md" disabled={!valid} aria-label="Create campaign"><Plus size={16} /></button>
          </form>
          <p className="popover-note">Documents, search and conversation memory are isolated per campaign.</p>
        </div>
      )}
    </div>
  );
}
