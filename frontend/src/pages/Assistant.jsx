import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import {
  ArrowUp, AudioLines, Briefcase, Check, CircleAlert, Copy, GraduationCap, HeartPulse, Info, Loader2, MapPin,
  Mic, PanelRight, RotateCcw, Search, SlidersHorizontal, Sparkles, Sprout, Square, TriangleAlert, Volume2, Waves, X, Zap,
} from "lucide-react";
import VoiceOrb, { Waveform } from "../components/VoiceOrb.jsx";
import Inspector from "../components/Inspector.jsx";
import { Badge, IconButton, Kbd, Select, Skeleton } from "../components/ui.jsx";
import { useSettings } from "../lib/context.jsx";
import { cx, ms, pages, titleCase } from "../lib/format.js";
import { sourceFor } from "../lib/thread.js";
import { ttsSupported } from "../lib/voice/tts.js";

const SUGGESTIONS = [
  { icon: MapPin, text: "I'm from Vijayawada." },
  { icon: HeartPulse, text: "What healthcare initiatives does the candidate propose for Vijayawada?" },
  { icon: GraduationCap, text: "Who is eligible for the Pratibha Scholarship?" },
  { icon: Sprout, text: "What is planned for chilli farmers in Guntur?" },
  { icon: Briefcase, text: "How many jobs will the Visakhapatnam IT corridor create?" },
  { icon: Waves, text: "What does the manifesto say about flood protection?" },
];
export const DISTRICTS = ["vijayawada", "guntur", "visakhapatnam", "krishna", "tirupati", "nellore", "kurnool"];
export const TOPICS = ["healthcare", "education", "employment", "agriculture", "infrastructure", "welfare", "governance"];

const isEditable = (el) => el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));

function useWide(query = "(min-width: 1281px)") {
  const [wide, setWide] = useState(() => window.matchMedia?.(query).matches ?? true);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return undefined;
    const on = () => setWide(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return wide;
}

export default function Assistant({ assistant, thread, filters, setFilters, onNewConversation, health }) {
  const voice = useSyncExternalStore(assistant.subscribe, assistant.getState);
  const { settings, update } = useSettings();
  const wide = useWide();
  const [sheet, setSheet] = useState(false); // narrow screens: details open as an overlay, not persisted
  const showInspector = wide ? settings.inspector : sheet;
  const setInspector = (open) => (wide ? update({ inspector: open }) : setSheet(open));
  const [selectedId, setSelectedId] = useState(null);
  const [inspectorTab, setInspectorTab] = useState("sources");
  const [focusSource, setFocusSource] = useState(null);
  const scrollRef = useRef(null);
  const stickRef = useRef(true);

  const assistantMsgs = thread.filter((m) => m.role === "assistant");
  const selected = thread.find((m) => m.id === selectedId) || assistantMsgs[assistantMsgs.length - 1];
  const memory = [...assistantMsgs].reverse().find((m) => m.conversation)?.conversation;

  // Keep the newest message in view while streaming, unless the user scrolled up to read.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [thread, voice.interim]);
  const onScroll = () => {
    const el = scrollRef.current;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };
  useEffect(() => { stickRef.current = true; setSelectedId(null); }, [thread.length]);

  // Hold Space to talk (release sends at once); Esc stops.
  useEffect(() => {
    let held = false;
    const down = (e) => {
      if (e.code === "Space" && !e.repeat && !isEditable(e.target) && !e.target.closest?.("button")) {
        e.preventDefault();
        held = true;
        const { phase } = assistant.getState();
        if (phase === "idle") assistant.listen();
        else if (phase === "thinking" || phase === "speaking") assistant.interrupt({ listen: true });
      } else if (e.key === "Escape" && assistant.getState().phase !== "idle") {
        assistant.interrupt();
      }
    };
    const up = (e) => {
      if (e.code === "Space" && held) {
        held = false;
        if (assistant.getState().phase === "listening") assistant.stopListening();
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [assistant]);

  const openSource = (message, n) => {
    setSelectedId(message.id);
    setInspectorTab("sources");
    setFocusSource(`S${n}`);
    setInspector(true);
  };

  const empty = thread.length === 0;
  return (
    <div className={cx("assistant", showInspector && "with-inspector")}>
      <section className="chat">
        <header className="topbar">
          <div className="topbar-title">
            <h1>Assistant</h1>
            {memory && (memory.district || memory.topic) && (
              <div className="memory" data-tip="What the assistant remembers from this conversation">
                <Sparkles size={13} aria-hidden />
                {memory.district && <span>{titleCase(memory.district)}</span>}
                {memory.district && memory.topic && <span className="sep">·</span>}
                {memory.topic && <span>{titleCase(memory.topic)}</span>}
              </div>
            )}
          </div>
          <div className="topbar-actions">
            <FiltersMenu filters={filters} setFilters={setFilters} />
            <IconButton label="New conversation" icon={RotateCcw} onClick={onNewConversation} disabled={empty} />
            <IconButton label={showInspector ? "Hide details" : "Show details"} icon={PanelRight}
                        active={showInspector} onClick={() => setInspector(!showInspector)} />
          </div>
        </header>

        <div className="chat-scroll" ref={scrollRef} onScroll={onScroll}>
          {empty ? (
            <Welcome voice={voice} assistant={assistant} health={health} />
          ) : (
            <div className="thread">
              {thread.map((m) => m.role === "user"
                ? <UserMessage key={m.id} m={m} />
                : <AssistantMessage key={m.id} m={m} selected={selected?.id === m.id && showInspector}
                                    onSelect={() => setSelectedId(m.id)} onCite={(n) => openSource(m, n)}
                                    onReplay={() => assistant.replay(m.id, m.text)}
                                    onShowSources={() => { setSelectedId(m.id); setInspectorTab("sources"); setInspector(true); }} />)}
            </div>
          )}
        </div>

        <div className="dock">
          <LiveBar voice={voice} assistant={assistant} />
          <Composer voice={voice} assistant={assistant} />
          <p className="dock-hint">
            Answers come only from this campaign's documents and cite them. <span className="hide-sm">Hold <Kbd>Space</Kbd> to talk, <Kbd>Esc</Kbd> to stop.</span>
          </p>
        </div>
      </section>

      {showInspector && !wide && <div className="inspector-scrim" onClick={() => setSheet(false)} aria-hidden />}
      {showInspector && (
        <Inspector message={selected} tab={inspectorTab} setTab={setInspectorTab} focusSource={focusSource}
                   onClose={() => setInspector(false)} health={health} />
      )}
    </div>
  );
}

// ── empty state ────────────────────────────────────────────────────────
function Welcome({ voice, assistant, health }) {
  const docs = health?.components?.documents?.count;
  return (
    <div className="welcome">
      <button className="welcome-orb" onClick={() => assistant.toggle()} aria-label="Start talking">
        <VoiceOrb phase={voice.phase} getLevel={assistant.level} size={148} />
      </button>
      <h2 className="welcome-title">Ask about the campaign</h2>
      <p className="welcome-sub">
        Tap the orb and speak, or type below. Every answer is drawn from the campaign's documents
        {docs ? ` (${docs} indexed)` : ""} and shows its sources.
      </p>
      <div className="suggestions">
        {SUGGESTIONS.map(({ icon: Icon, text }) => (
          <button key={text} className="suggestion" onClick={() => assistant.askText(text)}>
            <Icon size={16} aria-hidden />
            <span>{text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── messages ───────────────────────────────────────────────────────────
function UserMessage({ m }) {
  return (
    <div className="msg msg-user">
      <div className="bubble">
        {m.via === "voice" && <Mic size={13} className="via" aria-label="Spoken" />}
        {m.text}
      </div>
    </div>
  );
}

// Unverified claims, as ranges of the answer text. Claims arrive with their [S#] markers removed,
// so the text is mapped the same way (markers out, no space before punctuation) to find them.
function unverifiedRanges(text, verification) {
  const bad = (verification?.claims || []).filter((c) => c.verdict === "unsupported");
  if (!bad.length) return [];
  let plain = "";
  const map = [];
  const stripped = text.replace(/\s*\[S\d+\]/g, (mk) => "\u0000".repeat(mk.length));
  for (let i = 0; i < stripped.length; i++) {
    if (stripped[i] === "\u0000") continue;
    if (/\s/.test(stripped[i])) {
      let j = i;
      while (j < stripped.length && (/\s/.test(stripped[j]) || stripped[j] === "\u0000")) j++;
      if (/[.,;:!?]/.test(stripped[j] || "")) { i = j - 1; continue; }
    }
    plain += stripped[i];
    map.push(i);
  }
  return bad.flatMap((c) => {
    const at = plain.indexOf(c.text);
    if (at < 0) return [];
    return [{ start: map[at], end: map[at + c.text.length - 1] + 1, tip: `Not found in the sources: ${c.issues.join("; ")}` }];
  });
}

function AnswerText({ m, onCite }) {
  const ranges = m.status === "done" ? unverifiedRanges(m.text, m.verification) : [];
  const inRange = (i) => ranges.find((r) => r.start <= i && i < r.end);
  const pieces = [];
  const re = /\[S(\d+)\]/g;
  let last = 0;
  const pushText = (from, to) => {
    for (let i = from; i < to;) {
      const r = inRange(i);
      const next = r ? Math.min(r.end, to) : Math.min(to, ...ranges.filter((x) => x.start > i).map((x) => x.start));
      pieces.push({ text: m.text.slice(i, next), flag: r });
      i = next;
    }
  };
  for (let hit; (hit = re.exec(m.text));) {
    pushText(last, hit.index);
    pieces.push({ cite: Number(hit[1]) });
    last = hit.index + hit[0].length;
  }
  pushText(last, m.text.length);
  return (
    <div className="answer">
      {pieces.map((p, i) => {
        if (p.cite === undefined) {
          return p.flag
            ? <mark key={i} className="claim-flag" data-tip={p.flag.tip} tabIndex={0} aria-label={`${p.text} (${p.flag.tip})`}>{p.text}</mark>
            : <span key={i}>{p.text}</span>;
        }
        const n = p.cite;
        const src = sourceFor(m, n);
        const label = src ? `${src.document_name}${pages(src) ? ` · ${pages(src)}` : ""}${src.section ? ` · ${src.section}` : ""}` : `Source ${n}`;
        return (
          <button key={i} className="cite" onClick={(e) => { e.stopPropagation(); onCite(n); }} data-tip={label}
                  aria-label={`Source ${n}: ${label}`}>{n}</button>
        );
      })}
      {m.status === "streaming" && <span className="caret" aria-hidden />}
    </div>
  );
}

function AssistantMessage({ m, selected, onSelect, onCite, onReplay, onShowSources }) {
  const [copied, setCopied] = useState(false);
  const done = m.status === "done";
  const refused = done && m.answerable === false && m.refusal_reason;
  const skipped = m.trace?.strategy?.startsWith("skipped");
  const cited = (m.citations || []).filter((c) => c.cited).length;
  const vf = m.verification;
  const checked = vf ? vf.supported + vf.corrected + vf.unsupported : 0;
  const v = m.voice || {};
  const e2e = v.first_audio_ms !== undefined && v.endpoint_wait_ms != null ? v.endpoint_wait_ms + v.first_audio_ms : undefined;
  const firstWord = e2e ?? v.first_audio_ms ?? m.latency?.first_token;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(m.text.replace(/\s*\[S\d+\]/g, ""));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked */ }
  };

  return (
    <div className={cx("msg msg-assistant", selected && "selected")} onClick={onSelect}>
      <div className="avatar" aria-hidden><span /></div>
      <div className="msg-body">
        {m.status === "retrieving" && (
          <div className="thinking">
            <span className="thinking-label"><Search size={14} aria-hidden /> Searching campaign documents…</span>
            <Skeleton w="92%" /><Skeleton w="76%" /><Skeleton w="54%" />
          </div>
        )}
        {(m.status === "streaming" || done || m.status === "interrupted") && m.text && (
          refused ? (
            <div className="callout callout-info">
              <Info size={16} aria-hidden />
              <div>
                <div className="callout-title">Not covered by the campaign documents</div>
                <div>{m.text}</div>
              </div>
            </div>
          ) : <AnswerText m={m} onCite={onCite} />
        )}
        {m.status === "streaming" && !m.text && (
          <div className="thinking"><span className="thinking-label"><Loader2 size={14} className="spin" aria-hidden /> Writing the answer…</span></div>
        )}
        {m.status === "error" && (
          <div className="callout callout-error"><CircleAlert size={16} aria-hidden /><div>{m.error}</div></div>
        )}

        {(done || m.status === "interrupted") && (
          <div className="msg-meta">
            {m.status === "interrupted" && <Badge tone="warn" icon={Square}>Interrupted</Badge>}
            {done && !refused && !skipped && cited > 0 && (vf?.unsupported ? (
              <button className="badge badge-warn badge-btn" onClick={(e) => { e.stopPropagation(); onShowSources(); }}
                      data-tip="Underlined sentences could not be matched to the passage they cite">
                <TriangleAlert size={12} aria-hidden /> {vf.unsupported} of {checked} claim{checked === 1 ? "" : "s"} not found in sources
              </button>
            ) : (
              <button className="badge badge-ok badge-btn" onClick={(e) => { e.stopPropagation(); onShowSources(); }}
                      data-tip={vf ? `Each claim was checked against the passage it cites${vf.corrected ? `; ${vf.corrected} citation${vf.corrected === 1 ? " was" : "s were"} corrected` : ""}` : undefined}>
                <Check size={12} aria-hidden /> {vf && checked ? `Verified · ${checked} claim${checked === 1 ? "" : "s"}` : "Grounded"} · {cited} source{cited === 1 ? "" : "s"}
              </button>
            ))}
            {(v.cache === "hit" || v.cache === "stage1") && v.retrieval_saved_ms > 0 && (
              <Badge tone="accent" icon={Zap}
                     data-tip="Retrieval ran while you were still speaking, so the answer started sooner">
                Searched ahead · {ms(v.retrieval_saved_ms)} saved
              </Badge>
            )}
            {firstWord !== undefined && done && (
              <span className="meta-text" data-tip={e2e !== undefined
                ? "From your last word to the first spoken word, including the pause that tells it you'd finished"
                : v.first_audio_ms !== undefined ? "From the end of your question to the first spoken word" : "Time to the first word of the answer"}>
                {v.first_audio_ms !== undefined ? <Volume2 size={12} aria-hidden /> : <AudioLines size={12} aria-hidden />}
                {ms(firstWord)}
              </span>
            )}
            {m.llm?.fallback && <Badge tone="warn">Offline answer</Badge>}
            <span className="msg-actions">
              {ttsSupported && m.text && <IconButton size="sm" label="Read aloud" icon={Volume2} onClick={(e) => { e.stopPropagation(); onReplay(); }} />}
              <IconButton size="sm" label={copied ? "Copied" : "Copy"} icon={copied ? Check : Copy} onClick={(e) => { e.stopPropagation(); copy(); }} />
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── live voice bar ─────────────────────────────────────────────────────
function LiveBar({ voice, assistant }) {
  const { phase, interim, speculative, error, notice } = voice;
  if (error || notice) {
    return (
      <div className={cx("livebar", error ? "livebar-error" : "livebar-notice")} role="alert">
        {error ? <CircleAlert size={16} aria-hidden /> : <Info size={16} aria-hidden />}
        <span className="livebar-text">{error || notice}</span>
        <IconButton size="sm" label="Dismiss" icon={X} onClick={() => assistant.dismiss()} />
      </div>
    );
  }
  if (phase === "idle") return null;
  return (
    <div className={cx("livebar", `livebar-${phase}`)} aria-live="polite">
      {phase === "listening" && (
        <>
          <Waveform getLevel={assistant.level} active />
          <span className={cx("livebar-text", !interim && "muted")}>{interim || "Listening…"}</span>
          {speculative && (
            <span className="spec" data-tip={speculative.sources?.map((s) => s.document_name).join(", ")}>
              <Zap size={12} aria-hidden />
              {speculative.stage === "S2" ? "Ready" : "Searching ahead"} · {speculative.sources?.length || 0} sources
            </span>
          )}
        </>
      )}
      {phase === "transcribing" && (<><Loader2 size={16} className="spin" aria-hidden /><span className="livebar-text">Transcribing…</span></>)}
      {phase === "thinking" && (<><span className="dots" aria-hidden><i /><i /><i /></span><span className="livebar-text">Thinking…</span></>)}
      {phase === "speaking" && (
        <>
          <span className="eq" aria-hidden><i /><i /><i /><i /></span>
          <span className="livebar-text">Speaking<span className="hide-sm"> · {voice.listeningOver ? "tap the mic or start talking to interrupt" : "tap the mic to interrupt"}</span></span>
          <button className="btn btn-ghost btn-sm" onClick={() => assistant.interrupt()}><Square size={12} aria-hidden /> Stop</button>
        </>
      )}
    </div>
  );
}

// ── composer ───────────────────────────────────────────────────────────
function Composer({ voice, assistant }) {
  const [text, setText] = useState("");
  const ref = useRef(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text]);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "/" && !isEditable(e.target)) { e.preventDefault(); ref.current?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const submit = (e) => {
    e?.preventDefault();
    if (!text.trim()) return;
    assistant.askText(text);
    setText("");
  };
  const { phase } = voice;
  const micLabel = { idle: "Start talking", listening: "Stop and send", transcribing: "Transcribing", thinking: "Interrupt", speaking: "Interrupt" }[phase];
  const MicIcon = phase === "listening" ? Square : phase === "speaking" || phase === "thinking" ? AudioLines : Mic;
  const disabled = assistant.mode === "none";
  return (
    <form className="composer" onSubmit={submit}>
      <button type="button" className={cx("mic", `mic-${phase}`)} onClick={() => assistant.toggle()} aria-label={micLabel}
              data-tip={disabled ? "Voice input isn't available in this browser" : micLabel}
              disabled={disabled || phase === "transcribing"}>
        <span className="mic-ring" aria-hidden />
        <MicIcon size={20} aria-hidden />
      </button>
      <textarea ref={ref} rows={1} value={text} onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e); }}
                placeholder={phase === "listening" ? "Listening…" : "Ask a question…"} aria-label="Question" maxLength={1000} />
      <button type="submit" className="send" disabled={!text.trim()} aria-label="Send">
        <ArrowUp size={18} aria-hidden />
      </button>
    </form>
  );
}

function FiltersMenu({ filters, setFilters }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const active = Object.values(filters).filter(Boolean).length;
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => !ref.current?.contains(e.target) && setOpen(false);
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div className="popover-anchor" ref={ref}>
      <button className={cx("btn btn-ghost btn-sm", active && "btn-active")} onClick={() => setOpen(!open)} aria-expanded={open}>
        <SlidersHorizontal size={14} aria-hidden /> Filters{active ? ` · ${active}` : ""}
      </button>
      {open && (
        <div className="popover">
          <div className="popover-title">Limit answers to</div>
          <label className="field">
            <span className="field-label">District</span>
            <Select value={filters.district} onChange={(v) => setFilters({ ...filters, district: v })}
                    options={[{ value: "", label: "Any district (from conversation)" }, ...DISTRICTS.map((d) => ({ value: d, label: titleCase(d) }))]} />
          </label>
          <label className="field">
            <span className="field-label">Topic</span>
            <Select value={filters.topic} onChange={(v) => setFilters({ ...filters, topic: v })}
                    options={[{ value: "", label: "Any topic" }, ...TOPICS.map((t) => ({ value: t, label: titleCase(t) }))]} />
          </label>
          {active > 0 && <button className="btn btn-ghost btn-sm" onClick={() => setFilters({ district: "", topic: "" })}>Clear filters</button>}
        </div>
      )}
    </div>
  );
}
