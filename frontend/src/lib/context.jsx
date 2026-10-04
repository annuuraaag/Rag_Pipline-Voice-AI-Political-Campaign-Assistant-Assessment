import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Info, TriangleAlert, X, XCircle } from "lucide-react";
import { load, save } from "./storage.js";

// ── settings ─────────────────────────────────────────────────────────
export const DEFAULT_SETTINGS = {
  theme: "system",        // system | light | dark
  lang: "en-IN",          // speech recognition + preferred TTS accent
  voiceURI: "",           // "" = best available voice for `lang`
  rate: 1.05,
  autoSpeak: true,        // speak answers to spoken questions
  bargeIn: true,          // talking over a spoken answer interrupts it
  handsFree: false,       // conversation mode: keep listening after each answer
  endSilenceMs: 900,      // pause that ends an utterance (when smart detection is off or unsure)
  smartEndpointing: true, // shorter pause after a finished question, longer after "for…", "um…"
  speechInput: "auto",    // auto | browser | server: who recognises speech
  speechOutput: "auto",   // auto | browser | server: whose voice speaks the answer
  inspector: true,
};

const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_SETTINGS, ...load("crag-settings", {}) }));
  const update = useCallback((patch) => setSettings((s) => ({ ...s, ...patch })), []);
  useEffect(() => save("crag-settings", settings), [settings]);
  useEffect(() => {
    const root = document.documentElement;
    if (settings.theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", settings.theme);
  }, [settings.theme]);
  const value = useMemo(() => ({ settings, update }), [settings, update]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export const useSettings = () => useContext(SettingsContext);

// ── toasts ───────────────────────────────────────────────────────────
const ToastContext = createContext(() => {});
const ICONS = { success: CheckCircle2, error: XCircle, warning: TriangleAlert, info: Info };

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const seq = useRef(0);
  const dismiss = useCallback((id) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const toast = useCallback((message, { tone = "info", title, duration = 4500 } = {}) => {
    const id = ++seq.current;
    setToasts((t) => [...t.slice(-3), { id, message, tone, title }]);
    if (duration) setTimeout(() => dismiss(id), duration);
  }, [dismiss]);
  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div className="toaster" role="status" aria-live="polite">
        {toasts.map((t) => {
          const Icon = ICONS[t.tone] || Info;
          return (
            <div key={t.id} className={`toast toast-${t.tone}`}>
              <Icon size={18} className="toast-icon" aria-hidden />
              <div className="toast-body">
                {t.title && <div className="toast-title">{t.title}</div>}
                <div className="toast-msg">{t.message}</div>
              </div>
              <button className="icon-btn sm" onClick={() => dismiss(t.id)} aria-label="Dismiss"><X size={14} /></button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);
