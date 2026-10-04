import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { Menu } from "lucide-react";
import { api } from "./api.js";
import Sidebar, { Brand, systemStatus } from "./components/Sidebar.jsx";
import SettingsDialog from "./components/SettingsDialog.jsx";
import { Dot, Spinner } from "./components/ui.jsx";
import { SettingsProvider, ToastProvider, useSettings, useToast } from "./lib/context.jsx";
import { uid } from "./lib/format.js";
import { load, save } from "./lib/storage.js";
import { threadReducer } from "./lib/thread.js";
import { VoiceAssistant } from "./lib/voice/assistant.js";
import Assistant from "./pages/Assistant.jsx";
import Insights from "./pages/Insights.jsx";
import Knowledge from "./pages/Knowledge.jsx";

const ROUTES = ["assistant", "knowledge", "insights"];
const routeFromHash = () => {
  const r = window.location.hash.replace(/^#\/?/, "");
  return ROUTES.includes(r) ? r : "assistant";
};

export default function App() {
  return (
    <SettingsProvider>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </SettingsProvider>
  );
}

function Shell() {
  const { settings } = useSettings();
  const toast = useToast();
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [route, setRoute] = useState(routeFromHash);
  const [campaign, setCampaign] = useState(() => load("crag-campaign", "default"));
  const [sessionId, setSessionId] = useState(uid);
  const [filters, setFilters] = useState({ district: "", topic: "" });
  const [thread, dispatch] = useReducer(threadReducer, []);
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [assistant, setAssistant] = useState(null);

  const activeFilters = Object.fromEntries(Object.entries(filters).filter(([, v]) => v));
  const config = { campaign_id: campaign, session_id: sessionId, filters: Object.keys(activeFilters).length ? activeFilters : null };
  const configRef = useRef(config);
  configRef.current = config;

  // One voice assistant (and WebSocket) per app; created in an effect so StrictMode's
  // mount → unmount → mount cycle can't leave a destroyed instance behind.
  useEffect(() => {
    const a = new VoiceAssistant({
      dispatch,
      getSettings: () => settingsRef.current,
      transcribe: (blob, lang, c) => api.transcribe(blob, lang, c),
      fallbackAsk: (text, onEvent) => api.queryStream({ ...configRef.current, query: text }, onEvent),
    });
    setAssistant(a);
    return () => a.destroy();
  }, []);
  useEffect(() => { assistant?.configure(configRef.current); },
    [assistant, campaign, sessionId, filters.district, filters.topic]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refreshHealth = useCallback(() => api.health().then(setHealth).catch(() => setHealth(null)), []);
  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 15000);
    return () => clearInterval(t);
  }, [refreshHealth]);

  const reloadDocs = useCallback(() => {
    api.documents(campaign).then(setDocuments).catch(() => setDocuments([])).finally(() => setDocsLoading(false));
    refreshHealth();
  }, [campaign, refreshHealth]);
  useEffect(() => { setDocsLoading(true); reloadDocs(); }, [reloadDocs]);
  useEffect(() => {
    if (!documents.some((d) => !["indexed", "failed"].includes(d.status))) return undefined;
    const t = setTimeout(reloadDocs, 3000); // something is still indexing
    return () => clearTimeout(t);
  }, [documents, reloadDocs]);

  const newConversation = useCallback(() => {
    assistant?.interrupt();
    dispatch({ type: "reset" });
    setSessionId(uid());
  }, [assistant]);

  const switchCampaign = (c) => {
    if (c === campaign) return;
    save("crag-campaign", c);
    setCampaign(c);
    newConversation();
    toast(`Questions and uploads now use campaign “${c}”.`, { tone: "info", title: "Campaign switched" });
  };

  const navigate = (r) => { setRoute(r); setNavOpen(false); };
  const status = systemStatus(health);
  const indexed = documents.filter((d) => d.status === "indexed").length;

  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <Sidebar route={route} onNavigate={navigate} campaign={campaign} onCampaign={switchCampaign} health={health}
               docCount={indexed} onSettings={() => setSettingsOpen(true)} open={navOpen} onClose={() => setNavOpen(false)} />
      <div className="mobilebar">
        <button className="icon-btn md" onClick={() => setNavOpen(true)} aria-label="Open menu"><Menu size={18} /></button>
        <Brand />
        <Dot tone={status.tone} />
      </div>
      <main className="main" id="main">
        {!assistant ? (
          <div className="center-fill"><Spinner size={22} /></div>
        ) : (
          <>
            <div className="route" hidden={route !== "assistant"}>
              <Assistant assistant={assistant} thread={thread} filters={filters} setFilters={setFilters}
                         onNewConversation={newConversation} health={health} />
            </div>
            {route === "knowledge" && (
              <Knowledge campaign={campaign} documents={documents} loading={docsLoading} reload={reloadDocs} health={health} />
            )}
            {route === "insights" && <Insights health={health} refreshHealth={refreshHealth} />}
          </>
        )}
      </main>
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
