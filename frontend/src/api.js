// Thin client for the FastAPI backend. All paths go through /api (proxied by Vite / nginx).
const BASE = import.meta.env.VITE_API_BASE || "/api";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText || "Request failed";
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

// Only needed when the server sets API_KEY; kept for this browser session only.
const KEY_STORE = "campaign-rag-api-key";
export function getApiKey() {
  try { return sessionStorage.getItem(KEY_STORE) || ""; } catch { return ""; }
}
export function setApiKey(key) {
  try { key ? sessionStorage.setItem(KEY_STORE, key) : sessionStorage.removeItem(KEY_STORE); } catch { /* storage blocked */ }
}
const authHeaders = () => (getApiKey() ? { "X-API-Key": getApiKey() } : {});
const post = (path, body, signal) =>
  fetch(`${BASE}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });

export const api = {
  health: () => fetch(`${BASE}/health`).then(json),
  metrics: () => fetch(`${BASE}/metrics`).then(json),
  documents: (campaign) => fetch(`${BASE}/documents?campaign_id=${encodeURIComponent(campaign)}`).then(json),
  chunks: (id) => fetch(`${BASE}/documents/${encodeURIComponent(id)}/chunks`).then(json),
  deleteDocument: (id) =>
    fetch(`${BASE}/documents/${encodeURIComponent(id)}`, { method: "DELETE", headers: authHeaders() }).then(json),

  upload(file, meta) {
    const form = new FormData();
    form.append("file", file);
    Object.entries(meta).forEach(([k, v]) => v && form.append(k, v));
    return fetch(`${BASE}/upload`, { method: "POST", body: form, headers: authHeaders() }).then(json);
  },

  retrieve: (body) => post("/retrieve", body).then(json),

  transcribe(blob, language, campaign) {
    const form = new FormData();
    const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "m4a" : "webm";
    form.append("file", blob, `speech.${ext}`);
    if (language) form.append("language", language);
    if (campaign) form.append("campaign_id", campaign);
    return fetch(`${BASE}/transcribe`, { method: "POST", body: form }).then(json);
  },

  // Streams /query as Server-Sent Events; calls onEvent({type, ...}) for each event.
  async queryStream(body, onEvent, signal) {
    const res = await post("/query", { ...body, stream: true }, signal);
    if (!res.ok || !res.body) return json(res);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const data = frame.split("\n").find((l) => l.startsWith("data:"));
        if (data) onEvent(JSON.parse(data.slice(5)));
      }
    }
  },
};
