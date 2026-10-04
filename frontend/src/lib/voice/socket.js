// WebSocket client for /ws/voice: reconnects with backoff, replays its configuration after a
// reconnect, keeps the connection warm with pings, and never queues stale partial transcripts.

export function voiceSocketUrl(base = import.meta.env.VITE_API_BASE || "/api") {
  if (/^https?:\/\//.test(base)) return `${base.replace(/^http/, "ws").replace(/\/$/, "")}/ws/voice`;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${base.replace(/\/$/, "")}/ws/voice`;
}

export class VoiceSocket {
  constructor(url, { onEvent, onStatus } = {}) {
    this.url = url;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.queue = [];
    this.config = null;
    this.closed = false;
    this.attempt = 0;
    this.status = "connecting";
    this.info = null; // the server's "ready" payload
    this._connect();
  }

  _setStatus(status) {
    this.status = status;
    this.onStatus?.(status);
  }

  _connect() {
    this._setStatus(this.attempt ? "reconnecting" : "connecting");
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this._retry();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      if (this.config) ws.send(JSON.stringify({ type: "start", ...this.config }));
      for (const msg of this.queue.splice(0)) ws.send(JSON.stringify(msg));
      clearInterval(this._ping);
      this._ping = setInterval(() => this.send({ type: "ping", t: Date.now() }), 20000);
    };
    ws.onmessage = (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      if (event.type === "ready") {
        this.info = event;
        this._setStatus("open");
      }
      if (event.type !== "pong") this.onEvent?.(event);
    };
    ws.onclose = () => {
      clearInterval(this._ping);
      if (this.closed) return;
      this._setStatus("reconnecting");
      this._retry();
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  _retry() {
    const delay = Math.min(8000, 400 * 2 ** this.attempt++);
    clearTimeout(this._timer);
    this._timer = setTimeout(() => !this.closed && this._connect(), delay);
  }

  get open() {
    return this.ws?.readyState === WebSocket.OPEN && this.status === "open";
  }

  configure(config) {
    this.config = config;
    this.send({ type: "start", ...config });
  }

  send(msg) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    } else if (msg.type === "final" || msg.type === "cancel") {
      this.queue.push(msg); // partials are only useful live; finals must not be lost
    }
  }

  close() {
    this.closed = true;
    clearInterval(this._ping);
    clearTimeout(this._timer);
    this.ws?.close();
  }
}
