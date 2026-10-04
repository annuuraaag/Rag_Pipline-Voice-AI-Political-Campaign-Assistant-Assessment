// Voice assistant controller (framework-free; React subscribes via useSyncExternalStore).
//
//   idle ──tap/Space──▶ listening ──end of speech──▶ thinking ──first audio──▶ speaking ──▶ idle
//                          │  partials → server (speculative retrieval)          │
//                          └──── recorder fallback: transcribing ────────────────┘
//
// Barge-in: tapping the mic (or, in conversation mode, speaking over the answer) stops speech
// output at once, cancels the server-side answer, and starts listening. In conversation mode
// the recognizer keeps running while the assistant talks, so its own voice can be picked up;
// transcripts that mostly repeat what was just said aloud are treated as echo and ignored.

import { openMic, micErrorMessage, recordUtterance } from "./mic.js";
import { createRecognizer, speechRecognitionSupported, tidyTranscript } from "./recognizer.js";
import { SentenceSpeaker, pickVoice, ttsSupported } from "./tts.js";
import { VoiceSocket, voiceSocketUrl } from "./socket.js";

const uid = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/\W/g, "").slice(0, 10);
const words = (t) => t.toLowerCase().match(/[a-z0-9]+/g) || [];

/** True when `heard` is probably the assistant's own voice coming back through the mic. */
export function isEcho(heard, spoken) {
  const w = words(heard);
  if (w.length < 2) return true; // one word ("ok", "the") is never a deliberate interruption
  const pool = new Set(words(spoken));
  if (!pool.size) return false;
  return w.filter((x) => pool.has(x)).length / w.length >= 0.6;
}

export class VoiceAssistant {
  constructor({ dispatch, getSettings, transcribe, fallbackAsk }) {
    this.dispatch = dispatch;
    this.getSettings = getSettings;
    this.transcribe = transcribe;
    this.fallbackAsk = fallbackAsk;
    this.listeners = new Set();
    this.state = {
      phase: "idle", interim: "", speculative: null, error: null, notice: null,
      socket: "connecting", server: null,
    };
    this.turn = null;
    this.mic = null;
    this.rec = null;
    this.recording = null;
    this.config = null;
    this.speaker = new SentenceSpeaker({ onStart: (t) => this._firstAudio(t), onIdle: (t) => this._spoken(t) });
    this.socket = new VoiceSocket(voiceSocketUrl(), {
      onEvent: (e) => this._onEvent(e),
      onStatus: (s) => this._set({ socket: s, server: this.socket?.info ?? this.state.server }),
    });
  }

  // ── store interface ───────────────────────────────────────────────
  subscribe = (fn) => { this.listeners.add(fn); return () => this.listeners.delete(fn); };
  getState = () => this.state;
  _set(patch) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn());
  }

  level = () => (this.mic ? this.mic.level() : 0);

  get mode() {
    if (speechRecognitionSupported) return "webspeech";
    if (typeof window !== "undefined" && window.MediaRecorder && this.state.server?.transcribe) return "recorder";
    return "none";
  }

  configure(config) {
    this.config = config;
    this.socket.configure(config);
  }

  dismiss() { this._set({ error: null, notice: null }); }

  // ── user actions ──────────────────────────────────────────────────
  toggle() {
    const { phase } = this.state;
    if (phase === "idle") return this.listen();
    if (phase === "listening") return this.stopListening();
    if (phase === "thinking" || phase === "speaking") return this.interrupt({ listen: true });
  }

  async listen() {
    this.echoPool = null;
    this.speaker.unlock(); // inside the click/keypress: lets Safari speak later
    if (this.mode === "none") {
      this._set({ error: "Voice input needs Chrome, Edge or Safari (or a server speech-to-text key). You can still type." });
      return;
    }
    this._set({ error: null, notice: null, interim: "", speculative: null, phase: "listening" });
    if (!(await this._ensureMic())) return;
    if (this.mode === "webspeech") this._startRecognizer();
    else this._record();
  }

  stopListening() {
    if (this.rec) this.rec.stop();
    else if (this.recording) this.recording.stop();
  }

  /** Stop talking / cancel the answer. With listen=true, start listening right away (barge-in). */
  interrupt({ listen = false } = {}) {
    this._cancelTurn();
    this.rec?.abort();
    this.rec = null;
    if (listen) {
      this.listen();
    } else {
      this._closeMic();
      this._set({ phase: "idle", interim: "" });
    }
  }

  askText(text) {
    const q = text.trim();
    if (!q) return;
    this._cancelTurn();
    this.rec?.abort();
    this.rec = null;
    this.recording?.stop();
    this._closeMic();
    this._submit(q, { voice: false });
  }

  replay(turnId, text) {
    if (!ttsSupported) return;
    this.speaker.unlock();
    const s = this.getSettings();
    this._cancelTurn();
    this.turn = { id: `replay-${turnId}`, replay: true, speak: true, done: true };
    this.speaker.say(this.turn.id, text, { voice: pickVoice(s.lang, s.voiceURI), rate: s.rate, lang: s.lang });
    this._set({ phase: "speaking" });
  }

  newConversation(config) {
    this.interrupt();
    this.configure(config);
  }

  destroy() {
    this._cancelTurn();
    this.rec?.abort();
    this._closeMic();
    this.socket.close();
  }

  // ── microphone & recognition ──────────────────────────────────────
  async _ensureMic() {
    if (this.mic) return true;
    try {
      this.mic = await openMic();
      return true;
    } catch (err) {
      this._set({ phase: "idle", error: micErrorMessage(err) });
      return false;
    }
  }

  _closeMic() {
    this.mic?.close();
    this.mic = null;
  }

  _startRecognizer() {
    const s = this.getSettings();
    this.rec?.abort();
    const rec = createRecognizer({
      lang: s.lang,
      endSilenceMs: s.endSilenceMs,
      onPartial: (t) => this.rec === rec && this._onPartial(t),
      onSpeechEnd: () => this.rec === rec && this.state.phase === "listening" && this.socket.send({ type: "speech_end" }),
      onFinal: (t) => {
        if (this.rec !== rec) return;
        this.rec = null;
        if (this.state.phase !== "listening") {
          // Only echo was heard while the assistant talked: keep listening for a real interruption.
          if (this.turn) this._listenForBargeIn();
          return;
        }
        this._onUtterance(t);
      },
      onEnd: () => {
        if (this.rec !== rec) return;
        this.rec = null;
        if (this.state.phase === "listening") {
          this._closeMic();
          this._set({ phase: "idle", interim: "", notice: "I didn't catch that. Tap the mic and try again." });
        } else if (this.turn) {
          this._listenForBargeIn();
        }
      },
      onError: (code, message) => {
        if (this.rec !== rec) return;
        this.rec = null;
        if (this.state.phase === "listening") {
          this._closeMic();
          this._set({ phase: "idle", interim: "", error: message });
        }
      },
    });
    this.rec = rec;
    try {
      rec.start();
    } catch (err) {
      this.rec = null;
      this._closeMic();
      this._set({ phase: "idle", error: err.message });
    }
  }

  _onPartial(text) {
    const { phase } = this.state;
    if (phase === "thinking" || phase === "speaking") {
      // Conversation mode: the recognizer is live while the assistant answers.
      const spoken = `${this.speaker.recentText()} ${this.turn?.text || ""}`;
      if (isEcho(text, spoken)) return;
      this.echoPool = new Set(words(spoken)); // to trim echo picked up before the user cut in
      this._cancelTurn(); // the user is talking over the answer: stop and listen
    }
    const clean = this._trimEcho(text);
    this._set({ phase: "listening", interim: tidyTranscript(clean), error: null, notice: null });
    this.socket.send({ type: "partial", text: clean });
  }

  /** Drop leading words that repeat the answer (heard before the user cut in). */
  _trimEcho(text) {
    if (!this.echoPool) return text;
    const w = text.split(/\s+/);
    let i = 0;
    while (i < w.length - 2 && this.echoPool.has(w[i].toLowerCase().replace(/[^a-z0-9]/g, ""))) i++;
    return w.slice(i).join(" ");
  }

  _onUtterance(text) {
    this._closeMic();
    this._set({ interim: "" });
    const t = this._trimEcho(text);
    this.echoPool = null;
    this._submit(tidyTranscript(t, true), { voice: true });
  }

  async _record() {
    const s = this.getSettings();
    this.recording = recordUtterance(this.mic);
    const blob = await this.recording.done;
    this.recording = null;
    this._closeMic();
    if (this.state.phase !== "listening") return;
    if (!blob) {
      this._set({ phase: "idle", notice: "I didn't catch that. Tap the mic and try again." });
      return;
    }
    this._set({ phase: "transcribing" });
    try {
      const r = await this.transcribe(blob, s.lang, this.config?.campaign_id);
      if (!r.text) this._set({ phase: "idle", notice: "I couldn't make out any words. Please try again." });
      else this._submit(tidyTranscript(r.text, true), { voice: true });
    } catch (err) {
      this._set({ phase: "idle", error: err.message });
    }
  }

  // ── turns ─────────────────────────────────────────────────────────
  _submit(text, { voice }) {
    const s = this.getSettings();
    const id = uid();
    this.turn = { id, tFinal: performance.now(), voice, speak: voice && s.autoSpeak && ttsSupported, text: "", done: false };
    this.dispatch({ type: "user", id: `${id}-q`, text, via: voice ? "voice" : "text" });
    this.dispatch({ type: "assistant", turnId: id, voice });
    if (this.turn.speak) this.speaker.begin(id, { voice: pickVoice(s.lang, s.voiceURI), rate: s.rate, lang: s.lang });
    this._set({ phase: "thinking", speculative: null, interim: "" });

    if (this.socket.open) {
      this.socket.send({ type: "final", text, turn_id: id });
    } else {
      // Socket down (proxy without WebSocket support, server restarting): same answer over SSE.
      this.fallbackAsk(text, (e) => this._onEvent({ ...e, turn_id: id })).catch((err) =>
        this._onEvent({ type: "error", turn_id: id, message: err.message }));
    }
    if (voice && s.handsFree && this.mode === "webspeech") {
      // Listen while answering so the user can interrupt by speaking.
      setTimeout(() => this.turn?.id === id && !this.rec && this._listenForBargeIn(), 400);
    }
  }

  async _listenForBargeIn() {
    if (!(await this._ensureMic())) return;
    this._startRecognizer();
  }

  _cancelTurn() {
    const t = this.turn;
    this.speaker.cancel();
    if (t && !t.done) {
      t.done = true;
      if (!t.replay) {
        this.socket.send({ type: "cancel" });
        this.dispatch({ type: "interrupted", turnId: t.id });
      }
    }
    this.turn = null;
  }

  _onEvent(e) {
    if (e.type === "ready") {
      this._set({ server: e });
      return;
    }
    if (e.type === "speculative") {
      if (this.state.phase === "listening") this._set({ speculative: e });
      return;
    }
    const t = this.turn;
    if (e.type === "error" && !e.turn_id) {
      this._set({ error: e.message });
      return;
    }
    if (!t || e.turn_id !== t.id) return; // late events of a cancelled turn
    if (e.type === "retrieval") {
      this.dispatch({ type: "retrieval", turnId: t.id, retrieval: e.retrieval, cache: e.cache, latency: e.latency_ms });
    } else if (e.type === "token") {
      t.text += e.text;
      this.dispatch({ type: "token", turnId: t.id, text: e.text });
      if (t.speak) this.speaker.push(t.id, e.text);
    } else if (e.type === "done") {
      t.done = true;
      this.dispatch({ type: "done", turnId: t.id, payload: e });
      if (t.speak) this.speaker.flush(t.id);
      else this._finishTurn(t.id);
    } else if (e.type === "error") {
      t.done = true;
      this.speaker.cancel();
      this.dispatch({ type: "error", turnId: t.id, message: e.message });
      this._finishTurn(t.id);
    }
  }

  _firstAudio(turnId) {
    const t = this.turn;
    if (!t || t.id !== turnId) return;
    if (!t.replay) {
      this.dispatch({ type: "voice_metric", turnId, patch: { first_audio_ms: Math.round(performance.now() - t.tFinal) } });
    }
    if (this.state.phase === "thinking") this._set({ phase: "speaking" });
  }

  _spoken(turnId) {
    if (this.turn?.id === turnId) this._finishTurn(turnId);
  }

  _finishTurn(turnId) {
    if (this.turn?.id !== turnId) return;
    const voice = this.turn.voice;
    this.turn = null;
    if (this.state.phase === "listening") return; // already taken over by a new utterance
    const s = this.getSettings();
    if (voice && s.handsFree && this.mode === "webspeech") {
      // Fresh recognizer for the next question: the barge-in one may hold echo of the answer.
      this.rec?.abort();
      this.rec = null;
      this.listen();
    } else {
      if (this.rec) { this.rec.abort(); this.rec = null; }
      this._closeMic();
      this._set({ phase: "idle" });
    }
  }
}
