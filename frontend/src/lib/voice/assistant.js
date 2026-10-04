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
//
// Speech can be recognised and spoken in the browser (Web Speech API) or on the server:
//   input "server"   the microphone streams 16 kHz PCM over the socket; the server recognises,
//                    decides when the question is over, starts the answer (an `utterance` event)
//                    and, in conversation mode, detects barge-in itself (a `barge_in` event).
//   output "server"  the answer arrives as audio frames, played as they come (pcm.js).

import { openMic, micErrorMessage, recordUtterance } from "./mic.js";
import { PcmPlayer, pcmCaptureSupported, startPcmCapture } from "./pcm.js";
import { createRecognizer, speechRecognitionSupported, tidyTranscript } from "./recognizer.js";
import { SentenceSpeaker, pickVoice, ttsSupported } from "./tts.js";
import { VoiceSocket, voiceSocketUrl } from "./socket.js";

const uid = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/\W/g, "").slice(0, 10);
const words = (t) => t.toLowerCase().match(/[a-z0-9]+/g) || [];

/** Silence that ends the turn, from the server's end-of-turn verdict and the user's pause setting
 *  (which stays the wait whenever the server is unsure, and the most a finished question waits). */
export function endpointWait(e, userMs) {
  if (e.turn === "complete" || e.turn === "likely") return Math.min(e.wait_ms, userMs);
  if (e.turn === "incomplete") return Math.max(e.wait_ms, userMs);
  return userMs;
}

/** True when `heard` is probably the assistant's own voice coming back through the mic: its words
 *  follow what was said, in the same order. Word pairs, not single words, so a question that reuses
 *  some of the answer's words ("who is eligible for …") still counts as the user talking. */
export function isEcho(heard, spoken) {
  const w = words(heard);
  if (w.length < 2) return true; // one word ("ok", "the") is never a deliberate interruption
  const said = words(spoken);
  if (said.length < 2) return false;
  const pairs = new Set(said.slice(1).map((x, i) => `${said[i]} ${x}`));
  const heardPairs = w.slice(1).map((x, i) => `${w[i]} ${x}`);
  return heardPairs.filter((p) => pairs.has(p)).length / heardPairs.length >= 0.6;
}

export class VoiceAssistant {
  constructor({ dispatch, getSettings, transcribe, fallbackAsk }) {
    this.dispatch = dispatch;
    this.getSettings = getSettings;
    this.transcribe = transcribe;
    this.fallbackAsk = fallbackAsk;
    this.listeners = new Set();
    this.state = {
      phase: "idle", interim: "", speculative: null, endpoint: null, error: null, notice: null,
      socket: "connecting", server: null,
    };
    this.turn = null;
    this.mic = null;
    this.capture = null;
    this.rec = null;
    this.recording = null;
    this.config = null;
    this.speechKey = null;
    this.speaker = new SentenceSpeaker({ onStart: (t) => this._firstAudio(t), onIdle: (t) => this._spoken(t) });
    this.player = new PcmPlayer({ onStart: (t) => this._firstAudio(t), onIdle: (t) => this._spoken(t) });
    this.socket = new VoiceSocket(voiceSocketUrl(), {
      onEvent: (e) => this._onEvent(e),
      onAudio: (frame) => this._onAudio(frame),
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

  /** How speech is recognised: "server" | "webspeech" | "recorder" (clip → /transcribe) | "none". */
  get mode() {
    const pref = this.getSettings().speechInput || "auto";
    const server = Boolean(this.state.server?.speech?.stt) && pcmCaptureSupported && this.socket.open;
    if (pref === "server" && server) return "server";
    if (pref !== "server" && speechRecognitionSupported) return "webspeech";
    if (server) return "server";
    if (typeof window !== "undefined" && window.MediaRecorder && this.state.server?.transcribe) return "recorder";
    return "none";
  }

  /** Who speaks the answer: "server" (streamed audio) | "browser" (speechSynthesis) | "none". */
  get output() {
    const pref = this.getSettings().speechOutput || "auto";
    const server = Boolean(this.state.server?.speech?.tts) && this.socket.open;
    if (pref === "browser" && ttsSupported) return "browser";
    if (server) return "server";
    return ttsSupported ? "browser" : "none";
  }

  configure(config) {
    this.config = config;
    this.speechKey = null;
    this._syncSpeech();
  }

  /** Tell the server which side recognises and speaks (re-sent only when that changes). */
  _syncSpeech() {
    if (!this.config) return;
    const s = this.getSettings();
    const speech = {
      input: this.mode === "server" ? "server" : "browser",
      output: s.autoSpeak && this.output === "server" ? "server" : "browser",
      rate: s.rate, language: s.lang,
    };
    const key = JSON.stringify(speech);
    if (key === this.speechKey) return;
    this.speechKey = key;
    this.socket.configure({ ...this.config, speech });
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
    this.player.unlock();
    const mode = this.mode;
    if (mode === "none") {
      this._set({ error: "Voice input needs Chrome, Edge or Safari (or a server speech-to-text key). You can still type." });
      return;
    }
    this._set({ error: null, notice: null, interim: "", speculative: null, endpoint: null, phase: "listening" });
    if (!(await this._ensureMic())) return;
    if (mode === "server") this._listenServer();
    else if (mode === "webspeech") this._startRecognizer();
    else this._record();
  }

  stopListening() {
    if (this.mode === "server" && this.capture) this.socket.send({ type: "audio_stop" });
    else if (this.rec) this.rec.stop();
    else if (this.recording) this.recording.stop();
  }

  /** Stop talking / cancel the answer. With listen=true, start listening right away (barge-in). */
  interrupt({ listen = false } = {}) {
    if (this.capture && this.state.phase === "listening") this.socket.send({ type: "audio_cancel" });
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
    if (this.capture && this.state.phase === "listening") this.socket.send({ type: "audio_cancel" });
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
    this.capture?.stop();
    this.capture = null;
    this.mic?.close();
    this.mic = null;
  }

  // ── server recognition ────────────────────────────────────────────
  async _listenServer() {
    this._syncSpeech();
    this.socket.send({ type: "audio_start" });
    if (this.capture || !this.mic) return;
    try {
      const mic = this.mic;
      const capture = await startPcmCapture(mic.stream, (buf) => this.socket.sendBinary(buf));
      if (this.mic === mic) this.capture = capture;
      else capture.stop(); // the microphone was closed while the worklet loaded
    } catch (err) {
      this._closeMic();
      this._set({ phase: "idle", error: `Could not stream the microphone to the server: ${err.message}` });
    }
  }

  _onServerUtterance(e) {
    if (this.state.phase !== "listening") return;
    if (!e.text) {
      this._closeMic();
      this._set({ phase: "idle", interim: "", notice: "I didn't catch that. Tap the mic and try again." });
      return;
    }
    // In conversation mode the microphone keeps streaming: the server listens for barge-in.
    if (!this.getSettings().handsFree) this._closeMic();
    this._set({ interim: "", endpoint: null });
    this._submit(tidyTranscript(e.text, true), { voice: true, endpointMs: e.endpoint_ms, turnId: e.turn_id, started: true });
  }

  _onServerBargeIn(e) {
    const t = this.turn;
    if (t && e.turn_id === t.id) {
      this.speaker.cancel();
      this.player.cancel();
      t.done = true;
      this.dispatch({ type: "interrupted", turnId: t.id });
      this.turn = null;
    }
    // The server already treats the interruption as the next question.
    this._set({ phase: "listening", interim: "", speculative: null, error: null, notice: null });
  }

  _onAudio({ header, pcm }) {
    const t = this.turn;
    if (t && t.serverAudio && header.turn_id === t.id) this.player.push(t.id, header, pcm);
  }

  _startRecognizer() {
    const s = this.getSettings();
    this.rec?.abort();
    const rec = createRecognizer({
      lang: s.lang,
      endSilenceMs: s.endSilenceMs,
      onPartial: (t) => this.rec === rec && this._onPartial(t),
      onSpeechEnd: () => this.rec === rec && this.state.phase === "listening" && this.socket.send({ type: "speech_end" }),
      onFinal: (t, reason, waitedMs) => {
        if (this.rec !== rec) return;
        this.rec = null;
        if (this.state.phase !== "listening") {
          // Only echo was heard while the assistant talked: keep listening for a real interruption.
          if (this.turn) this._listenForBargeIn();
          return;
        }
        this._onUtterance(t, reason === "silence" ? waitedMs : null);
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
      const spoken = `${this.speaker.recentText()} ${this.player.recentText()} ${this.turn?.text || ""}`;
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

  _onUtterance(text, endpointMs = null) {
    this._closeMic();
    this._set({ interim: "", endpoint: null });
    const t = this._trimEcho(text);
    this.echoPool = null;
    this._submit(tidyTranscript(t, true), { voice: true, endpointMs });
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
  /** A question becomes a turn. `started`: the server already began answering it (server recognition). */
  _submit(text, { voice, endpointMs = null, turnId = null, started = false }) {
    const s = this.getSettings();
    if (!started) this._syncSpeech();
    const id = turnId || uid();
    const out = this.output;
    const serverAudio = voice && s.autoSpeak && out === "server" && this.socket.open;
    this.turn = {
      id, tFinal: performance.now(), voice, text: "", done: false,
      speak: voice && s.autoSpeak && out === "browser", serverAudio,
    };
    this.dispatch({ type: "user", id: `${id}-q`, text, via: voice ? "voice" : "text" });
    this.dispatch({ type: "assistant", turnId: id, voice });
    if (endpointMs != null) this.dispatch({ type: "voice_metric", turnId: id, patch: { endpoint_wait_ms: endpointMs } });
    if (this.turn.speak) this.speaker.begin(id, { voice: pickVoice(s.lang, s.voiceURI), rate: s.rate, lang: s.lang });
    if (serverAudio) this.player.begin(id);
    this._set({ phase: "thinking", speculative: null, interim: "" });

    if (started) {
      // nothing to send: the server is answering already
    } else if (this.socket.open) {
      this.socket.send({ type: "final", text, turn_id: id, speak: serverAudio });
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
    this.player.cancel();
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
      this._syncSpeech(); // now that the server's speech providers are known
      return;
    }
    if (e.type === "speculative") {
      if (this.state.phase === "listening") this._set({ speculative: e });
      return;
    }
    if (e.type === "endpoint") {
      if (this.state.phase !== "listening") return;
      this._set({ endpoint: e });
      const s = this.getSettings();
      if (this.rec && s.smartEndpointing !== false) this.rec.hint(e.text, endpointWait(e, s.endSilenceMs));
      return;
    }
    if (e.type === "transcript") {
      if (this.state.phase === "listening") this._set({ interim: tidyTranscript(e.text) });
      return;
    }
    if (e.type === "utterance") return this._onServerUtterance(e);
    if (e.type === "barge_in") return this._onServerBargeIn(e);
    if (e.type === "voice_metrics") {
      // Server-side timings of the spoken answer (may arrive after the turn has finished playing).
      const { type, turn_id: turnId, ...m } = e;
      this.dispatch({ type: "voice_metric", turnId, patch: { server: m } });
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
      else if (!t.serverAudio) this._finishTurn(t.id); // server audio: finished when it has played
    } else if (e.type === "audio_end") {
      this.player.end(t.id);
    } else if (e.type === "audio_error") {
      this._set({ notice: "The server voice failed; the answer is shown as text." });
      this.player.end(t.id);
    } else if (e.type === "error") {
      t.done = true;
      this.speaker.cancel();
      this.player.cancel();
      this.dispatch({ type: "error", turnId: t.id, message: e.message });
      this._finishTurn(t.id);
    }
  }

  _firstAudio(turnId) {
    const t = this.turn;
    if (!t || t.id !== turnId) return;
    if (!t.replay) {
      this.dispatch({ type: "voice_metric", turnId, patch: { first_audio_ms: Math.round(performance.now() - t.tFinal) } });
      // While the answer is heard the server listens for barge-in (server recognition).
      this.socket.send({ type: "playback", playing: true });
    }
    if (this.state.phase === "thinking") this._set({ phase: "speaking" });
  }

  _spoken(turnId) {
    if (this.turn?.id === turnId) this._finishTurn(turnId);
  }

  _finishTurn(turnId) {
    if (this.turn?.id !== turnId) return;
    const voice = this.turn.voice;
    if (!this.turn.replay) this.socket.send({ type: "playback", playing: false });
    this.turn = null;
    if (this.state.phase === "listening") return; // already taken over by a new utterance
    const s = this.getSettings();
    if (voice && s.handsFree && (this.mode === "webspeech" || this.mode === "server")) {
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
