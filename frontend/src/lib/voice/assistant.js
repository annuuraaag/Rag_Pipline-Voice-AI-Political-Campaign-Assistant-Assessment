// Voice assistant controller (framework-free; React subscribes via useSyncExternalStore).
//
//   idle ──tap/Space──▶ listening ──end of speech──▶ thinking ──first audio──▶ speaking ──▶ idle
//                          │  partials → server (speculative retrieval)          │
//                          └──── recorder fallback: transcribing ────────────────┘
//
// Barge-in: tapping the mic, or speaking over the answer ("Interrupt by speaking", on by
// default), stops speech output at once, cancels the server-side answer, and starts listening.
// The recognizer keeps running while the assistant talks, so its own voice can be picked up;
// words that follow what was just said aloud are treated as echo and ignored (echo.js).
// "Stop" / "wait" only stops the answer. Conversation mode also keeps listening after each answer.
//
// Speech can be recognised and spoken in the browser (Web Speech API) or on the server:
//   input "server"   the microphone streams 16 kHz PCM over the socket; the server recognises,
//                    decides when the question is over, starts the answer (an `utterance` event)
//                    and, in conversation mode, detects barge-in itself (a `barge_in` event).
//   output "server"  the answer arrives as audio frames, played as they come (pcm.js).

import { openMic, micErrorMessage, recordUtterance } from "./mic.js";
import { PcmPlayer, pcmCaptureSupported, startPcmCapture } from "./pcm.js";
import { dropWords, interruption, isStopCommand } from "./echo.js";
import { createRecognizer, speechRecognitionSupported, tidyTranscript } from "./recognizer.js";
import { SentenceSpeaker, pickVoice, ttsSupported } from "./tts.js";
import { VoiceSocket, voiceSocketUrl } from "./socket.js";

const uid = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/\W/g, "").slice(0, 10);

/** Silence that ends the turn, from the server's end-of-turn verdict and the user's pause setting
 *  (which stays the wait whenever the server is unsure, and the most a finished question waits). */
export function endpointWait(e, userMs) {
  if (e.turn === "complete" || e.turn === "likely") return Math.min(e.wait_ms, userMs);
  if (e.turn === "incomplete") return Math.max(e.wait_ms, userMs);
  return userMs;
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
      listeningOver: false, // listening while the answer is spoken: talking interrupts it
    };
    this.skipWords = 0; // echo words at the start of the barge-in recognizer's transcript
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

  /** Speaking over the answer interrupts it (also always on in conversation mode). */
  get bargeIn() {
    const s = this.getSettings();
    return s.bargeIn !== false || Boolean(s.handsFree);
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
    this.skipWords = 0;
    this.speaker.unlock(); // inside the click/keypress: lets Safari speak later
    this.player.unlock();
    const mode = this.mode;
    if (mode === "none") {
      this._set({ error: "Voice input needs Chrome, Edge or Safari (or a server speech-to-text key). You can still type." });
      return;
    }
    this._set({ error: null, notice: null, interim: "", speculative: null, endpoint: null, phase: "listening", listeningOver: false });
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
      this._set({ phase: "idle", interim: "", listeningOver: false });
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
      // The browser's recognizer opens the microphone itself: ours only drives the level meter.
      this.mic = await openMic({ processing: this.mode !== "webspeech" });
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
    if (isStopCommand(e.text)) {
      // "Stop": the server took it for a question; it isn't one.
      this.socket.send({ type: "cancel" });
      this._afterStop();
      return;
    }
    // While the answer is spoken the microphone keeps streaming: the server listens for barge-in.
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
    this._set({ phase: "listening", interim: "", speculative: null, error: null, notice: null, listeningOver: false });
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
      // While the answer is spoken, silence only clears the echo heard so far.
      keepAlive: () => this.rec === rec && (this.state.phase === "thinking" || this.state.phase === "speaking"),
      onPartial: (t) => this.rec === rec && this._onPartial(t),
      onSpeechEnd: () => this.rec === rec && this.state.phase === "listening" && this.socket.send({ type: "speech_end" }),
      onFinal: (t, reason, waitedMs) => {
        if (this.rec !== rec) return;
        this.rec = null;
        if (this.state.phase !== "listening") {
          // The browser kept ending the session while the answer played: talking can't interrupt.
          this._set({ listeningOver: false });
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
        } else {
          this._set({ listeningOver: false });
        }
      },
      onError: (code, message) => {
        if (this.rec !== rec) return;
        this.rec = null;
        if (this.state.phase === "listening") {
          this._closeMic();
          this._set({ phase: "idle", interim: "", error: message });
        } else {
          this._set({ listeningOver: false });
        }
      },
    });
    this.rec = rec;
    try {
      rec.start();
    } catch (err) {
      this.rec = null;
      if (this.state.phase === "listening") {
        this._closeMic();
        this._set({ phase: "idle", error: err.message });
      } else {
        this._set({ listeningOver: false }); // the answer goes on; only talking can't interrupt it
      }
    }
  }

  _onPartial(text) {
    const { phase } = this.state;
    if (phase === "thinking" || phase === "speaking") {
      // The recognizer is live while the assistant answers: is this the user, or the answer's echo?
      const spoken = `${this.speaker.recentText()} ${this.player.recentText()} ${this.turn?.text || ""}`;
      const cut = interruption(text, spoken);
      if (!cut) return;
      this.skipWords = cut.skip; // the echo heard before the user cut in
      this._cancelTurn(); // the user is talking over the answer: stop and listen
    }
    const clean = dropWords(text, this.skipWords);
    this._set({ phase: "listening", interim: tidyTranscript(clean), error: null, notice: null, listeningOver: false });
    this.socket.send({ type: "partial", text: clean });
  }

  _onUtterance(text, endpointMs = null) {
    const t = dropWords(text, this.skipWords);
    this.skipWords = 0;
    this._set({ interim: "", endpoint: null });
    if (isStopCommand(t)) {
      this._afterStop();
      return;
    }
    this._submit(tidyTranscript(t, true), { voice: true, endpointMs });
  }

  /** After "stop": nothing to answer. Conversation mode keeps listening; otherwise back to idle. */
  _afterStop() {
    if (this.rec) { this.rec.abort(); this.rec = null; }
    if (this.getSettings().handsFree) {
      this.listen();
    } else {
      this._closeMic();
      this._set({ phase: "idle", interim: "", listeningOver: false });
    }
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
    // Listen while the answer is spoken, so the user can interrupt by talking.
    const mode = this.mode;
    const over = voice && (this.turn.speak || serverAudio) && this.bargeIn && (mode === "webspeech" || mode === "server");
    if (!over) this._closeMic();
    this._set({ phase: "thinking", speculative: null, interim: "", listeningOver: over && mode === "server" });

    if (started) {
      // nothing to send: the server is answering already
    } else if (this.socket.open) {
      this.socket.send({ type: "final", text, turn_id: id, speak: serverAudio });
    } else {
      // Socket down (proxy without WebSocket support, server restarting): same answer over SSE.
      this.fallbackAsk(text, (e) => this._onEvent({ ...e, turn_id: id })).catch((err) =>
        this._onEvent({ type: "error", turn_id: id, message: err.message }));
    }
    if (over && mode === "webspeech") {
      // A moment for the question's recognizer to release the microphone.
      setTimeout(() => this.turn?.id === id && !this.rec && this._listenForBargeIn(), 300);
    }
  }

  async _listenForBargeIn() {
    if (!(await this._ensureMic())) return;
    if (!this.turn) return; // the answer finished meanwhile
    this.skipWords = 0;
    this._startRecognizer();
    if (this.rec) this._set({ listeningOver: true });
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
    this._set({ listeningOver: false });
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
