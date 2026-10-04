// Web Speech API recognizer with our own end-of-utterance detection ("endpointing").
//
// Chrome's single-shot mode (continuous = false) ends the utterance at the first short pause,
// which cuts questions in half ("what healthcare … schemes are there"). We run it continuously
// instead and decide the end ourselves: no new words for `endSilenceMs` after speech → final.
// Every result event also yields the full running transcript as a partial, which the server
// uses for speculative retrieval. The server answers each partial with an end-of-turn verdict;
// hint() then shortens the wait when the question sounds finished and lengthens it when it ends
// on "for", "the" or "um".

const SR = typeof window !== "undefined" ? window.SpeechRecognition || window.webkitSpeechRecognition : null;
export const speechRecognitionSupported = Boolean(SR);

export const RECOGNITION_ERRORS = {
  "not-allowed": "Microphone access is blocked. Allow it in the browser's site settings, then try again.",
  "service-not-allowed": "This browser doesn't allow speech recognition here. Try Chrome or Edge, or type your question.",
  "audio-capture": "No microphone was found. Check that one is connected.",
  network: "The browser's speech service is unreachable (Chrome needs an internet connection for it).",
  "language-not-supported": "This language isn't supported for speech recognition. Change it in Settings.",
};

/** Make an ASR transcript read like a sentence (Chrome returns lower case, no punctuation). */
export function tidyTranscript(text, final = false) {
  let t = text.replace(/\s+/g, " ").trim();
  if (!t) return t;
  t = t[0].toUpperCase() + t.slice(1);
  if (final && !/[.?!]$/.test(t)) {
    const question = /^(what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|will|would|should|shall|has|have|tell me)\b/i;
    t += question.test(t) ? "?" : ".";
  }
  return t;
}

const norm = (t) => (t.toLowerCase().match(/[a-z0-9']+/g) || []).join(" ");

export function createRecognizer({ lang = "en-IN", endSilenceMs = 900, noSpeechMs = 8000, onPartial, onFinal,
  onSpeechStart, onSpeechEnd, onError, onEnd }) {
  if (!SR) throw new Error("Speech recognition is not supported in this browser.");
  const rec = new SR();
  rec.lang = lang;
  rec.continuous = true;
  rec.interimResults = true;
  rec.maxAlternatives = 1;

  let finals = "";
  let interim = "";
  let finished = false;
  let running = false;
  let silenceTimer = null;
  let noSpeechTimer = null;
  let lastResultAt = 0;

  const text = () => `${finals} ${interim}`.replace(/\s+/g, " ").trim();
  const clearTimers = () => { clearTimeout(silenceTimer); clearTimeout(noSpeechTimer); };
  const armSilence = (ms) => {
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(() => finish("silence"), Math.max(0, ms));
  };

  function finish(reason) {
    if (finished) return;
    finished = true;
    clearTimers();
    const t = text();
    try { rec.stop(); } catch { /* already stopped */ }
    // waitedMs: last recognised words → end of turn (the endpointing share of the latency).
    if (t) onFinal?.(t, reason, lastResultAt ? Math.round(performance.now() - lastResultAt) : null);
    else onEnd?.("no-speech");
  }

  rec.onstart = () => {
    running = true;
    noSpeechTimer = setTimeout(() => !text() && finish("timeout"), noSpeechMs);
  };
  rec.onspeechstart = () => onSpeechStart?.();
  rec.onspeechend = () => onSpeechEnd?.();
  rec.onresult = (event) => {
    interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const r = event.results[i];
      if (r.isFinal) finals += ` ${r[0].transcript}`;
      else interim += ` ${r[0].transcript}`;
    }
    const t = text();
    if (!t) return;
    clearTimeout(noSpeechTimer);
    lastResultAt = performance.now();
    // Wait a little longer while words are still provisional: Chrome may yet revise them.
    armSilence(interim.trim() ? endSilenceMs + 400 : endSilenceMs);
    onPartial?.(t);
  };
  rec.onerror = (event) => {
    if (event.error === "no-speech" || event.error === "aborted") return; // handled by onend
    finished = true;
    clearTimers();
    onError?.(event.error, RECOGNITION_ERRORS[event.error] || `Speech recognition error: ${event.error}`);
  };
  rec.onend = () => {
    running = false;
    if (!finished) finish("ended"); // the browser stopped on its own: whatever we have is final
  };

  return {
    start() {
      finals = ""; interim = ""; finished = false;
      rec.start();
    },
    /** User pressed stop: treat what we have as the final transcript. */
    stop() { finish("manual"); },
    /** Discard everything (e.g. the assistant was interrupted by a click, not speech). */
    abort() {
      finished = true;
      clearTimers();
      try { rec.abort(); } catch { /* not started */ }
    },
    /** End-of-turn hint for the transcript ending in `forText`: end the turn after `waitMs` of
     *  silence instead of the default. Ignored once newer words have arrived. */
    hint(forText, waitMs) {
      const now = norm(text());
      const heard = norm(forText || "");
      if (finished || !heard || !now.endsWith(heard)) return false;
      const provisional = interim.trim() ? 150 : 0;
      armSilence(waitMs + provisional - (performance.now() - lastResultAt));
      return true;
    },
    get running() { return running; },
    get transcript() { return text(); },
  };
}
