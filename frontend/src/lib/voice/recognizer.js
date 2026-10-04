// Web Speech API recognizer with our own end-of-utterance detection ("endpointing").
//
// Chrome's single-shot mode (continuous = false) ends the utterance at the first short pause,
// which cuts questions in half ("what healthcare … schemes are there"). We run it continuously
// instead and decide the end ourselves: no new words for `endSilenceMs` after speech → final.
// Every result event also yields the full running transcript as a partial, which the server
// uses for speculative retrieval. The server answers each partial with an end-of-turn verdict;
// hint() then shortens the wait when the question sounds finished and lengthens it when it ends
// on "for", "the" or "um".
//
// Browsers differ: Edge's speech service (and Chrome's after a long silence or a network blip)
// ends a "continuous" session on its own. A session that ends while we are still waiting for
// words is restarted and its transcript kept, so a pause never cuts a question in half and a
// quiet start never gives up early. Sessions that end straight away back off, then give up.
//
// keepAlive(): while it returns true (the assistant is answering and we only listen for an
// interruption), silence does not end the utterance: what was heard (echo of the answer) is
// discarded instead, and the session is kept running.

const SR = typeof window !== "undefined" ? window.SpeechRecognition || window.webkitSpeechRecognition : null;
export const speechRecognitionSupported = Boolean(SR);

export const RECOGNITION_ERRORS = {
  "not-allowed": "Microphone access is blocked. Allow it in the browser's site settings, then try again.",
  "service-not-allowed": "This browser doesn't allow speech recognition here. Try Chrome or Edge, or type your question.",
  "audio-capture": "No microphone was found. Check that one is connected.",
  network: "The browser's speech service is unreachable (Chrome and Edge need an internet connection for it).",
  "language-not-supported": "This language isn't supported for speech recognition. Change it in Settings.",
};
// Errors after which retrying cannot help.
const FATAL = new Set(["not-allowed", "service-not-allowed", "audio-capture", "language-not-supported", "bad-grammar"]);
const MAX_QUICK_ENDS = 5;

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

/** Append a final result. Some engines (Chrome on Android) repeat the earlier finals in each new
 *  one ("what is" → "what is the plan"): then the new one replaces them instead. */
function joinFinal(finals, t) {
  const a = norm(finals);
  return a && norm(t).startsWith(a) ? t : `${finals} ${t}`;
}

export function createRecognizer({ lang = "en-IN", endSilenceMs = 900, noSpeechMs = 8000, keepAlive = () => false,
  onPartial, onFinal, onSpeechStart, onSpeechEnd, onError, onEnd }) {
  if (!SR) throw new Error("Speech recognition is not supported in this browser.");
  const rec = new SR();
  rec.lang = lang;
  rec.continuous = true;
  rec.interimResults = true;
  rec.maxAlternatives = 1;

  let base = "";      // words of earlier sessions of this utterance
  let finals = "";
  let interim = "";
  let skip = 0;       // results of this session to ignore (discarded echo)
  let seen = 0;       // results of this session so far
  let finished = false;
  let running = false;
  let silenceTimer = null;
  let noSpeechTimer = null;
  let restartTimer = null;
  let lastResultAt = 0;
  let sessionAt = 0;
  let quickEnds = 0;
  let lastError = null;
  let hinted = null; // { text, waitMs }: the server's verdict on these exact words

  const text = () => `${base} ${finals} ${interim}`.replace(/\s+/g, " ").trim();
  const clearTimers = () => { clearTimeout(silenceTimer); clearTimeout(noSpeechTimer); clearTimeout(restartTimer); };
  const armSilence = (ms) => {
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(onSilence, Math.max(0, ms));
  };
  const armNoSpeech = () => {
    clearTimeout(noSpeechTimer);
    noSpeechTimer = setTimeout(() => {
      if (keepAlive()) armNoSpeech();
      else if (!text()) finish("timeout");
    }, noSpeechMs);
  };

  function onSilence() {
    if (finished) return;
    if (keepAlive()) {
      // Only the answer's echo was heard: forget it and keep listening for the user.
      base = ""; finals = ""; interim = ""; hinted = null;
      skip = seen;
      return;
    }
    finish("silence");
  }

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

  function fail(code) {
    if (finished) return;
    finished = true;
    clearTimers();
    try { rec.abort(); } catch { /* already stopped */ }
    onError?.(code, RECOGNITION_ERRORS[code] || `Speech recognition error: ${code}`);
  }

  function session() {
    if (finished) return;
    skip = 0; seen = 0;
    sessionAt = performance.now();
    try {
      rec.start();
    } catch {
      // The previous session has not fully stopped yet: try again shortly.
      ended();
    }
  }

  /** The browser ended a session. Restart it while we still need words, unless it keeps failing. */
  function ended() {
    running = false;
    if (finished) return;
    quickEnds = performance.now() - sessionAt < 1000 ? quickEnds + 1 : 0;
    if (quickEnds >= MAX_QUICK_ENDS) {
      if (!text() && lastError) fail(lastError);
      else finish("ended");
      return;
    }
    base = text(); finals = ""; interim = "";
    restartTimer = setTimeout(session, quickEnds ? 150 * 2 ** quickEnds : 0);
  }

  rec.onstart = () => { running = true; };
  rec.onspeechstart = () => onSpeechStart?.();
  rec.onspeechend = () => onSpeechEnd?.();
  rec.onresult = (event) => {
    // Rebuilt from the whole result list (it holds every result of the session), so no engine
    // quirk in resultIndex can drop or repeat words.
    seen = event.results.length;
    let f = "";
    let it = "";
    for (let i = skip; i < event.results.length; i++) {
      const r = event.results[i];
      const t = r[0]?.transcript || "";
      if (r.isFinal) f = joinFinal(f, t);
      else it += ` ${t}`;
    }
    finals = f;
    interim = it;
    const t = text();
    if (!t) return;
    lastResultAt = performance.now();
    if (hinted && hinted.text === norm(t)) {
      // Same words as the last verdict (typically the final result repeating the interim one):
      // the server sends no new hint for them, so keep that one.
      armSilence(hinted.waitMs + (interim.trim() ? 150 : 0));
    } else {
      // Wait a little longer while words are still provisional: the engine may yet revise them.
      armSilence(interim.trim() ? endSilenceMs + 400 : endSilenceMs);
    }
    onPartial?.(t);
  };
  rec.onerror = (event) => {
    // no-speech, aborted, network, no-match: the session ends (onend follows) and is restarted
    // with back-off. Only errors retrying cannot fix are reported.
    if (event.error !== "no-speech" && event.error !== "aborted") lastError = event.error;
    if (FATAL.has(event.error)) fail(event.error);
  };
  rec.onend = ended;

  return {
    start() {
      base = ""; finals = ""; interim = ""; finished = false; hinted = null; quickEnds = 0; lastResultAt = 0;
      lastError = null;
      skip = 0; seen = 0;
      sessionAt = performance.now();
      armNoSpeech();
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
      hinted = { text: now, waitMs };
      armSilence(waitMs + (interim.trim() ? 150 : 0) - (performance.now() - lastResultAt));
      return true;
    },
    get running() { return running; },
    get transcript() { return text(); },
  };
}
