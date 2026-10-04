// Streaming text-to-speech on the browser's SpeechSynthesis.
//
// Tokens arrive from the LLM a few characters at a time. Waiting for the whole answer before
// speaking would add the full generation time to the perceived latency, so text is cut into
// sentences as it streams and each sentence is spoken as soon as it is complete. Short
// utterances also dodge Chrome's habit of silently stopping long ones after ~15 s.

const CITATIONS = /\s*(\[S\d+\])+/g;
const ABBREVIATIONS = /\b(?:Dr|Mr|Mrs|Ms|Prof|Rs|St|No|Nos|vs|approx|etc|e\.g|i\.e|p|pp|Sr|Jr|Govt|Dept)\.$/i;
const MIN_SENTENCE = 24;
const MAX_CLAUSE = 220;

export const ttsSupported = typeof window !== "undefined" && "speechSynthesis" in window;

/** Text as it should be heard: no citation markers, markdown or ellipsis artefacts. */
export function speakable(text) {
  return text
    .replace(CITATIONS, "")
    .replace(/[*_#`>|]/g, "")
    .replace(/\s+…$/, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Split off complete sentences; returns [sentences, rest]. Decimals ("6.5") and common
 *  abbreviations ("Dr.") never end a sentence. */
export function splitSentences(buffer, force = false) {
  const out = [];
  let rest = buffer;
  for (;;) {
    let cut = -1;
    const re = /[.!?;](?=\s+["'(]?[A-Z0-9])|[.!?](?=\s*$)/g;
    let m;
    while ((m = re.exec(rest))) {
      const end = m.index + 1;
      const head = rest.slice(0, end);
      if (head.trim().length < MIN_SENTENCE) continue;
      if (ABBREVIATIONS.test(head)) continue;
      if (/\d\.$/.test(head) && /^\d/.test(rest.slice(end))) continue;
      if (!force && end === rest.length) break; // a trailing "." may still be "6." of "6.5"
      cut = end;
      break;
    }
    if (cut < 0 && rest.length > MAX_CLAUSE) {
      const comma = rest.lastIndexOf(", ", MAX_CLAUSE);
      if (comma > MIN_SENTENCE) cut = comma + 1;
    }
    if (cut < 0) break;
    const sentence = rest.slice(0, cut).trim();
    if (sentence) out.push(sentence);
    rest = rest.slice(cut);
  }
  if (force && rest.trim()) {
    out.push(rest.trim());
    rest = "";
  }
  return [out, rest];
}

const PREFERRED = /(natural|neural|premium|enhanced|google|online)/i;

export function listVoices() {
  if (!ttsSupported) return [];
  return window.speechSynthesis.getVoices().filter((v) => v.lang?.toLowerCase().startsWith("en"));
}

export function pickVoice(lang = "en-IN", voiceURI = "") {
  const voices = listVoices();
  if (!voices.length) return null;
  if (voiceURI) {
    const exact = voices.find((v) => v.voiceURI === voiceURI);
    if (exact) return exact;
  }
  const byLang = (l) => voices.filter((v) => v.lang.toLowerCase().replace("_", "-") === l.toLowerCase());
  for (const l of [lang, "en-GB", "en-US"]) {
    const group = byLang(l);
    const good = group.find((v) => PREFERRED.test(v.name));
    if (good || group[0]) return good || group[0];
  }
  return voices.find((v) => PREFERRED.test(v.name)) || voices[0];
}

/** Speaks one answer turn sentence by sentence. Events: onStart (first audio), onIdle (done). */
export class SentenceSpeaker {
  constructor({ onStart, onIdle } = {}) {
    this.onStart = onStart;
    this.onIdle = onIdle;
    this.buffer = "";
    this.queue = [];
    this.active = null;     // utterance being spoken (kept referenced: Chrome GCs orphans mid-speech)
    this.turn = null;
    this.started = false;
    this.flushed = false;
    this.recent = [];       // last sentences spoken, for echo detection during barge-in
    this.voice = null;
    this.rate = 1.0;
    this.lang = "en-IN";
    this._keepAlive = null;
  }

  get speaking() {
    return Boolean(this.active || this.queue.length);
  }

  /** Must run inside a user gesture once (iOS/Safari only allow speech after one). */
  unlock() {
    if (!ttsSupported || this._unlocked) return;
    const u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    window.speechSynthesis.speak(u);
    this._unlocked = true;
  }

  begin(turn, { voice, rate, lang }) {
    this.cancel();
    this.turn = turn;
    this.voice = voice;
    this.rate = rate;
    this.lang = lang;
    this.started = false;
    this.flushed = false;
  }

  push(turn, delta) {
    if (!ttsSupported || turn !== this.turn) return;
    this.buffer += delta;
    const [sentences, rest] = splitSentences(this.buffer);
    this.buffer = rest;
    sentences.forEach((s) => this._enqueue(s));
  }

  flush(turn) {
    if (turn !== this.turn) return;
    const [sentences] = splitSentences(this.buffer, true);
    this.buffer = "";
    sentences.forEach((s) => this._enqueue(s));
    this.flushed = true;
    if (!this.speaking) this._idle();
  }

  /** Speak a whole text now (e.g. "replay answer"). */
  say(turn, text, opts) {
    this.begin(turn, opts);
    this.push(turn, text);
    this.flush(turn);
  }

  cancel() {
    this.queue = [];
    this.buffer = "";
    this.active = null;
    this.turn = null;
    clearInterval(this._keepAlive);
    this._keepAlive = null;
    if (ttsSupported) window.speechSynthesis.cancel();
  }

  recentText() {
    return this.recent.join(" ");
  }

  _enqueue(sentence) {
    const text = speakable(sentence);
    if (!text) return;
    this.queue.push(text);
    if (!this.active) this._next();
  }

  _next() {
    const text = this.queue.shift();
    if (!text) {
      this.active = null;
      if (this.flushed) this._idle();
      return;
    }
    const u = new SpeechSynthesisUtterance(text);
    if (this.voice) u.voice = this.voice;
    u.lang = this.voice?.lang || this.lang;
    u.rate = this.rate;
    const turn = this.turn;
    u.onstart = () => {
      this.recent = [...this.recent.slice(-2), text];
      if (!this.started && turn === this.turn) {
        this.started = true;
        this.onStart?.(turn);
      }
    };
    const done = () => {
      if (this.active !== u) return; // cancelled or superseded
      this.active = null;
      this._next();
    };
    u.onend = done;
    u.onerror = done;
    this.active = u;
    window.speechSynthesis.speak(u);
    if (!this._keepAlive) {
      // Chrome can stall the speech queue after long sessions; a periodic resume() un-sticks it.
      this._keepAlive = setInterval(() => window.speechSynthesis.speaking && window.speechSynthesis.resume(), 5000);
    }
  }

  _idle() {
    clearInterval(this._keepAlive);
    this._keepAlive = null;
    const turn = this.turn;
    this.turn = null;
    this.onIdle?.(turn);
  }
}
