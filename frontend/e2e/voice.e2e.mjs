// End-to-end test of the voice flow in a real browser against a running stack.
//
// Headless Chromium has no speech engine, so the Web Speech API is replaced with a scripted
// recognizer (emits word-by-word partial transcripts like Chrome does) and a fake
// speechSynthesis that records what would be spoken. Everything else is real: the React app,
// the WebSocket, speculative retrieval on the server, streaming, TTS sentence splitting,
// barge-in and the inspector.
//
//   BASE_URL=http://localhost:5173 node e2e/voice.e2e.mjs [--shots ./shots]
//
// Requires the `playwright` package (npm i -D playwright, or a global install).
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL || "http://localhost:5173";
const shotsArg = process.argv.indexOf("--shots");
const SHOTS = shotsArg > 0 ? process.argv[shotsArg + 1] : null;
if (SHOTS) mkdirSync(SHOTS, { recursive: true });

const FAKE_VOICE = () => {
  // Browser recognition and the browser voice, even if the server also offers speech (see server-speech.e2e.mjs).
  if (!localStorage.getItem("crag-settings")) localStorage.setItem("crag-settings", JSON.stringify({ speechInput: "browser", speechOutput: "browser" }));
  window.__voice = { next: "", queue: [], edge: false, spoken: [], cancels: 0, msPerChar: 12 };
  // One scripted session per start(): it says `__voice.next` (once), then stays open like Chrome's
  // continuous mode. `__voice.active.emit(text)` makes the running session hear more words later
  // (the answer's echo, or the user talking over it), as a new result of the same session.
  class FakeRecognition {
    start() {
      window.__voice.active = this;
      this.results = [];
      this.timers = [];
      setTimeout(() => this.onstart?.(), 30);
      const text = window.__voice.next || window.__voice.queue.shift() || "";
      window.__voice.next = "";
      if (text) this.emit(text);
    }
    emit(text, msPerWord = 140) {
      const words = text.split(" ").filter(Boolean);
      const idx = this.results.length;
      this.results.push(Object.assign([{ transcript: "" }], { isFinal: false }));
      this.onspeechstart?.();
      let i = 0;
      const t = setInterval(() => {
        i += 1;
        const final = i > words.length;
        this.results[idx] = Object.assign([{ transcript: words.slice(0, i).join(" ") }], { isFinal: final });
        this.onresult?.({ resultIndex: idx, results: this.results.slice() });
        if (final) {
          clearInterval(t);
          this.onspeechend?.();
          // Edge's speech service ends a "continuous" session on its own after a short pause.
          if (window.__voice.edge) this.timers.push(setTimeout(() => this.onend?.(), 500));
        }
      }, msPerWord);
      this.timers.push(t);
    }
    stop() { this.timers.forEach(clearInterval); setTimeout(() => this.onend?.(), 10); }
    abort() { this.timers.forEach(clearInterval); setTimeout(() => this.onend?.(), 10); }
  }
  window.SpeechRecognition = FakeRecognition;
  window.webkitSpeechRecognition = FakeRecognition;

  const synth = {
    speaking: false, pending: false, paused: false, queue: [],
    getVoices: () => [{ name: "Test Voice", lang: "en-IN", voiceURI: "test-voice", default: true }],
    speak(u) { this.queue.push(u); if (!this.speaking) this._run(); },
    _run() {
      const u = this.queue.shift();
      if (!u) { this.speaking = false; return; }
      this.speaking = true;
      setTimeout(() => {
        window.__voice.spoken.push(u.text);
        u.onstart?.();
        setTimeout(() => { u.onend?.(); this._run(); }, window.__voice.msPerChar * u.text.length);
      }, 15);
    },
    cancel() { this.queue = []; this.speaking = false; window.__voice.cancels += 1; },
    resume() {}, pause() {}, addEventListener() {}, removeEventListener() {},
  };
  Object.defineProperty(window, "speechSynthesis", { value: synth, configurable: true });
  window.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
};

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
});
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, permissions: ["microphone"] });
await context.addInitScript(FAKE_VOICE);
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && !/favicon|ERR_CONNECTION/.test(m.text()) && errors.push(m.text()));
const shot = async (name) => SHOTS && page.screenshot({ path: `${SHOTS}/${name}.png` });

try {
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector(".welcome");
  check("welcome screen renders", true);
  await shot("01-welcome");

  // ── 1. a spoken question with speculative retrieval ─────────────────
  await page.evaluate(() => { window.__voice.next = "what healthcare initiatives are proposed for vijayawada"; });
  await page.click(".mic");
  await page.waitForSelector(".livebar-listening", { timeout: 5000 });
  check("mic opens the listening bar", true);
  const spec = await page.waitForSelector(".spec", { timeout: 6000 }).then(() => true).catch(() => false);
  check("speculative search runs while speaking", spec);
  await shot("02-listening");
  await page.waitForSelector(".msg-user .via", { timeout: 8000 });
  const question = await page.textContent(".msg-user .bubble");
  check("final transcript becomes a tidy question", /^What healthcare initiatives are proposed for vijayawada\?$/i.test(question.trim()), question.trim());
  await page.waitForSelector(".msg-assistant .msg-meta", { timeout: 20000 });
  const answer = await page.textContent(".msg-assistant .answer");
  check("answer streams in with citations", (await page.$$(".msg-assistant .cite")).length > 0, answer.slice(0, 80));
  const ahead = await page.$(".msg-assistant .badge-accent");
  check("answer reused the speculative retrieval", Boolean(ahead), ahead ? await ahead.textContent() : "no badge");
  await page.waitForFunction(() => window.__voice.spoken.length > 0, null, { timeout: 8000 });
  const spoken = await page.evaluate(() => window.__voice.spoken);
  check("answer is spoken sentence by sentence", spoken.length >= 1 && !spoken.join(" ").includes("[S"), `${spoken.length} utterance(s)`);
  await shot("03-answer");

  // ── 2. inspector ─────────────────────────────────────────────────────
  await page.click(".msg-assistant .cite");
  check("citation opens its source", Boolean(await page.$(".source.focused")));
  await page.click(".tab:has-text('Retrieval')");
  await page.waitForSelector(".funnel");
  await shot("04-retrieval");
  await page.click(".tab:has-text('Latency')");
  await page.waitForSelector(".hero-num");
  const voiceKv = await page.textContent(".latency");
  check("latency tab shows the voice pipeline", /Speculative retrieval/.test(voiceKv) && /first spoken word/i.test(voiceKv));
  const endpointMs = await page.evaluate(() => {
    const dt = [...document.querySelectorAll(".latency dt")].find((d) => /Last word → end of question/.test(d.textContent));
    const m = dt && /([\d.]+)\s*(ms|s)/.exec(dt.nextElementSibling.textContent);
    return m ? Number(m[1]) * (m[2] === "s" ? 1000 : 1) : null;
  });
  check("a finished question ends after a short pause", endpointMs !== null && endpointMs < 600, `${endpointMs} ms`);
  await shot("05-latency");

  // ── 3. typed follow-up uses conversation memory ──────────────────────
  await page.waitForFunction(() => !document.querySelector(".livebar-speaking"), null, { timeout: 20000 });
  await page.fill(".composer textarea", "What about education?");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelectorAll(".msg-assistant .msg-meta").length >= 2, null, { timeout: 20000 });
  await page.click(".tab:has-text('Retrieval')");
  const rewrite = await page.textContent(".q-rewrite").catch(() => "");
  check("follow-up is rewritten with the remembered district", /vijayawada/i.test(rewrite), rewrite.trim());
  await shot("06-followup");

  // ── 4. barge-in: tap the mic while the assistant is talking ──────────
  await page.evaluate(() => { window.__voice.msPerChar = 60; window.__voice.next = "what is planned for chilli farmers in guntur"; });
  await page.click(".mic");
  await page.waitForSelector(".livebar-speaking", { timeout: 25000 });
  const cancelsBefore = await page.evaluate(() => window.__voice.cancels);
  await page.evaluate(() => { window.__voice.next = "who is eligible for the pratibha scholarship"; });
  await page.click(".mic"); // barge in
  await page.waitForSelector(".livebar-listening", { timeout: 3000 });
  const cancelsAfter = await page.evaluate(() => window.__voice.cancels);
  check("barge-in stops speech immediately", cancelsAfter > cancelsBefore);
  await page.waitForFunction(() => document.querySelectorAll(".msg-user").length >= 4, null, { timeout: 10000 });
  await page.waitForFunction(() => document.querySelectorAll(".msg-assistant .msg-meta").length >= 4, null, { timeout: 20000 });
  const lastQ = await page.$$eval(".msg-user .bubble", (els) => els[els.length - 1].textContent);
  const third = await page.$$eval(".msg-assistant .answer", (els) => els[2]?.textContent || "");
  // The text answer had finished streaming; only its speech was cut. It stays complete.
  check("barge-in takes the new question and keeps the earlier answer", /pratibha/i.test(lastQ) && third.length > 20, lastQ.trim());
  await page.evaluate(() => { window.__voice.msPerChar = 12; });
  await shot("07-barge-in");

  // ── 4b. barge-in by speaking, with the answer's echo in the microphone ─
  await page.waitForFunction(() => !document.querySelector(".livebar-speaking"), null, { timeout: 30000 });
  await page.evaluate(() => { window.__voice.msPerChar = 60; window.__voice.next = "what healthcare schemes are there in guntur"; });
  const usersBefore = await page.$$eval(".msg-user", (els) => els.length);
  await page.click(".mic");
  await page.waitForSelector(".livebar-speaking", { timeout: 25000 });
  await page.waitForFunction(() => /start talking to interrupt/.test(document.querySelector(".livebar")?.textContent || ""), null, { timeout: 3000 })
    .catch(() => {});
  check("speaking bar invites talking to interrupt", /start talking to interrupt/.test(await page.textContent(".livebar")));
  // Laptop speakers: the recognizer hears the answer itself. That must not interrupt it.
  const echo = () => window.__voice.spoken.at(-1).replace(/[^\w\s']/g, " ").split(/\s+/).filter(Boolean).slice(0, 12).join(" ");
  const echoed = await page.evaluate((src) => { const e = (0, eval)(src)(); window.__voice.active.emit(e, 60); return e; }, `(${echo})`);
  await page.waitForTimeout(12 * 60 + 1500);
  const stillSpeaking = Boolean(await page.$(".livebar-speaking"));
  check("the answer's own echo does not interrupt it", stillSpeaking && (await page.$$eval(".msg-user", (els) => els.length)) === usersBefore + 1,
    echoed);
  // More echo, then the user talks over it before any pause: only the user's words count.
  const cancels2 = await page.evaluate(() => window.__voice.cancels);
  await page.evaluate((src) => {
    const v = window.__voice;
    v.active.emit((0, eval)(src)(), 50);
    setTimeout(() => v.active.emit("what about education in vijayawada"), 12 * 50 + 250);
  }, `(${echo})`);
  await page.waitForFunction((n) => document.querySelectorAll(".msg-user").length >= n, usersBefore + 2, { timeout: 15000 });
  const spokenQ = (await page.$$eval(".msg-user .bubble", (els) => els[els.length - 1].textContent)).trim();
  check("talking over the answer stops it and asks the new question",
    (await page.evaluate(() => window.__voice.cancels)) > cancels2 && /^What about education in vijayawada\?$/i.test(spokenQ), spokenQ);
  await shot("07b-spoken-barge-in");
  // "Stop" only stops the answer.
  await page.waitForSelector(".livebar-speaking", { timeout: 25000 });
  const usersNow = await page.$$eval(".msg-user", (els) => els.length);
  await page.waitForFunction(() => /start talking/.test(document.querySelector(".livebar")?.textContent || ""), null, { timeout: 3000 }).catch(() => {});
  await page.evaluate(() => window.__voice.active.emit("stop"));
  const stopped = await page.waitForFunction(() => !document.querySelector(".livebar-speaking") && !document.querySelector(".livebar-listening"),
    null, { timeout: 5000 }).then(() => true).catch(() => false);
  check("saying “stop” stops the answer without asking anything", stopped && (await page.$$eval(".msg-user", (els) => els.length)) === usersNow);
  await page.evaluate(() => { window.__voice.msPerChar = 12; });

  // ── 4c. a browser that ends the session at a pause (Edge) ────────────
  await page.waitForFunction(() => !document.querySelector(".livebar-speaking") && !document.querySelector(".livebar-listening"), null, { timeout: 30000 });
  await page.evaluate(() => {
    Object.assign(window.__voice, { edge: true, queue: ["what healthcare initiatives are proposed for", "vijayawada"] });
  });
  await page.click(".mic");
  await page.waitForFunction((n) => document.querySelectorAll(".msg-user").length > n, usersNow, { timeout: 10000 });
  const edgeQ = (await page.$$eval(".msg-user .bubble", (els) => els[els.length - 1].textContent)).trim();
  check("a pause that ends the browser's session doesn't cut the question", /^What healthcare initiatives are proposed for vijayawada\?$/i.test(edgeQ), edgeQ);
  await page.evaluate(() => { window.__voice.edge = false; window.__voice.queue = []; });
  await page.waitForSelector(".msg-assistant:last-child .msg-meta", { timeout: 20000 }).catch(() => {});

  // ── 5. other pages, dark mode, mobile ────────────────────────────────
  await page.click("a[href='#/knowledge']");
  await page.waitForSelector("table.docs");
  check("knowledge base lists documents", (await page.$$("table.docs tbody tr")).length >= 10);
  await shot("08-knowledge");
  await page.click("table.docs tbody tr");
  await page.waitForSelector(".chunk");
  await shot("09-chunks");
  await page.keyboard.press("Escape");
  await page.click("a[href='#/insights']");
  await page.waitForSelector(".health-card");
  await shot("10-insights");
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await page.click("a[href='#/assistant']");
  await shot("11-dark-assistant");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  await shot("12-mobile");
  await page.click(".msg-assistant .cite");
  await page.waitForSelector(".inspector .source.focused");
  await page.waitForTimeout(400); // let the sheet finish sliding in
  await shot("13-mobile-sources");
  await page.click(".inspector-head [aria-label='Close details']");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  check("no horizontal overflow on mobile", !overflow);

  check("no console errors", errors.length === 0, errors.slice(0, 3).join(" | "));
} catch (err) {
  check(`unexpected failure: ${err.message.split("\n")[0]}`, false);
  await shot("99-failure");
} finally {
  await browser.close();
}
const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
