// End-to-end test of server-side speech in a real browser: the microphone streams PCM over the
// WebSocket, the server recognises it, decides when the question is over, answers, and streams
// its spoken answer back; a second question spoken over the answer must interrupt it (barge-in).
//
// Chromium's fake microphone plays a WAV file instead of a real one, so the speech is real audio
// end to end. Make the file with the server's own voice, then run against a stack started with
// STT_PROVIDER=local TTS_PROVIDER=local:
//
//   python scripts/make_speech_wav.py --out /tmp/questions.wav \
//     "What healthcare initiatives are proposed for Vijayawada?" "Who is eligible for the Pratibha Scholarship?"
//   MIC_WAV=/tmp/questions.wav BASE_URL=http://localhost:5173 node e2e/server-speech.e2e.mjs [--shots ./shots]
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL || "http://localhost:5173";
const WAV = process.env.MIC_WAV;
if (!WAV) {
  console.error("Set MIC_WAV to a WAV file with two spoken questions (see the header of this file).");
  process.exit(2);
}
const shotsArg = process.argv.indexOf("--shots");
const SHOTS = shotsArg > 0 ? process.argv[shotsArg + 1] : null;
if (SHOTS) mkdirSync(SHOTS, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${WAV}%noloop`, "--autoplay-policy=no-user-gesture-required"],
});
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, permissions: ["microphone"] });
// Server recognition and the server voice, default settings: talking over an answer interrupts it.
await context.addInitScript(() => {
  localStorage.setItem("crag-settings", JSON.stringify({ speechInput: "server", speechOutput: "server" }));
  window.__audio = { frames: 0, stopped: 0 };
  const stop = AudioBufferSourceNode.prototype.stop;
  AudioBufferSourceNode.prototype.stop = function (...args) {
    window.__audio.stopped += 1; // the player stops queued answer audio on barge-in
    return stop.apply(this, args);
  };
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function (data) {
    if (typeof data !== "string") window.__audio.frames += 1;
    return send.call(this, data);
  };
});
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && !/favicon|ERR_CONNECTION/.test(m.text()) && errors.push(m.text()));
const shot = async (name) => SHOTS && page.screenshot({ path: `${SHOTS}/${name}.png` });

try {
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector(".welcome");
  const speech = await page.evaluate(() => fetch("/api/health").then((r) => r.json()).then((h) => h.components.voice.server_speech));
  check("server has speech providers", /^local\//.test(speech.stt) && /^local\//.test(speech.tts), JSON.stringify(speech));

  await page.click(".mic");
  await page.waitForSelector(".livebar-listening", { timeout: 5000 });
  check("mic opens and streams audio", await page.waitForFunction(() => window.__audio.frames > 10, null, { timeout: 5000 })
    .then(() => true).catch(() => false));
  const interim = await page.waitForFunction(() => {
    const t = document.querySelector(".livebar-text")?.textContent || "";
    return /health|vijay/i.test(t) && t;
  }, null, { timeout: 15000 }).then((h) => h.jsonValue()).catch(() => "");
  check("server transcript appears while speaking", Boolean(interim), interim);
  await shot("01-listening");

  await page.waitForSelector(".msg-user .bubble", { timeout: 15000 });
  const q1 = (await page.textContent(".msg-user .bubble")).trim();
  check("server decides the question is over", /health/i.test(q1) && /[.?]$/.test(q1), q1);
  check("misheard place name is corrected", /vijayawada/i.test(q1), q1);
  await page.waitForSelector(".msg-assistant .answer", { timeout: 20000 });
  const spoke = await page.waitForSelector(".livebar-speaking", { timeout: 20000 }).then(() => true).catch(() => false);
  check("answer is spoken with the server voice", spoke);
  await shot("02-speaking");

  // The WAV's second question starts while the first answer is still playing.
  const barged = await page.waitForFunction(() => document.querySelectorAll(".msg-user").length >= 2, null, { timeout: 30000 })
    .then(() => true).catch(() => false);
  const q2 = barged ? (await page.$$eval(".msg-user .bubble", (els) => els[1].textContent)).trim() : "";
  const stopped = await page.evaluate(() => window.__audio.stopped);
  check("speaking over the answer stops its audio", barged && stopped > 0, `${stopped} queued chunk(s) stopped`);
  check("the interruption becomes the next question", /scholarship|eligible/i.test(q2), q2);
  await page.waitForFunction(() => document.querySelectorAll(".msg-assistant .msg-meta").length >= 2, null, { timeout: 30000 });
  await page.waitForSelector(".livebar-speaking", { timeout: 20000 }).catch(() => {});
  await shot("03-barge-in");

  await page.click(".tab:has-text('Latency')").catch(() => {});
  const latency = await page.waitForFunction(() => {
    const t = document.querySelector(".latency")?.textContent || "";
    return /Last word → end of question/.test(t) && /first spoken word/i.test(t) && t;
  }, null, { timeout: 15000 }).then((h) => h.jsonValue()).catch(() => "");
  check("latency tab shows the server voice pipeline", Boolean(latency),
    (await page.textContent(".hero-num").catch(() => "")).trim());
  await shot("04-latency");
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
