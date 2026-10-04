// Server-side speech in the browser: stream the microphone to the server as 16 kHz PCM16, and
// play the server's spoken answer as it arrives (PCM16 chunks, scheduled back to back).
// no-inline: a data: URL would be blocked by the Content-Security-Policy (script-src 'self').
import workletUrl from "./pcm-worklet.js?url&no-inline";

export const pcmCaptureSupported = typeof window !== "undefined" && "AudioWorkletNode" in window;

/** Streams `stream` (a getUserMedia stream) to onFrame(ArrayBuffer) in 40 ms frames. */
export async function startPcmCapture(stream, onFrame) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  await ctx.audioWorklet.addModule(workletUrl);
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (e) => onFrame(e.data);
  source.connect(node);
  // A worklet only runs while connected towards the output; a muted gain keeps it inaudible.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  node.connect(mute).connect(ctx.destination);
  if (ctx.state === "suspended") await ctx.resume().catch(() => {});
  return {
    stop() {
      node.port.onmessage = null;
      source.disconnect();
      node.disconnect();
      ctx.close().catch(() => {});
    },
  };
}

/** Binary frame from /ws/voice: uint32 LE header length, JSON header, PCM16 mono. */
export function parseAudioFrame(buf) {
  const n = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { header, pcm: new Int16Array(buf, 4 + n, (buf.byteLength - 4 - n) >> 1) };
}

/** Plays one answer turn at a time. Events: onStart(turn) at the first sound, onIdle(turn) when done. */
export class PcmPlayer {
  constructor({ onStart, onIdle } = {}) {
    this.onStart = onStart;
    this.onIdle = onIdle;
    this.ctx = null;
    this.turn = null;
    this.sources = new Set();
    this.endAt = 0;
    this.started = false;
    this.ended = false;
    this.recent = []; // sentences played, for echo detection during barge-in
    this.timers = new Set();
  }

  /** Must run inside a user gesture once: browsers only start audio after one. */
  unlock() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    this.ctx ||= new Ctx();
    if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
  }

  begin(turn) {
    this.cancel();
    this.unlock();
    this.turn = turn;
    this.started = false;
    this.ended = false;
  }

  push(turn, header, pcm) {
    if (turn !== this.turn || !this.ctx || !pcm.length) return;
    const data = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
    const buf = this.ctx.createBuffer(1, data.length, header.rate);
    buf.copyToChannel(data, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const at = Math.max(this.ctx.currentTime + 0.03, this.endAt);
    src.start(at);
    this.endAt = at + buf.duration;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      this._maybeIdle();
    };
    const delay = Math.max(0, (at - this.ctx.currentTime) * 1000);
    if (header.text) this._later(delay, () => turn === this.turn && (this.recent = [...this.recent.slice(-2), header.text]));
    if (!this.started) {
      this.started = true;
      this._later(delay, () => turn === this.turn && this.onStart?.(turn));
    }
  }

  /** The server has sent all audio of `turn`: report idle once it has played. */
  end(turn) {
    if (turn !== this.turn) return;
    this.ended = true;
    this._maybeIdle();
  }

  cancel() {
    for (const src of this.sources) {
      src.onended = null;
      try { src.stop(); } catch { /* not started */ }
    }
    this.sources.clear();
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.endAt = 0;
    this.turn = null;
  }

  get speaking() {
    return this.sources.size > 0;
  }

  recentText() {
    return this.recent.join(" ");
  }

  _later(ms, fn) {
    const t = setTimeout(() => { this.timers.delete(t); fn(); }, ms);
    this.timers.add(t);
  }

  _maybeIdle() {
    if (!this.ended || this.sources.size || !this.turn) return;
    const turn = this.turn;
    this.turn = null;
    this.onIdle?.(turn);
  }
}
