// AudioWorklet (runs on the audio thread): microphone audio → 16 kHz 16-bit PCM in 40 ms frames,
// posted to the main thread, which sends them to the server's speech recogniser.
// `sampleRate` is the AudioContext's rate (a global in worklet scope); audio is resampled here
// rather than by asking for a 16 kHz context, which not every browser can connect a mic to.

const OUT_RATE = 16000;
const FRAME = 640; // 40 ms

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / OUT_RATE;
    this.pos = 0; // read position (in input samples) relative to the current block
    this.prev = 0; // last sample of the previous block, for interpolation across blocks
    this.frame = new Int16Array(FRAME);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input || !input.length) return true;
    let i = this.pos;
    while (i < input.length - 1) {
      const k = Math.floor(i);
      const f = i - k;
      const a = k < 0 ? this.prev : input[k];
      const s = a + (input[k + 1] - a) * f;
      this.frame[this.n++] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
      if (this.n === FRAME) {
        this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
        this.frame = new Int16Array(FRAME);
        this.n = 0;
      }
      i += this.step;
    }
    this.pos = i - input.length;
    this.prev = input[input.length - 1];
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
