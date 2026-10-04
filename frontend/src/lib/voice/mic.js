// Microphone access: a level meter for the voice orb, and a recorder with simple voice-activity
// detection for browsers without the Web Speech API (the clip is sent to POST /transcribe).

/**
 * processing: echo cancellation, noise suppression and automatic gain, for audio that is
 * recognised from this stream (server recognition). When the browser recognises speech it opens
 * the microphone itself and this stream only drives the level meter: then it is opened raw,
 * because on Windows and macOS Chrome/Edge's automatic gain changes the system microphone level
 * (which the browser's recognizer hears too) and voice processing can turn other audio down.
 */
export async function openMic({ processing = true } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot access a microphone.");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: processing, noiseSuppression: processing, autoGainControl: processing },
  });
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  let smooth = 0;

  return {
    stream,
    /** 0..1, perceptual (dB-scaled) and smoothed. */
    level() {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const db = 20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-8);
      const v = Math.min(1, Math.max(0, (db + 58) / 42));
      smooth = v > smooth ? smooth * 0.4 + v * 0.6 : smooth * 0.85 + v * 0.15;
      return smooth;
    },
    close() {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
    },
  };
}

export function micErrorMessage(err) {
  if (err?.name === "NotAllowedError" || err?.name === "SecurityError")
    return "Microphone access is blocked. Allow it in the browser's site settings, then try again.";
  if (err?.name === "NotFoundError") return "No microphone was found. Check that one is connected.";
  if (err?.name === "NotReadableError") return "The microphone is in use by another app.";
  return err?.message || "Could not open the microphone.";
}

function pickMime() {
  const types = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4", "audio/webm"];
  return types.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || "";
}

/**
 * Record one utterance: starts immediately, ends after `silenceMs` of quiet following speech,
 * after `maxMs`, or when stop() is called. Resolves with a Blob, or null if nobody spoke.
 */
export function recordUtterance(mic, { maxMs = 15000, silenceMs = 1100, noSpeechMs = 7000, threshold = 0.28 } = {}) {
  const mimeType = pickMime();
  const recorder = new MediaRecorder(mic.stream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  let heard = false;
  let loudSince = 0;
  let quietSince = 0;
  let raf = 0;
  const t0 = performance.now();

  const done = new Promise((resolve) => {
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
      cancelAnimationFrame(raf);
      resolve(heard && chunks.length ? new Blob(chunks, { type: recorder.mimeType || "audio/webm" }) : null);
    };
  });

  const stop = () => recorder.state !== "inactive" && recorder.stop();
  const tick = () => {
    const now = performance.now();
    const level = mic.level();
    if (level > threshold) {
      loudSince ||= now;
      quietSince = 0;
      if (now - loudSince > 120) heard = true;
    } else {
      loudSince = 0;
      quietSince ||= now;
    }
    if ((heard && quietSince && now - quietSince > silenceMs) || now - t0 > maxMs || (!heard && now - t0 > noSpeechMs)) {
      stop();
      return;
    }
    raf = requestAnimationFrame(tick);
  };
  recorder.start(250);
  raf = requestAnimationFrame(tick);
  return { done, stop: () => { heard = heard || performance.now() - t0 > 600; stop(); } };
}
