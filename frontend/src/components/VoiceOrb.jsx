import { useEffect, useRef } from "react";

/**
 * Animated voice orb. The microphone level drives a CSS variable from requestAnimationFrame
 * (no React re-render per frame); each phase has its own motion:
 * idle = slow breathing, listening = follows your voice, thinking = rotating sheen,
 * speaking = soft rhythmic pulse.
 */
export default function VoiceOrb({ phase = "idle", getLevel, size = 160 }) {
  const ref = useRef(null);
  useEffect(() => {
    let raf = 0;
    let t = 0;
    const el = ref.current;
    const tick = () => {
      t += 1;
      let lvl = 0;
      if (phase === "listening") lvl = getLevel?.() ?? 0;
      else if (phase === "speaking") lvl = 0.35 + 0.25 * Math.sin(t / 7) * Math.sin(t / 13);
      el?.style.setProperty("--lvl", lvl.toFixed(3));
      raf = requestAnimationFrame(tick);
    };
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!reduced) raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase, getLevel]);

  return (
    <div className="orb" data-phase={phase} ref={ref} style={{ "--size": `${size}px` }} aria-hidden>
      <div className="orb-glow" />
      <div className="orb-ring" />
      <div className="orb-core">
        <div className="orb-swirl" />
        <div className="orb-shine" />
      </div>
    </div>
  );
}

/** Compact live waveform for the listening bar. */
export function Waveform({ getLevel, active, bars = 28 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!active) return undefined;
    const el = ref.current;
    const history = new Array(bars).fill(0);
    let raf = 0;
    let frame = 0;
    const tick = () => {
      frame += 1;
      if (frame % 2 === 0) {
        history.shift();
        history.push(getLevel?.() ?? 0);
        el?.childNodes.forEach((n, i) => {
          n.style.transform = `scaleY(${Math.max(0.12, history[i]).toFixed(3)})`;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, getLevel, bars]);
  return (
    <div className="wave" ref={ref} aria-hidden>
      {Array.from({ length: bars }, (_, i) => <span key={i} />)}
    </div>
  );
}
