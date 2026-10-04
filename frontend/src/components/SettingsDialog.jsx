import { useEffect, useState } from "react";
import { Monitor, Moon, Sun, Volume2 } from "lucide-react";
import { getApiKey, setApiKey } from "../api.js";
import { Button, Dialog, Field, Select, Toggle } from "./ui.jsx";
import { DEFAULT_SETTINGS, useSettings } from "../lib/context.jsx";
import { speechRecognitionSupported } from "../lib/voice/recognizer.js";
import { SentenceSpeaker, listVoices, pickVoice, ttsSupported } from "../lib/voice/tts.js";

const LANGS = [
  { value: "en-IN", label: "English (India)" },
  { value: "en-US", label: "English (US)" },
  { value: "en-GB", label: "English (UK)" },
];
const preview = new SentenceSpeaker();

export default function SettingsDialog({ open, onClose }) {
  const { settings, update } = useSettings();
  const [voices, setVoices] = useState(listVoices);
  const [key, setKey] = useState(getApiKey);
  useEffect(() => {
    if (!ttsSupported) return undefined;
    const refresh = () => setVoices(listVoices());
    window.speechSynthesis.addEventListener?.("voiceschanged", refresh);
    refresh();
    return () => window.speechSynthesis.removeEventListener?.("voiceschanged", refresh);
  }, []);

  const auto = pickVoice(settings.lang, "");
  const test = () => preview.say("preview", "Hello! I can answer questions about the campaign's plans for your district.",
    { voice: pickVoice(settings.lang, settings.voiceURI), rate: settings.rate, lang: settings.lang });

  return (
    <Dialog open={open} onClose={onClose} title="Settings" description="Stored in this browser only."
            footer={<Button variant="ghost" onClick={() => update({ ...DEFAULT_SETTINGS, inspector: settings.inspector })}>Reset to defaults</Button>}>
      <div className="settings-section">
        <h3>Voice</h3>
        {!speechRecognitionSupported && (
          <p className="callout callout-info small">This browser has no built-in speech recognition; spoken questions are recorded and transcribed on the server when it supports that. Chrome, Edge and Safari stream your words live.</p>
        )}
        <Field label="Recognition language" hint="Indian English recognises local place names best.">
          <Select value={settings.lang} onChange={(v) => update({ lang: v })} options={LANGS} />
        </Field>
        {ttsSupported && (
          <Field label="Assistant voice">
            <div className="row">
              <Select className="grow" value={settings.voiceURI} onChange={(v) => update({ voiceURI: v })}
                      options={[{ value: "", label: `Automatic${auto ? ` (${auto.name})` : ""}` },
                        ...voices.map((v) => ({ value: v.voiceURI, label: `${v.name} · ${v.lang}` }))]} />
              <Button icon={Volume2} onClick={test}>Test</Button>
            </div>
          </Field>
        )}
        <Field label={`Speaking rate · ${settings.rate.toFixed(2)}×`}>
          <input type="range" min="0.8" max="1.4" step="0.05" value={settings.rate} onChange={(e) => update({ rate: Number(e.target.value) })} />
        </Field>
        <Field label={`End-of-question pause · ${settings.endSilenceMs} ms`} hint="Shorter answers faster; longer lets you pause mid-sentence.">
          <input type="range" min="600" max="1600" step="100" value={settings.endSilenceMs} onChange={(e) => update({ endSilenceMs: Number(e.target.value) })} />
        </Field>
        <Toggle checked={settings.autoSpeak} onChange={(v) => update({ autoSpeak: v })} label="Speak answers aloud"
                description="Answers to spoken questions are read out as they are written." />
        <Toggle checked={settings.handsFree} onChange={(v) => update({ handsFree: v })} label="Conversation mode"
                description="Keeps listening after each answer, and you can interrupt by speaking. Headphones work best." />
      </div>

      <div className="settings-section">
        <h3>Appearance</h3>
        <div className="segmented" role="radiogroup" aria-label="Theme">
          {[["system", "System", Monitor], ["light", "Light", Sun], ["dark", "Dark", Moon]].map(([v, label, Icon]) => (
            <button key={v} role="radio" aria-checked={settings.theme === v} className={settings.theme === v ? "active" : ""}
                    onClick={() => update({ theme: v })}><Icon size={14} aria-hidden /> {label}</button>
          ))}
        </div>
      </div>

      <div className="settings-section">
        <h3>Access</h3>
        <Field label="API key" hint="Only needed when the server sets API_KEY (uploads and deletions). Kept for this browser session.">
          <input className="input" type="password" value={key} autoComplete="off" placeholder="Not set"
                 onChange={(e) => { setKey(e.target.value); setApiKey(e.target.value); }} />
        </Field>
      </div>
    </Dialog>
  );
}
