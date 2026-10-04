// Conversation thread state: user turns and streamed assistant turns.

export function threadReducer(state, a) {
  switch (a.type) {
    case "reset":
      return [];
    case "user":
      return [...state, { id: a.id, role: "user", text: a.text, via: a.via, at: Date.now() }];
    case "assistant":
      return [...state, { id: a.turnId, role: "assistant", status: "retrieving", text: "", voice: { spoken: a.voice }, at: Date.now() }];
    default:
      return state.map((m) => (m.id === a.turnId ? update(m, a) : m));
  }
}

function update(m, a) {
  switch (a.type) {
    case "retrieval":
      return { ...m, status: "streaming", trace: a.retrieval, latency: a.latency, voice: { ...m.voice, cache: a.cache } };
    case "token":
      return { ...m, status: "streaming", text: m.text + a.text };
    case "done": {
      const p = a.payload;
      return {
        ...m, status: "done", text: p.answer, citations: p.citations, answerable: p.answerable,
        refusal_reason: p.refusal_reason, llm: p.llm, latency: p.latency_ms, conversation: p.conversation,
        verification: p.verification,
        voice: { ...m.voice, ...(p.voice || {}), cache: p.cache ?? m.voice?.cache },
      };
    }
    case "error":
      return { ...m, status: "error", error: a.message };
    case "interrupted":
      return m.status === "done" || m.status === "error" ? m : { ...m, status: "interrupted" };
    case "voice_metric":
      return { ...m, voice: { ...m.voice, ...a.patch } };
    default:
      return m;
  }
}

/** [S#] → source, during streaming (from the retrieval trace) and after (from citations). */
export function sourceFor(message, n) {
  const c = message.citations?.find((x) => x.source_id === `S${n}`);
  if (c) return c;
  const r = message.trace?.results?.[n - 1];
  return r ? { ...r, source_id: `S${n}`, snippet: r.text?.slice(0, 220) } : null;
}
