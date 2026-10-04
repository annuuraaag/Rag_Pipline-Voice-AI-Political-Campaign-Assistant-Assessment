// Telling the user's voice from the assistant's own voice coming back through the microphone.
//
// While an answer is spoken the recognizer keeps listening, so the user can interrupt by talking.
// Through laptop speakers it also hears the answer itself ("echo"). The recognizer's transcript
// then holds echo first and, once the user cuts in, the user's words after it. Only the words
// after the last stretch of echo are judged, so echo heard earlier never hides an interruption.

export const words = (t) => t.toLowerCase().match(/[a-z0-9]+/g) || [];

const pairsOf = (w) => new Set(w.slice(1).map((x, i) => `${w[i]} ${x}`));

const STOP = new Set(["stop", "wait", "cancel", "enough", "quiet", "pause"]);
const FILLER = new Set(["ok", "okay", "please", "thanks", "thank", "you", "hold", "on", "just", "hey", "no", "sorry",
  "that", "s", "it", "shut", "up", "now"]);

/** "stop", "okay stop", "wait please": stop talking, but this is not a question to answer. */
export function isStopCommand(text) {
  const w = words(text);
  return w.length > 0 && w.length <= 4 && w.some((x) => STOP.has(x)) && w.every((x) => STOP.has(x) || FILLER.has(x));
}

/**
 * Has the user started talking over the answer? `heard` is everything the recognizer has heard
 * since it started listening during the answer; `spoken` is what the assistant has said recently.
 *
 * Echo shows up as word pairs of the answer in the same order. The user's words are the tail
 * after the last such pair. It counts as an interruption when that tail is 3+ words with at
 * least 2 the answer doesn't use (one or two misheard echo words are common), or 5+ words, or
 * a stop word. Returns { skip: echo words to drop from the start, text: what the user said } or null.
 */
export function interruption(heard, spoken) {
  const w = words(heard);
  if (!w.length) return null;
  const said = words(spoken);
  const pairs = pairsOf(said);
  const bag = new Set(said);
  let last = -1;
  for (let i = 1; i < w.length; i++) if (pairs.has(`${w[i - 1]} ${w[i]}`)) last = i;
  const tail = w.slice(last + 1);
  const novel = tail.filter((x) => !bag.has(x)).length;
  const stop = tail.some((x) => STOP.has(x) && !bag.has(x));
  if ((tail.length >= 3 && novel >= 2) || tail.length >= 5 || stop) {
    return { skip: w.length - tail.length, text: tail.join(" ") };
  }
  return null;
}

/** `text` without its first `n` words (as counted by words()), keeping the rest as written. */
export function dropWords(text, n) {
  if (!n) return text;
  const tokens = text.trim().split(/\s+/);
  let seen = 0;
  let i = 0;
  while (i < tokens.length && seen < n) seen += words(tokens[i++]).length;
  return tokens.slice(i).join(" ");
}
