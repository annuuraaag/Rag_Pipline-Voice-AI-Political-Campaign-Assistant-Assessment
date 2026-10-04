export const cx = (...parts) => parts.filter(Boolean).join(" ");

export function ms(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (v >= 10000) return `${(v / 1000).toFixed(1)} s`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ms`;
}

export function bytes(n) {
  if (!n && n !== 0) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function timeAgo(iso) {
  const t = new Date(iso).getTime();
  if (!t) return "";
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export const titleCase = (s) => (s || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export function pages(c) {
  if (!c?.page) return "";
  return c.page_end && c.page_end !== c.page ? `pp. ${c.page}–${c.page_end}` : `p. ${c.page}`;
}

export const uid = () =>
  (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`);
