"""Fix recogniser near-misses of names a general English model can't know.

Speech models write "Vijayawada" as "Vid, awada" or "Vijawada", "Guntur" as "Gunter",
"Visakhapatnam" as "Visika Putnam". District filters, retrieval and the end-of-turn rules all key
on these names, so one wrong name costs more than one wrong word.

Vocabulary: the district gazetteer, plus the campaign's own proper names read from its documents:
words capitalised mid-sentence that never appear in lower case ("Pratibha", "Udayam", "Poshana"),
and the multi-word names they belong to ("Pratibha Scholarship").

A span of 1-4 words is replaced by a name only if
  * it contains a word the campaign's documents never use (a misheard name rarely is a real word
    of the corpus, while "office" or "last" always are), and it neither starts nor ends with such
    a known word unless the name does ("the Pratiba scholarship" → "the Pratibha Scholarship"), and
  * its rough phonetic spelling (w=v, z=j, dg=j, ph=f, c=k; h after a consonant and doubled letters
    dropped) is close to the name's: Levenshtein similarity ≥ 0.72, and
  * the name has no more words than the span (a correction never adds words).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from app.domain import MetadataFilter
from app.lexicon import DISTRICT_ALIASES, DISTRICT_DISPLAY

_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")
_CAP_RUN = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of\s+|and\s+)?[A-Z][a-z]+)*")
_LEADING = {"the", "in", "for", "of", "and", "our", "a", "an", "about", "this", "other"}
THRESHOLD = 0.72


def phonetic(text: str) -> str:
    s = re.sub(r"[^a-z]", "", text.lower())
    for a, b in (("ph", "f"), ("dg", "j"), ("dj", "j"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("c", "k"), ("w", "v"),
                 ("z", "j")):
        s = s.replace(a, b)
    s = re.sub(r"(?<=[^aeiou])h", "", s)   # aspirates and digraphs: dh→d, bh→b, kh→k, sh→s
    return re.sub(r"(.)\1+", r"\1", s)     # doubled letters


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def district_terms() -> dict[str, str]:
    """Spelling → name to write. A misspelling alias ("vijaywada") writes the district's name; an
    alternative name ("bezawada", "vizag") is kept as said (the query rewriter maps it)."""
    out: dict[str, str] = {}
    for slug, aliases in DISTRICT_ALIASES.items():
        name = DISTRICT_DISPLAY[slug]
        out[name] = name
        for alias in aliases:
            if " " in alias:
                continue
            out[alias.title()] = name if similarity(phonetic(alias), phonetic(name)) >= THRESHOLD else alias.title()
    return out


def corpus_words(texts: Iterable[str]) -> set[str]:
    return {w.lower().removesuffix("'s") for t in texts for w in _TOKEN.findall(t)}


def corpus_names(texts: Iterable[str], titles: Iterable[str] = ()) -> list[str]:
    """Proper names of a corpus: words capitalised mid-sentence and never written in lower case,
    and the multi-word names (also from titles) that contain one ("Kisan Price Guarantee Fund")."""
    texts, titles = list(texts), list(titles)
    lower = {w for t in texts + titles for w in re.findall(r"\b[a-z][a-z']+\b", t)}
    capital = {w for t in texts for w in re.findall(r"(?<=[a-z,;:] )[A-Z][a-z]{2,}\b", t)}
    proper = {w for w in capital if w.lower() not in lower}
    names = set(proper)
    for t in texts + titles:
        for run in _CAP_RUN.findall(t):
            words = run.split()
            while words and words[0].lower() in _LEADING:
                words = words[1:]
            if 1 < len(words) <= 4 and any(w in proper for w in words):  # spans are at most 4 words
                names.add(" ".join(words))
    return sorted(names)


class VocabularyCorrector:
    def __init__(self, names: Iterable[str] | dict[str, str], known_words: Iterable[str] = ()):
        mapping = names if isinstance(names, dict) else {n: n for n in names}
        self.names = sorted(set(mapping.values()), key=lambda n: (-len(n.split()), n))
        self.terms: list[tuple[str, str, int, str, str]] = []   # (phonetic key, name, words, first, last word)
        seen = set()
        for spelling, name in mapping.items():
            key = phonetic(spelling)
            if len(key) >= 4 and key not in seen:
                seen.add(key)
                words = name.lower().split()
                self.terms.append((key, name, len(spelling.split()), words[0], words[-1]))
        self.known = {w.lower() for w in known_words}
        self.correct = lru_cache(maxsize=1024)(self._correct)

    def __len__(self) -> int:
        return len(self.terms)

    def _best(self, span: str, n: int, first_known: str | None, last_known: str | None) -> tuple[float, str] | None:
        key = phonetic(span)
        if len(key) < 4:
            return None
        best = None
        for tkey, name, words, first, last in self.terms:
            # Cheap filters first: misheard names nearly always keep their first sound.
            if words > n or key[0] != tkey[0] or not 0.75 <= len(key) / len(tkey) <= 1.33:
                continue
            if (first_known and first_known != first) or (last_known and last_known != last):
                continue
            sim = similarity(key, tkey)
            if sim >= THRESHOLD and (best is None or sim > best[0]):
                best = (sim, name)
        return best

    def _correct(self, text: str) -> str:
        if not self.terms:
            return text
        tokens = list(_TOKEN.finditer(text))
        low = [t.group(0).lower().removesuffix("'s") for t in tokens]
        unknown = [w not in self.known for w in low]
        out, pos, i = [], 0, 0
        while i < len(tokens):
            hit = None
            for n in (4, 3, 2, 1):
                if i + n > len(tokens) or not any(unknown[i:i + n]):
                    continue
                start, end = tokens[i].start(), tokens[i + n - 1].end()
                first = None if unknown[i] else low[i]
                last = None if unknown[i + n - 1] else low[i + n - 1]
                found = self._best(text[start:end], n, first, last)
                if found and (hit is None or found[0] > hit[0]):
                    hit = (found[0], found[1], n, start, end)
            if hit is None:
                i += 1
                continue
            _, name, n, start, end = hit
            if text[start:end].lower() != name.lower():   # casing alone is not worth changing
                out.append(text[pos:start])
                out.append(name)
                pos = end
            i += n
        out.append(text[pos:])
        return "".join(out)


def build_corrector(chunks: Iterable) -> VocabularyCorrector:
    """From a campaign's chunks (app.domain.Chunk): its names, and every word it uses."""
    chunks = list(chunks)
    texts = [c.text for c in chunks]
    titles = [c.metadata.section for c in chunks] + [c.metadata.document_title for c in chunks]
    names = {**{n: n for n in corpus_names(texts, titles)}, **district_terms()}
    return VocabularyCorrector(names, known_words=corpus_words(texts + titles) - {k.lower() for k in names})


class VocabularyCache:
    """One corrector per campaign, rebuilt when its documents change (corpus revision)."""

    def __init__(self, store, revision):
        self.store = store
        self.revision = revision
        self._cache: dict[str, tuple[int, VocabularyCorrector]] = {}

    def get(self, campaign: str) -> VocabularyCorrector:
        rev = self.revision()
        hit = self._cache.get(campaign)
        if hit is None or hit[0] != rev:
            hit = (rev, build_corrector(self.store.iter_chunks(MetadataFilter(campaign_id=campaign))))
            self._cache[campaign] = hit
        return hit[1]
