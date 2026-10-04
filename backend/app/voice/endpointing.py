"""Semantic end-of-turn detection ("endpointing") for spoken questions.

A fixed silence timeout has to be long enough for the slowest mid-sentence pause, so it
taxes every question: with 900 ms, a finished question still waits 900 ms before the
assistant may answer. But the transcript usually tells us whether the user is done:

    "what is planned for chilli farmers in"        → clearly unfinished: wait longer
    "what is planned for chilli farmers in guntur" → a complete question: answer soon

This module scores the transcript (no audio, no model, ~0.1 ms) and returns how long a
silence must last before the turn counts as over:

    complete    300 ms   a full question/request that lands on its subject (district, topic,
                         scheme, or a pronoun the conversation resolves), small talk, or
                         "I'm from Vijayawada"
    likely      550 ms   a full clause without such a landing ("... will be filled"), or a
                         landing without a verb ("vizag IT corridor how many jobs")
    unsure      900 ms   the old fixed timeout: nothing points either way
    incomplete 1600 ms   ends on a filler or a word that needs a continuation ("for", "the",
                         "and", "does", "how many")

The browser path uses it as a hint for the client's silence timer; the server speech-to-text
path uses it directly. eval/run_latency_benchmark.py measures the trade-off: the wait after
complete questions vs. how often a mid-sentence pause would end the turn early.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.config import Settings
from app.conversation.state import ConversationState
from app.conversation.understanding import understand
from app.lexicon import DISTRICT_ALIASES, TOPIC_KEYWORDS

Turn = Literal["complete", "likely", "unsure", "incomplete"]

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "hm", "ah", "mm", "like", "so", "well"}
# Words that need a continuation when they end an utterance.
DANGLING = {
    # prepositions
    "for", "in", "on", "at", "to", "of", "about", "with", "from", "by", "into", "onto", "under", "over", "near",
    "regarding", "than", "between", "among", "across", "through", "during", "after", "before", "towards", "toward",
    "per", "via", "including", "against", "within", "without", "around",
    # articles, determiners, possessives, quantifiers that introduce a noun
    "a", "an", "the", "my", "our", "your", "their", "his", "her", "its", "any", "some", "every", "each", "both",
    "either", "neither", "many", "several", "such", "what's", "which",
    # conjunctions
    "and", "or", "but", "because", "if", "whether", "although", "though", "while", "unless", "since", "nor", "as",
    "plus", "then",
    # auxiliaries, modals, copulas
    "is", "are", "was", "were", "be", "been", "being", "am", "will", "would", "can", "could", "should", "shall",
    "may", "might", "must", "do", "does", "did", "has", "have", "had", "isn't", "aren't", "won't", "can't",
    "doesn't", "don't", "didn't",
    # question openers and clause-opening pronouns
    "what", "who", "whom", "whose", "where", "when", "why", "how", "i", "we", "they", "he", "she", "i'm",
    # verbs and adverbs that announce more to come
    "tell", "know", "want", "wanted", "need", "very", "really", "just", "not", "also", "explain", "describe",
}
QUESTION_OPENERS = {"what", "what's", "which", "who", "who's", "whom", "whose", "when", "where", "where's", "why",
                    "how", "how's", "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would",
                    "should", "shall", "has", "have", "any"}
REQUEST_OPENERS = {"tell", "explain", "describe", "list", "give", "show", "compare", "summarise", "summarize",
                   "share", "find", "please"}
AUX_DO = {"do", "does", "did"}
AUX_OTHER = {"is", "are", "was", "were", "am", "will", "would", "can", "could", "should", "shall", "has", "have",
             "had", "may", "might", "must", "what's", "who's", "where's", "how's", "it's", "that's", "there's",
             "isn't", "aren't", "won't", "can't"}
MAIN_VERBS = {
    "propose", "proposes", "proposed", "plan", "plans", "planned", "offer", "offers", "offered", "provide", "provides",
    "provided", "cover", "covers", "covered", "create", "creates", "created", "build", "builds", "built", "include",
    "includes", "get", "gets", "got", "give", "gives", "gave", "say", "says", "said", "mean", "means", "apply",
    "applies", "qualify", "qualifies", "cost", "costs", "help", "helps", "happen", "happens", "benefit", "benefits",
    "promise", "promises", "support", "supports", "fund", "funds", "improve", "improves", "increase", "reduce",
    "fix", "open", "opens", "start", "starts", "won", "win", "wins", "made", "make", "makes", "do", "doing", "done",
    "go", "goes", "went", "come", "comes", "came", "pay", "pays", "paid", "spend", "spends", "spent", "run", "runs",
    "talk", "talks", "need", "needs", "want", "wants", "live", "lives", "work", "works", "stand", "stands", "think",
    "protect", "protects", "monitor", "fill", "filled", "change", "changes", "address", "tackle", "solve", "deal",
}
PARTICLES = {"up", "out", "off", "back", "down", "today", "now", "please", "there's"}
PREPOSITIONS = {"for", "in", "on", "at", "to", "of", "about", "with", "from", "by", "near", "regarding", "across",
                "around", "towards", "toward", "into", "under", "within", "like", "and"}
PRONOUN_ENDS = {"it", "that", "this", "there", "them", "those", "these", "one"}

_ANCHOR_WORDS = {w for aliases in DISTRICT_ALIASES.values() for a in aliases for w in a.split()} | {
    w for kws in TOPIC_KEYWORDS.values() for k in kws for w in k.split() if len(w) > 2
}


@dataclass(frozen=True)
class EndOfTurn:
    turn: Turn
    p: float            # rough probability that the user has finished (for display and tuning)
    wait_ms: int        # silence after the last word before the turn counts as over
    reason: str

    def as_event(self, text: str) -> dict:
        # The tail of the transcript: clients match it against what they have heard since.
        return {"type": "endpoint", "text": text[-300:], "turn": self.turn, "p": self.p, "wait_ms": self.wait_ms,
                "reason": self.reason}


class Endpointer:
    def __init__(self, complete_ms: int = 300, likely_ms: int = 550, unsure_ms: int = 900, incomplete_ms: int = 1600):
        self.waits: dict[Turn, int] = {"complete": complete_ms, "likely": likely_ms, "unsure": unsure_ms,
                                       "incomplete": incomplete_ms}

    def _result(self, turn: Turn, p: float, reason: str) -> EndOfTurn:
        return EndOfTurn(turn, p, self.waits[turn], reason)

    def assess(self, text: str, state: ConversationState | None = None) -> EndOfTurn:
        raw = text.strip()
        words = _WORD.findall(raw.lower().replace("’", "'"))
        if not words:
            return self._result("incomplete", 0.05, "no words yet")
        if words[-1] in FILLERS:
            return self._result("incomplete", 0.1, f"trailing filler '{words[-1]}'")
        u = understand(raw)
        if u.smalltalk:
            return self._result("complete", 0.9, f"small talk ({u.smalltalk})")
        content = [w for w in words if w not in FILLERS]
        last = content[-1]
        if last in DANGLING:
            return self._result("incomplete", 0.1, f"ends on '{last}'")
        if len(content) >= 2 and " ".join(content[-2:]) in ("how many", "how much", "tell me", "is there",
                                                           "are there", "what about", "how about"):
            return self._result("incomplete", 0.1, f"ends on '{' '.join(content[-2:])}'")
        if u.intro_only:
            return self._result("complete", 0.9, "self-introduction")

        # The word that carries the meaning at the end ("coming up" → "coming").
        tail = content[:]
        while len(tail) > 1 and tail[-1] in PARTICLES:
            tail.pop()
        end = tail[-1]
        context = bool(state and (state.district or state.topic or state.entity))
        lands = _lands(tail, raw, u.entity, context)
        verbs = [w for w in content if w in MAIN_VERBS or (len(w) > 4 and w.endswith(("ed", "ing")))]
        has_do = any(w in AUX_DO for w in content)
        has_verb = bool(verbs) or any(w in AUX_OTHER for w in content)
        # "What healthcare initiatives does the candidate ..." – do-support still owes a main verb.
        owes_verb = has_do and not [w for w in verbs if w not in AUX_DO]
        question = content[0] in QUESTION_OPENERS or any(w in QUESTION_OPENERS - {"is", "are", "do", "can"}
                                                         for w in content[1:])
        request = content[0] in REQUEST_OPENERS
        follow_up = u.is_follow_up and context
        punctuated = raw.endswith(("?", "."))

        if owes_verb:
            return self._result("unsure", 0.35, "question still needs its verb")
        if lands and (has_verb or request or follow_up or (punctuated and question)):
            return self._result("complete", 0.85, f"complete {'request' if request else 'question'} ending on '{end}'")
        if (question or request) and has_verb and len(content) >= 4:
            return self._result("likely", 0.65, "full clause")
        if lands and len(content) >= 3:
            return self._result("likely", 0.6, f"ends on '{end}'")
        if punctuated and len(content) >= 3:
            return self._result("likely", 0.6, "punctuated by the recogniser")
        return self._result("unsure", 0.4, "no strong cue")


def _lands(tail: list[str], raw: str, entity: str | None, context: bool) -> bool:
    """Does the utterance end on the subject of the question?

    A place or topic word counts only when it ends the phrase after a preposition
    ("... for chilli farmers", "... in Guntur"): in "what are the education ..." it is still
    modifying a noun to come. A scheme name the understanding found at the very end counts,
    and so does a pronoun the conversation can resolve ("who is eligible for it").
    """
    end = tail[-1]
    if entity and raw.rstrip(" ?.!").endswith(entity):
        return True
    if end in PRONOUN_ENDS:
        return context
    if end not in _ANCHOR_WORDS:
        return False
    return any(w in PREPOSITIONS for w in tail[-5:-1])


def build_endpointer(s: Settings) -> Endpointer | None:
    if not s.voice_adaptive_endpointing:
        return None
    return Endpointer(s.voice_endpoint_complete_ms, s.voice_endpoint_likely_ms, s.voice_endpoint_unsure_ms,
                      s.voice_endpoint_incomplete_ms)


def endpointing_info(s: Settings) -> dict:
    return {"adaptive": s.voice_adaptive_endpointing,
            "wait_ms": {"complete": s.voice_endpoint_complete_ms, "likely": s.voice_endpoint_likely_ms,
                        "unsure": s.voice_endpoint_unsure_ms, "incomplete": s.voice_endpoint_incomplete_ms}}
