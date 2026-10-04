"""Conversation memory, kept *structured* rather than as a transcript.

Retrieval needs a few facts (where the citizen is from, what we're talking about,
which scheme "it" refers to), not the whole chat. Storing those as fields keeps
rewriting deterministic and prompts small; a bounded window of recent turns is
kept only for the LLM's conversational context.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from app.conversation.understanding import Understanding

MAX_TURNS = 6  # 3 exchanges; older turns are dropped to bound prompt size and latency


@dataclass
class Turn:
    role: str  # user | assistant
    text: str


@dataclass
class ConversationState:
    session_id: str
    district: str | None = None
    topic: str | None = None
    entity: str | None = None
    last_query: str | None = None      # last *resolved* retrieval query
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    updated_at: float = field(default_factory=time.time)

    def observe_user(self, u: Understanding, resolved_query: str) -> None:
        """Update memory from a user utterance (after it has been resolved)."""
        if u.districts:
            self.district = u.districts[0]
        if u.topic:
            self.topic = u.topic
        if u.entity:
            self.entity = u.entity
        self.last_query = resolved_query
        self.turns.append(Turn("user", u.original))
        self.updated_at = time.time()

    def observe_assistant(self, text: str) -> None:
        self.turns.append(Turn("assistant", text[:400]))
        self.updated_at = time.time()

    def summary(self) -> dict:
        return {"session_id": self.session_id, "district": self.district, "topic": self.topic,
                "entity": self.entity, "turns": len(self.turns)}

    def copy(self) -> ConversationState:
        c = ConversationState(self.session_id, self.district, self.topic, self.entity, self.last_query)
        c.turns = deque(self.turns, maxlen=MAX_TURNS)
        return c


class SessionStore:
    """In-memory sessions with TTL and an LRU cap. Production: Redis with the same interface."""

    def __init__(self, ttl_s: float = 1800, max_sessions: int = 2000):
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._data: OrderedDict[str, ConversationState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> ConversationState:
        now = time.time()
        with self._lock:
            st = self._data.get(session_id)
            if st is None or now - st.updated_at > self.ttl_s:
                st = ConversationState(session_id)
                self._data[session_id] = st
            self._data.move_to_end(session_id)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)
            return st

    def peek(self, session_id: str) -> ConversationState | None:
        with self._lock:
            st = self._data.get(session_id)
            return st.copy() if st and time.time() - st.updated_at <= self.ttl_s else None

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)
