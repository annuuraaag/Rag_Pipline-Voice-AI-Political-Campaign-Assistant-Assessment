import asyncio

from app.conversation.rewriter import LLMRewriter, rewrite_with_rules
from app.conversation.state import ConversationState, SessionStore
from app.conversation.understanding import clean_utterance, understand
from app.generation.llm.base import LLMError


def _state(*history: str) -> ConversationState:
    st = ConversationState("s")
    for h in history:
        rw = rewrite_with_rules(h, st)
        st.observe_user(rw.understanding, rw.query)
    return st


def test_understanding_extracts_district_topic_entity_and_follow_up():
    u = understand("I'm from Bezawada")
    assert u.districts == ["vijayawada"] and u.is_self_intro
    assert understand("What about education?").is_follow_up
    assert understand("What is the Pratibha Scholarship?").entity == "Pratibha Scholarship"
    assert understand("healthcare schemes").topic == "healthcare"


def test_voice_disfluencies_are_removed():
    assert clean_utterance("uh so what are you guys doing for, um, farmers near Guntur") == \
        "What are you doing for, farmers near Guntur"


def test_old_place_names_get_the_canonical_name_added():
    rw = rewrite_with_rules("I live in bezawada, any new hospitals coming up", None)
    assert "bezawada (Vijayawada)" in rw.query and rw.filters.district == "vijayawada"
    assert rewrite_with_rules("Vijayawada hospitals", None).query == "Vijayawada hospitals"


def test_district_from_memory_is_added_and_filtered():
    rw = rewrite_with_rules("What healthcare schemes are available?", _state("I'm from Vijayawada."))
    assert rw.query == "What healthcare schemes are available in Vijayawada?"
    assert rw.filters.district == "vijayawada"
    assert rw.original == "What healthcare schemes are available?"


def test_follow_up_and_pronoun_resolution():
    st = _state("I'm from Vijayawada.", "What healthcare schemes are available?")
    assert rewrite_with_rules("What about education?", st).query == \
        "What are the education plans and schemes in Vijayawada?"
    assert rewrite_with_rules("Who is eligible for it?", _state("What is the Pratibha Scholarship?")).query == \
        "Who is eligible for Pratibha Scholarship?"
    assert rewrite_with_rules("What is planned for farmers there?", _state("Tell me about Guntur.")).query == \
        "What is planned for farmers in Guntur?"


def test_new_district_in_utterance_overrides_memory_and_two_districts_disable_filter():
    st = _state("I'm from Vijayawada.")
    assert rewrite_with_rules("What about hospitals in Guntur?", st).filters.district == "guntur"
    assert rewrite_with_rules("Compare Guntur and Vizag hospitals", st).filters.district is None


def test_without_state_the_query_is_only_cleaned():
    rw = rewrite_with_rules("Kisan Price Guarantee Fund", None)
    assert rw.query == "Kisan Price Guarantee Fund" and rw.filters.is_empty() and not rw.changed


def test_memory_window_is_bounded_and_sessions_expire():
    st = _state(*[f"question {i}" for i in range(10)])
    assert len(st.turns) == 6
    store = SessionStore(ttl_s=0)
    s1 = store.get("a")
    s1.district = "guntur"
    s1.updated_at -= 1
    assert store.get("a").district is None  # expired → fresh state
    assert store.peek("missing") is None


class _LLM:
    name, model, is_fallback = "fake", "m", False

    def __init__(self, reply=None, fail=False, delay=0.0):
        self.reply, self.fail, self.delay = reply, fail, delay

    async def complete(self, messages, max_tokens=None, grounding=None):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise LLMError("down")
        return self.reply


def test_llm_rewriter_falls_back_to_rules_on_error_timeout_or_garbage():
    st = _state("I'm from Vijayawada.", "What healthcare schemes are available?")
    rules = rewrite_with_rules("What about education?", st).query
    for llm in (_LLM(fail=True), _LLM(reply="x" * 500), _LLM(reply="ok", delay=0.5)):
        rw = asyncio.run(LLMRewriter(llm, timeout_s=0.1).rewrite("What about education?", st))
        assert rw.query == rules and rw.method == "rules"
    rw = asyncio.run(LLMRewriter(_LLM(reply="Education plans for Vijayawada")).rewrite("What about education?", st))
    assert rw.query == "Education plans for Vijayawada" and rw.method == "llm"
    assert rw.filters.district == "vijayawada"  # filters always come from the deterministic tier
