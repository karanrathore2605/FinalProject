import pytest
from unittest.mock import patch, MagicMock

from app.config import MAX_HISTORY_MESSAGES
from app.decision_layer import DecisionLayer, AgentResult
from app.memory import ConversationManager, get_conversation_manager, reset_conversation_manager
from app.rag_pipeline import RAGPipeline
from app.retriever import HybridRetriever
from app.vector_store import VectorStore


@pytest.fixture(autouse=True)
def reset_memory():
    """Reset the global conversation manager before and after each test."""
    reset_conversation_manager()
    yield
    reset_conversation_manager()


def create_test_agent():
    """Create a lightweight agent with empty vector store."""
    vs = VectorStore()
    retriever = HybridRetriever(vs)
    pipeline = RAGPipeline(retriever)
    return DecisionLayer(pipeline)


# =============================================================================
# REQUIREMENT 14 TESTS: TEST 1 - TEST 6
# =============================================================================

def test_1_python_followup_and_its_features():
    """
    TEST 1:
    'What is Python?' -> answers about Python.
    Then: 'and its features' -> answers about Python features.
    """
    agent = create_test_agent()
    session_id = "test_session_python"

    # Turn 1: "What is Python?"
    res1 = agent.execute("What is Python?", session_id=session_id)
    assert res1.success is True
    assert res1.route == "Direct LLM"
    assert "python" in res1.answer.lower()

    # Verify history was recorded
    mgr = get_conversation_manager()
    history = mgr.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is Python?"
    assert history[1]["role"] == "assistant"

    # Turn 2: "and its features"
    res2 = agent.execute("and its features", session_id=session_id)
    assert res2.success is True
    assert res2.route == "Direct LLM"
    # Should resolve to Python features and explain Python features
    assert any(term in res2.answer.lower() for term in ["python", "readable", "syntax", "interpreted", "language", "features", "object"])

    # Verify history now has 4 messages
    history_after = mgr.get_history(session_id)
    assert len(history_after) == 4
    assert history_after[2]["content"] == "and its features"


def test_2_fastapi_followup_why_is_it_used():
    """
    TEST 2:
    'What is FastAPI?'
    Then: 'Why is it used?' -> understands 'it' = FastAPI.
    """
    agent = create_test_agent()
    session_id = "test_session_fastapi"

    # Turn 1
    res1 = agent.execute("What is FastAPI?", session_id=session_id)
    assert res1.success is True
    assert "fastapi" in res1.answer.lower() or "api" in res1.answer.lower()

    # Turn 2
    res2 = agent.execute("Why is it used?", session_id=session_id)
    assert res2.success is True
    # The answer or contextualized query should clearly reference FastAPI
    assert "fastapi" in res2.answer.lower() or "api" in res2.answer.lower() or "performance" in res2.answer.lower() or "async" in res2.answer.lower()


def test_3_rag_followup_how_does_it_work():
    """
    TEST 3:
    'What is RAG?'
    Then: 'How does it work?' -> understands the previous topic (RAG).
    """
    agent = create_test_agent()
    session_id = "test_session_rag"

    # Turn 1
    res1 = agent.execute("What is RAG?", session_id=session_id)
    assert res1.success is True

    # Turn 2
    res2 = agent.execute("How does it work?", session_id=session_id)
    assert res2.success is True
    # Should explain retrieval augmented generation mechanism
    assert any(w in res2.answer.lower() for w in ["rag", "retriev", "generat", "document", "vector", "llm", "knowledge", "augment"])


def test_4_no_conversation_history_works_normally():
    """
    TEST 4:
    No conversation history:
    'What is Python?' -> should work normally without errors.
    """
    agent = create_test_agent()

    # Execute with session_id=None (no history, no session)
    res = agent.execute("What is Python?")
    assert res.success is True
    assert res.route == "Direct LLM"
    assert "python" in res.answer.lower()

    # Execute with a fresh session_id that has no prior messages
    fresh_session = "fresh_empty_session"
    res_fresh = agent.execute("What is Python?", session_id=fresh_session)
    assert res_fresh.success is True
    assert "python" in res_fresh.answer.lower()


def test_5_new_chat_does_not_use_old_context():
    """
    TEST 5:
    Previous conversation: 'What is Python?'
    Start new chat.
    'What are its features?' -> should NOT incorrectly use the old Python context.
    """
    agent = create_test_agent()
    session_old = "session_old"
    session_new = "session_new"

    # Turn in old session
    agent.execute("What is Python?", session_id=session_old)

    # In new session (new chat), ask "What are its features?"
    res = agent.execute("What are its features?", session_id=session_new)
    assert res.success is True
    # Without context, it should either ask for clarification or not talk about Python
    # It must not have Python history in its session
    mgr = get_conversation_manager()
    new_history = mgr.get_history(session_new)
    # The session history should only contain the current query and answer, NOT Python
    assert len(new_history) == 2
    assert new_history[0]["content"] == "What are its features?"


def test_6_calculator_routing_with_history():
    """
    TEST 6:
    'What is Python?'
    Then: 'Calculate 25 * 4' -> should still route to the calculator tool.
    """
    agent = create_test_agent()
    session_id = "test_session_calc"

    # Turn 1: Conversational question
    res1 = agent.execute("What is Python?", session_id=session_id)
    assert res1.route == "Direct LLM"

    # Turn 2: Math query
    res2 = agent.execute("Calculate 25 * 4", session_id=session_id)
    assert res2.success is True
    assert res2.route == "Calculator"
    assert res2.tool_used == "calculator"
    assert res2.calculation_result == "100"
    assert "100" in res2.answer


# =============================================================================
# SESSION ISOLATION & MEMORY LIMIT TESTS
# =============================================================================

def test_session_isolation():
    """Verify different sessions do not share history."""
    mgr = ConversationManager(max_history=5)
    mgr.add_turn("session_A", "What is Java?", "Java is a programming language.")
    mgr.add_turn("session_B", "What is Rust?", "Rust is a systems programming language.")

    hist_a = mgr.get_history("session_A")
    hist_b = mgr.get_history("session_B")

    assert len(hist_a) == 2
    assert hist_a[0]["content"] == "What is Java?"
    assert len(hist_b) == 2
    assert hist_b[0]["content"] == "What is Rust?"
    assert hist_a != hist_b


def test_history_limit_truncation():
    """Verify that get_history respects the MAX_HISTORY_MESSAGES limit."""
    mgr = ConversationManager(max_history=4)
    # Add 4 turns (8 messages)
    for i in range(1, 5):
        mgr.add_turn("session_limit", f"Query {i}", f"Answer {i}")

    # Should only return the last 4 messages (Query 3, Answer 3, Query 4, Answer 4)
    history = mgr.get_history("session_limit")
    assert len(history) == 4
    assert history[0]["content"] == "Query 3"
    assert history[1]["content"] == "Answer 3"
    assert history[2]["content"] == "Query 4"
    assert history[3]["content"] == "Answer 4"


def test_clear_session():
    """Verify that clearing a session removes all history for that session."""
    mgr = ConversationManager(max_history=10)
    mgr.add_turn("session_to_clear", "Hello", "Hi there!")
    assert len(mgr.get_history("session_to_clear")) == 2

    assert mgr.clear_session("session_to_clear") is True
    assert len(mgr.get_history("session_to_clear")) == 0
