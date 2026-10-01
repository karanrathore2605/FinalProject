import pytest
from fastapi.testclient import TestClient
from app.api import app, get_shared_pipeline
from app.vector_store import VectorStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_shared_store():
    """Ensure clean vector store before and after tests."""
    vs, _, _, _ = get_shared_pipeline()
    vs.clear()
    yield
    vs.clear()


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "AI Document Assistant API is running" in data["message"]
    assert "docs_url" in data
    assert any(m in data["embedding_model"] for m in ["MiniLM", "all-", "sentence", "Qwen"])


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data
    assert "llm_model" in data


def test_models_endpoint():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert any(m in data["embedding"]["requested_model"] for m in ["MiniLM", "all-", "sentence", "Qwen"])
    assert data["chunking"]["strategy"] == "RecursiveCharacterTextSplitter"


def test_recursive_splitter_endpoint():
    text = (
        "Paragraph 1: Introduction to AI systems and embeddings.\n\n"
        "Paragraph 2: Second section detailing vector retrieval.\n\n"
        "Paragraph 3: Third section explaining mathematical decision layers."
    )
    resp = client.post("/api/split", json={"text": text, "chunk_size": 80, "chunk_overlap": 20})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] >= 3
    assert all(isinstance(c, str) and len(c) > 0 for c in data["chunks"])


def test_calculator_endpoint():
    resp = client.post("/api/calculator", json={"expression": "25 * 8"})
    assert resp.status_code == 200
    assert resp.json()["result"] == "200"

    resp_pct = client.post("/api/calculator", json={"expression": "500 * 15%"})
    assert resp_pct.status_code == 200
    assert resp_pct.json()["result"] == "75"


def test_index_text_and_search_endpoint():
    doc_content = (
        "Acme Global Inc. employs 500 people worldwide.\n\n"
        "Exactly 15% of employees work remotely from home.\n\n"
        "The engineering team consists of 120 senior developers."
    )
    # 1. Index document
    index_resp = client.post(
        "/api/documents/index-text",
        json={"title": "acme_policy.txt", "content": doc_content},
    )
    assert index_resp.status_code == 200
    assert index_resp.json()["success"] is True
    assert index_resp.json()["total_chunks"] > 0

    # 2. Check document status
    status_resp = client.get("/api/documents/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_empty"] is False
    assert status_resp.json()["total_chunks"] > 0

    # 3. Hybrid search
    search_resp = client.post(
        "/api/search",
        json={"query": "How many employees work remotely?", "top_k": 2},
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["count"] > 0
    assert any("remotely" in r["text"] or "500" in r["text"] for r in search_data["results"])


def test_chat_calculator_route():
    resp = client.post("/api/chat", json={"query": "25 * 8"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["route"] == "Calculator"
    assert data["calculation_result"] == "200"


def test_chat_conversation_history_api():
    """Test conversation history, follow-up, and clear endpoints over API."""
    sess_id = "test_api_conv_sess"

    # Turn 1
    resp1 = client.post("/api/chat", json={"query": "What is Python?", "session_id": sess_id})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["success"] is True
    assert data1["session_id"] == sess_id
    assert "python" in data1["answer"].lower()

    # Check history endpoint
    hist_resp1 = client.get(f"/api/chat/history/{sess_id}")
    assert hist_resp1.status_code == 200
    hist_data1 = hist_resp1.json()
    assert hist_data1["count"] == 2
    assert hist_data1["messages"][0]["content"] == "What is Python?"

    # Turn 2: Follow-up question
    resp2 = client.post("/api/chat", json={"query": "and its features", "session_id": sess_id})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["success"] is True

    # Check updated history
    hist_resp2 = client.get(f"/api/chat/history/{sess_id}")
    assert hist_resp2.status_code == 200
    assert hist_resp2.json()["count"] == 4

    # Clear history via POST
    clear_resp = client.post("/api/chat/clear", json={"session_id": sess_id})
    assert clear_resp.status_code == 200
    assert clear_resp.json()["success"] is True

    # Verify history is now 0
    hist_resp3 = client.get(f"/api/chat/history/{sess_id}")
    assert hist_resp3.status_code == 200
    assert hist_resp3.json()["count"] == 0


def test_chat_session_isolation_api():
    """Test that two different session IDs maintain isolated conversation histories via API."""
    sess_a = "sess_user_alpha"
    sess_b = "sess_user_beta"

    client.post("/api/chat", json={"query": "What is Python?", "session_id": sess_a})
    client.post("/api/chat", json={"query": "What is Java?", "session_id": sess_b})

    hist_a = client.get(f"/api/chat/history/{sess_a}").json()
    hist_b = client.get(f"/api/chat/history/{sess_b}").json()

    assert hist_a["count"] == 2
    assert hist_b["count"] == 2
    assert "Python" in hist_a["messages"][0]["content"]
    assert "Java" in hist_b["messages"][0]["content"]
