import pytest
from fastapi.testclient import TestClient
from app.api import app, get_shared_pipeline
from app.retrieval.vector_store import VectorStore

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
    assert "all-MiniLM-L6-v2" in data["embedding_model"]


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
    assert "all-MiniLM-L6-v2" in data["embedding"]["requested_model"]
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
