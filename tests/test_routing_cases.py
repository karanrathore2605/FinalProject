import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pytest
from app.loaders.chunker import TextChunker
from app.services.decision_layer import DecisionLayer, RouteDecision
from app.loaders.document_loader import DocumentLoader
from app.services.rag_pipeline import RAGPipeline
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def setup_krishna_agent():
    pdf_path = PROJECT_DIR / "data" / "documents" / "Krishna Resume Offline.pdf"
    doc = DocumentLoader.load_from_path(pdf_path)
    chunks = TextChunker().chunk_document(doc)

    vs = VectorStore()
    vs.build_from_chunks(chunks)

    retriever = HybridRetriever(vs)
    pipeline = RAGPipeline(retriever)
    agent = DecisionLayer(pipeline)
    return agent, doc.source


def test_case_1_general_rag(setup_krishna_agent):
    """TEST 1: 'What is RAG?' should route to LLM, not DOCUMENT_SEARCH."""
    agent, doc_name = setup_krishna_agent
    decision: RouteDecision = agent.decide_route("What is RAG?", has_documents=True, doc_name=doc_name)
    assert decision.route == "DIRECT_LLM"
    assert decision.tool == "llm"


def test_case_2_resume_skills(setup_krishna_agent):
    """TEST 2: 'What programming languages are mentioned in the resume?' should route to DOCUMENT_SEARCH."""
    agent, doc_name = setup_krishna_agent
    decision: RouteDecision = agent.decide_route(
        "What programming languages are mentioned in the resume?",
        has_documents=True,
        doc_name=doc_name,
    )
    assert decision.route == "RAG"
    assert decision.tool == "document_search"


def test_case_3_krishna_projects(setup_krishna_agent):
    """TEST 3: 'According to the uploaded resume, what projects has Krishna worked on?' -> DOCUMENT_SEARCH."""
    agent, doc_name = setup_krishna_agent
    decision: RouteDecision = agent.decide_route(
        "According to the uploaded resume, what projects has Krishna worked on?",
        has_documents=True,
        doc_name=doc_name,
    )
    assert decision.route == "RAG"
    assert decision.tool == "document_search"
    assert decision.is_explicit_doc is True


def test_case_4_calculator(setup_krishna_agent):
    """TEST 4: 'What is 25 * 8?' -> CALCULATOR (200)."""
    agent, _ = setup_krishna_agent
    decision: RouteDecision = agent.decide_route("What is 25 * 8?", has_documents=True)
    assert decision.route == "CALCULATOR"
    assert decision.tool == "calculator"
    assert "25 * 8" in decision.expression

    result = agent.execute("What is 25 * 8?")
    assert result.success is True
    assert result.calculation_result == "200"


def test_case_5_explain_embeddings(setup_krishna_agent):
    """TEST 5: 'Explain embeddings in simple words.' -> LLM."""
    agent, doc_name = setup_krishna_agent
    decision: RouteDecision = agent.decide_route(
        "Explain embeddings in simple words.",
        has_documents=True,
        doc_name=doc_name,
    )
    assert decision.route == "DIRECT_LLM"
    assert decision.tool == "llm"


def test_case_6_explicit_doc_rag(setup_krishna_agent):
    """TEST 6: 'According to the uploaded document, what is RAG?' -> DOCUMENT_SEARCH with explicit doc flag."""
    agent, doc_name = setup_krishna_agent
    decision: RouteDecision = agent.decide_route(
        "According to the uploaded document, what is RAG?",
        has_documents=True,
        doc_name=doc_name,
    )
    assert decision.route == "RAG"
    assert decision.tool == "document_search"
    assert decision.is_explicit_doc is True
