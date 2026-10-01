import pytest
from app.calculator_tool import calculate, calculator_tool
from app.chunker import TextChunker
from app.decision_layer import DecisionLayer
from app.document_loader import Document, DocumentLoader
from app.rag_pipeline import RAGPipeline
from app.retriever import HybridRetriever
from app.vector_store import VectorStore


def test_calculator_basic_operations():
    assert calculate("25 * 8") == 200
    assert calculate("100 / 4") == 25
    assert calculate("500 * 0.15") == 75
    assert calculate("500 * 15%") == 75
    assert calculate("(20 + 30) * 2") == 100
    assert calculator_tool("25 * 8") == "200"


def test_calculator_safety():
    with pytest.raises(ValueError):
        calculate("__import__('os').system('ls')")

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calculate("10 / 0")

    with pytest.raises(ValueError):
        calculate("open('/etc/passwd').read()")


def test_document_loader_txt(tmp_path):
    txt_file = tmp_path / "test_doc.txt"
    txt_file.write_text("Acme Corporation has 500 employees. 15% work remotely.", encoding="utf-8")

    doc = DocumentLoader.load_from_path(txt_file)
    assert doc.source == "test_doc.txt"
    assert "500 employees" in doc.content


def test_chunking_and_retrieval(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "Company Remote Work Policy.\n\n"
        "Section 1: The company has 500 employees worldwide.\n\n"
        "Section 2: Exactly 15% of the employees work remotely full time.\n\n"
        "Section 3: Product engineering has 250 employees with 20% senior developers.",
        encoding="utf-8",
    )

    doc = DocumentLoader.load_from_path(txt_file)
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 0

    vector_store = VectorStore()
    vector_store.build_from_chunks(chunks)
    assert not vector_store.is_empty()

    retriever = HybridRetriever(vector_store, top_k=2)
    results = retriever.retrieve("How many employees work remotely?")
    assert len(results) > 0
    assert any("500 employees" in r.text or "remotely" in r.text for r in results)


def test_meta_query_retrieval(tmp_path):
    txt_file = tmp_path / "summary_test.txt"
    txt_file.write_text(
        "Executive Overview of AI Platform.\n\n"
        "The AI Document Assistant provides hybrid search and tool calling.\n\n"
        "It supports PDF, TXT, and DOCX documents with automated citations.",
        encoding="utf-8",
    )
    doc = DocumentLoader.load_from_path(txt_file)
    chunks = TextChunker().chunk_document(doc)
    vs = VectorStore()
    vs.build_from_chunks(chunks)
    retriever = HybridRetriever(vs)

    results = retriever.retrieve("Explain this in simple words")
    assert len(results) > 0
    assert "Executive Overview" in results[0].text or "AI Document Assistant" in results[0].text


def test_decision_layer_fast_path():
    vs = VectorStore()
    retriever = HybridRetriever(vs)
    pipeline = RAGPipeline(retriever)
    agent = DecisionLayer(pipeline)

    # Pure arithmetic
    route, expr = agent.decide_route("25 * 8", has_documents=False)
    assert route == "CALCULATOR"
    assert "25 * 8" in expr

    result = agent.execute("25 * 8")
    assert result.success is True
    assert result.route == "Calculator"
    assert result.calculation_result == "200"
