import unittest
from unittest.mock import MagicMock, patch
import pytest

from app.services.decision_layer import AgentResult, DecisionLayer, RouteDecision
from app.loaders.document_loader import Document, DocumentLoader
from app.services.rag_pipeline import RAGPipeline, RAGResponse
from app.retrieval.retriever import HybridRetriever, RetrievalResult
from app.retrieval.vector_store import VectorStore
from app.loaders.chunker import Chunk, TextChunker
from app.services.llm_service import generate_answer


class TestErrorHandling(unittest.TestCase):
    """
    Comprehensive test suite verifying that raw internal exceptions,
    Groq API errors, tool errors, and stack traces are NEVER displayed to the user,
    and that exact friendly mappings are returned instead.
    """

    def setUp(self):
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.rag_pipeline = RAGPipeline(self.retriever)
        self.agent = DecisionLayer(self.rag_pipeline)

    # -------------------------------------------------------------------------
    # Case 1: Empty Question
    # -------------------------------------------------------------------------
    def test_empty_question(self):
        """Empty or whitespace-only questions must prompt the user to enter a question."""
        for empty_query in ["", "   ", "\t\n  "]:
            result: AgentResult = self.agent.execute(empty_query)
            self.assertFalse(result.success)
            self.assertEqual(result.answer, "Please enter a question.")
            self.assertIsNone(result.error)

    # -------------------------------------------------------------------------
    # Case 2: No Document Processed
    # -------------------------------------------------------------------------
    def test_no_document_processed(self):
        """When vector store is empty, asking document/general questions must inform the user to process a document first."""
        self.vector_store.clear()
        self.assertTrue(self.vector_store.is_empty())

        questions = [
            "What is Krishna's experience?",
            "Summarize the document",
            "What is RAG?",
            "Explain the company policy",
        ]
        for q in questions:
            result: AgentResult = self.agent.execute(q)
            self.assertFalse(result.success)
            self.assertEqual(
                result.answer,
                "Please process a document before asking a question.",
                f"Failed for query '{q}'",
            )
            self.assertIsNone(result.error)

    # -------------------------------------------------------------------------
    # Case 3: Calculator Standalone Fast-Path (Works even without document)
    # -------------------------------------------------------------------------
    def test_calculator_standalone_without_document(self):
        """Pure math questions like '25 * 8' evaluate cleanly via calculator tool without needing a document."""
        self.vector_store.clear()
        result: AgentResult = self.agent.execute("25 * 8")
        self.assertTrue(result.success)
        self.assertEqual(result.tool_used, "calculator")
        self.assertEqual(result.calculation_result, "200")
        self.assertIn("200", result.answer)

    # -------------------------------------------------------------------------
    # Case 4: Calculator / Tool Failure (No Raw Exceptions)
    # -------------------------------------------------------------------------
    def test_calculator_tool_failure(self):
        """Division by zero or invalid expressions must return a user-friendly message without raw exceptions."""
        result = self.agent.execute("10 / 0")
        self.assertFalse(result.success)
        self.assertEqual(
            result.answer,
            "Sorry, I couldn't calculate that expression. Please check the mathematical formula and try again.",
        )
        self.assertIsNone(result.calculation_result)
        # Ensure raw Python ZeroDivisionError is NOT in user answer
        self.assertNotIn("ZeroDivisionError", result.answer)
        self.assertNotIn("division by zero", result.answer.lower())
        self.assertIsNone(result.error)

    def _populate_mock_doc(self, text: str = "Acme Corp employs 500 team members worldwide. 15% work remotely.", source: str = "doc.txt"):
        chunk = Chunk(chunk_id=1, text=text, source=source)
        self.vector_store.chunks = [chunk]
        self.vector_store.index = MagicMock()
        self.vector_store.total_chunks = 1

    # -------------------------------------------------------------------------
    # Case 5: Valid Document Question
    # -------------------------------------------------------------------------
    def test_valid_document_question(self):
        """When a document is loaded and question is answerable, grounded response is returned."""
        self._populate_mock_doc("Acme Corp employs 500 team members worldwide. 15% work remotely.", "acme_policy.txt")
        chunk = self.vector_store.chunks[0]

        with patch("app.services.decision_layer.generate_answer") as mock_gen, \
             patch("app.services.rag_pipeline.generate_answer") as mock_rag_gen:
            mock_gen.return_value = '{"tool": "document_search", "reason": "Document query", "expression": null}'
            mock_rag_gen.return_value = "Acme Corp employs 500 team members worldwide."

            with patch.object(self.rag_pipeline, "search_documents") as mock_search:
                mock_search.return_value = [
                    RetrievalResult(chunk=chunk, semantic_score=0.9, keyword_score=0.8, hybrid_score=0.87)
                ]
                result = self.agent.execute("How many employees does Acme Corp employ?")
                self.assertTrue(result.success)
                self.assertIn("500", result.answer)
                self.assertIsNone(result.error)

    # -------------------------------------------------------------------------
    # Case 6: Question Not Present in Document
    # -------------------------------------------------------------------------
    def test_question_not_present_in_document_no_chunks(self):
        """When document does not contain relevant chunks, exact friendly message is returned."""
        self._populate_mock_doc("Acme Corp employs 500 team members worldwide.", "acme_policy.txt")

        with patch("app.services.decision_layer.generate_answer") as mock_gen:
            mock_gen.return_value = '{"tool": "document_search", "reason": "Document query", "expression": null}'

            with patch.object(self.rag_pipeline, "search_documents") as mock_search:
                mock_search.return_value = []
                result = self.agent.execute("According to the document, what is the stock price?")
                self.assertFalse(result.success)
                self.assertEqual(
                    result.answer,
                    "Sorry, I couldn't find relevant information in the uploaded document. Please try asking a question related to the document.",
                )
                self.assertIsNone(result.error)

    def test_question_not_present_in_document_model_cannot_find(self):
        """When LLM RAG response indicates info is missing, message is normalized to required text."""
        self._populate_mock_doc("Acme Corp employs 500 team members worldwide.", "acme_policy.txt")
        chunk = self.vector_store.chunks[0]

        with patch("app.services.decision_layer.generate_answer") as mock_gen, \
             patch("app.services.rag_pipeline.generate_answer") as mock_rag_gen:
            mock_gen.return_value = '{"tool": "document_search", "reason": "Document query", "expression": null}'
            mock_rag_gen.return_value = "I could not find information about stock price in the uploaded document."

            with patch.object(self.rag_pipeline, "search_documents") as mock_search:
                mock_search.return_value = [
                    RetrievalResult(chunk=chunk, semantic_score=0.4, keyword_score=0.2, hybrid_score=0.34)
                ]
                result = self.agent.execute("What is the company stock price?")
                self.assertFalse(result.success)
                self.assertEqual(
                    result.answer,
                    "Sorry, I couldn't find relevant information in the uploaded document. Please try asking a question related to the document.",
                )

    # -------------------------------------------------------------------------
    # Case 7: LLM / Groq Service Failure
    # -------------------------------------------------------------------------
    def test_llm_groq_failure_in_rag(self):
        """When Groq API fails during RAG generation, exact friendly error is shown without raw exceptions."""
        self._populate_mock_doc("Some text", "doc.txt")
        chunk = self.vector_store.chunks[0]

        with patch("app.services.decision_layer.generate_answer") as mock_gen, \
             patch("app.services.rag_pipeline.generate_answer") as mock_rag_gen:
            mock_gen.return_value = '{"tool": "document_search", "reason": "Document query", "expression": null}'
            mock_rag_gen.side_effect = RuntimeError("Groq API request failed: 500 Internal Server Error")

            with patch.object(self.rag_pipeline, "search_documents") as mock_search:
                mock_search.return_value = [
                    RetrievalResult(chunk=chunk, semantic_score=0.8, keyword_score=0.7, hybrid_score=0.77)
                ]
                result = self.agent.execute("Explain the document.")
                self.assertFalse(result.success)
                self.assertEqual(
                    result.answer,
                    "Sorry, I couldn't process your request right now. Please try again.",
                )
                self.assertNotIn("500", result.answer)
                self.assertNotIn("Groq API", result.answer)
                self.assertNotIn("RuntimeError", result.answer)

    def test_llm_groq_failure_in_direct_llm(self):
        """When Groq API fails during direct LLM, exact friendly error is shown without raw exceptions."""
        self._populate_mock_doc("Some text", "doc.txt")

        with patch("app.services.decision_layer.generate_answer") as mock_gen:
            # First call for router succeeds, second call for direct LLM fails
            mock_gen.side_effect = [
                '{"tool": "llm", "reason": "General question", "expression": null}',
                RuntimeError("Groq API request failed: Connection reset by peer"),
            ]
            result = self.agent.execute("What is Python?")
            self.assertFalse(result.success)
            self.assertEqual(
                result.answer,
                "Sorry, I couldn't process your request right now. Please try again.",
            )
            self.assertNotIn("Connection reset", result.answer)
            self.assertNotIn("RuntimeError", result.answer)

    # -------------------------------------------------------------------------
    # Case 8: Groq Invalid Tool Call Failure Handled Gracefully
    # -------------------------------------------------------------------------
    def test_groq_invalid_tool_call_handled(self):
        """
        When Groq returns 'Tool choice is none, but model called a tool: repo_browser.open_file',
        generate_answer attempts plain-text recovery; if it still fails, user sees only friendly message.
        """
        from app.services import llm_service

        mock_client = MagicMock()
        tool_err = Exception(
            "Error code: 400 - {'error': {'message': \"Tool choice is none, but model called a tool: repo_browser.open_file with arguments {'path': ''}\", 'type': 'invalid_request_error', 'code': 'tool_use_failed'}}"
        )
        mock_client.chat.completions.create.side_effect = tool_err

        with patch.object(llm_service, "get_client", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                llm_service.generate_answer("Open file")
            # Backend exception is raised cleanly
            self.assertIn("Groq API request failed", str(exc_info.value))

        # Now test that when agent executes this, the UI receives friendly message, NEVER the 400 error!
        self._populate_mock_doc("Some text", "doc.txt")

        with patch("app.services.decision_layer.generate_answer") as mock_gen:
            mock_gen.side_effect = RuntimeError(f"Groq API request failed: {tool_err}")
            result = self.agent.execute("Open repo")
            self.assertFalse(result.success)
            self.assertEqual(
                result.answer,
                "Sorry, I couldn't process your request right now. Please try again.",
            )
            self.assertNotIn("repo_browser", result.answer)
            self.assertNotIn("tool_use_failed", result.answer)
            self.assertNotIn("Tool choice is none", result.answer)
            self.assertNotIn("400", result.answer)

    # -------------------------------------------------------------------------
    # Case 9: Unexpected Exception Caught by Top-Level Handler
    # -------------------------------------------------------------------------
    def test_unexpected_error_handling(self):
        """Any unexpected runtime error in execute returns the required generic friendly message."""
        with patch.object(self.agent, "decide_route", side_effect=Exception("Critical disk failure")):
            result = self.agent.execute("Test query")
            self.assertFalse(result.success)
            self.assertEqual(
                result.answer,
                "Something went wrong while processing your request. Please try again.",
            )
            self.assertNotIn("Critical disk failure", result.answer)
