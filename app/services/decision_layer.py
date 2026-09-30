import json
import logging
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sys

from app.tools.calculator_tool import calculate, calculator_tool, preprocess_expression
from app.services.llm_service import generate_answer
from app.services.rag_pipeline import RAGPipeline, RAGResponse
from app.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


# Fast-path regex for pure arithmetic questions like "25 * 8", "100 / 4", "(20+30)*2"
PURE_MATH_PATTERN = re.compile(
    r"^(?:what\s+is\s+|calculate\s+|solve\s+|compute\s+)?[\d\s+\-*/%().^×÷]+[?]?$",
    re.IGNORECASE,
)

# Math detection within text
HAS_MATH_OR_PERCENT = re.compile(
    r"(\b\d+\s*[%]|(\b\d+\s*[\+\-\*\/×÷]\s*\d+\b)|\b(calculate|compute|solve|how many|percentage|ratio|total)\b)",
    re.IGNORECASE,
)

# Reference to document in query
DOCUMENT_REFERENCE = re.compile(
    r"\b(document|pdf|policy|file|text|passage|section|article|context|according to|stated in|summary|key points|explain this)\b",
    re.IGNORECASE,
)

# Explicit references demanding facts specifically from the uploaded document
EXPLICIT_DOC_PATTERN = re.compile(
    r"\b(according to the|in the (?:uploaded )?(?:document|resume|file|pdf|text)|from the (?:uploaded )?(?:document|resume|file|pdf|text)|stated in the|mentioned in the|as per the (?:uploaded )?(?:document|resume|file|pdf|text)|what does the (?:document|resume|file|pdf) say)\b",
    re.IGNORECASE,
)


class RouteDecision(tuple):
    """
    Backwards-compatible 2-tuple (route, expression) with rich routing metadata:
    - route: "CALCULATOR", "RAG", "RAG_AND_CALCULATOR", "DIRECT_LLM"
    - expression: Optional math expression string
    - reason: Explanatory reasoning for the chosen route
    - tool: "calculator", "document_search", "document_search + calculator", "llm"
    - is_explicit_doc: True if user specifically requested document-grounded facts
    """
    def __new__(cls, route: str, expression: Optional[str] = None, reason: str = "", tool: str = "llm", is_explicit_doc: bool = False):
        obj = super().__new__(cls, (route, expression))
        obj.route = route
        obj.expression = expression
        obj.reason = reason
        obj.tool = tool
        obj.is_explicit_doc = is_explicit_doc
        return obj

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "route": self.route,
            "reason": self.reason,
            "expression": self.expression,
            "is_explicit_document_query": self.is_explicit_doc,
        }


@dataclass
class AgentResult:
    """Standardized output from the Decision Layer agent."""
    success: bool
    route: str  # "CALCULATOR", "RAG", "RAG_AND_CALCULATOR", "DIRECT_LLM"
    answer: str
    tool_used: str
    expression: Optional[str] = None
    calculation_result: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    error: Optional[str] = None
    reason: Optional[str] = None


class DecisionLayer:
    """
    Intelligent Decision Layer that analyzes user intent and dynamically orchestrates
    RAG (document search), Calculator Tool, and Direct LLM response generation.
    Distinguishes between document-specific queries, general knowledge questions, and calculations.
    """

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline

    @staticmethod
    def _log_routing(query: str, decision: RouteDecision) -> None:
        """Log routing decisions in structured standard format."""
        display_route = (
            "DOCUMENT_SEARCH" if decision.route == "RAG"
            else ("LLM" if decision.route == "DIRECT_LLM"
            else decision.route)
        )
        logger.info(
            "\nQuery: %s\nRoute: %s\nReason: %s",
            query,
            display_route,
            decision.reason or "No reason provided",
        )

    def decide_route(
        self,
        query: str,
        has_documents: bool,
        doc_name: Optional[str] = None,
    ) -> RouteDecision:
        """
        Determine the execution route for the query.
        Returns a RouteDecision (unpackable as (route, expression)).
        Routes:
        - 'CALCULATOR': Standalone math expression (Tool: calculator)
        - 'RAG_AND_CALCULATOR': Document-grounded math problem (Tool: document_search + calculator)
        - 'RAG': Document information / explanation / summary (Tool: document_search)
        - 'DIRECT_LLM': General conversation / knowledge (Tool: llm)
        """
        cleaned_query = query.strip()
        lower_q = cleaned_query.lower()

        if doc_name is None and has_documents:
            vs = getattr(self.rag_pipeline.retriever, "vector_store", None)
            if vs and vs.chunks:
                doc_name = vs.chunks[0].source

        # 1. Fast Path: Pure Math Expression (e.g. "25 * 8", "what is 100 / 4")
        if PURE_MATH_PATTERN.match(cleaned_query):
            expr = re.sub(r"(?i)^(what\s+is\s+|calculate\s+|solve\s+|compute\s+)", "", cleaned_query)
            expr = expr.rstrip("?").strip()
            if any(char.isdigit() for char in expr):
                decision = RouteDecision(
                    route="CALCULATOR",
                    expression=expr,
                    reason="Mathematical expression",
                    tool="calculator",
                    is_explicit_doc=False,
                )
                self._log_routing(cleaned_query, decision)
                return decision

        # 2. Fast Path: User explicitly asks for general LLM / external knowledge
        if any(cue in lower_q for cue in ["using llm", "use llm", "external source", "use external", "general knowledge", "without document"]):
            decision = RouteDecision(
                route="DIRECT_LLM",
                expression=None,
                reason="User explicitly requested general knowledge / LLM response",
                tool="llm",
                is_explicit_doc=False,
            )
            self._log_routing(cleaned_query, decision)
            return decision

        # 3. Rule Check: Document-dependent math
        has_math_cues = bool(HAS_MATH_OR_PERCENT.search(cleaned_query))
        has_explicit_doc = bool(EXPLICIT_DOC_PATTERN.search(cleaned_query))

        if has_documents and has_math_cues and (has_explicit_doc or ("how many" in lower_q and ("in the" in lower_q or "resume" in lower_q or "document" in lower_q))):
            decision = RouteDecision(
                route="RAG_AND_CALCULATOR",
                expression=None,
                reason="Document-grounded mathematical calculation",
                tool="document_search + calculator",
                is_explicit_doc=True,
            )
            self._log_routing(cleaned_query, decision)
            return decision

        # 4. LLM-Based Intent Classification
        routing_prompt = f"""You are an intelligent query router that classifies user queries into the appropriate tool.

Available Tools:
1. "calculator":
   - The query is a direct, standalone math expression or calculation (e.g., "25 * 8", "calculate 100 / 4").
2. "document_search":
   - The query asks about specific information, facts, qualifications, skills, experience, or projects from the uploaded document or the specific person/entity it describes (e.g., "What skills are mentioned in Krishna's resume?", "What is Krishna's experience?", "Which projects are mentioned in the uploaded resume?").
   - OR the query explicitly references the uploaded document (e.g., "According to the uploaded document, what is RAG?", "What does the file say about leave policy?").
3. "llm":
   - The query asks a general knowledge, conceptual, or theoretical question that does NOT specifically request document context (e.g., "What is RAG?", "What is Python?", "Explain embeddings in simple words", "What is FAISS?").
   - General conversation, greetings, or common questions.
4. "document_search + calculator":
   - The query requires retrieving facts/numbers from the uploaded document AND performing a calculation (e.g., "If the document says there are 500 employees and 15% work remotely, how many employees work remotely?").

CRITICAL RULES:
- General knowledge or definition questions (like "What is RAG?", "What is Python?", "Explain embeddings in simple words", "What is FAISS?") MUST be classified as "llm", EVEN IF a document is loaded, UNLESS the user explicitly asks what the uploaded document states about it.
- If the query specifically mentions the person or entity from the uploaded document (e.g. "Krishna" when document is "{doc_name or 'None'}"), classify as "document_search".

Uploaded Document Loaded: {'YES' if has_documents else 'NO'}
Uploaded Document Name: "{doc_name or 'None'}"
User Query: "{cleaned_query}"

Respond ONLY with a JSON object in this exact format:
{{
  "tool": "calculator" | "document_search" | "llm" | "document_search + calculator",
  "reason": "Clear 1-sentence reason for this decision",
  "expression": null,
  "is_explicit_document_query": false
}}
"""
        tool_to_route = {
            "calculator": "CALCULATOR",
            "document_search": "RAG",
            "document_search + calculator": "RAG_AND_CALCULATOR",
            "llm": "DIRECT_LLM",
        }

        try:
            raw_response = generate_answer(
                routing_prompt,
                system_prompt="You are a strict JSON query router. Return valid JSON only.",
                temperature=0.0,
            )
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                tool = parsed.get("tool", "llm").strip().lower()
                reason = parsed.get("reason", "")
                expr = parsed.get("expression")
                is_explicit = bool(parsed.get("is_explicit_document_query", False)) or has_explicit_doc

                if tool in tool_to_route:
                    route = tool_to_route[tool]
                    decision = RouteDecision(
                        route=route,
                        expression=expr,
                        reason=reason or ("Document-specific question" if route == "RAG" else "General knowledge question"),
                        tool=tool,
                        is_explicit_doc=is_explicit,
                    )
                    self._log_routing(cleaned_query, decision)
                    return decision
        except Exception:
            logger.exception("LLM router classification error. Using heuristic router.")

        # 5. Fallback Heuristics
        doc_entity_match = False
        if doc_name:
            stem = Path(doc_name).stem.lower().replace("_", " ").replace("-", " ")
            stem_words = [w for w in stem.split() if len(w) > 2 and w not in ["resume", "offline", "document", "file", "pdf", "text"]]
            if any(w in lower_q for w in stem_words):
                doc_entity_match = True

        if has_documents and (has_explicit_doc or doc_entity_match):
            decision = RouteDecision(
                route="RAG",
                expression=None,
                reason="Document-specific question",
                tool="document_search",
                is_explicit_doc=has_explicit_doc,
            )
        else:
            decision = RouteDecision(
                route="DIRECT_LLM",
                expression=None,
                reason="General knowledge question",
                tool="llm",
                is_explicit_doc=False,
            )

        self._log_routing(cleaned_query, decision)
        return decision

    def execute(self, query: str) -> AgentResult:
        """Execute the full agent workflow based on decided route."""
        try:
            cleaned = query.strip() if query else ""
            if not cleaned:
                return AgentResult(
                    success=False,
                    route="NONE",
                    answer="Please enter a question.",
                    tool_used="None",
                )

            has_docs = not self.rag_pipeline.retriever.vector_store.is_empty()
            doc_name = (
                self.rag_pipeline.retriever.vector_store.chunks[0].source
                if has_docs and self.rag_pipeline.retriever.vector_store.chunks
                else None
            )
            decision = self.decide_route(cleaned, has_documents=has_docs, doc_name=doc_name)
            route = decision.route
            expression = decision.expression

            logger.info("Executing Query: '%s' | Route: %s | Tool: %s", cleaned, route, decision.tool)

            # Check if document has not been processed for document-dependent queries
            if not has_docs and route != "CALCULATOR":
                return AgentResult(
                    success=False,
                    route=route or "NONE",
                    answer="Please process a document before asking a question.",
                    tool_used="None",
                    reason="Document not processed",
                )

            # =====================================================
            # ROUTE 1: CALCULATOR
            # =====================================================
            if route == "CALCULATOR":
                calc_expr = expression or cleaned
                calc_expr = re.sub(r"(?i)^(what\s+is\s+|calculate\s+|solve\s+|compute\s+)", "", calc_expr).rstrip("?").strip()

                try:
                    result_val = calculator_tool(calc_expr)
                    formatted_answer = f"**Calculation Result:**\n\n`{calc_expr} = {result_val}`"
                    return AgentResult(
                        success=True,
                        route="Calculator",
                        answer=formatted_answer,
                        tool_used="calculator",
                        expression=calc_expr,
                        calculation_result=str(result_val),
                        reason=decision.reason,
                    )
                except Exception as exc:
                    logger.exception("Calculator tool execution failed on '%s'", calc_expr)
                    return AgentResult(
                        success=False,
                        route="Calculator",
                        answer="Sorry, I couldn't calculate that expression. Please check the mathematical formula and try again.",
                        tool_used="calculator",
                        expression=calc_expr,
                        calculation_result=None,
                        error=None,
                        reason=decision.reason,
                    )

            # =====================================================
            # ROUTE 2: RAG + CALCULATOR (Document Context + Math)
            # =====================================================
            if route == "RAG_AND_CALCULATOR":
                if not has_docs:
                    return AgentResult(
                        success=False,
                        route="RAG + Calculator",
                        answer="Please process a document before asking a question.",
                        tool_used="None",
                        reason=decision.reason,
                    )

                # Step 1: Retrieve relevant context from document
                try:
                    retrieval_results = self.rag_pipeline.search_documents(cleaned)
                except Exception:
                    logger.exception("Retrieval failed in RAG_AND_CALCULATOR")
                    retrieval_results = []

                if not retrieval_results:
                    return AgentResult(
                        success=False,
                        route="RAG + Calculator",
                        answer="Sorry, I couldn't find relevant information in the uploaded document. Please try asking a question related to the document.",
                        tool_used="document_search",
                        sources=[],
                    )

                context_str = "\n\n".join(
                    f"[Source: {r.source} | Chunk {r.chunk_id}]: {r.text}" for r in retrieval_results
                )

                # Step 2: Ask LLM to extract numerical facts and formulate math expression
                formula_prompt = f"""You are a mathematical reasoning assistant.
Based on the retrieved document context below, identify the relevant numbers and formulate the exact mathematical expression needed to answer the user's question.

Retrieved Context:
{context_str}

User Question:
{cleaned}

Instructions:
1. Extract the specific numbers from the context (e.g., total = 500, percentage = 15%).
2. Output the exact arithmetic expression for the calculator (e.g. "500 * 0.15" or "250 * 0.20").
3. Respond ONLY in this JSON format:
{{"expression": "500 * 0.15", "context_facts": "The document states there are 500 employees and 15% work remotely."}}
"""
                calc_expr = None
                context_facts = ""
                calc_result = None

                try:
                    llm_formula_resp = generate_answer(
                        formula_prompt,
                        system_prompt="You extract math expressions from context. Return valid JSON only.",
                        temperature=0.0,
                    )
                    json_match = re.search(r"\{.*\}", llm_formula_resp, re.DOTALL)
                    if json_match:
                        formula_data = json.loads(json_match.group(0))
                        calc_expr = formula_data.get("expression")
                        context_facts = formula_data.get("context_facts", "")
                except Exception as exc:
                    logger.warning("Could not formulate math expression via LLM: %s", exc)

                # Fallback regex extraction if LLM didn't return a clean expression
                if not calc_expr:
                    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned + " " + context_str)
                    if len(numbers) >= 2:
                        calc_expr = f"{numbers[0]} * ({numbers[1]} / 100)"

                # Step 3: Run the Calculator Tool safely!
                if calc_expr:
                    try:
                        calc_result = calculator_tool(calc_expr)
                        logger.info("Calculator Tool computed: %s = %s", calc_expr, calc_result)
                    except Exception as exc:
                        logger.exception("Calculator tool execution failed on '%s'", calc_expr)
                        calc_result = None

                # Step 4: Synthesize grounded final explanation incorporating the calculation
                if calc_result is not None:
                    synthesis_prompt = f"""You are an AI Document Assistant answering a document question with a verified calculation.

Document Context:
{context_str}

User Question:
{cleaned}

Retrieved Facts:
{context_facts}

Calculator Execution:
Formula: {calc_expr}
Verified Result: {calc_result}

Instructions:
Provide a clear, grounded answer that:
1. Explains the facts found in the document.
2. Explains the calculation performed by the calculator tool ({calc_expr} = {calc_result}).
3. States the final answer clearly.
Do not hallucinate. Ground your answer in the retrieved context.
"""
                else:
                    synthesis_prompt = f"""You are an AI Document Assistant answering a document question.

Document Context:
{context_str}

User Question:
{cleaned}

Retrieved Facts:
{context_facts}

Instructions:
Answer the question based on the document context and retrieved facts. State the final answer clearly.
Do not hallucinate. Ground your answer in the retrieved context.
"""
                try:
                    final_answer = generate_answer(
                        synthesis_prompt,
                        system_prompt="You provide clear grounded answers using document context.",
                        temperature=0.1,
                    )
                except Exception:
                    logger.exception("LLM synthesis failed in RAG_AND_CALCULATOR")
                    return AgentResult(
                        success=False,
                        route="RAG + Calculator",
                        answer="Sorry, I couldn't process your request right now. Please try again.",
                        tool_used="document_search + calculator",
                    )

                sources = list(dict.fromkeys(r.source for r in retrieval_results))

                return AgentResult(
                    success=True,
                    route="RAG + Calculator",
                    answer=final_answer,
                    tool_used="document_search + calculator",
                    expression=calc_expr,
                    calculation_result=str(calc_result) if calc_result is not None else None,
                    sources=sources,
                    retrieval_results=retrieval_results,
                )

            # =====================================================
            # ROUTE 3: RAG (Document Search & Answering)
            # =====================================================
            if route == "RAG":
                if not has_docs:
                    return AgentResult(
                        success=False,
                        route="RAG",
                        answer="Please process a document before asking a question.",
                        tool_used="document_search",
                        reason=decision.reason,
                    )

                retrieval_results = self.rag_pipeline.search_documents(cleaned)

                # Zero chunks found by retriever
                if not retrieval_results:
                    return AgentResult(
                        success=False,
                        route="RAG",
                        answer="Sorry, I couldn't find relevant information in the uploaded document. Please try asking a question related to the document.",
                        tool_used="document_search",
                        sources=[],
                        reason=decision.reason,
                    )

                # Chunks retrieved -> synthesize grounded answer
                rag_resp: RAGResponse = self.rag_pipeline.answer_question(cleaned)

                if not rag_resp.success:
                    return AgentResult(
                        success=False,
                        route="RAG",
                        answer=rag_resp.answer,
                        tool_used="document_search",
                        sources=rag_resp.sources,
                        retrieval_results=rag_resp.retrieval_results,
                        reason=decision.reason,
                    )

                # Check if answer indicates missing information in document
                negative_phrases = [
                    "could not find",
                    "couldn't find",
                    "not found in the uploaded document",
                    "not found in the provided",
                    "not mentioned in the context",
                    "not mentioned in the document",
                    "does not contain",
                    "cannot be determined from the context",
                    "no information",
                    "sorry, i couldn't find",
                    "sorry, i could not find",
                ]
                ans_lower = rag_resp.answer.lower()
                not_in_document = any(p in ans_lower for p in negative_phrases)

                if not_in_document:
                    return AgentResult(
                        success=False,
                        route="RAG",
                        answer="Sorry, I couldn't find relevant information in the uploaded document. Please try asking a question related to the document.",
                        tool_used="document_search",
                        sources=rag_resp.sources,
                        retrieval_results=rag_resp.retrieval_results,
                        reason=decision.reason,
                    )

                return AgentResult(
                    success=rag_resp.success,
                    route="RAG",
                    answer=rag_resp.answer,
                    tool_used="document_search",
                    sources=rag_resp.sources,
                    retrieval_results=rag_resp.retrieval_results,
                    reason=decision.reason,
                )

            # =====================================================
            # ROUTE 4: DIRECT LLM
            # =====================================================
            try:
                direct_ans = generate_answer(cleaned, temperature=0.3)
                return AgentResult(
                    success=True,
                    route="Direct LLM",
                    answer=direct_ans,
                    tool_used="llm",
                    sources=[],
                    reason=decision.reason,
                )
            except Exception:
                logger.exception("LLM service failed in direct LLM route")
                return AgentResult(
                    success=False,
                    route="Direct LLM",
                    answer="Sorry, I couldn't process your request right now. Please try again.",
                    tool_used="llm",
                    error=None,
                    reason=decision.reason,
                )

        except Exception:
            logger.exception("Unexpected error in decision layer execute")
            return AgentResult(
                success=False,
                route="ERROR",
                answer="Something went wrong while processing your request. Please try again.",
                tool_used="None",
                error=None,
            )
