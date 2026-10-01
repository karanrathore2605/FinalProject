import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.calculator_tool import calculate, calculator_tool, preprocess_expression
from app.llm_service import generate_answer
from app.memory import get_conversation_manager
from app.rag_pipeline import RAGPipeline, RAGResponse
from app.retriever import RetrievalResult

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
    session_id: Optional[str] = None
    contextualized_query: Optional[str] = None


class DecisionLayer:
    """
    LLM Decision Layer that analyzes user intent and dynamically orchestrates
    RAG, Calculator Tool, and Direct LLM response generation.
    """

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline

    def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Understand the current user query using previous conversation context.
        Reformulates contextual follow-up questions (e.g. 'its features', 'why is it used?')
        into clear standalone questions for accurate routing and retrieval.
        If the query is already independent or no history exists, returns the query unchanged.
        """
        cleaned = query.strip()
        if not history or not cleaned:
            return cleaned

        # Fast-path for standalone math - never contextualize pure math expressions
        if PURE_MATH_PATTERN.match(cleaned):
            return cleaned

        recent_history = history[-6:]
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in recent_history
        )

        contextualize_prompt = f"""Given a conversation history and a user query, determine if the query is a follow-up that depends on the previous conversation (e.g. uses pronouns like 'it', 'its', 'they', 'them', or phrases like 'and its...', 'why is it...', 'what are its...', 'how does it work?').

If it is a follow-up, rephrase it into a clear standalone question that fully expresses the user's intent with context.
If the query is already independent, complete, or starts an entirely new topic, return it exactly as is.
Do NOT answer the question. Only return the rewritten standalone question.

Conversation History:
{history_text}

User Query: {cleaned}

Standalone Question:"""

        try:
            rewritten = generate_answer(
                contextualize_prompt,
                system_prompt="You are a query contextualizer. Return ONLY the rewritten question.",
                temperature=0.0,
            ).strip()
            # Strip prefixes or quotes if any returned
            rewritten = re.sub(r'^(Standalone Question:|"|\')', "", rewritten).strip().strip('"\'')
            if rewritten:
                return rewritten
        except Exception as exc:
            logger.warning("Contextualization failed: %s. Using original query.", exc)

        return cleaned

    def decide_route(
        self,
        query: str,
        has_documents: bool,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Determine the execution route for the query.
        Returns (route_name, optional_extracted_math_expression).
        Routes:
        - 'CALCULATOR': Standalone math expression
        - 'RAG_AND_CALCULATOR': Document-grounded math problem
        - 'RAG': Document information / explanation / summary
        - 'DIRECT_LLM': General conversation / knowledge
        """
        cleaned_query = query.strip()

        # 1. Fast Path: Pure Math Expression (e.g. "25 * 8", "what is 100 / 4")
        if PURE_MATH_PATTERN.match(cleaned_query):
            # Extract the raw math expression
            expr = re.sub(r"(?i)^(what\s+is\s+|calculate\s+|solve\s+|compute\s+)", "", cleaned_query)
            expr = expr.rstrip("?").strip()
            if any(char.isdigit() for char in expr):
                logger.info("Fast-path route: CALCULATOR for '%s'", expr)
                return "CALCULATOR", expr

        # Apply conversational context resolution if history is passed directly
        if history:
            cleaned_query = self.contextualize_query(cleaned_query, history)

        # 2. Fast Path: User explicitly asks for LLM / external knowledge
        lower_q = cleaned_query.lower()
        if any(cue in lower_q for cue in ["using llm", "use llm", "external source", "use external", "general knowledge", "without document"]):
            logger.info("Explicit route: DIRECT_LLM based on user instruction in query")
            return "DIRECT_LLM", None

        # 3. Rule Check: Document-dependent math (contains math/percentage and document keywords or numerical questions when document is loaded)
        has_math_cues = bool(HAS_MATH_OR_PERCENT.search(cleaned_query))
        has_doc_cues = bool(DOCUMENT_REFERENCE.search(cleaned_query))

        if has_documents and has_math_cues and (has_doc_cues or "how many" in cleaned_query.lower() or "%" in cleaned_query):
            logger.info("Heuristic route: RAG_AND_CALCULATOR")
            return "RAG_AND_CALCULATOR", None

        # 4. LLM-Based Intent Classification
        routing_prompt = f"""You are an intelligent AI router that classifies user queries into the appropriate processing route.

Available Routes:
1. "CALCULATOR": The query is a direct, standalone math expression or calculation without needing document context (e.g., "25 * 8", "calculate 100 / 4").
2. "RAG_AND_CALCULATOR": The query requires retrieving facts/numbers from the uploaded document AND performing a calculation or percentage computation (e.g., "If the document says there are 500 employees and 15% work remotely, how many employees work remotely?", "What is the total revenue minus cost according to the report?").
3. "RAG": The query asks about the uploaded document's content, policies, explanations, summaries, or key points (e.g., "Explain this in simple words", "Give key points", "What is the leave policy?"). NOTE: General knowledge questions about computer science or AI concepts (such as "What is RAG (Retrieval-Augmented Generation)?" or "How does RAG work?") are "DIRECT_LLM" unless explicitly asking about an uploaded document.
4. "DIRECT_LLM": General knowledge question unrelated to any uploaded document or math (e.g., "What is Python?", "What is RAG?", "How does it work?", "Hello!").

Document currently loaded: {'YES' if has_documents else 'NO'}
User Query: "{cleaned_query}"

Respond ONLY with a JSON object in this exact format:
{{"route": "CALCULATOR"|"RAG"|"RAG_AND_CALCULATOR"|"DIRECT_LLM", "expression": null}}
If route is CALCULATOR, put the math expression in "expression".
"""
        try:
            raw_response = generate_answer(
                routing_prompt,
                system_prompt="You are a strict JSON classifier. Output valid JSON only.",
                temperature=0.0,
            )
            # Parse JSON from LLM output
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                route = parsed.get("route", "").upper()
                expr = parsed.get("expression")
                if route in {"CALCULATOR", "RAG", "RAG_AND_CALCULATOR", "DIRECT_LLM"}:
                    if not has_documents and route in {"RAG", "RAG_AND_CALCULATOR"}:
                        logger.info("No documents are loaded; re-routing %s to DIRECT_LLM", route)
                        route = "DIRECT_LLM"
                    logger.info("LLM Decision Layer selected route: %s", route)
                    return route, expr
        except Exception as exc:
            logger.warning("LLM router classification error: %s. Falling back to heuristics.", exc)

        # 5. Fallback Heuristics
        if has_documents:
            if has_math_cues:
                return "RAG_AND_CALCULATOR", None
            return "RAG", None

        return "DIRECT_LLM", None

    def execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResult:
        """Execute the full agent workflow based on decided route with conversational memory."""
        cleaned = query.strip()
        if not cleaned:
            return AgentResult(
                success=False,
                route="NONE",
                answer="Please enter a question or instruction.",
                tool_used="None",
                session_id=session_id,
            )

        # 1. Retrieve conversation history for session if not explicitly provided
        conv_mgr = get_conversation_manager()
        if history is None and session_id:
            history = conv_mgr.get_history(session_id)
        elif history is None:
            history = []

        # 2. Safe debug logging (Requirement 12 - NO API keys or secrets logged)
        logger.info(
            "Conversation ID: %s | History messages: %d | Current query: '%s'",
            session_id or "none",
            len(history),
            cleaned,
        )

        # 3. Understand Current Query Using Context (Requirement 6)
        contextualized_query = self.contextualize_query(cleaned, history)
        if contextualized_query != cleaned:
            logger.info("Contextualized query: '%s' -> '%s'", cleaned, contextualized_query)

        # 4. Context-aware routing
        has_docs = not self.rag_pipeline.retriever.vector_store.is_empty()
        route, expression = self.decide_route(contextualized_query, has_documents=has_docs)

        logger.info("Executing Query: '%s' | Route: %s", cleaned, route)

        result: Optional[AgentResult] = None

        # =====================================================
        # ROUTE 1: CALCULATOR
        # =====================================================
        if route == "CALCULATOR":
            calc_expr = expression or contextualized_query or cleaned
            # Strip non-math words if needed
            calc_expr = re.sub(r"(?i)^(what\s+is\s+|calculate\s+|solve\s+|compute\s+)", "", calc_expr).rstrip("?").strip()

            try:
                result_val = calculator_tool(calc_expr)
                formatted_answer = f"**Calculation Result:**\n\n`{calc_expr} = {result_val}`"
                result = AgentResult(
                    success=True,
                    route="Calculator",
                    answer=formatted_answer,
                    tool_used="calculator",
                    expression=calc_expr,
                    calculation_result=str(result_val),
                    session_id=session_id,
                    contextualized_query=contextualized_query if contextualized_query != cleaned else None,
                )
            except Exception as exc:
                result = AgentResult(
                    success=False,
                    route="Calculator",
                    answer=f"Calculator error: {exc}",
                    tool_used="calculator",
                    expression=calc_expr,
                    error=str(exc),
                    session_id=session_id,
                    contextualized_query=contextualized_query if contextualized_query != cleaned else None,
                )

        # =====================================================
        # ROUTE 2: RAG + CALCULATOR (Document Context + Math)
        # =====================================================
        elif route == "RAG_AND_CALCULATOR":
            if not has_docs:
                result = AgentResult(
                    success=False,
                    route="RAG + Calculator",
                    answer="Please upload and process a document first to answer document-based calculation questions.",
                    tool_used="None",
                    session_id=session_id,
                )
            else:
                # Step 1: Retrieve relevant context from document using contextualized query
                retrieval_results = self.rag_pipeline.search_documents(contextualized_query)

                if not retrieval_results:
                    result = AgentResult(
                        success=False,
                        route="RAG + Calculator",
                        answer="I could not find relevant numerical facts in the uploaded document to solve this problem.",
                        tool_used="document_search",
                        sources=[],
                        session_id=session_id,
                    )
                else:
                    context_str = "\n\n".join(
                        f"[Source: {r.source} | Chunk {r.chunk_id}]: {r.text}" for r in retrieval_results
                    )

                    # Step 2: Ask LLM to extract numerical facts and formulate math expression
                    formula_prompt = f"""You are a mathematical reasoning assistant.
Based on the retrieved document context below, identify the relevant numbers and formulate the exact mathematical expression needed to answer the user's question.

Retrieved Context:
{context_str}

User Question:
{contextualized_query}

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
                            logger.error("Calculator tool execution failed on '%s': %s", calc_expr, exc)
                            calc_result = f"Error: {exc}"

                    # Step 4: Synthesize grounded final explanation incorporating the calculation
                    synthesis_prompt = f"""You are an AI Document Assistant answering a document question with a verified calculation.

Document Context:
{context_str}

User Question:
{contextualized_query}

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
                    final_answer = generate_answer(
                        synthesis_prompt,
                        system_prompt="You provide clear grounded answers using document context and verified calculations.",
                        temperature=0.1,
                    )

                    sources = list(dict.fromkeys(r.source for r in retrieval_results))

                    result = AgentResult(
                        success=True,
                        route="RAG + Calculator",
                        answer=final_answer,
                        tool_used="document_search + calculator",
                        expression=calc_expr,
                        calculation_result=str(calc_result) if calc_result is not None else None,
                        sources=sources,
                        retrieval_results=retrieval_results,
                        session_id=session_id,
                        contextualized_query=contextualized_query if contextualized_query != cleaned else None,
                    )

        # =====================================================
        # ROUTE 3: RAG (Document Search & Answering)
        # =====================================================
        elif route == "RAG":
            if not has_docs:
                result = AgentResult(
                    success=False,
                    route="RAG",
                    answer="Please upload and process a document first so I can answer questions about it.",
                    tool_used="document_search",
                    session_id=session_id,
                )
            else:
                rag_resp: RAGResponse = self.rag_pipeline.answer_question(contextualized_query, history=history)

                result = AgentResult(
                    success=rag_resp.success,
                    route="RAG",
                    answer=rag_resp.answer,
                    tool_used="document_search",
                    sources=rag_resp.sources,
                    retrieval_results=rag_resp.retrieval_results,
                    session_id=session_id,
                    contextualized_query=contextualized_query if contextualized_query != cleaned else None,
                )

        # =====================================================
        # ROUTE 4: DIRECT LLM (Requirement 3: structured messages)
        # =====================================================
        else:
            try:
                # Structure: system message -> previous user/assistant messages -> current user message
                messages: List[Dict[str, str]] = []
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": cleaned})

                direct_ans = generate_answer(
                    messages=messages,
                    temperature=0.3,
                )
                result = AgentResult(
                    success=True,
                    route="Direct LLM",
                    answer=direct_ans,
                    tool_used="llm",
                    sources=[],
                    session_id=session_id,
                    contextualized_query=contextualized_query if contextualized_query != cleaned else None,
                )
            except Exception as exc:
                result = AgentResult(
                    success=False,
                    route="Direct LLM",
                    answer=f"LLM service error: {exc}",
                    tool_used="llm",
                    error=str(exc),
                    session_id=session_id,
                )

        # Save conversation turn to memory if session is active and successful (Requirement 6)
        if session_id and result.success:
            conv_mgr.add_turn(session_id, user_query=cleaned, assistant_response=result.answer)

        return result
