import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.llm_service import generate_answer
from app.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    success: bool
    answer: str
    sources: List[str]
    retrieval_results: List[RetrievalResult]


class RAGPipeline:
    """
    Orchestrates context retrieval, prompt construction with grounding constraints,
    and LLM answer generation.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def search_documents(self, query: str) -> List[RetrievalResult]:
        """Search documents for relevant chunks."""
        return self.retriever.retrieve(query)

    def answer_question(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """Execute grounded RAG answering flow with optional conversational history."""
        results = self.search_documents(question)

        if not results:
            return RAGResponse(
                success=False,
                answer="I could not find relevant information in the uploaded document to answer this question.",
                sources=[],
                retrieval_results=[],
            )

        # Assemble rich, labeled context
        context_blocks = []
        for res in results:
            context_blocks.append(
                f"[Document: {res.source} | Chunk {res.chunk_id} | Similarity: {res.hybrid_score:.2f}]\n{res.text}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        user_content = f"""Answer the user's question using ONLY the provided context below.

Rules:
1. Ground your answer strictly in the facts provided in the Context.
2. If the answer cannot be determined from the context, explicitly state: "I could not find this information in the uploaded document."
3. Do not assume or fabricate facts not mentioned in the context.
4. When stating facts, you may mention the source document name if relevant.

Context:
{context}

User Question:
{question}
"""
        system_prompt = (
            "You are a strict, grounded AI document assistant. "
            "You never hallucinate facts and only use facts from the provided context."
        )

        messages: List[Dict[str, str]] = []
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_content})

        try:
            answer = generate_answer(
                system_prompt=system_prompt,
                messages=messages,
                temperature=0.1,
            )
            sources = list(dict.fromkeys(res.source for res in results))

            return RAGResponse(
                success=True,
                answer=answer,
                sources=sources,
                retrieval_results=results,
            )
        except Exception as exc:
            logger.error("RAG pipeline answer generation failed: %s", exc)
            return RAGResponse(
                success=False,
                answer=f"Error generating answer: {exc}",
                sources=[],
                retrieval_results=results,
            )
