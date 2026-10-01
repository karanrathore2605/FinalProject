from typing import Any, Dict, List
from app.retriever import HybridRetriever, RetrievalResult


class DocumentSearchTool:
    """
    Standard tool interface for document search.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def execute(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Execute document search and return structured output."""
        try:
            results: List[RetrievalResult] = self.retriever.retrieve(query, top_k=top_k)
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "chunk_id": r.chunk_id,
                        "source": r.source,
                        "content": r.text,
                        "score": r.hybrid_score,
                        "semantic_score": r.semantic_score,
                        "keyword_score": r.keyword_score,
                    }
                    for r in results
                ],
            }
        except Exception as exc:
            return {
                "success": False,
                "query": query,
                "error": str(exc),
                "results": [],
            }
