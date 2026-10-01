import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from app.chunker import Chunk
from app.config import HYBRID_ALPHA, SIMILARITY_THRESHOLD, TOP_K
from app.embeddings import embed_query
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Patterns for document-level meta queries (e.g. "explain this in simple words", "give key points")
META_QUERY_PATTERNS = [
    r"\bexplain\s+(?:this|it|the\s+document|all\s+this)\b",
    r"\bgive\s+(?:me\s+)?(?:key\s+points|main\s+points|bullet\s+points|highlights|takeaways)\b",
    r"\bkey\s+points\b",
    r"\bmain\s+points\b",
    r"\bsummarize\s+(?:this|the\s+document|it)?\b",
    r"\bsummary\s+of\s+(?:this|the\s+document)\b",
    r"\boverview\s+of\s+(?:this|the\s+document)\b",
    r"\bwhat\s+is\s+this\s+(?:document\s+)?about\b",
]


@dataclass
class RetrievalResult:
    """Detailed retrieval result containing chunk and component scores."""
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    hybrid_score: float

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def chunk_id(self) -> int:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    """
    Retrieves most relevant chunks using combined semantic and keyword scoring,
    with smart query expansion and support for document-level meta queries.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        alpha: float = HYBRID_ALPHA,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha

    def is_meta_query(self, query: str) -> bool:
        """Check if query is asking for a general document overview, key points, or summary."""
        q_lower = query.lower().strip()
        return any(re.search(pat, q_lower) for pat in META_QUERY_PATTERNS)

    def expand_query(self, query: str) -> str:
        """Expand queries with common resume/document synonyms for higher recall."""
        q_lower = query.lower().strip()
        extra_terms: List[str] = []

        if self.is_meta_query(query):
            extra_terms.extend(["summary", "overview", "purpose", "introduction", "main points", "executive summary"])

        if "name" in q_lower or "who" in q_lower:
            extra_terms.extend(["name", "candidate name", "personal details", "full name"])

        if "skill" in q_lower or "technolog" in q_lower:
            extra_terms.extend(["skills", "technical skills", "IT skills", "technologies", "tools"])

        if "education" in q_lower or "college" in q_lower or "degree" in q_lower:
            extra_terms.extend(["education", "qualification", "academic", "degree", "institute"])

        if "project" in q_lower:
            extra_terms.extend(["projects", "project description", "role", "backend"])

        if extra_terms:
            return f"{query} {' '.join(extra_terms)}"
        return query

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Execute hybrid search on vector store."""
        limit = top_k or self.top_k

        if self.vector_store.is_empty():
            logger.info("Retriever: vector store is empty.")
            return []

        is_meta = self.is_meta_query(query)
        effective_query = self.expand_query(query)

        query_emb = embed_query(effective_query)
        if self.vector_store.total_chunks <= 6:
            candidate_count = self.vector_store.total_chunks
            limit = min(self.vector_store.total_chunks, max(limit, 4))
        else:
            candidate_count = min(max(limit * 3, 5), self.vector_store.total_chunks)
        semantic_matches = self.vector_store.search_semantic(query_emb, candidate_count)

        candidates: List[RetrievalResult] = []
        for idx, raw_score in semantic_matches:
            chunk = self.vector_store.chunks[idx]
            norm_sem = max(0.0, min(1.0, float(raw_score)))
            kw_score = self.vector_store.compute_keyword_score(effective_query, idx)
            hybrid = self.alpha * norm_sem + (1.0 - self.alpha) * kw_score

            candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=round(norm_sem, 4),
                    keyword_score=round(kw_score, 4),
                    hybrid_score=round(hybrid, 4),
                )
            )

        # Sort by hybrid score descending
        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)

        top_cand = candidates[0] if candidates else None

        # Filter by threshold
        passed = [c for c in candidates if c.hybrid_score >= self.similarity_threshold][:limit]

        # For small documents (e.g. uploaded resume with <= 5 chunks total):
        # If strict threshold dropped it, but candidates exist, provide top candidates
        if not passed and self.vector_store.total_chunks <= 5 and candidates:
            logger.info("Small document recall: passing top candidate to LLM for grounded inspection.")
            passed = candidates[:limit]

        # For meta queries (e.g., "Explain this in simple words", "Give key points"):
        # If strict threshold filtered out generic wording, supply the initial representative chunks
        elif not passed and is_meta and self.vector_store.chunks:
            logger.info("Meta query fallback: retrieving initial document chunks for overview.")
            meta_chunks = self.vector_store.chunks[:limit]
            passed = [
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=round(candidates[idx].semantic_score if idx < len(candidates) else 0.5, 4),
                    keyword_score=0.5,
                    hybrid_score=round(max(self.similarity_threshold, 0.50), 4),
                )
                for idx, chunk in enumerate(meta_chunks)
            ]

        # Log detailed retrieval statistics
        logger.info("--- Hybrid Retrieval ---")
        logger.info("Query: %s (Is Meta: %s)", query, is_meta)
        if top_cand:
            logger.info("Top result source: %s (Chunk %d)", top_cand.source, top_cand.chunk_id)
            logger.info("Semantic: %.4f | Keyword: %.4f | Hybrid: %.4f",
                        top_cand.semantic_score, top_cand.keyword_score, top_cand.hybrid_score)
        logger.info("Threshold: %.2f | Passed: %d chunks", self.similarity_threshold, len(passed))

        return passed
