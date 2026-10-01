import logging
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

from app.chunker import Chunk
from app.config import STORAGE_DIR
from app.embeddings import embed_texts

logger = logging.getLogger(__name__)

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves", "explain", "tell", "simple", "word", "words", "please", "give", "key", "points"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and filter stopwords, preserving numeric identifiers."""
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [w for w in words if (len(w) > 1 or w.isdigit()) and w not in STOPWORDS]


class VectorStore:
    """
    FAISS-backed vector store with chunk metadata preservation and
    IDF table creation for hybrid retrieval.
    """

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []
        self.chunk_tokens: List[Set[str]] = []
        self.df: Counter = Counter()
        self.total_chunks: int = 0

    def is_empty(self) -> bool:
        """Check if vector store is initialized with indexed chunks."""
        return self.index is None or len(self.chunks) == 0

    def clear(self) -> None:
        """Clear the vector store index and chunks."""
        self.index = None
        self.chunks = []
        self.chunk_tokens = []
        self.df = Counter()
        self.total_chunks = 0

    def build_from_chunks(self, chunks: List[Chunk]) -> None:
        """Build FAISS index and keyword statistics from given chunks."""
        if not chunks:
            self.clear()
            return

        self.chunks = chunks
        self.total_chunks = len(chunks)

        # 1. Build Keyword / IDF Table
        self.chunk_tokens = [set(tokenize(c.text)) for c in self.chunks]
        self.df = Counter()
        for tokens in self.chunk_tokens:
            for token in tokens:
                self.df[token] += 1

        # 2. Compute Embeddings
        texts = [chunk.text for chunk in self.chunks]
        embeddings = embed_texts(texts)

        # 3. Create FAISS Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logger.info("Built FAISS index with %d chunks (dim=%d)", self.total_chunks, dimension)

    def search_semantic(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Search top_k nearest neighbors by cosine similarity."""
        if self.is_empty():
            return []

        k = min(top_k, self.total_chunks)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def compute_keyword_score(self, query: str, chunk_index: int) -> float:
        """Compute keyword similarity score using direct overlap, phrase matching, and IDF weighting."""
        query_tokens = tokenize(query)
        if not query_tokens or self.total_chunks == 0 or chunk_index >= self.total_chunks:
            return 0.0

        chunk_tokens = self.chunk_tokens[chunk_index]
        chunk_text_lower = self.chunks[chunk_index].text.lower() if chunk_index < len(self.chunks) else ""
        query_lower = query.lower()

        # 1. Exact alphanumeric/number phrases (e.g. "day 5", "week 1", "chapter 3")
        number_phrases = re.findall(r"\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b", query_lower)
        has_num_phrase_query = bool(number_phrases)
        matched_num_phrase = any(phrase in chunk_text_lower for phrase in number_phrases)

        # 2. General bigram matches (e.g., "gradient descent", "vector search")
        bigrams = [
            f"{query_tokens[i]} {query_tokens[i+1]}"
            for i in range(len(query_tokens) - 1)
        ]
        matched_bigrams = [bg for bg in bigrams if bg in chunk_text_lower]

        # 3. Direct token overlap
        matched_tokens = [
            token for token in query_tokens
            if token in chunk_tokens or any(
                (len(c_tok) >= 4 and len(token) >= 4 and (c_tok.startswith(token) or token.startswith(c_tok)))
                for c_tok in chunk_tokens
            )
        ]
        overlap_score = len(matched_tokens) / len(query_tokens)

        # Exact number phrase matching takes highest priority for entity queries
        if has_num_phrase_query:
            if matched_num_phrase:
                return 1.0
            else:
                # Chunks missing the specific requested entity phrase are penalised
                return overlap_score * 0.25

        # For general queries, reward bigram sequence matches
        phrase_bonus = 0.35 if matched_bigrams else 0.0
        base_kw = min(1.0, overlap_score + phrase_bonus)

        # 4. IDF weighting for larger corpora
        if self.total_chunks > 5:
            total_query_weight = 0.0
            matched_weight = 0.0

            for token in query_tokens:
                doc_freq = self.df.get(token, 0)
                idf = math.log((self.total_chunks - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                total_query_weight += idf

                if token in chunk_tokens or any(
                    (len(c_tok) >= 4 and len(token) >= 4 and (c_tok.startswith(token) or token.startswith(c_tok)))
                    for c_tok in chunk_tokens
                ):
                    matched_weight += idf

            idf_score = (matched_weight / total_query_weight) if total_query_weight > 0 else 0.0
            return min(1.0, max(0.0, 0.4 * base_kw + 0.6 * idf_score))

        return min(1.0, max(0.0, base_kw))

    def save(self, directory: Optional[Path] = None) -> None:
        """Persist index and chunks to disk."""
        target_dir = directory or STORAGE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(target_dir / "index.faiss"))
            with open(target_dir / "chunks.pkl", "wb") as f:
                pickle.dump(self.chunks, f)
            logger.info("Saved index and chunks to %s", target_dir)

    def load(self, directory: Optional[Path] = None) -> bool:
        """Load index and chunks from disk if available."""
        target_dir = directory or STORAGE_DIR
        index_file = target_dir / "index.faiss"
        chunks_file = target_dir / "chunks.pkl"

        if not (index_file.exists() and chunks_file.exists()):
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            self.total_chunks = len(self.chunks)
            self.chunk_tokens = [set(tokenize(c.text)) for c in self.chunks]
            self.df = Counter()
            for tokens in self.chunk_tokens:
                for token in tokens:
                    self.df[token] += 1
            logger.info("Loaded FAISS index with %d chunks from %s", self.total_chunks, target_dir)
            return True
        except Exception as exc:
            logger.warning("Failed to load saved index: %s", exc)
            return False
