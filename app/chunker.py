import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.document_loader import Document

logger = logging.getLogger(__name__)

# Try importing RecursiveCharacterTextSplitter from langchain_text_splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False


@dataclass
class Chunk:
    """Represents a text chunk for vector indexing and citation."""
    chunk_id: int
    text: str
    source: str
    metadata: Dict = field(default_factory=dict)


class _FallbackRecursiveSplitter:
    """
    Pure Python recursive text splitter fallback in case
    langchain_text_splitters is unavailable.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else ""
        next_seps = separators[1:] if len(separators) > 1 else []

        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part) + (len(sep) if current else 0)
            if current_len + part_len <= self.chunk_size:
                current.append(part)
                current_len += part_len
            else:
                if current:
                    merged = sep.join(current).strip()
                    if merged:
                        chunks.append(merged)
                    # Keep overlap from the end
                    overlap_parts = []
                    overlap_len = 0
                    for rev_part in reversed(current):
                        if overlap_len + len(rev_part) <= self.chunk_overlap:
                            overlap_parts.insert(0, rev_part)
                            overlap_len += len(rev_part) + len(sep)
                        else:
                            break
                    current = overlap_parts
                    current_len = sum(len(p) for p in current) + max(0, len(current) - 1) * len(sep)

                if len(part) > self.chunk_size:
                    if next_seps:
                        sub_chunks = self._split(part, next_seps)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(part[: self.chunk_size])
                else:
                    current.append(part)
                    current_len += len(part) + (len(sep) if len(current) > 1 else 0)

        if current:
            merged = sep.join(current).strip()
            if merged:
                chunks.append(merged)

        return chunks


class TextChunker:
    """
    Recursive Character Text Splitter for documents.
    Splits text hierarchically using double newlines, single newlines, sentences,
    and spaces with configurable chunk size and overlap.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        if HAS_LANGCHAIN_SPLITTER:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            logger.info("Using built-in fallback recursive text splitter.")
            self._splitter = _FallbackRecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )

    def split_text(self, text: str) -> List[str]:
        """Split raw text into clean chunks using recursive character splitting."""
        # Clean form-feeds and normalize linebreaks
        cleaned = text.replace("\x0c", "\n")
        cleaned = re.sub(r"\r\n|\r", "\n", cleaned)

        raw_chunks = self._splitter.split_text(cleaned)
        # Filter out empty or whitespace-only chunks
        return [c.strip() for c in raw_chunks if c.strip()]

    def chunk_document(self, document: Document, start_chunk_id: int = 1) -> List[Chunk]:
        """Split a single Document into a list of coherent Chunks."""
        raw_chunks = self.split_text(document.content)

        chunks: List[Chunk] = []
        current_id = start_chunk_id

        for idx, c_text in enumerate(raw_chunks, start=1):
            chunks.append(
                Chunk(
                    chunk_id=current_id,
                    text=c_text,
                    source=document.source,
                    metadata={
                        "source": document.source,
                        "chunk_index": idx,
                        "total_chunks_in_doc": len(raw_chunks),
                        "char_count": len(c_text),
                        "word_count": len(c_text.split()),
                    },
                )
            )
            current_id += 1

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a collection of documents with consecutive chunk IDs."""
        all_chunks: List[Chunk] = []
        next_id = 1
        for doc in documents:
            doc_chunks = self.chunk_document(doc, start_chunk_id=next_id)
            all_chunks.extend(doc_chunks)
            next_id += len(doc_chunks)
        return all_chunks
