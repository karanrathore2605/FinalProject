"""
Pydantic Request and Response schemas for the FastAPI endpoints.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    embedding_model: Dict[str, Any]
    llm_model: str
    total_chunks: int
    is_document_loaded: bool
    active_document: Optional[str]


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to embed")
    is_query: bool = Field(False, description="Whether the texts are search queries")


class EmbedResponse(BaseModel):
    model: str
    count: int
    dimension: int
    embeddings: List[List[float]]


class SplitRequest(BaseModel):
    text: str = Field(..., description="Raw text content to split")
    chunk_size: Optional[int] = Field(None, description="Max chunk size in characters")
    chunk_overlap: Optional[int] = Field(None, description="Overlap in characters")


class SplitResponse(BaseModel):
    total_chunks: int
    chunks: List[str]


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve")


class SearchResultItem(BaseModel):
    chunk_id: int
    source: str
    hybrid_score: float
    semantic_score: float
    keyword_score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchResultItem]


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or mathematical instruction")


class ChatResponse(BaseModel):
    success: bool
    query: str
    route: str
    tool_used: str
    answer: str
    sources: List[str]
    retrieval_results: List[SearchResultItem]
    expression: Optional[str] = None
    calculation_result: Optional[str] = None
    error: Optional[str] = None


class CalculatorRequest(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate (e.g. '25 * 8')")


class CalculatorResponse(BaseModel):
    expression: str
    result: str


class IndexTextRequest(BaseModel):
    title: str = Field(..., description="Document title / filename identifier")
    content: str = Field(..., description="Full text content of the document")


class DocumentStatusResponse(BaseModel):
    is_empty: bool
    total_chunks: int
    source_documents: List[str]
