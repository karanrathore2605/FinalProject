import io
import logging
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.calculator_tool import calculator_tool
from app.chunker import TextChunker
from app.config import (
    API_HOST,
    API_PORT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    HYBRID_ALPHA,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.decision_layer import AgentResult, DecisionLayer
from app.document_loader import Document, DocumentLoader
from app.embeddings import embed_query, embed_texts, get_embedding_info
from app.memory import get_conversation_manager
from app.rag_pipeline import RAGPipeline
from app.retriever import HybridRetriever, RetrievalResult
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="AI Document Assistant API",
    description=(
        "REST API backend for AI Document Assistant featuring Qwen 3-Embedding-8B, "
        "Recursive Character Text Splitter, Hybrid RAG Retrieval, and AST Calculator."
    ),
    version="1.0.0",
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Pipeline Singletons
_vector_store: Optional[VectorStore] = None
_retriever: Optional[HybridRetriever] = None
_rag_pipeline: Optional[RAGPipeline] = None
_decision_agent: Optional[DecisionLayer] = None


def get_shared_pipeline():
    """Retrieve or initialize the shared singleton RAG and Decision pipeline."""
    global _vector_store, _retriever, _rag_pipeline, _decision_agent
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.load()
        _retriever = HybridRetriever(
            _vector_store,
            top_k=TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
            alpha=HYBRID_ALPHA,
        )
        _rag_pipeline = RAGPipeline(_retriever)
        _decision_agent = DecisionLayer(_rag_pipeline)
    return _vector_store, _retriever, _rag_pipeline, _decision_agent


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    embedding_model: Dict[str, Any]
    llm_model: str
    total_chunks: int
    is_document_loaded: bool
    active_document: Optional[str]


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to embed")
    is_query: bool = Field(False, description="Whether the texts are search queries (applies query prompt for Qwen)")


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
    session_id: Optional[str] = Field(None, description="Optional conversation session ID for conversational memory")


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
    session_id: Optional[str] = Field(None, description="Active conversation session ID")
    contextualized_query: Optional[str] = Field(None, description="Contextualized query if reformulated from context")


class ClearChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID whose conversation history will be cleared")


class HistoryMessageItem(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    count: int
    messages: List[HistoryMessageItem]


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


# -----------------------------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    """Root endpoint providing system information and link to API docs."""
    return {
        "message": "AI Document Assistant API is running.",
        "docs_url": "/docs",
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": GROQ_MODEL,
        "chunking": "Recursive Character Text Splitter",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Diagnostic health check endpoint."""
    vs, _, _, _ = get_shared_pipeline()
    emb_info = get_embedding_info()
    active_doc = vs.chunks[0].source if vs.chunks else None

    return HealthResponse(
        status="healthy",
        embedding_model=emb_info,
        llm_model=GROQ_MODEL,
        total_chunks=vs.total_chunks,
        is_document_loaded=not vs.is_empty(),
        active_document=active_doc,
    )


@app.get("/api/models", tags=["General"])
def model_info():
    """Get active model details and configuration parameters."""
    emb_info = get_embedding_info()
    return {
        "embedding": emb_info,
        "llm": {
            "model": GROQ_MODEL,
            "provider": "Groq",
        },
        "chunking": {
            "strategy": "RecursiveCharacterTextSplitter",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    }


@app.post("/api/embed", response_model=EmbedResponse, tags=["Embeddings"])
def generate_embeddings(req: EmbedRequest):
    """
    Generate normalized dense vector embeddings using Qwen 3-Embedding-8B.
    Automatically applies query instructions when is_query=True.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="The 'texts' list cannot be empty.")

    try:
        emb_matrix = embed_texts(req.texts, is_query=req.is_query)
        emb_list = emb_matrix.tolist()
        dim = int(emb_matrix.shape[1]) if len(emb_matrix.shape) > 1 else 0

        info = get_embedding_info()
        return EmbedResponse(
            model=info["active_model"],
            count=len(emb_list),
            dimension=dim,
            embeddings=emb_list,
        )
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {exc}",
        )


@app.post("/api/split", response_model=SplitResponse, tags=["Chunking"])
def preview_text_split(req: SplitRequest):
    """
    Preview splitting of raw text using the Recursive Character Text Splitter.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    c_size = req.chunk_size or CHUNK_SIZE
    c_overlap = req.chunk_overlap or CHUNK_OVERLAP

    chunker = TextChunker(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = chunker.split_text(req.text)

    return SplitResponse(
        total_chunks=len(chunks),
        chunks=chunks,
    )


@app.post("/api/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX), split it using Recursive Character Text Splitter,
    generate Qwen embeddings, and index into FAISS vector store.
    """
    vs, _, _, _ = get_shared_pipeline()
    filename = file.filename or "uploaded_file"
    file_bytes = await file.read()

    try:
        doc = DocumentLoader.load_from_bytes(file_bytes, filename)
        chunker = TextChunker()
        chunks = chunker.chunk_document(doc)

        vs.build_from_chunks(chunks)
        vs.save()

        return {
            "success": True,
            "filename": filename,
            "total_chunks": len(chunks),
            "char_count": len(doc.content),
            "message": f"Successfully indexed '{filename}' with {len(chunks)} chunks using Recursive Splitter and Qwen embeddings.",
        }
    except Exception as exc:
        logger.error("Error uploading document %s: %s", filename, exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/documents/index-text", tags=["Documents"])
def index_raw_text(req: IndexTextRequest):
    """
    Index raw text directly as a document.
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    vs, _, _, _ = get_shared_pipeline()
    try:
        doc = Document(content=req.content.strip(), source=req.title.strip())
        chunker = TextChunker()
        chunks = chunker.chunk_document(doc)

        vs.build_from_chunks(chunks)
        vs.save()

        return {
            "success": True,
            "title": req.title,
            "total_chunks": len(chunks),
            "message": f"Successfully indexed '{req.title}' with {len(chunks)} chunks.",
        }
    except Exception as exc:
        logger.error("Error indexing text: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/documents/status", response_model=DocumentStatusResponse, tags=["Documents"])
def get_document_status():
    """Retrieve indexed document information and chunk count."""
    vs, _, _, _ = get_shared_pipeline()
    sources = list(dict.fromkeys(c.source for c in vs.chunks))
    return DocumentStatusResponse(
        is_empty=vs.is_empty(),
        total_chunks=vs.total_chunks,
        source_documents=sources,
    )


@app.delete("/api/documents/clear", tags=["Documents"])
def clear_documents():
    """Clear all indexed document chunks from memory and vector store."""
    vs, _, _, _ = get_shared_pipeline()
    vs.clear()
    vs.save()
    return {"success": True, "message": "Vector store and document index cleared."}


@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
def search_documents(req: SearchRequest):
    """
    Perform hybrid retrieval (dense semantic Qwen embeddings + sparse keyword scoring)
    over the indexed document chunks.
    """
    _, retriever, _, _ = get_shared_pipeline()

    if retriever.vector_store.is_empty():
        return SearchResponse(query=req.query, count=0, results=[])

    k = req.top_k or TOP_K
    results: List[RetrievalResult] = retriever.retrieve(req.query, top_k=k)

    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            source=r.source,
            hybrid_score=round(r.hybrid_score, 4),
            semantic_score=round(r.semantic_score, 4),
            keyword_score=round(r.keyword_score, 4),
            text=r.text,
        )
        for r in results
    ]

    return SearchResponse(query=req.query, count=len(items), results=items)


@app.post("/api/chat", response_model=ChatResponse, tags=["AI Assistant"])
@app.post("/api/ask", response_model=ChatResponse, tags=["AI Assistant"])
def chat_or_ask(req: ChatRequest):
    """
    Execute full intelligent query flow through LLM Decision Layer with conversational memory:
    - Standalone math -> Calculator Tool
    - Document grounded question -> Hybrid RAG Search
    - Document math -> RAG Context + Calculator Tool
    - General inquiry -> Direct LLM
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Determine or generate session ID for memory isolation
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:10]}"

    _, _, _, agent = get_shared_pipeline()
    result: AgentResult = agent.execute(req.query, session_id=session_id)

    retrieval_items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            source=r.source,
            hybrid_score=round(r.hybrid_score, 4),
            semantic_score=round(r.semantic_score, 4),
            keyword_score=round(r.keyword_score, 4),
            text=r.text,
        )
        for r in result.retrieval_results
    ]

    return ChatResponse(
        success=result.success,
        query=req.query,
        route=result.route,
        tool_used=result.tool_used,
        answer=result.answer,
        sources=result.sources,
        retrieval_results=retrieval_items,
        expression=result.expression,
        calculation_result=result.calculation_result,
        error=result.error,
        session_id=session_id,
        contextualized_query=result.contextualized_query,
    )


@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse, tags=["AI Assistant"])
def get_chat_history(session_id: str):
    """Retrieve conversation history messages for a specific session."""
    conv_mgr = get_conversation_manager()
    history = conv_mgr.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        count=len(history),
        messages=[HistoryMessageItem(role=m["role"], content=m["content"]) for m in history],
    )


@app.post("/api/chat/clear", tags=["AI Assistant"])
def clear_chat_history_post(req: ClearChatRequest):
    """Clear conversation history for a given session."""
    conv_mgr = get_conversation_manager()
    conv_mgr.clear_session(req.session_id)
    return {
        "success": True,
        "session_id": req.session_id,
        "message": f"Conversation history for session '{req.session_id}' cleared.",
    }


@app.delete("/api/chat/history/{session_id}", tags=["AI Assistant"])
def clear_chat_history_delete(session_id: str):
    """Clear conversation history for a given session via DELETE."""
    conv_mgr = get_conversation_manager()
    conv_mgr.clear_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "message": f"Conversation history for session '{session_id}' cleared.",
    }


@app.post("/api/calculator", response_model=CalculatorResponse, tags=["Calculator"])
def calculate_expression(req: CalculatorRequest):
    """
    Directly evaluate a mathematical expression using the sandboxed AST calculator.
    """
    try:
        val = calculator_tool(req.expression)
        return CalculatorResponse(expression=req.expression, result=val)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
