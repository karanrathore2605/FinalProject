import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

import uuid
from app.chunker import TextChunker
from app.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    HYBRID_ALPHA,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.decision_layer import AgentResult, DecisionLayer
from app.document_loader import Document, DocumentLoader
from app.memory import get_conversation_manager
from app.rag_pipeline import RAGPipeline
from app.retriever import HybridRetriever
from app.vector_store import VectorStore

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS matching the requested clean UI style
st.markdown(
    """
    <style>
    /* Chat message spacing & styling */
    [data-testid="stChatMessage"] {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        border: 1px solid #edf0f2;
    }
    .tool-caption {
        color: #718096;
        font-size: 13px;
        margin-bottom: 4px;
        margin-top: 14px;
        font-family: inherit;
    }
    .doc-ready-card {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 12px 14px;
        color: #2e7d32;
        font-weight: 500;
        font-size: 15px;
        margin: 14px 0 10px 0;
    }
    /* Increase font size of the question box (st.chat_input) */
    [data-testid="stChatInput"] textarea {
        font-size: 18px !important;
        line-height: 1.5 !important;
        font-family: inherit !important;
        min-height: 52px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        font-size: 18px !important;
        color: #9ca3af !important;
    }
    .stTextInput input, div[data-testid="stTextInput"] input {
        font-size: 18px !important;
    }
    /* User question text inside the chat feed */
    .user-question-text {
        font-size: 18px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        color: #1a202c !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# APPLICATION SINGLETONS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_shared_vector_store() -> VectorStore:
    vs = VectorStore()
    vs.load()
    return vs


@st.cache_resource
def get_agent_pipeline():
    vs = get_shared_vector_store()
    retriever = HybridRetriever(
        vs,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        alpha=HYBRID_ALPHA,
    )
    rag_pipe = RAGPipeline(retriever)
    decision_agent = DecisionLayer(rag_pipe)
    return vs, retriever, rag_pipe, decision_agent


vector_store, retriever, rag_pipeline, agent = get_agent_pipeline()

# Session State for Document & Chat Search History
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st_session_{uuid.uuid4().hex[:10]}"

if "active_doc_name" not in st.session_state:
    if vector_store.chunks:
        st.session_state.active_doc_name = vector_store.chunks[0].source
    else:
        st.session_state.active_doc_name = None

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("AI Study Assistant")
    st.caption("Upload a TXT or PDF document and ask questions about it.")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        if "session_id" in st.session_state:
            get_conversation_manager().clear_session(st.session_state.session_id)
        st.session_state.session_id = f"st_session_{uuid.uuid4().hex[:10]}"
        st.session_state.history = []
        st.rerun()

    # Upload document widget
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "docx"],
        help="Upload a TXT, PDF, or DOCX document",
    )

    # Optional sample files
    sample_files = []
    if DOCUMENTS_DIR.exists():
        sample_files = [
            f.name for f in DOCUMENTS_DIR.iterdir()
            if f.suffix.lower() in [".txt", ".pdf", ".docx"]
        ]
    selected_sample = None
    if sample_files:
        with st.expander("Or choose a sample document", expanded=False):
            selected_sample = st.selectbox(
                "Sample file:",
                options=["None"] + sample_files,
                index=0,
            )

    # Process Document Button
    if st.button("Process Document", type="primary", use_container_width=True):
        doc: Optional[Document] = None
        try:
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                doc = DocumentLoader.load_from_bytes(file_bytes, uploaded_file.name)
                st.session_state.active_doc_name = uploaded_file.name
            elif selected_sample and selected_sample != "None":
                sample_path = DOCUMENTS_DIR / selected_sample
                doc = DocumentLoader.load_from_path(sample_path)
                st.session_state.active_doc_name = selected_sample
            else:
                st.warning("Please upload a file or choose a sample document.")

            if doc:
                with st.spinner("Indexing document..."):
                    chunks = TextChunker().chunk_document(doc)
                    vector_store.build_from_chunks(chunks)
                    vector_store.save()
                    if "session_id" in st.session_state:
                        get_conversation_manager().clear_session(st.session_state.session_id)
                    st.session_state.session_id = f"st_session_{uuid.uuid4().hex[:10]}"
                    st.session_state.history = []  # fresh search history for new doc
                    st.rerun()
        except Exception as exc:
            st.error(f"Error processing document: {exc}")

    # Document ready status box
    if not vector_store.is_empty():
        doc_display_name = st.session_state.active_doc_name or (
            vector_store.chunks[0].source if vector_store.chunks else "document.pdf"
        )
        st.markdown(
            f"""
            <div class="doc-ready-card">
                Document ready
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"**File:** {doc_display_name}")
        st.write(f"**Chunks:** {vector_store.total_chunks}")

        if st.button("Clear Document", use_container_width=True):
            vector_store.clear()
            st.session_state.active_doc_name = None
            if "session_id" in st.session_state:
                get_conversation_manager().clear_session(st.session_state.session_id)
            st.session_state.session_id = f"st_session_{uuid.uuid4().hex[:10]}"
            st.session_state.history = []
            st.rerun()

    # Clear chat history button if messages exist
    if st.session_state.history:
        if st.button("Clear Search History", use_container_width=True):
            if "session_id" in st.session_state:
                get_conversation_manager().clear_session(st.session_state.session_id)
            st.session_state.session_id = f"st_session_{uuid.uuid4().hex[:10]}"
            st.session_state.history = []
            st.rerun()


# -----------------------------------------------------------------------------
# MAIN CHAT & QUERY SEARCH HISTORY FEED
# -----------------------------------------------------------------------------

# If no search history yet, show friendly suggestions
if not st.session_state.history:
    st.markdown("### Ask questions about your document")
    st.caption("Type any question below, or click a suggestion to start:")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("📝 Explain simple", use_container_width=True):
        st.session_state.pending_query = "Explain this in simple words"
        st.rerun()
    if col2.button("📌 Key points", use_container_width=True):
        st.session_state.pending_query = "Give key points"
        st.rerun()
    if col3.button("🧮 25 * 8", use_container_width=True):
        st.session_state.pending_query = "25 * 8"
        st.rerun()
    if col4.button("📊 Remote math", use_container_width=True):
        st.session_state.pending_query = "The document says there are 500 employees. If 15% work remotely, how many employees work remotely?"
        st.rerun()

# Render all previous query search history on the screen
for turn in st.session_state.history:
    # 1. Tool used label
    tool_label = turn.get("tool_used", "document_search")
    st.markdown(f'<div class="tool-caption">Tool used: {tool_label}</div>', unsafe_allow_html=True)

    # 2. User query bubble
    with st.chat_message("user", avatar="🔴"):
        st.markdown(f'<div class="user-question-text">{turn["query"]}</div>', unsafe_allow_html=True)

    # 3. Assistant response bubble
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(turn["answer"])

        # Collapsible Sources expander
        with st.expander("Sources", expanded=False):
            if turn.get("retrieval_results"):
                for idx, r in enumerate(turn["retrieval_results"], start=1):
                    st.write(f"- **{r.source}** (Chunk {r.chunk_id}) &mdash; *Similarity: {r.hybrid_score:.2f}*")
                    st.caption(f'"{r.text[:280]}..."')
            elif turn.get("sources"):
                for s in turn["sources"]:
                    st.write(f"- {s}")
            else:
                st.caption("No document context used.")

        # Show calculation formula if applicable
        if turn.get("expression") and turn.get("calculation_result") is not None:
            st.caption(f"**Calculation Formula:** `{turn['expression']} = {turn['calculation_result']}`")


# -----------------------------------------------------------------------------
# CHAT INPUT & EXECUTION
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask something...")

# Check if a suggestion chip was clicked
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query and user_query.strip():
    clean_q = user_query.strip()

    # Determine tool and generate answer with session conversational memory
    with st.spinner("Searching..."):
        res: AgentResult = agent.execute(clean_q, session_id=st.session_state.session_id)

    # Append to query search history
    st.session_state.history.append({
        "query": clean_q,
        "answer": res.answer,
        "tool_used": res.tool_used,
        "sources": res.sources,
        "retrieval_results": res.retrieval_results,
        "expression": res.expression,
        "calculation_result": res.calculation_result,
        "route": res.route,
    })

    # Rerun to cleanly update the chronological history feed
    st.rerun()
