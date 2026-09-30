"""
AI Study Assistant - Streamlit Frontend Application
Clean, modular entry point orchestrating UI, Services, and Core components.
"""
import logging
import os
from pathlib import Path
import sys
from typing import Optional

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    HYBRID_ALPHA,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import VectorStore
from app.services.decision_layer import AgentResult, DecisionLayer
from app.services.rag_pipeline import RAGPipeline
from app.ui.components import (
    render_chat_history,
    render_sidebar,
    render_suggestion_chips,
)
from app.ui.styles import inject_custom_css

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide",
)

inject_custom_css()


# -----------------------------------------------------------------------------
# APPLICATION SINGLETONS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_shared_vector_store() -> VectorStore:
    """Load and cache the vector store across app reruns."""
    vs = VectorStore()
    vs.load()
    return vs


@st.cache_resource
def get_agent_pipeline():
    """Build and cache the retrieval, RAG pipeline, and intelligent agent."""
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

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
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
render_sidebar(vector_store)


# -----------------------------------------------------------------------------
# MAIN CHAT & QUERY SEARCH HISTORY FEED
# -----------------------------------------------------------------------------
if not st.session_state.history:
    render_suggestion_chips()
else:
    render_chat_history(st.session_state.history)


# -----------------------------------------------------------------------------
# CHAT INPUT & EXECUTION
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask something...")

# Check if a suggestion chip was clicked
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query is not None:
    clean_q = user_query.strip()
    if not clean_q:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching..."):
                res: AgentResult = agent.execute(clean_q)
        except Exception:
            logger.exception("Unexpected error during agent execution")
            res = AgentResult(
                success=False,
                route="ERROR",
                answer="Something went wrong while processing your request. Please try again.",
                tool_used="None",
            )

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

        st.rerun()
