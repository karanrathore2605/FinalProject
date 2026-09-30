"""
UI Components module for Streamlit interface.
Provides reusable components for the sidebar, suggestion chips,
and chat message feeds.
"""
import logging
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st

from app.core.config import DOCUMENTS_DIR
from app.loaders.document_loader import Document, DocumentLoader
from app.loaders.chunker import TextChunker
from app.retrieval.embeddings import get_active_model_name

logger = logging.getLogger(__name__)


def render_suggestion_chips() -> None:
    """Render friendly suggestion buttons when chat history is empty."""
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
        st.session_state.pending_query = (
            "The document says there are 500 employees. If 15% work remotely, how many employees work remotely?"
        )
        st.rerun()


def render_chat_turn(turn: dict) -> None:
    """Render a single turn in the chat history feed."""
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
                    source_name = getattr(r, "source", "Unknown")
                    chunk_id = getattr(r, "chunk_id", idx)
                    hybrid_score = getattr(r, "hybrid_score", 0.0)
                    text_snippet = getattr(r, "text", "")[:280]
                    st.write(f"- **{source_name}** (Chunk {chunk_id}) &mdash; *Similarity: {hybrid_score:.2f}*")
                    st.caption(f'"{text_snippet}..."')
            elif turn.get("sources"):
                for s in turn["sources"]:
                    st.write(f"- {s}")
            else:
                st.caption("No document context used.")

        # Show calculation formula if applicable
        if turn.get("expression") and turn.get("calculation_result") is not None:
            calc_val = str(turn["calculation_result"])
            if not calc_val.lower().startswith("error"):
                st.caption(f"**Calculation Formula:** `{turn['expression']} = {calc_val}`")


def render_chat_history(history: List[dict]) -> None:
    """Render all previous query search history turns."""
    for turn in history:
        render_chat_turn(turn)


def render_sidebar(vector_store: Any) -> None:
    """Render the application sidebar for document upload, processing, and management."""
    with st.sidebar:
        st.title("AI Study Assistant")
        st.caption("Upload a TXT, PDF, or DOCX document and ask questions about it.")

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
                    logger.info("[1] Document uploaded: '%s' (%d bytes)", uploaded_file.name, len(file_bytes))
                    logger.info("[2] Text extraction started: '%s'", uploaded_file.name)
                    doc = DocumentLoader.load_from_bytes(file_bytes, uploaded_file.name)
                    logger.info("[3] Text extraction completed: '%s' (%d chars)", uploaded_file.name, len(doc.content))
                    st.session_state.active_doc_name = uploaded_file.name
                elif selected_sample and selected_sample != "None":
                    sample_path = DOCUMENTS_DIR / selected_sample
                    logger.info("[1] Document uploaded (sample): '%s'", selected_sample)
                    logger.info("[2] Text extraction started: '%s'", selected_sample)
                    doc = DocumentLoader.load_from_path(sample_path)
                    logger.info("[3] Text extraction completed: '%s' (%d chars)", selected_sample, len(doc.content))
                    st.session_state.active_doc_name = selected_sample
                else:
                    st.warning("Please upload a file or choose a sample document.")

                if doc:
                    with st.spinner("Indexing document..."):
                        logger.info("[4] Chunking started for '%s'", doc.source)
                        chunks = TextChunker().chunk_document(doc)
                        logger.info("[5] Number of chunks created: %d chunks", len(chunks))

                        # Build index
                        vector_store.build_from_chunks(chunks)
                        vector_store.save()

                        st.session_state.history = []  # fresh search history for new doc
                        st.rerun()
            except ValueError as val_err:
                logger.exception("Validation error processing document: %s", val_err)
                st.error("Could not process the uploaded file. Please ensure it is a valid, readable TXT, PDF, or DOCX document.")
            except Exception:
                logger.exception("Unexpected error processing document")
                st.error("Something went wrong while processing your request. Please try again.")

        # Document ready status box
        if not vector_store.is_empty():
            doc_display_name = st.session_state.active_doc_name or (
                vector_store.chunks[0].source if vector_store.chunks else "document.pdf"
            )
            st.markdown(
                """
                <div class="doc-ready-card">
                    Document ready
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"**File:** {doc_display_name}")
            st.write(f"**Chunks:** {vector_store.total_chunks}")
            st.write(f"**Embedding Model:** `{get_active_model_name()}`")

            if st.button("Clear Document", use_container_width=True):
                vector_store.clear()
                st.session_state.active_doc_name = None
                st.session_state.history = []
                st.rerun()

        # Clear chat history button if messages exist
        if st.session_state.get("history"):
            if st.button("Clear Search History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
