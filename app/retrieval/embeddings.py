import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedding_model: Optional[SentenceTransformer] = None
_model_dimension: Optional[int] = None


def _get_optimal_device() -> str:
    """Detect whether CUDA is available or default to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sole SentenceTransformer embedding model (all-MiniLM-L6-v2).
    Uses CPU when GPU/CUDA is unavailable.
    """
    global _embedding_model, _model_dimension

    if _embedding_model is not None:
        return _embedding_model

    device = _get_optimal_device()
    target_model = EMBEDDING_MODEL

    try:
        logger.info("Initializing embedding model: %s (device: %s)", target_model, device)
        model = SentenceTransformer(target_model, device=device)

        dim = None
        if hasattr(model, "get_embedding_dimension"):
            dim = model.get_embedding_dimension()
        elif hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
        if dim is None:
            sample_emb = model.encode(["test dimension check"], convert_to_numpy=True)
            dim = int(sample_emb.shape[1])

        _embedding_model = model
        _model_dimension = int(dim)

        logger.info("Embedding model: %s", target_model)
        logger.info("Embedding dimension: %d", _model_dimension)
        return _embedding_model

    except Exception as exc:
        err_msg = (
            f"Embedding model '{target_model}' failed to initialize on device '{device}': {exc}. "
            "Please verify network access or that 'all-MiniLM-L6-v2' is installed correctly."
        )
        logger.exception("Embedding model failed to initialize: %s", exc)
        raise RuntimeError("Embedding model initialization failed") from exc


def get_active_model_name() -> str:
    """Return currently active embedding model name."""
    return EMBEDDING_MODEL


def get_model_dimension() -> int:
    """Return embedding vector dimension dynamically obtained from loaded model."""
    global _model_dimension
    if _model_dimension is not None:
        return _model_dimension
    model = get_embedding_model()
    return _model_dimension or (model.get_sentence_embedding_dimension() if hasattr(model, "get_sentence_embedding_dimension") else 384)


def is_fallback_active() -> bool:
    """Return False since fallback architecture is permanently removed."""
    return False


def get_fallback_reason() -> Optional[str]:
    """Return None since fallback architecture is permanently removed."""
    return None


def get_embedding_info() -> Dict[str, Any]:
    """Retrieve diagnostic information regarding the active embedding model."""
    dim = get_model_dimension()
    return {
        "requested_model": EMBEDDING_MODEL,
        "active_model": EMBEDDING_MODEL,
        "is_fallback": False,
        "device": _get_optimal_device(),
        "dimension": dim,
    }


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    Generate normalized float32 embeddings for a batch of texts using all-MiniLM-L6-v2.
    The exact same model and encoding pipeline is used for documents and queries.
    """
    if not texts:
        dim = get_model_dimension()
        return np.empty((0, dim), dtype="float32")

    try:
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")
    except Exception as exc:
        logger.exception("Embedding generation failed for %d texts", len(texts))
        raise RuntimeError("Embedding generation failed") from exc


def embed_query(query: str) -> np.ndarray:
    """Generate normalized float32 embedding for a single query."""
    return embed_texts([query], is_query=True)
