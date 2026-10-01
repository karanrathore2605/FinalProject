import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_API_URL,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_embedding_model: Optional[SentenceTransformer] = None
_active_model_name: Optional[str] = None
_is_fallback_active: bool = False
_model_dimension: Optional[int] = None


def _get_optimal_device() -> str:
    """Detect whether CUDA is available or default to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _is_qwen_embedding_model(model_name: str) -> bool:
    """Check if model belongs to Qwen/GTE embedding family."""
    lower = model_name.lower()
    return "qwen" in lower or "gte" in lower


def _call_remote_embedding_api(texts: List[str]) -> Optional[np.ndarray]:
    """Call an external OpenAI-compatible or Ollama embedding API if configured."""
    if not EMBEDDING_API_URL:
        return None

    import httpx

    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(EMBEDDING_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if "data" in data:
                # OpenAI-compatible response format
                embeddings = [item["embedding"] for item in data["data"]]
                arr = np.asarray(embeddings, dtype="float32")
                # Normalize
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
    except Exception as exc:
        logger.warning("Remote embedding API call failed: %s. Falling back to local loader.", exc)

    return None


def get_embedding_model() -> SentenceTransformer:
    """
    Load or retrieve the cached SentenceTransformer model.
    Uses sentence-transformers (all-MiniLM-L6-v2) for fast, lightweight dense vector embeddings.
    """
    global _embedding_model, _active_model_name, _is_fallback_active, _model_dimension

    if _embedding_model is not None:
        return _embedding_model

    device = _get_optimal_device()
    target_model = EMBEDDING_MODEL

    logger.info("Attempting to load embedding model: %s (device: %s)", target_model, device)

    try:
        _embedding_model = SentenceTransformer(
            target_model,
            device=device,
        )
        _active_model_name = target_model
        _is_fallback_active = False
        logger.info("Successfully loaded embedding model: %s", target_model)
    except Exception as exc:
        logger.warning(
            "Could not load primary embedding model '%s': %s. "
            "Falling back to '%s' to ensure uninterrupted functionality.",
            target_model,
            exc,
            EMBEDDING_FALLBACK_MODEL,
        )
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_FALLBACK_MODEL, device=device)
            _active_model_name = EMBEDDING_FALLBACK_MODEL
            _is_fallback_active = True
            logger.info("Successfully loaded fallback embedding model: %s", EMBEDDING_FALLBACK_MODEL)
        except Exception as fallback_exc:
            logger.error("Failed to load fallback embedding model: %s", fallback_exc)
            raise fallback_exc

    # Determine embedding dimension
    try:
        sample_emb = _embedding_model.encode(["test"], convert_to_numpy=True)
        _model_dimension = int(sample_emb.shape[1])
    except Exception:
        _model_dimension = None

    return _embedding_model


def get_embedding_info() -> Dict[str, Any]:
    """Retrieve diagnostic information regarding the active embedding model."""
    global _active_model_name, _is_fallback_active, _model_dimension
    return {
        "requested_model": EMBEDDING_MODEL,
        "active_model": _active_model_name or EMBEDDING_MODEL,
        "is_fallback": _is_fallback_active,
        "fallback_model": EMBEDDING_FALLBACK_MODEL,
        "device": _get_optimal_device(),
        "dimension": _model_dimension,
        "remote_api_configured": bool(EMBEDDING_API_URL),
    }


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    Generate normalized float32 embeddings for a batch of texts.
    Supports Qwen3-Embedding query prompting when is_query=True.
    """
    if not texts:
        return np.empty((0, _model_dimension or 384), dtype="float32")

    # Check for external API first
    remote_embeddings = _call_remote_embedding_api(texts)
    if remote_embeddings is not None:
        return remote_embeddings

    model = get_embedding_model()
    is_qwen = _is_qwen_embedding_model(_active_model_name or EMBEDDING_MODEL)

    # Qwen embedding models utilize prompt_name='query' for search queries
    if is_query and is_qwen:
        try:
            embeddings = model.encode(
                texts,
                prompt_name="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype="float32")
        except Exception:
            # Fall back to standard encode if prompt_name is not supported by the module
            pass

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Generate normalized float32 embedding for a single query."""
    return embed_texts([query], is_query=True)
