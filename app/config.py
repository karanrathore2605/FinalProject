import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# Directory configurations
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents")))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))

# Ensure necessary directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

# Conversational Memory Configuration
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# Embedding & Vector Store Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip().strip("\"'")
EMBEDDING_FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL", "all-MiniLM-L6-v2").strip().strip("\"'")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "").strip()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
TOP_K = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.08"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))

# Chunking settings (Recursive Character Text Splitter)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Backend API server configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


def validate_config() -> None:
    """Validate critical environment variables."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            f"GROQ_API_KEY is missing. Please add your Groq API key to:\n{ENV_FILE}"
        )
    if not GROQ_API_KEY.startswith("gsk_"):
        raise RuntimeError(
            "GROQ_API_KEY does not look like a valid Groq API key (expected key starting with 'gsk_')."
        )
