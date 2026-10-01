import logging
from typing import Dict, List, Optional
from groq import Groq

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    validate_config,
)

logger = logging.getLogger(__name__)

_client: Optional[Groq] = None


def get_client() -> Groq:
    """Retrieve or initialize the Groq client."""
    global _client
    validate_config()
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_answer(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Generate an answer from the configured Groq LLM model.
    Handles content and reasoning fallbacks gracefully.
    Supports either a single prompt string or structured chat messages.
    """
    client = get_client()

    sys_content = system_prompt or (
        "You are an accurate, helpful, and concise AI assistant. "
        "Adhere strictly to instructions and facts provided."
    )

    if messages is not None:
        chat_messages: List[Dict[str, str]] = []
        if not any(m.get("role") == "system" for m in messages):
            chat_messages.append({"role": "system", "content": sys_content})
        chat_messages.extend(messages)
    else:
        chat_messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt or ""},
        ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=chat_messages,
            temperature=temperature,
        )

        message = response.choices[0].message
        content = message.content or ""
        # Handle reasoning models where output might be in message.reasoning if content is empty
        if not content.strip() and getattr(message, "reasoning", None):
            content = message.reasoning or ""

        return content.strip()

    except Exception as error:
        error_text = str(error)
        logger.error("Groq API error: %s", error_text)

        if "401" in error_text or "invalid_api_key" in error_text:
            raise RuntimeError(
                "Groq authentication failed. Please verify your GROQ_API_KEY in the .env file."
            ) from error

        if "rate_limit" in error_text.lower() or "429" in error_text:
            raise RuntimeError(
                "Groq rate limit exceeded. Please wait a few moments before making another request."
            ) from error

        raise RuntimeError(
            f"Groq API request failed: {error}"
        ) from error
