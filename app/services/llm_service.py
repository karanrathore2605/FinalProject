import logging
from typing import Optional
from groq import Groq

from app.core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    validate_config,
)

import sys

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
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    Generate an answer from the configured Groq LLM model.
    Handles content and reasoning fallbacks gracefully.
    """
    client = get_client()

    sys_content = system_prompt or (
        "You are an accurate, helpful, and concise AI assistant. "
        "Adhere strictly to instructions and facts provided."
    )
    if "tools" not in sys_content.lower() and "function" not in sys_content.lower():
        sys_content = sys_content.rstrip() + " Do not call or invoke any external tools, functions, or repo browsers. Respond in plain text only."

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ],
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
        logger.exception("Groq API error encountered: %s", error_text)

        # Handle specific Groq tool-calling failure:
        # "Tool choice is none, but model called a tool" or repo_browser.open_file
        is_tool_call_error = any(
            pattern in error_text.lower()
            for pattern in [
                "tool_use_failed",
                "model called a tool",
                "tool choice is none",
                "repo_browser",
            ]
        )

        if is_tool_call_error:
            logger.warning(
                "Groq model attempted an invalid tool call (%s). Retrying with strict text-only instruction.",
                error_text,
            )
            try:
                strict_sys = (
                    "CRITICAL: You are a pure text model. You do NOT have any tools, repo browsers, "
                    "or file opening capabilities. Output ONLY plain text directly answering the prompt. "
                    "Do NOT call tools or use XML tags."
                )
                retry_resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": strict_sys},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                retry_msg = retry_resp.choices[0].message
                retry_content = retry_msg.content or ""
                if not retry_content.strip() and getattr(retry_msg, "reasoning", None):
                    retry_content = retry_msg.reasoning or ""
                if retry_content.strip():
                    logger.info("Successfully recovered from Groq invalid tool call via plain-text retry.")
                    return retry_content.strip()
            except Exception as retry_err:
                logger.exception("Retry after invalid tool call also failed: %s", retry_err)

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
