"""
pipeline/chat.py
----------------
Core chat pipeline:
1. Build prompt from memory + retrieved chunks
2. LLM call with streaming support
3. Collect full response after stream
4. Generate thread title (first message only)

Two modes:
- stream=True  → yields tokens one by one via SSE
- stream=False → returns complete response at once
"""

import time
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _build_llm(streaming: bool = False) -> ChatOpenAI:
    if settings.use_ollama:
        if ChatOllama is not None:
            return ChatOllama(
                model=settings.OLLAMA_CHAT_MODEL,
                base_url=settings.ollama_url,
                temperature=0.3,
                max_tokens=1024,
                streaming=streaming,
            )
        return ChatOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_url.rstrip('/')}/v1",
            model=settings.OLLAMA_CHAT_MODEL,
            temperature=0.3,
            max_tokens=1024,
            streaming=streaming,
        )
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MAP_MODEL,
        temperature=0.3,
        max_tokens=1024,
        streaming=streaming,
    )


def generate_thread_title(first_query: str) -> str:
    """
    Auto-generate a short thread title from the first message.
    Called only once per thread — on first message.
    """
    try:
        llm = _build_llm(streaming=False)
        messages = [
            SystemMessage(content="Generate a short 4-6 word title for this conversation. Return only the title, nothing else."),
            HumanMessage(content=first_query),
        ]
        response = llm.invoke(messages)
        title = response.content.strip().strip('"').strip("'")
        return title[:100]
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return first_query[:50]


def chat_complete(prompt: str) -> dict:
    """
    Non-streaming LLM call.
    Returns complete response dict with content and token usage.
    """
    llm = _build_llm(streaming=False)
    start = time.time()

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        elapsed = round(time.time() - start, 2)

        return {
            "content":           response.content.strip(),
            "prompt_tokens":     getattr(response.usage_metadata, "input_tokens", None),
            "completion_tokens": getattr(response.usage_metadata, "output_tokens", None),
            "elapsed_sec":       elapsed,
        }
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise RuntimeError(f"LLM call failed: {e}")


async def chat_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Streaming LLM call.
    Yields tokens one by one as SSE events.
    Also collects full response for storage after stream ends.

    Usage:
        async for token in chat_stream(prompt):
            yield f"data: {token}\n\n"
    """
    llm = _build_llm(streaming=True)
    full_response = []

    try:
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                full_response.append(token)
                yield token

        # Signal stream end
        yield "[DONE]"

    except Exception as e:
        logger.error("Streaming LLM call failed: %s", e)
        yield f"[ERROR]: {e}"

    # Store full response on the generator object for saving to DB later
    chat_stream.last_response = "".join(full_response)
