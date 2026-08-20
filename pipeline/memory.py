"""
pipeline/memory.py
------------------
Chat memory management using ConversationBufferWindowMemory.

Loads past N turns from chat_messages table and injects them into prompt.
This is how the chatbot remembers conversation history.

Window = last N turns only (not entire history) to avoid token overflow.
"""
from typing import Optional
from sqlalchemy.orm import Session
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

from db.models import ChatMessage
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# How many past turns to include in context
# 1 turn = 1 human + 1 AI message
MEMORY_WINDOW_SIZE = 5


def load_memory_from_db(thread_id: str, tenant_id: str, org_unit_id: str, db: Session) -> ConversationBufferWindowMemory:
    """
    Load last N turns from chat_messages table.
    Returns a ConversationBufferWindowMemory with history injected.

    This is called on every request — memory is rebuilt from DB each time.
    This ensures consistency even if server restarts.

    tenant_id + org_unit_id: filtered here even though thread_id alone is
    already effectively unique to one tenant+org_unit (every
    ChatMessage.thread_id points at exactly one ChatThread, which belongs
    to exactly one tenant and department, and the caller already
    validated thread ownership via get_thread() before reaching this
    call). Filtering by both anyway means this query stays correct even
    if that upstream check is ever refactored away.
    """
    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_SIZE,
        return_messages=True,
        memory_key="chat_history",
        input_key="query",
        output_key="answer",
    )

    try:
        # Load last N*2 messages (N turns = N human + N AI)
        messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.thread_id == thread_id,
                ChatMessage.tenant_id == tenant_id,
                ChatMessage.org_unit_id == org_unit_id,
            )
            .order_by(ChatMessage.created_at.asc())
            .limit(MEMORY_WINDOW_SIZE * 2)
            .all()
        )

        # Inject messages into memory in pairs (human, ai)
        i = 0
        while i < len(messages) - 1:
            human_msg = messages[i]
            ai_msg    = messages[i + 1]

            if human_msg.role == "human" and ai_msg.role == "ai":
                memory.save_context(
                    {"query": human_msg.content},
                    {"answer": ai_msg.content},
                )
                i += 2
            else:
                i += 1

        logger.info(
            "Loaded %d messages from thread %s into memory",
            len(messages), str(thread_id)[:8],
        )

    except Exception as e:
        logger.warning("Memory load failed for thread %s: %s", thread_id, e)

    return memory


def build_prompt_with_history(
    query: str,
    memory: ConversationBufferWindowMemory,
    context_chunks: list[dict],
    doc_filter: Optional[str] = None,
) -> str:
    """
    Build the full prompt string:
    - System instructions
    - Retrieved document context
    - Conversation history
    - Current query
    """
    from typing import Optional

    # System message
    system = """You are Avabodh, an intelligent document assistant.
Answer questions based strictly on the provided document context.
Some context comes from images (charts, tables, diagrams, photos) that were
described by GPT-4o Vision — you can see and describe what those images show
just as confidently as text content.
If the answer is not in the context, say clearly: "I don't have enough information in the documents to answer this."
Always be specific, cite which document your answer comes from.
Keep answers concise but complete."""

    # Document context from retrieved chunks — text and image chunks are
    # formatted differently so the LLM knows which parts came from GPT-4o
    # Vision's understanding of an image versus the document's actual text.
    if context_chunks:
        context_parts = []
        for chunk in context_chunks:
            if chunk.get("role") == "image":
                image_type = chunk.get("image_type") or "image"
                caption = chunk.get("image_caption") or chunk["chunk_text"]
                context_parts.append(
                    f"[IMAGE from: {chunk['doc_name']}]\n"
                    f"Image type: {image_type}\n"
                    f"Caption: {caption}"
                )
            else:
                heading_line = (
                    f"\n[Section: {chunk['section_heading']}]" if chunk.get("section_heading") else ""
                )
                context_parts.append(
                    f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}]{heading_line}\n"
                    f"{chunk['chunk_text']}"
                )
        context_str = "\n\n".join(context_parts)
    else:
        context_str = "No relevant document context found."

    # Conversation history from memory
    history_messages = memory.chat_memory.messages
    history_str = ""
    if history_messages:
        history_parts = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                history_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        history_str = "\n".join(history_parts)

    # Build full prompt
    prompt = f"""{system}

DOCUMENT CONTEXT:
{context_str}

CONVERSATION HISTORY:
{history_str if history_str else "No previous conversation."}

CURRENT QUESTION:
{query}

ANSWER:"""

    return prompt
