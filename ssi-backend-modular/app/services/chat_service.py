# app/services/chat_service.py
from __future__ import annotations

import json
import re
from typing import List

from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.openai_client import (
    generate_structured_text,
    generate_chat_reply,
    build_vector_store_context,
)
from app.repositories.user_repository import get_user_by_id
from app.repositories.chat_repository import (
    create_chat_session,
    get_chat_session_by_id,
    list_user_chat_sessions,
    update_chat_session_title,
    delete_chat_session,
    list_chat_messages,
    create_message,
)
from app.repositories.flashcard_repository import create_flashcard
from app.services.knowledge_source_service import get_knowledge_source_filter_policy

from app.schemas.chat_schema import (
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionOut,
    MessageCreate,
    MessageOut,
    SendMessageOut,
)


# -----------------------------
# Structured Output Schema
# -----------------------------

class FlashcardCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., description="Flashcard question (front).")
    answer: str = Field(..., description="Flashcard answer (back).")


class AssistantStructuredResponse(BaseModel):
    """
    Professor requirement:
    - Entire assistant answer in ONE field
    - Flashcards in other fields
    """
    model_config = ConfigDict(extra="forbid")

    assistant_text: str = Field(..., description="The full assistant answer to show in chat.")
    flashcards: List[FlashcardCandidate] = Field(
        ...,
        description="Up to 5 flashcards derived from the assistant_text.",
    )


def _system_prompt() -> str:
    return (
        "You are a medical-surgical nursing tutor focused ONLY on surgical site infection (SSI) prevention.\n"
        "Answer ONLY questions that are clearly about SSI prevention in perioperative/med-surg nursing.\n"
        "Allowed scope includes prevention measures pre-op, intra-op, and post-op, aseptic technique, skin prep,\n"
        "antibiotic prophylaxis timing, hair removal, glucose control, normothermia, wound care, bundles, and patient education.\n"
        "If the question is not specifically about SSI prevention (including unrelated medical topics, diagnosis,\n"
        "treatment beyond prevention, or non-nursing topics), refuse briefly and redirect the user to ask an SSI prevention question.\n"
        "When refusing, use 1-2 sentences and do NOT answer the off-topic request.\n"
        "Answer clearly and accurately when in scope.\n"
        "Then generate up to 5 helpful study flashcards based ONLY on your answer.\n"
        "If not suitable, return an empty flashcards list.\n"
    )


def _rag_context_intro_prompt() -> str:
    return (
        "Use ONLY the knowledge base sources below to answer. "
        "If the sources do not contain the answer, say you don't have enough "
        "information in the knowledge base and ask for a more specific question. "
        "Do NOT use outside knowledge."
    )


def _rag_no_context_prompt() -> str:
    return (
        "No knowledge base sources were found for this question. "
        "Tell the user you don't have enough information in the knowledge base "
        "and ask for a more specific question or additional sources."
    )


def _rag_context_prompt(context_chunk: str) -> str:
    return f"Knowledge base source:\n{context_chunk}"


def _clip_flashcards(cards: List[FlashcardCandidate], max_cards: int = 5) -> List[FlashcardCandidate]:
    cleaned: List[FlashcardCandidate] = []
    for c in cards:
        q = (c.question or "").strip()
        a = (c.answer or "").strip()
        if not q or not a:
            continue
        cleaned.append(FlashcardCandidate(question=q, answer=a))
        if len(cleaned) >= max_cards:
            break
    return cleaned


# -----------------------------
# Chat title helpers
# -----------------------------

def _is_default_title(title: str | None) -> bool:
    if title is None:
        return True
    normalized = title.strip().lower()
    return normalized in {"", "new chat", "new conversation", "chat", "untitled"}


def _clean_title(raw: str) -> str:
    title = (raw or "").strip().strip("\"'").strip()
    title = re.sub(r"\s+", " ", title)
    title = title.rstrip(" .:;!,-")
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title


async def _generate_chat_title(user_messages: List[str], assistant_text: str | None) -> str:
    # Use only the first few user messages to form the topic.
    samples = [m for m in user_messages if m.strip()][:3]
    if not samples:
        return ""

    assistant_snippet = (assistant_text or "").strip()
    if len(assistant_snippet) > 280:
        assistant_snippet = assistant_snippet[:277].rstrip() + "..."

    user_blob = "\n".join([f"{i+1}. {m}" for i, m in enumerate(samples)])
    prompt = (
        "Create a concise chat title (3-7 words) summarizing the topic.\n"
        "Use sentence case. No quotes. No punctuation at the end.\n"
        "Return only the title.\n\n"
        f"User messages:\n{user_blob}\n"
    )
    if assistant_snippet:
        prompt += f"\nAssistant reply (for context):\n{assistant_snippet}\n"

    title = await generate_chat_reply(
        messages_for_model=[
            {"role": "system", "content": "You generate short, specific chat titles."},
            {"role": "user", "content": prompt},
        ],
        model_name="gpt-4o-mini",
        temperature=0.2,
    )

    return _clean_title(title)


# -----------------------------
# Guards
# -----------------------------

async def ensure_user_exists(db: AsyncSession, user_id: int):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def ensure_chat_session_exists(db: AsyncSession, chat_id: int):
    chat = await get_chat_session_by_id(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return chat


# -----------------------------
# Chat session services
# -----------------------------

async def create_chat_session_for_user(
    db: AsyncSession,
    user_id: int,
    payload: ChatSessionCreate,
) -> ChatSessionOut:
    await ensure_user_exists(db, user_id)

    model_name = payload.model_name or "gpt-4o-mini"
    chat = await create_chat_session(
        db,
        user_id=user_id,
        title=payload.title,
        model_name=model_name,
    )
    return ChatSessionOut.model_validate(chat, from_attributes=True)


async def list_chats_for_user(
    db: AsyncSession,
    user_id: int,
) -> List[ChatSessionOut]:
    await ensure_user_exists(db, user_id)
    chats = await list_user_chat_sessions(db, user_id)
    return [ChatSessionOut.model_validate(c, from_attributes=True) for c in chats]


def _sanitize_manual_chat_title(title: str) -> str:
    normalized = re.sub(r"\s+", " ", (title or "").strip())
    if not normalized:
        raise HTTPException(status_code=400, detail="Chat title cannot be empty")
    if len(normalized) > 255:
        raise HTTPException(status_code=400, detail="Chat title is too long")
    return normalized


async def rename_chat_session_for_user(
    db: AsyncSession,
    user_id: int,
    chat_id: int,
    payload: ChatSessionUpdate,
) -> ChatSessionOut:
    await ensure_user_exists(db, user_id)
    chat = await ensure_chat_session_exists(db, chat_id)
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to edit this chat session")

    title = _sanitize_manual_chat_title(payload.title)
    updated = await update_chat_session_title(db, chat, title=title)
    return ChatSessionOut.model_validate(updated, from_attributes=True)


async def delete_chat_session_for_user(
    db: AsyncSession,
    user_id: int,
    chat_id: int,
) -> None:
    await ensure_user_exists(db, user_id)
    chat = await ensure_chat_session_exists(db, chat_id)
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this chat session")

    await delete_chat_session(db, chat)


# -----------------------------
# Message services
# -----------------------------

async def list_messages_in_chat(
    db: AsyncSession,
    user_id: int,
    chat_id: int,
) -> List[MessageOut]:
    await ensure_user_exists(db, user_id)
    chat = await ensure_chat_session_exists(db, chat_id)
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to access this chat session")

    msgs = await list_chat_messages(db, chat_id)
    return [MessageOut.model_validate(m, from_attributes=True) for m in msgs]


async def send_message_and_get_reply(
    db: AsyncSession,
    user_id: int,
    chat_id: int,
    payload: MessageCreate,   # <-- NO SendMessageIn in your schemas
) -> SendMessageOut:
    """
    Flow:
    1) Save user message
    2) Build conversation history
    3) Call LLM ONCE with Structured Outputs => {assistant_text, flashcards[]}
    4) Save assistant message
    5) Save up to 5 flashcards linked to assistant message
    6) Return both messages
    """
    await ensure_user_exists(db, user_id)
    chat = await ensure_chat_session_exists(db, chat_id)
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to use this chat session")

    # 1) Save user message (repository signature: no payload=)
    user_msg = await create_message(
        db,
        chat_id=chat_id,
        sender_role="user",
        content=payload.content,
        model_name=None,
        evidence_source=None,
    )

    # 2) History for model
    history = await list_chat_messages(db, chat_id)
    messages_for_model = [{"role": "system", "content": _system_prompt()}]

    vector_store_id = settings.OPENAI_VECTOR_STORE_ID
    rag_context_chunks: List[str] = []
    evidence_payload: dict | None = None
    if vector_store_id:
        try:
            source_filter_policy = await get_knowledge_source_filter_policy(db)
            rag_context_chunks, evidence_payload = build_vector_store_context(
                query=payload.content,
                vector_store_id=vector_store_id,
                max_results=6,
                source_filter_policy=source_filter_policy,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Vector store search failed") from exc

        if rag_context_chunks:
            messages_for_model.append({"role": "system", "content": _rag_context_intro_prompt()})
            for chunk in rag_context_chunks:
                messages_for_model.append({"role": "system", "content": _rag_context_prompt(chunk)})
        else:
            messages_for_model.append({"role": "system", "content": _rag_no_context_prompt()})
    else:
        messages_for_model.append(
            {
                "role": "system",
                "content": (
                    "The knowledge base is not configured (OPENAI_VECTOR_STORE_ID is missing). "
                    "Tell the user to configure it and do not answer from general knowledge."
                ),
            }
        )

    for m in history:
        messages_for_model.append({"role": m.sender_role, "content": m.content})

    # 3) Structured Outputs call (single call)
    structured = await generate_structured_text(
        messages_for_model=messages_for_model,
        response_model=AssistantStructuredResponse,
        model_name=chat.model_name or "gpt-4o-mini",
    )

    assistant_text = (structured.assistant_text or "").strip()
    flashcards = _clip_flashcards(structured.flashcards, max_cards=5)

    # 4) Save assistant message
    evidence_source = None
    if vector_store_id:
        if evidence_payload is None:
            evidence_payload = {
                "vector_store_id": vector_store_id,
                "query": payload.content,
                "sources": [],
            }
        evidence_source = json.dumps(evidence_payload, ensure_ascii=True)

    assistant_msg = await create_message(
        db,
        chat_id=chat_id,
        sender_role="assistant",
        content=assistant_text,
        model_name=chat.model_name or "gpt-4o-mini",
        evidence_source=evidence_source,
    )

    # 4b) Auto-title chat after first exchange (if still default)
    if _is_default_title(chat.title):
        try:
            user_messages = [m.content for m in history if m.sender_role == "user"]
            new_title = await _generate_chat_title(user_messages, assistant_text)
            if new_title:
                chat.title = new_title
                db.add(chat)
                await db.commit()
                await db.refresh(chat)
        except Exception:
            # Title generation should never block the main chat response
            pass

    # 5) Save flashcards linked to assistant message
    for fc in flashcards:
        await create_flashcard(
            db,
            user_id=user_id,
            question=fc.question,
            answer=fc.answer,
            chat_session_id=chat_id,
            source_message_id=assistant_msg.id,
        )

    # 6) Return response
    return SendMessageOut(
        user_message=MessageOut.model_validate(user_msg, from_attributes=True),
        assistant_message=MessageOut.model_validate(assistant_msg, from_attributes=True),
    )
