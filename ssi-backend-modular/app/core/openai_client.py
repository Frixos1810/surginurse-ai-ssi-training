# app/core/openai_client.py
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

T = TypeVar("T", bound=BaseModel)


def _safe_console_print(text: Any) -> None:
    """
    Print without letting console encoding errors break the request flow.
    """
    stream = getattr(sys, "stdout", None)
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        print(str(text))
    except UnicodeEncodeError:
        sanitized = str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(sanitized)


def _normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures each message is in the format:
      {"role": "system"|"user"|"assistant", "content": "..."}
    """
    normalized: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role is None or content is None:
            raise ValueError(f"Invalid message: {m}")
        normalized.append({"role": role, "content": content})
    return normalized


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_text_from_result(result: Any) -> str:
    parts = _get_attr(result, "content", None) or []
    texts: List[str] = []
    for part in parts:
        if isinstance(part, str):
            part_text = part.strip()
            if part_text:
                texts.append(part_text)
            continue
        text = _get_attr(part, "text", "")
        if text and str(text).strip():
            texts.append(str(text).strip())
    return "\n".join(texts).strip()


def _normalize_source_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def list_vector_store_files(
    *,
    vector_store_id: str,
) -> List[Dict[str, Any]]:
    status_filters: List[Optional[str]] = [
        None,
        "in_progress",
        "completed",
        "failed",
        "cancelled",
    ]
    collected_by_file_id: Dict[str, Dict[str, Any]] = {}

    for status_filter in status_filters:
        after_cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            list_kwargs: Dict[str, Any] = {
                "vector_store_id": vector_store_id,
                "limit": 100,
                "order": "desc",
            }
            if status_filter is not None:
                list_kwargs["filter"] = status_filter
            if after_cursor:
                list_kwargs["after"] = after_cursor

            page = client.vector_stores.files.list(**list_kwargs)
            page_items = list(_get_attr(page, "data", None) or [])
            if not page_items:
                break

            for vector_store_file in page_items:
                vector_store_file_id = _get_attr(vector_store_file, "id", None)
                file_id = _get_attr(vector_store_file, "file_id", None) or vector_store_file_id
                if not file_id:
                    continue

                filename = None
                file_status = _get_attr(vector_store_file, "status", None)
                file_error = _get_attr(vector_store_file, "last_error", None)
                usage_bytes = _get_attr(vector_store_file, "usage_bytes", None)
                created_at = _get_attr(vector_store_file, "created_at", None)

                try:
                    file_obj = client.files.retrieve(file_id)
                    filename = _get_attr(file_obj, "filename", None)
                except Exception:
                    filename = None

                collected_by_file_id[str(file_id)] = {
                    "file_id": file_id,
                    "vector_store_file_id": vector_store_file_id,
                    "filename": filename or file_id or "unknown",
                    "status": file_status,
                    "usage_bytes": usage_bytes,
                    "created_at": created_at,
                    "last_error": (
                        {
                            "code": _get_attr(file_error, "code", None),
                            "message": _get_attr(file_error, "message", None),
                        }
                        if file_error is not None
                        else None
                    ),
                }

            if len(page_items) < 100:
                break

            next_cursor = str(_get_attr(page_items[-1], "id", "")).strip()
            if not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            after_cursor = next_cursor

    files = list(collected_by_file_id.values())
    files.sort(
        key=lambda item: (
            -int(item.get("created_at") or 0),
            str(item.get("filename") or "").lower(),
        )
    )
    return files


def list_processed_account_files(
    *,
    purpose: str | None = None,
) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    after_cursor: str | None = None
    seen_cursors: set[str] = set()
    normalized_purpose = str(purpose or "").strip().lower()

    while True:
        list_kwargs: Dict[str, Any] = {
            "limit": 100,
            "order": "desc",
        }
        if after_cursor:
            list_kwargs["after"] = after_cursor

        page = client.files.list(**list_kwargs)
        page_items = list(_get_attr(page, "data", None) or [])
        if not page_items:
            break

        for item in page_items:
            file_id = _get_attr(item, "id", None)
            file_purpose = str(_get_attr(item, "purpose", "")).strip().lower()
            file_status = str(_get_attr(item, "status", "")).strip().lower()
            if not file_id or file_status != "processed":
                continue
            if normalized_purpose and file_purpose != normalized_purpose:
                continue
            files.append(
                {
                    "file_id": file_id,
                    "filename": _get_attr(item, "filename", file_id),
                    "purpose": file_purpose,
                    "status": file_status,
                    "created_at": _get_attr(item, "created_at", None),
                }
            )

        if len(page_items) < 100:
            break

        next_cursor = str(_get_attr(page_items[-1], "id", "")).strip()
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        after_cursor = next_cursor

    return files


def attach_files_to_vector_store(
    *,
    vector_store_id: str,
    file_ids: Sequence[str],
) -> int:
    attached_count = 0
    for file_id in file_ids:
        normalized = str(file_id or "").strip()
        if not normalized:
            continue
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=normalized,
        )
        attached_count += 1
    return attached_count


def build_vector_store_context(
    *,
    query: str,
    vector_store_id: str,
    max_results: int = 6,
    max_chars_per_result: int = 1200,
    source_filter_policy: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], Dict[str, Any]]:
    if not query or not query.strip():
        return [], {"vector_store_id": vector_store_id, "query": query, "sources": []}

    results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=max_results,
    )

    data = _get_attr(results, "data", []) or []
    sources: List[Dict[str, Any]] = []
    context_chunks: List[str] = []

    policy = source_filter_policy or {}
    has_registry_rows = bool(policy.get("has_registry_rows"))
    enabled_refs = {
        _normalize_source_key(v) for v in (policy.get("enabled_refs") or set()) if _normalize_source_key(v)
    }
    verified_refs = {
        _normalize_source_key(v) for v in (policy.get("verified_refs") or set()) if _normalize_source_key(v)
    }
    strict_verified_only = bool(policy.get("strict_verified_only"))
    filtered_out_disabled = 0
    filtered_out_unverified = 0

    selected_rows: List[Dict[str, Any]] = []

    for idx, result in enumerate(data):
        text = _extract_text_from_result(result)
        if not text:
            continue
        if len(text) > max_chars_per_result:
            text = text[:max_chars_per_result].rstrip() + "..."
        file_id = _get_attr(result, "file_id", None)
        filename = (
            _get_attr(result, "filename", None)
            or _get_attr(result, "file_name", None)
            or file_id
            or "unknown"
        )
        candidate_keys = {
            _normalize_source_key(file_id),
            _normalize_source_key(filename),
        }
        candidate_keys.discard("")

        verified_match = False
        if has_registry_rows:
            if not candidate_keys or enabled_refs.isdisjoint(candidate_keys):
                filtered_out_disabled += 1
                continue
            verified_match = bool(candidate_keys & verified_refs)
            if strict_verified_only and not verified_match:
                filtered_out_unverified += 1
                continue

        selected_rows.append(
            {
                "index": idx,
                "verified_match": verified_match,
                "source": {
                    "file_id": file_id,
                    "filename": filename,
                    "score": _get_attr(result, "score", None),
                    "snippet": text,
                },
            }
        )

    # Rank boost verified sources when verification exists and strict mode is off.
    if selected_rows and verified_refs and not strict_verified_only:
        selected_rows.sort(key=lambda row: (not row["verified_match"], row["index"]))

    for row in selected_rows:
        source = row["source"]
        if row["verified_match"]:
            source["verified_match"] = True
        sources.append(source)
        context_chunks.append(f"[{len(context_chunks) + 1}] {source['filename']}\n{source['snippet']}")

    evidence = {
        "vector_store_id": vector_store_id,
        "query": query,
        "search_query": _get_attr(results, "search_query", None),
        "sources": sources,
        "source_filter": {
            "registry_enforced": has_registry_rows,
            "strict_verified_only": strict_verified_only,
            "filtered_out_disabled": filtered_out_disabled,
            "filtered_out_unverified": filtered_out_unverified,
        },
    }
    _safe_console_print("\nRetrieved Chunk Snippets:")
    for src in sources:
        _safe_console_print(f"\n--- {src['filename']} ---")
        _safe_console_print(src["snippet"])

    return context_chunks, evidence


async def generate_chat_reply(
    *,
    messages_for_model: Sequence[Dict[str, Any]],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Plain text generation (non-structured).
    """
    msgs = _normalize_messages(messages_for_model)
    resp = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


async def generate_structured_text(
    *,
    messages_for_model: Sequence[Dict[str, Any]],
    response_model: Type[T],
    model_name: str = "gpt-4o-mini",
) -> T:
    """
    TRUE Structured Outputs:
    Uses OpenAI SDK structured parsing. Model is forced to match `response_model`.
    """
    msgs = _normalize_messages(messages_for_model)

    resp = client.responses.parse(
        model=model_name,
        input=msgs,
        text_format=response_model,
    )

    parsed = resp.output_parsed
    if parsed is None:
        # Covers refusal / incomplete cases.
        raise RuntimeError("No structured output parsed (refusal or incomplete response).")

    return parsed


# Backwards-compatible alias (your code was importing this name)
async def generate_structured_output(
    *,
    messages_for_model: Sequence[Dict[str, Any]],
    response_model: Type[T],
    model_name: str = "gpt-4o-mini",
) -> T:
    return await generate_structured_text(
        messages_for_model=messages_for_model,
        response_model=response_model,
        model_name=model_name,
    )
