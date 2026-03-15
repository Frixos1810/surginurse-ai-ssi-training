from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from typing import Any

from dotenv import load_dotenv
from fastapi import HTTPException

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal
from app.repositories.flashcard_repository import list_user_flashcards
from app.schemas.chat_schema import ChatSessionCreate, MessageCreate
from app.schemas.quiz_schema import QuizCreate
from app.services.chat_service import create_chat_session_for_user, send_message_and_get_reply
from app.services.quiz_service import create_auto_mcq_quiz_for_user


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation questions through the existing chat pipeline and export "
            "answers, retrieved snippets, flashcards, and MCQ outputs to JSON."
        )
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="Existing user id to execute the evaluation as.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Path to questions file (.json, .jsonl, or line-based text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file path. Defaults to "
            "'evaluation_report_YYYYMMDD_HHMMSS.json' in the project root."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        help="Model name used when creating chat sessions.",
    )
    parser.add_argument(
        "--chat-title",
        default="Evaluation Session",
        help="Base title for generated chat sessions.",
    )
    parser.add_argument(
        "--reuse-chat",
        action="store_true",
        help="Reuse one chat session for all questions (default: isolate each question).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on the number of questions loaded from file.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any question fails.",
    )
    parser.add_argument(
        "--retrieval-retries",
        type=int,
        default=2,
        help="Number of retry attempts for vector-store retrieval failures per question.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds; retries use exponential backoff.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _normalize_questions_payload(payload: Any) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        if "questions" not in payload:
            raise ValueError("JSON object must contain a 'questions' key.")
        payload = payload["questions"]

    if not isinstance(payload, list):
        raise ValueError("Questions payload must be a list.")

    for item in payload:
        if isinstance(item, str):
            text = item.strip()
            if text:
                questions.append({"question": text})
            continue

        if isinstance(item, dict):
            candidate = item.get("question") or item.get("prompt") or item.get("text")
            if isinstance(candidate, str) and candidate.strip():
                normalized = dict(item)
                normalized["question"] = candidate.strip()
                questions.append(normalized)
                continue

        raise ValueError(
            "Each question item must be either a string or an object containing "
            "'question', 'prompt', or 'text'."
        )

    if not questions:
        raise ValueError("No non-empty questions found.")

    return questions


def _read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        xml_data = zf.read("word/document.xml")

    root = ET.fromstring(xml_data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts: list[str] = []
        for text_node in para.findall(".//w:t", ns):
            if text_node.text:
                parts.append(text_node.text)
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


def _parse_questions_docx(path: Path) -> list[dict[str, Any]]:
    lines = _read_docx_paragraphs(path)
    if not lines:
        raise ValueError("DOCX file does not contain readable paragraph text.")

    question_specs: list[dict[str, Any]] = []
    current_guideline: str | None = None
    current_expected_answerable: bool | None = None
    current_section_label: str | None = None
    question_pattern = re.compile(r"^\s*\d+\s*[.)]?\s*(.+?)\s*$")

    for raw_line in lines:
        line = raw_line.strip()
        normalized = re.sub(r"\s+", " ", line).strip()
        lowered = normalized.lower()

        if "questions from the guidelines" in lowered or "questions included in the guidelines" in lowered:
            current_expected_answerable = True
            current_section_label = normalized
            continue

        if "questions not included in the guidelines" in lowered:
            current_expected_answerable = False
            current_section_label = normalized
            continue

        if (
            (lowered.endswith("guidelines:") or lowered.endswith("guideline:"))
            and not lowered.startswith("questions ")
        ):
            current_guideline = normalized.rstrip(":").strip()
            current_expected_answerable = None
            current_section_label = None
            continue

        match = question_pattern.match(normalized)
        if not match:
            continue

        question_text = match.group(1).strip()
        if not question_text:
            continue
        question_specs.append(
            {
                "question": question_text,
                "guideline": current_guideline,
                "section_label": current_section_label,
                "expected_answerable": current_expected_answerable,
            }
        )

    if not question_specs:
        raise ValueError("No questions detected in DOCX file.")
    return question_specs


def _load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Questions path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix == ".docx":
        return _parse_questions_docx(path)

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError("Questions file is empty.")

    if suffix == ".json":
        return _normalize_questions_payload(json.loads(raw))

    if suffix == ".jsonl":
        questions: list[dict[str, Any]] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, str):
                if obj.strip():
                    questions.append({"question": obj.strip()})
                continue
            if isinstance(obj, dict):
                candidate = obj.get("question") or obj.get("prompt") or obj.get("text")
                if isinstance(candidate, str) and candidate.strip():
                    normalized = dict(obj)
                    normalized["question"] = candidate.strip()
                    questions.append(normalized)
                    continue
            raise ValueError("Invalid JSONL line; expected string or object with question/prompt/text.")

        if not questions:
            raise ValueError("No non-empty questions found in JSONL file.")
        return questions

    questions = [
        {"question": line.strip()}
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not questions:
        raise ValueError("No non-empty questions found in text file.")
    return questions


def _parse_evidence(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "raw_evidence_source": raw}

    if isinstance(parsed, dict):
        return parsed
    return {"raw_evidence_payload": parsed}


def _extract_retrieved_chunks(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    sources = evidence.get("sources")
    if not isinstance(sources, list):
        return []

    chunks: list[dict[str, Any]] = []
    for idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        chunks.append(
            {
                "rank": idx,
                "file_id": source.get("file_id"),
                "filename": source.get("filename"),
                "score": source.get("score"),
                "verified_match": bool(source.get("verified_match", False)),
                "snippet": source.get("snippet"),
            }
        )
    return chunks


def _is_retriable_retrieval_failure(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        if exc.status_code == 500 and "Vector store search failed" in str(exc.detail):
            return True
    message = str(exc)
    return "Vector store search failed" in message


async def _run_evaluation(
    *,
    user_id: int,
    question_specs: list[dict[str, Any]],
    model_name: str,
    chat_title: str,
    reuse_chat: bool,
    stop_on_error: bool,
    retrieval_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    successful_questions = 0
    failed_questions = 0

    async with AsyncSessionLocal() as db:
        shared_chat_id: int | None = None
        if reuse_chat:
            shared_chat = await create_chat_session_for_user(
                db,
                user_id=user_id,
                payload=ChatSessionCreate(title=chat_title, model_name=model_name),
            )
            shared_chat_id = shared_chat.id

        max_attempts = 1 + max(0, retrieval_retries)

        for idx, question_spec in enumerate(question_specs, start=1):
            question = str(question_spec.get("question", "")).strip()
            metadata = {
                key: value
                for key, value in question_spec.items()
                if key != "question"
            }
            retry_events: list[dict[str, Any]] = []
            send_out = None
            used_chat_id: int | None = None

            try:
                for attempt in range(1, max_attempts + 1):
                    if shared_chat_id is not None and attempt == 1:
                        attempt_chat_id = shared_chat_id
                    else:
                        suffix = f" #{idx}" if attempt == 1 else f" #{idx} (retry {attempt - 1})"
                        chat = await create_chat_session_for_user(
                            db,
                            user_id=user_id,
                            payload=ChatSessionCreate(
                                title=f"{chat_title}{suffix}",
                                model_name=model_name,
                            ),
                        )
                        attempt_chat_id = chat.id

                    try:
                        send_out = await send_message_and_get_reply(
                            db,
                            user_id=user_id,
                            chat_id=attempt_chat_id,
                            payload=MessageCreate(content=question),
                        )
                        used_chat_id = attempt_chat_id
                        break
                    except Exception as exc:
                        retriable = _is_retriable_retrieval_failure(exc)
                        event: dict[str, Any] = {
                            "attempt": attempt,
                            "chat_session_id": attempt_chat_id,
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                            "retriable_retrieval_failure": retriable,
                        }

                        if retriable and attempt < max_attempts:
                            backoff_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
                            event["backoff_seconds"] = backoff_seconds
                            retry_events.append(event)
                            await asyncio.sleep(backoff_seconds)
                            continue

                        retry_events.append(event)
                        raise

                if send_out is None or used_chat_id is None:
                    raise RuntimeError("Message send did not produce a response.")

                assistant_message = send_out.assistant_message

                evidence = _parse_evidence(assistant_message.evidence_source)
                retrieved_chunks = _extract_retrieved_chunks(evidence)

                flashcards = await list_user_flashcards(
                    db,
                    user_id=user_id,
                    source_message_id=assistant_message.id,
                )
                flashcards = sorted(flashcards, key=lambda card: card.id)

                flashcards_payload = [
                    {
                        "id": card.id,
                        "question": card.question,
                        "answer": card.answer,
                        "chat_session_id": card.chat_session_id,
                        "source_message_id": card.source_message_id,
                        "created_at": _to_iso(card.created_at),
                    }
                    for card in flashcards
                ]

                mcq_payload: dict[str, Any] | None = None
                mcq_error: dict[str, Any] | None = None
                if flashcards_payload:
                    flashcard_ids = [card["id"] for card in flashcards_payload]
                    try:
                        quiz_detail = await create_auto_mcq_quiz_for_user(
                            db,
                            user_id=user_id,
                            payload=QuizCreate(
                                title=f"Evaluation MCQ #{idx}",
                                flashcard_ids=flashcard_ids,
                            ),
                        )
                        mcq_payload = quiz_detail.model_dump(mode="json")
                    except Exception as exc:
                        mcq_error = {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        }

                results.append(
                    {
                        "status": "ok",
                        "question_index": idx,
                        "chat_session_id": used_chat_id,
                        "attempts": len(retry_events) + 1,
                        "retry_events": retry_events,
                        "user_question": question,
                        "question_metadata": metadata,
                        "user_message": {
                            "id": send_out.user_message.id,
                            "content": send_out.user_message.content,
                            "created_at": _to_iso(send_out.user_message.created_at),
                        },
                        "assistant_message": {
                            "id": assistant_message.id,
                            "content": assistant_message.content,
                            "created_at": _to_iso(assistant_message.created_at),
                            "model_name": assistant_message.model_name,
                        },
                        "retrieval": {
                            "vector_store_id": evidence.get("vector_store_id"),
                            "search_query": evidence.get("search_query"),
                            "source_filter": evidence.get("source_filter"),
                            "retrieved_chunks": retrieved_chunks,
                            "raw_evidence_source": evidence,
                        },
                        "flashcards": flashcards_payload,
                        "mcq": mcq_payload,
                        "mcq_error": mcq_error,
                    }
                )
                successful_questions += 1
            except Exception as exc:
                failed_questions += 1
                results.append(
                    {
                        "status": "error",
                        "question_index": idx,
                        "chat_session_id": used_chat_id,
                        "attempts": len(retry_events) if retry_events else 1,
                        "retry_events": retry_events,
                        "user_question": question,
                        "question_metadata": metadata,
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                )
                if stop_on_error:
                    break

    return {
        "generated_at_utc": _utc_now_iso(),
        "user_id": user_id,
        "total_questions": len(question_specs),
        "successful_questions": successful_questions,
        "failed_questions": failed_questions,
        "reuse_chat": reuse_chat,
        "retrieval_retries": max(0, retrieval_retries),
        "retry_backoff_seconds": retry_backoff_seconds,
        "results": results,
    }


async def _async_main() -> int:
    args = _parse_args()
    question_specs = _load_questions(args.questions_file)
    if args.max_questions is not None:
        if args.max_questions <= 0:
            raise ValueError("--max-questions must be greater than zero.")
        question_specs = question_specs[: args.max_questions]
    if not question_specs:
        raise ValueError("No questions left to evaluate after applying filters.")

    expected_answerable_counts = {
        "true": sum(1 for q in question_specs if q.get("expected_answerable") is True),
        "false": sum(1 for q in question_specs if q.get("expected_answerable") is False),
        "unknown": sum(1 for q in question_specs if q.get("expected_answerable") is None),
    }

    output_path = args.output
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / f"evaluation_report_{stamp}.json"
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    report = await _run_evaluation(
        user_id=args.user_id,
        question_specs=question_specs,
        model_name=args.model_name,
        chat_title=args.chat_title,
        reuse_chat=args.reuse_chat,
        stop_on_error=args.stop_on_error,
        retrieval_retries=args.retrieval_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    report["questions_file"] = str(args.questions_file.resolve())
    report["question_distribution"] = {
        "expected_answerable": expected_answerable_counts
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Evaluation completed. "
        f"Questions: {report['total_questions']}, "
        f"success: {report['successful_questions']}, "
        f"failed: {report['failed_questions']}. "
        f"Output: {output_path}"
    )
    return 0


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
