from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REFUSAL_MARKERS = (
    "don't have enough information in the knowledge base",
    "do not have enough information in the knowledge base",
    "knowledge base is not configured",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an evaluation report to only keep questions that were answered "
            "from knowledge sources and keep evaluation-relevant fields."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the full evaluation report JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for filtered JSON output. Defaults to <input>_answerable_only.json.",
    )
    parser.add_argument(
        "--min-retrieved-chunks",
        type=int,
        default=1,
        help="Minimum retrieved chunks required to keep a record.",
    )
    parser.add_argument(
        "--include-non-expected-answerable",
        action="store_true",
        help="If set, do not require question_metadata.expected_answerable == true.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_refusal_answer(assistant_text: str) -> bool:
    lowered = (assistant_text or "").strip().lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _is_answered_from_knowledge_source(
    item: dict[str, Any],
    *,
    require_expected_answerable: bool,
    min_retrieved_chunks: int,
) -> bool:
    if item.get("status") != "ok":
        return False

    metadata = item.get("question_metadata") or {}
    if require_expected_answerable and metadata.get("expected_answerable") is not True:
        return False

    retrieval = item.get("retrieval") or {}
    chunks = retrieval.get("retrieved_chunks") or []
    if len(chunks) < min_retrieved_chunks:
        return False

    assistant_text = ((item.get("assistant_message") or {}).get("content") or "").strip()
    if _is_refusal_answer(assistant_text):
        return False

    return True


def _trim_record(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("question_metadata") or {}
    retrieval = item.get("retrieval") or {}
    mcq = item.get("mcq") or {}
    mcq_quiz = mcq.get("quiz") or {}
    mcq_questions = mcq.get("questions") or []

    trimmed_mcq_questions: list[dict[str, Any]] = []
    for q in mcq_questions:
        mcq_options = q.get("mcq_options") or {}
        trimmed_mcq_questions.append(
            {
                "question_text": q.get("question_text"),
                "flashcard_id": q.get("flashcard_id"),
                "options": mcq_options.get("options") or [],
                "correct_label": mcq_options.get("correct_label"),
            }
        )

    return {
        "question_index": item.get("question_index"),
        "guideline": metadata.get("guideline"),
        "user_question": item.get("user_question"),
        "assistant_answer": (item.get("assistant_message") or {}).get("content"),
        "retrieved_chunks": retrieval.get("retrieved_chunks") or [],
        "flashcards": [
            {
                "question": card.get("question"),
                "answer": card.get("answer"),
            }
            for card in (item.get("flashcards") or [])
        ],
        "mcq": {
            "quiz_title": mcq_quiz.get("title"),
            "questions": trimmed_mcq_questions,
        },
    }


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_answerable_only.json")


def main() -> int:
    args = _parse_args()
    if args.min_retrieved_chunks < 0:
        raise ValueError("--min-retrieved-chunks must be >= 0.")

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else _default_output_path(input_path)

    report = _load_json(input_path)
    source_results = report.get("results") or []
    if not isinstance(source_results, list):
        raise ValueError("Input JSON does not contain a valid 'results' list.")

    require_expected_answerable = not args.include_non_expected_answerable
    filtered_records = [
        _trim_record(item)
        for item in source_results
        if _is_answered_from_knowledge_source(
            item,
            require_expected_answerable=require_expected_answerable,
            min_retrieved_chunks=args.min_retrieved_chunks,
        )
    ]

    output_payload = {
        "generated_at_utc": _utc_now_iso(),
        "source_report_path": str(input_path),
        "source_report_generated_at_utc": report.get("generated_at_utc"),
        "user_id": report.get("user_id"),
        "filter_criteria": {
            "status": "ok",
            "require_expected_answerable": require_expected_answerable,
            "min_retrieved_chunks": args.min_retrieved_chunks,
            "exclude_knowledge_base_refusals": True,
        },
        "counts": {
            "source_total_results": len(source_results),
            "kept_results": len(filtered_records),
            "removed_results": len(source_results) - len(filtered_records),
        },
        "results": filtered_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        "Filtering completed. "
        f"Kept {len(filtered_records)} of {len(source_results)} records. "
        f"Output: {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
