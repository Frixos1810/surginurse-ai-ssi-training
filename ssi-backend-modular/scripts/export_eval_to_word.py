from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a filtered evaluation JSON into a nurse-friendly Word (.docx) "
            "evaluation workbook."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to filtered evaluation JSON (answerable-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .docx path. Defaults to <input>_nurse_eval.docx.",
    )
    parser.add_argument(
        "--title",
        default="Nurse Evaluation Workbook - SSI RAG Assistant",
        help="Document title shown in the Word file.",
    )
    return parser.parse_args()


def _xml_text(value: Any) -> str:
    text = str(value if value is not None else "")
    cleaned: list[str] = []
    for ch in text:
        code = ord(ch)
        is_valid = (
            ch in ("\t", "\n", "\r")
            or 0x20 <= code <= 0xD7FF
            or 0xE000 <= code <= 0xFFFD
            or 0x10000 <= code <= 0x10FFFF
        )
        cleaned.append(ch if is_valid else "?")
    return escape("".join(cleaned), quote=False)


class _DocBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add_paragraph(self, text: str = "", *, bold: bool = False) -> None:
        if text is None or text == "":
            self.parts.append("<w:p/>")
            return
        run_props = "<w:rPr><w:b/></w:rPr>" if bold else ""
        self.parts.append(
            "<w:p><w:r>"
            f"{run_props}<w:t xml:space=\"preserve\">{_xml_text(text)}</w:t>"
            "</w:r></w:p>"
        )

    def add_page_break(self) -> None:
        self.parts.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def to_document_xml(self) -> str:
        body_xml = "".join(self.parts)
        section = (
            "<w:sectPr>"
            "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
            "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" "
            "w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
            "</w:sectPr>"
        )
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
            f"<w:body>{body_xml}{section}</w:body>"
            "</w:document>"
        )


def _build_content(builder: _DocBuilder, report: dict[str, Any], title: str) -> None:
    results = report.get("results") or []
    generated = report.get("generated_at_utc", "")
    source_path = report.get("source_report_path", "")
    counts = report.get("counts") or {}

    builder.add_paragraph(title, bold=True)
    builder.add_paragraph(f"Generated UTC: {generated}")
    builder.add_paragraph(f"Source JSON: {source_path}")
    builder.add_paragraph(
        f"Included questions: {counts.get('kept_results', len(results))} "
        f"(from {counts.get('source_total_results', len(results))} source records)"
    )
    builder.add_paragraph()

    builder.add_paragraph("How To Evaluate (A1, A3, A4, B1, D1)", bold=True)
    builder.add_paragraph("A1 Correctness: score each answer as 1 / 0.5 / 0.")
    builder.add_paragraph("A3 Helpfulness: Likert 1-5 (helpful, clear, confidence).")
    builder.add_paragraph("A4 Unsupported claims: unsupported claims / total claims.")
    builder.add_paragraph()
    builder.add_paragraph("B1 Precision@k (nurse steps):", bold=True)
    builder.add_paragraph("1) Read the user question and assistant answer.")
    builder.add_paragraph("2) For each retrieved chunk, mark Relevant or Not relevant to that question.")
    builder.add_paragraph("3) Count how many chunks are Relevant.")
    builder.add_paragraph("4) Compute B1 = Relevant chunks / k (k = total retrieved chunks shown).")
    builder.add_paragraph()
    builder.add_paragraph("D1 MCQ validity rate (no users needed):", bold=True)
    builder.add_paragraph("For each MCQ, check:")
    builder.add_paragraph("- Exactly 4 options.")
    builder.add_paragraph("- Exactly 1 correct option.")
    builder.add_paragraph("- Correct option text matches the flashcard answer.")
    builder.add_paragraph("- No duplicate option texts.")
    builder.add_paragraph("- Labels are valid (A/B/C/D).")
    builder.add_paragraph("D1 per question = MCQs that pass all checks / total MCQs.")
    builder.add_paragraph(
        "This workbook contains answerable questions only, so abstention metrics are not included."
    )
    builder.add_paragraph()

    for idx, item in enumerate(results, start=1):
        if idx > 1:
            builder.add_page_break()

        question_index = item.get("question_index")
        guideline = item.get("guideline") or "N/A"
        user_question = item.get("user_question") or ""
        assistant_answer = item.get("assistant_answer") or ""
        chunks = item.get("retrieved_chunks") or []
        flashcards = item.get("flashcards") or []
        mcq = item.get("mcq") or {}
        mcq_questions = mcq.get("questions") or []

        builder.add_paragraph(f"Question {idx} (Original index: {question_index})", bold=True)
        builder.add_paragraph(f"Guideline: {guideline}")
        builder.add_paragraph(f"User question: {user_question}")
        builder.add_paragraph()

        builder.add_paragraph("Assistant answer", bold=True)
        builder.add_paragraph(assistant_answer)
        builder.add_paragraph()

        builder.add_paragraph(f"Retrieved chunks (k={len(chunks)})", bold=True)
        if not chunks:
            builder.add_paragraph("No retrieved chunks were recorded.")
        else:
            for c_idx, chunk in enumerate(chunks, start=1):
                header = (
                    f"Chunk {c_idx} | File: {chunk.get('filename')} | "
                    f"Score: {chunk.get('score')}"
                )
                builder.add_paragraph(header)
                snippet = chunk.get("snippet") or ""
                for line in str(snippet).splitlines():
                    builder.add_paragraph(f"  {line}")
                builder.add_paragraph("Nurse relevance: [ ] Relevant   [ ] Not relevant")
                builder.add_paragraph()

        builder.add_paragraph("Flashcards generated", bold=True)
        if not flashcards:
            builder.add_paragraph("No flashcards generated.")
        else:
            for f_idx, card in enumerate(flashcards, start=1):
                builder.add_paragraph(f"Flashcard {f_idx} question: {card.get('question')}")
                builder.add_paragraph(f"Flashcard {f_idx} answer: {card.get('answer')}")
                builder.add_paragraph()

        builder.add_paragraph("MCQ generated", bold=True)
        builder.add_paragraph(f"Quiz title: {mcq.get('quiz_title')}")
        if not mcq_questions:
            builder.add_paragraph("No MCQ questions recorded.")
        else:
            builder.add_paragraph("D1 checklist should be completed for each MCQ below.")
            for m_idx, mq in enumerate(mcq_questions, start=1):
                builder.add_paragraph(f"MCQ {m_idx}: {mq.get('question_text')}")
                for option in mq.get("options") or []:
                    builder.add_paragraph(f"  - {option.get('label')}: {option.get('text')}")
                builder.add_paragraph(
                    f"  Correct label (reference): {mq.get('correct_label')}"
                )
                builder.add_paragraph("  D1 checks:")
                builder.add_paragraph("  [ ] 4 unique options")
                builder.add_paragraph("  [ ] 1 correct option only")
                builder.add_paragraph("  [ ] Correct option equals flashcard answer")
                builder.add_paragraph("  [ ] No duplicate option text")
                builder.add_paragraph("  [ ] Labels are well-formed (A/B/C/D)")
                builder.add_paragraph("  MCQ valid overall: [ ] Yes   [ ] No")
                builder.add_paragraph()

        builder.add_paragraph("Nurse scoring form", bold=True)
        builder.add_paragraph("A1 Correctness: [ ] 1   [ ] 0.5   [ ] 0")
        builder.add_paragraph("A3 Helpfulness (1-5): ____")
        builder.add_paragraph("A4 Unsupported claims: unsupported ____ / total claims ____")
        builder.add_paragraph("B1 Precision@k: relevant chunks ____ / k ____ = ____")
        builder.add_paragraph("D1 MCQ validity: valid MCQs ____ / total MCQs ____ = ____")
        builder.add_paragraph("Notes:")
        builder.add_paragraph("____________________________________________________________")
        builder.add_paragraph("____________________________________________________________")


def _build_docx_bytes(document_xml: str, title: str) -> bytes:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    content_types = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/docProps/core.xml\" "
        "ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
        "<Override PartName=\"/docProps/app.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>"
        "</Types>"
    )

    package_rels = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "<Relationship Id=\"rId2\" "
        "Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" "
        "Target=\"docProps/core.xml\"/>"
        "<Relationship Id=\"rId3\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" "
        "Target=\"docProps/app.xml\"/>"
        "</Relationships>"
    )

    document_rels = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"/>"
    )

    core = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties "
        "xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        f"<dc:title>{_xml_text(title)}</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{timestamp}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{timestamp}</dcterms:modified>"
        "</cp:coreProperties>"
    )

    app = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" "
        "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
        "<Application>Codex Script</Application>"
        "</Properties>"
    )

    import io

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", package_rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
    return buffer.getvalue()


def main() -> int:
    args = _parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_nurse_eval.docx")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    report = json.loads(input_path.read_text(encoding="utf-8"))

    builder = _DocBuilder()
    _build_content(builder, report, args.title)
    document_xml = builder.to_document_xml()
    docx_bytes = _build_docx_bytes(document_xml, args.title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(docx_bytes)

    print(f"Word export completed: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
