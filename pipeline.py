"""LLM-first Shared-Pass MELT for grounded long-document question answering.

Every cold run uses an LLM to plan all questions, extract shared evidence from
the document, and write all final answers. Python is limited to document
structure, evidence grounding, monotone storage, deterministic calculations,
coverage checks, scheduling, and recovery.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import html
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from config import Settings
from prompts import (
    ANSWER_SYSTEM,
    AUDIT_SYSTEM,
    CENSUS_SYSTEM,
    EXTRACTION_SYSTEMS,
    PLANNING_SYSTEM,
    audit_prompt,
    answer_repair_prompt,
    census_prompt,
    extraction_prompt,
    planning_prompt,
    single_answer_prompt,
)
from records import DocumentRecord, RecordLayout, detect_records
from schemas import (
    AnswerBatch,
    CoverageEntry,
    DocumentChunk,
    GlobalEvidence,
    PlanBatch,
    Question,
    QuestionPlan,
    RunReport,
    ScanDelta,
    ScanPack,
    SharedEvidencePlan,
    SharedRequirement,
    Submission,
    SubmissionAnswer,
)


LOGGER = logging.getLogger("llm_first_melt")
PIPELINE_VERSION = "llm-first-shared-v8.4-phase-aligned-evidence"
ANSWER_VERSION = "llm-grounded-v8.4"
PAGE_RE = re.compile(r"<!--\s*page-start-marker-(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]{2,}|\d+", re.UNICODE)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
T = TypeVar("T", bound=BaseModel)


CATEGORY_TEMPLATES: dict[str, dict[str, Any]] = {
    "aggregation": {
        "operations": ["normalize", "deduplicate", "filter", "count", "list"],
        "required_passes": ["facts"],
        "closure_rules": ["all document packs inspected", "all candidates resolved"],
    },
    "superlative": {
        "operations": ["normalize_units", "compare_all", "select_extreme"],
        "required_passes": ["facts"],
        "closure_rules": ["all document packs inspected", "all comparable values resolved"],
    },
    "absence": {
        "operations": ["topic_matrix", "positive_controls", "absence"],
        "required_passes": ["topics"],
        "closure_rules": ["all document packs inspected", "ambiguous mentions audited"],
    },
    "contradiction": {
        "operations": ["group_claim_slots", "preserve_both_sides", "verify_conflict"],
        "required_passes": ["claims"],
        "closure_rules": ["all document packs inspected", "both claims sourced"],
    },
    "cross_section": {
        "operations": ["join_entities", "verify_relation", "preserve_qualifiers"],
        "required_passes": ["facts", "claims"],
        "closure_rules": ["all document packs inspected", "relations directly sourced"],
    },
    "global_synthesis": {
        "operations": ["order_observations", "group_themes", "preserve_exceptions"],
        "required_passes": ["topics"],
        "closure_rules": ["all document packs inspected", "multiple source regions represented"],
    },
    "needle": {
        "operations": ["rank_lexical_candidates", "verify_direct_quote"],
        "required_passes": ["needle"],
        "closure_rules": ["direct source quote verified"],
    },
}


class PipelineError(RuntimeError):
    pass


class BudgetExceeded(PipelineError):
    pass


@dataclass(frozen=True)
class DocumentInfo:
    path: Path
    text: str
    sha256: str


def stable_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.count("\\n") > 100 and text.count("\n") < 10:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return "".join(
        character
        for character in text
        if character in "\n\t" or unicodedata.category(character) != "Cc"
    )


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PipelineError("PDF input requires pypdf; install requirements.txt") from exc
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:
        raise PipelineError(f"Could not open PDF {path}: {exc}") from exc
    if bool(getattr(reader, "is_encrypted", False)):
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise PipelineError(f"PDF is password-protected: {path}") from exc
        if not unlocked:
            raise PipelineError(f"PDF is password-protected: {path}")
    pages = []
    usable_pages = 0
    usable_characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = _normalize_text(page.extract_text() or "").strip()
        except Exception as exc:
            raise PipelineError(
                f"Could not extract text from PDF page {page_number}: {exc}"
            ) from exc
        usable = len(re.sub(r"\s+", "", page_text))
        usable_characters += usable
        usable_pages += int(usable >= 20)
        pages.append(f"<!-- page-start-marker-{page_number} -->\n\n{page_text}")
    if not pages:
        raise PipelineError(f"PDF contains no pages: {path}")
    if usable_characters < 100 or usable_pages == 0:
        raise PipelineError(
            "PDF has no usable text layer and is likely image-only; apply OCR first"
        )
    if len(pages) >= 10 and usable_pages / len(pages) < 0.10:
        raise PipelineError(
            "PDF text layer is too sparse for reliable full-document coverage; apply OCR first"
        )
    return "\n\n".join(pages)


def normalize_document(path: Path) -> DocumentInfo:
    if not path.is_file():
        raise PipelineError(f"Document file does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        try:
            text = _normalize_text(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise PipelineError(f"Could not read UTF-8 document {path}: {exc}") from exc
    else:
        raise PipelineError("Supported document types are UTF-8 txt/md and searchable PDF")
    if not text.strip():
        raise PipelineError(f"Document is empty: {path}")
    return DocumentInfo(path=path, text=text, sha256=stable_hash(text))


@dataclass
class _Block:
    text: str
    start: int
    end: int
    page: Optional[int]
    headings: list[str]
    unit: Optional[str]


def detect_document_records(document: DocumentInfo) -> RecordLayout:
    """Recover the document's repeating record structure, if it has one."""

    pages = [match.start() for match in PAGE_RE.finditer(document.text)]
    return detect_records(document.text, pages)


def _document_blocks(text: str, layout: Optional[RecordLayout] = None) -> list[_Block]:
    blocks: list[_Block] = []
    page: Optional[int] = None
    heading_stack: dict[int, str] = {}
    unit: Optional[str] = None
    unit_counter = 0
    # A detected record is a real, corroborated census unit. Heading depth is
    # only a fallback: text recovered from a PDF routinely flattens every
    # heading to one level, which would otherwise mint hundreds of spurious
    # "units" out of per-entry sidebar titles.
    records = layout.records if layout and layout.detected else ()
    record_starts = [item.start for item in records]
    pattern = re.compile(r"(?:[^\n].*?(?:\n(?=\n)|\Z)|<!--\s*page-start-marker-\d+\s*-->)", re.S)
    for match in pattern.finditer(text):
        raw = match.group(0).strip()
        if not raw:
            continue
        if records:
            index = bisect.bisect_right(record_starts, match.start()) - 1
            unit = (
                records[index].id
                if index >= 0 and match.start() < records[index].end
                else None
            )
        marker = PAGE_RE.fullmatch(raw)
        if marker:
            page = int(marker.group(1))
            blocks.append(_Block(raw, match.start(), match.end(), page, list(heading_stack.values()), unit))
            continue
        first_line = raw.splitlines()[0].strip()
        heading = HEADING_RE.match(first_line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = {key: value for key, value in heading_stack.items() if key < level}
            heading_stack[level] = title
            if not records and level <= 2:
                unit_counter += 1
                unit = f"section_{unit_counter:04d}"
        blocks.append(
            _Block(raw, match.start(), match.end(), page, list(heading_stack.values()), unit)
        )
    if not blocks:
        blocks.append(_Block(text, 0, len(text), None, [], None))
    return blocks


def chunk_document(
    document: DocumentInfo,
    settings: Settings,
    layout: Optional[RecordLayout] = None,
) -> list[DocumentChunk]:
    blocks = _document_blocks(document.text, layout)
    periodic = bool(layout and layout.detected)
    chunks: list[DocumentChunk] = []
    current: list[_Block] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        text = "\n\n".join(item.text for item in current)
        pages = [item.page for item in current if item.page is not None]
        headings = current[-1].headings or current[0].headings
        units = list(dict.fromkeys(item.unit for item in current if item.unit))
        order = len(chunks)
        chunks.append(
            DocumentChunk(
                id=f"chunk_{order:04d}_{stable_hash(text)[:10]}",
                order=order,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                heading_path=headings,
                primary_units=units,
                record_ids=units if periodic else [],
                text=text,
                source_start=current[0].start,
                source_end=current[-1].end,
            )
        )
        current = []
        current_chars = 0

    for block in blocks:
        if len(block.text) > settings.chunk_max_chars:
            flush()
            start = 0
            while start < len(block.text):
                end = min(start + settings.chunk_max_chars, len(block.text))
                if end < len(block.text):
                    boundary = block.text.rfind(". ", start, end)
                    if boundary > start + settings.chunk_target_chars // 2:
                        end = boundary + 1
                fragment = _Block(
                    block.text[start:end], block.start + start, block.start + end,
                    block.page, block.headings, block.unit,
                )
                current = [fragment]
                current_chars = len(fragment.text)
                flush()
                start = end
            continue
        # A chunk that mixes two records cannot be attributed to either during
        # a census, so a record boundary always closes the current chunk.
        crosses_record = bool(
            periodic and current and current[-1].unit != block.unit
        )
        projected = current_chars + len(block.text) + (2 if current else 0)
        starts_major_heading = (
            not periodic and bool(HEADING_RE.match(block.text.splitlines()[0]))
        )
        if current and (
            crosses_record
            or projected > settings.chunk_max_chars
            or (starts_major_heading and current_chars >= settings.chunk_target_chars * 0.65)
        ):
            flush()
        current.append(block)
        current_chars += len(block.text) + (2 if len(current) > 1 else 0)
        # Splitting inside a record is allowed when the record is very large,
        # but a record that fits stays whole so the census sees it complete.
        if current_chars >= settings.chunk_target_chars and not periodic:
            flush()
        elif periodic and current_chars >= settings.chunk_max_chars:
            flush()
    flush()
    return chunks


def _pack_text(chunks: Sequence[DocumentChunk]) -> str:
    return "\n\n".join(
        f"<<<CHUNK {chunk.id}>>>\n{chunk.text}\n<<<END CHUNK {chunk.id}>>>"
        for chunk in chunks
    )


def _build_pack(prefix: str, order: int, group: Sequence[DocumentChunk]) -> ScanPack:
    text = _pack_text(group)
    pages = [page for item in group for page in (item.page_start, item.page_end) if page]
    return ScanPack(
        id=f"{prefix}_{order:03d}_{stable_hash([item.id for item in group])[:10]}",
        order=order,
        chunk_ids=[item.id for item in group],
        primary_units=list(dict.fromkeys(unit for item in group for unit in item.primary_units)),
        record_ids=list(dict.fromkeys(item for chunk in group for item in chunk.record_ids)),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        text=text,
        character_count=len(text),
        expected_records=len({item for chunk in group for item in chunk.record_ids}),
    )


def make_scan_packs(
    chunks: Sequence[DocumentChunk],
    pass_count: int,
    settings: Settings,
) -> list[ScanPack]:
    """Wide packs for the claims and topics channels, which read for meaning."""

    target = settings.pack_target_chars
    minimum_target = target
    while True:
        groups: list[list[DocumentChunk]] = []
        current: list[DocumentChunk] = []
        size = 0
        for chunk in chunks:
            overhead = len(chunk.id) * 2 + 40
            projected = size + len(chunk.text) + overhead
            if current and projected > target:
                groups.append(current)
                current = []
                size = 0
            current.append(chunk)
            size += len(chunk.text) + overhead
        if current:
            groups.append(current)
        if len(groups) * max(pass_count, 1) <= settings.max_extraction_calls:
            break
        if target >= settings.pack_max_chars:
            # A document simply larger than the call cap is not an error. Packs
            # cannot grow past what the extraction model reads reliably, so the
            # honest response is to report the shortfall and let the run
            # proceed: the total-call cap and the deadline are the real limits,
            # and reading part of a large document beats refusing to start.
            LOGGER.warning(
                "Document needs %d extraction calls per pass at the maximum pack size, "
                "above MAX_EXTRACTION_CALLS=%d. Proceeding — the total-call cap and the "
                "time budget now bound the run, and later packs may go unread.",
                len(groups) * max(pass_count, 1), settings.max_extraction_calls,
            )
            break
        target = min(settings.pack_max_chars, target + 2_000)
        if target == minimum_target:
            LOGGER.warning("Could not increase the scan-pack target; using %d chars", target)
            break
    return [_build_pack("pack", order, group) for order, group in enumerate(groups)]


def _census_groups(
    ordered: Sequence[str],
    by_record: dict[str, list[DocumentChunk]],
    per_pack: int,
    settings: Settings,
) -> list[list[DocumentChunk]]:
    groups: list[list[DocumentChunk]] = []
    current: list[DocumentChunk] = []
    current_size = 0
    current_records = 0

    def rendered_size(chunks: Sequence[DocumentChunk]) -> int:
        return sum(len(item.text) + len(item.id) * 2 + 40 for item in chunks)

    def flush() -> None:
        nonlocal current, current_size, current_records
        if current:
            groups.append(current)
        current = []
        current_size = 0
        current_records = 0

    for record_id in ordered:
        record_chunks = by_record[record_id]
        record_size = rendered_size(record_chunks)
        # Keep normal records intact. An unusually large single record is the
        # only case where its chunks may be split across census calls.
        if record_size > settings.pack_max_chars:
            flush()
            groups.extend(_split_to_pack_limit(record_chunks, settings))
            continue
        if current and (
            current_records >= per_pack
            or current_size + record_size > settings.pack_max_chars
        ):
            flush()
        current.extend(record_chunks)
        current_size += record_size
        current_records += 1
    flush()
    return groups


def _split_to_pack_limit(
    chunks: Sequence[DocumentChunk], settings: Settings
) -> list[list[DocumentChunk]]:
    groups: list[list[DocumentChunk]] = []
    current: list[DocumentChunk] = []
    size = 0
    for chunk in chunks:
        chunk_size = len(chunk.text) + len(chunk.id) * 2 + 40
        if current and size + chunk_size > settings.pack_max_chars:
            groups.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += chunk_size
    if current:
        groups.append(current)
    return groups


def _windowed_census_packs(
    chunks: Sequence[DocumentChunk],
    settings: Settings,
    reserved_calls: int = 0,
) -> list[ScanPack]:
    """The census unit for a document with no repeating structure.

    A record gives the census a natural small unit, and that smallness is what
    makes it exhaustive: asked to enumerate everything inside a wide pack, the
    extraction model writes a few good rows and stops. A document with no
    records used to fall back to one pass of full-width packs, which discards
    exactly that property — and with it the aggregation, superlative and
    synthesis questions that need every figure, not a representative sample.

    There is no structure to align to here, so the unit is a fixed window: as
    narrow as the leftover extraction budget affords, never wider than a
    reading pack. It buys the same exhaustiveness by the same mechanism,
    without assuming anything about the document's shape.
    """

    if not chunks:
        return []

    total = sum(len(chunk.text) for chunk in chunks)
    budget = max(1, settings.max_extraction_calls - max(0, reserved_calls))
    # Narrow enough that nothing gets skipped silently, wide enough that the
    # window still carries the context a figure needs to be interpretable.
    floor_width = max(4000, settings.pack_target_chars // 6)
    width = max(floor_width, -(-total // budget))
    width = min(width, settings.pack_target_chars)

    groups: list[list[DocumentChunk]] = []
    current: list[DocumentChunk] = []
    size = 0
    for chunk in chunks:
        if current and size + len(chunk.text) > width:
            groups.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += len(chunk.text)
    if current:
        groups.append(current)

    LOGGER.info(
        "No record structure: censusing %d windows of about %d characters "
        "(%d extraction calls affordable)",
        len(groups), width, budget,
    )
    return [_build_pack("census", order, group) for order, group in enumerate(groups)]


def make_census_packs(
    chunks: Sequence[DocumentChunk],
    layout: Optional[RecordLayout],
    settings: Settings,
    reserved_calls: int = 0,
) -> list[ScanPack]:
    """Narrow packs for the facts channel, which performs an exhaustive census.

    A small model asked to enumerate every profiled unit inside a large pack
    writes a few good rows and stops. Bounding each pack to a handful of whole
    records keeps the unit of work small enough that no unit can be skipped
    silently, and gives the coverage check an exact denominator.

    One record per call is the ideal, but a document with hundreds of short
    records would spend the entire call budget on the census alone. When that
    happens the grouping widens until the census fits the budget left over
    after the other channels, rather than silently overrunning it.
    """

    if not (layout and layout.detected):
        return _windowed_census_packs(chunks, settings, reserved_calls)
    by_record: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.record_ids:
            by_record[chunk.record_ids[0]].append(chunk)
    ordered = [record.id for record in layout.records if record.id in by_record]
    # Front matter, indexes and appendices sit outside the phase-aligned record
    # spans. They remain available to wide shared and question-focus passes,
    # but they are not census members: asking the record census to invent a
    # record name for an index created false population members and ownership
    # edges. Only detected records determine the census denominator.
    # The census keeps at least half the extraction budget. Subtracting an
    # oversized reservation for the reading passes would otherwise collapse it
    # to one pack holding every record, which is the failure it exists to
    # prevent.
    floor = max(1, settings.max_extraction_calls // 2)
    affordable = max(floor, settings.max_extraction_calls - max(0, reserved_calls))
    per_pack = max(1, settings.census_records_per_pack)
    groups = _census_groups(ordered, by_record, per_pack, settings)
    while len(groups) > affordable and per_pack < len(ordered):
        per_pack += 1
        groups = _census_groups(ordered, by_record, per_pack, settings)
    if len(groups) > affordable:
        # More records than calls is a large document, not a broken one. The
        # grouping is already as wide as the pack limit allows, so the run
        # proceeds and the deadline decides how much of it gets read.
        LOGGER.warning(
            "Census needs %d calls against %d affordable: proceeding at %d records "
            "per pack, and later packs may go unread.",
            len(groups), affordable, per_pack,
        )
    if per_pack != max(1, settings.census_records_per_pack):
        LOGGER.warning(
            "Census widened to %d records per pack: %d records would not fit %d "
            "affordable extraction calls at the configured grouping",
            per_pack, len(ordered), affordable,
        )
    return [_build_pack("census", order, group) for order, group in enumerate(groups)]


def make_transposed_packs(
    document: DocumentInfo,
    chunks: Sequence[DocumentChunk],
    layout: Optional[RecordLayout],
    settings: Settings,
) -> list[ScanPack]:
    """Packs that gather one repeated section across every record.

    The per-record census reads one entry at a time, so whether a given field
    comes back is decided independently for each entry — and measurably, each
    run drops a different handful. Reading the statistics section of every
    entry *together* removes that: the figures are all in one response, so an
    entry cannot be skipped without it being visible in the same view.

    This supplements the per-record census rather than replacing it. A marker
    that appears inline instead of as its own block is invisible here, and only
    the per-record pass covers those entries.
    """

    if not (layout and layout.detected):
        return []
    slots = layout.slots(document.text)
    if not slots:
        return []
    slot = slots[0]
    # A section that is mostly prose is not a statistics block, and gathering
    # it would just rebuild the document in a different order.
    if slot.digit_density < 0.02 or slot.count < max(4, layout.count // 3):
        return []
    by_offset = sorted(chunks, key=lambda item: item.source_start)
    starts = [item.source_start for item in by_offset]

    def chunk_for(offset: int) -> Optional[DocumentChunk]:
        index = bisect.bisect_right(starts, offset) - 1
        return by_offset[index] if index >= 0 else None

    # A gathered statistics block names its own fields but usually not the entry
    # it belongs to — "Highest point: <a summit> (m)" says nothing about which
    # entry that summit is in. Without it the model builds a name out of the
    # component, and one entry comes back under several names, which then
    # counts as several entries. The prose immediately preceding the block
    # names the entry far more reliably than the record boundary does, because
    # a detected boundary sits on a marker rather than on the entry's heading.
    marker_labels = {_norm(marker.label) for marker in layout.markers}

    def context_before(offset: int) -> str:
        """Recover likely entry headings without naming a document schema."""

        record = layout.record_at(offset)
        if record is not None:
            # The repetition detector is phase-aligned to the record's actual
            # opening page. Preserve that opening beside every transposed slot
            # so a compact statistics view cannot attach a component or value
            # to the adjacent entry. This is stronger than guessing from the
            # last few headings, which are usually sidebar titles.
            opening = document.text[
                record.start:min(record.end, record.start + 1_600)
            ]
            opening = PAGE_RE.sub(" ", opening)
            opening = re.sub(r"\s+", " ", opening).strip()
            if opening:
                return opening[:1_200]

        floor = max(0, offset - 24_000)
        window = document.text[floor:offset]
        headings = []
        for match in re.finditer(r"(?m)^#{1,6}\s+([^\n]{2,120})$", window):
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" #")
            normalized = _norm(candidate)
            if (
                normalized
                and normalized not in marker_labels
                and not candidate.isdigit()
                and any(character.isalpha() for character in candidate)
            ):
                headings.append(candidate)
        if headings:
            # A title can wrap or be followed by one non-repeating subheading.
            # Supplying several bounded candidates lets the extractor copy a
            # source name instead of manufacturing one from a component value.
            return " … ".join(dict.fromkeys(headings[-4:]))[:700]
        nearby = PAGE_RE.sub(" ", document.text[max(0, offset - 900) : offset])
        lines = [line.strip(" #") for line in nearby.split("\n")]
        lines = [
            line for line in lines
            if len(line) > 12 and not line.isdigit() and any(ch.isupper() for ch in line)
        ]
        return " … ".join(lines[-2:])[:320]

    # Bounded by block count as well as characters: the value of this view is
    # that one response holds many entries, and a response that runs past the
    # output ceiling loses the tail entries — the very thing it exists to stop.
    per_pack = max(1, settings.transposed_records_per_pack)
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for start, end in slot.pieces:
        holder = chunk_for(start)
        if holder is None:
            continue
        body = document.text[start:end].strip()
        if not body:
            continue
        opening = context_before(start)
        if opening:
            body = f"[entry context: {opening}]\n{body}"
        rendered_size = len(body) + len(holder.id) * 2 + 48
        if current and (
            len(current) >= per_pack
            or size + rendered_size > settings.pack_target_chars
        ):
            groups.append(current)
            current = []
            size = 0
        current.append((holder.id, body))
        size += rendered_size
    if current:
        groups.append(current)

    packs = []
    for order, group in enumerate(groups):
        text = "\n\n".join(
            f"<<<CHUNK {chunk_id}>>>\n{body}\n<<<END CHUNK {chunk_id}>>>"
            for chunk_id, body in group
        )
        packs.append(
            ScanPack(
                id=f"slot_{order:03d}_{stable_hash([chunk_id for chunk_id, _ in group])[:10]}",
                order=order,
                chunk_ids=list(dict.fromkeys(chunk_id for chunk_id, _ in group)),
                primary_units=[],
                record_ids=[],
                page_start=None,
                page_end=None,
                text=text,
                character_count=len(text),
                expected_records=len(group),
            )
        )
    LOGGER.info(
        "Transposed view: slot %r gathers %d sections into %d pack(s) (%d chars, density %.3f)",
        slot.label, slot.count, len(packs), slot.characters, slot.digit_density,
    )
    return packs


def packs_by_pass(
    passes: Sequence[str],
    scan_packs: Sequence[ScanPack],
    census_packs: Sequence[ScanPack],
) -> dict[str, list[ScanPack]]:
    """Route each shared channel to the pack shape its job actually needs."""

    return {
        name: list(census_packs if name == "facts" else scan_packs)
        for name in passes
    }


def load_questions(path: Path) -> list[Question]:
    if not path.is_file():
        raise PipelineError(f"Questions file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"Could not read questions YAML: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("questions", raw.get("items", raw))
    if isinstance(raw, dict):
        raw = [{"id": key, "question": value} for key, value in raw.items()]
    if not isinstance(raw, list):
        raise PipelineError("Questions YAML must contain a list")
    # Deliberately destroy evaluation metadata at the input boundary. YAML
    # comments are already gone after safe_load; this projection also drops
    # labels, rubrics, expected answers, and arbitrary annotations before a
    # Question object, prompt, hash, or cache key can observe them.
    safe_items = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PipelineError(
                f"Invalid question record at index {index}: expected a mapping"
            )
        safe_items.append({"id": item.get("id"), "question": item.get("question")})
    try:
        questions = [Question.model_validate(item) for item in safe_items]
    except ValidationError as exc:
        raise PipelineError(f"Invalid question record: {exc}") from exc
    ids = [item.id for item in questions]
    if len(ids) != len(set(ids)):
        raise PipelineError("Question IDs must be unique")
    return questions


def questions_hash(questions: Sequence[Question]) -> str:
    # Only the exact question contract influences caches and derived state.
    return stable_hash(
        {
            "pipeline_version": PIPELINE_VERSION,
            "questions": [
                {"id": item.id, "question": item.question}
                for item in questions
            ],
        }
    )


def _category(question: Question) -> str:
    """Infer the execution template from question wording alone."""

    text = _norm(question.question)
    contradiction_cues = (
        "contradict", "inconsisten", "conflicting claim", "disagree", "mismatch",
        "discrepancy", "different units", "defined differently", "contrary to",
        "but gives another", "but table", "but its note", "one of those did not",
        "two measurement bases",
    )
    if any(term in text for term in contradiction_cues) or re.search(
        r"\bbut\b.{0,100}\b(?:larger|smaller|higher|lower|different|differently)\b",
        text,
    ):
        return "contradiction"
    if re.search(r"\b(?:never|absent|nowhere)\b", text) or any(
        term in text for term in (
            "not mentioned", "not substantively", "which one is not",
            "which is not substantively",
        )
    ):
        return "absence"
    if any(term in text for term in (
        "comparing the", "compare the", "reconcile its statement",
        "connect historically", "far apart", "two entries", "two park entries",
        "several different senses", "formally paired", "international border",
        "how far apart", "between the opening", "with box", "from different parts",
    )):
        return "cross_section"
    if any(term in text for term in (
        "taking the book as a whole", "taking the document as a whole",
        "recurs throughout", "recurring story", "recurring pattern",
        "across its entries", "across the different", "economy wide pattern",
        "across several sections", "aggregate conditions coexist",
        "as a whole how", "what work does it do", "how does the book handle",
    )):
        return "global_synthesis"
    if any(term in text for term in (
        "largest", "smallest", "changed the most", "most visited",
        "greatest", "maximum", "minimum",
    )):
        return "superlative"
    if any(term in text for term in ("how many", "number of", "counting", "count of")):
        return "aggregation"
    if re.search(
        r"\b(?:at least|at most|more than|less than|or more|or fewer|above|below)\b|(?:>=|<=)",
        text,
    ) and any(term in text for term in ("which", "list", "entries", "items", "figures")):
        return "aggregation"
    if any(term in text for term in ("highest", "lowest")):
        return "superlative"
    if any(term in text for term in (
        "across", "which entries", "which profiles", "which parks are",
        "which figures are", "what are the two", "how does the report connect",
    )):
        return "cross_section"
    return "needle"


def enabled_passes(questions: Sequence[Question]) -> list[str]:
    passes = []
    for question in questions:
        for pass_name in CATEGORY_TEMPLATES[_category(question)]["required_passes"]:
            if pass_name != "needle" and pass_name not in passes:
                passes.append(pass_name)
    # An aggregation question also wants any total the document states about the
    # collection itself, which lives in prose the per-record census never reads.
    # That is a claim, so the claims channel has to run even when no
    # contradiction or cross-section question would otherwise enable it.
    if "claims" not in passes and any(
        _category(item) == "aggregation" for item in questions
    ):
        passes.append("claims")
    return passes


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _explicit_alternatives(question: str) -> list[str]:
    """Recover an explicitly enumerated alternative set without knowing its domain."""

    candidates: list[str] = []
    dash_parts = re.findall(r"(?:—|--)(.+?)(?:—|--)", question)
    if dash_parts:
        candidates.append(dash_parts[0])
    patterns = (
        r"\bof\s+(?:these|the following)\s+(.+?),?\s+which\b",
        r"\bamong\s+(?:these|the following)\s+(.+?),?\s+which\b",
        r"\b(?:these|the following)\s+(.+?),?\s+which\b",
        r"\bof\s+(.+?),?\s+which(?:\s+one)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.I | re.S)
        if match:
            candidates.append(match.group(1))
    for candidate in candidates:
        candidate = re.sub(r"^[^:]{0,80}:\s*", "", candidate.strip())
        # Commas/semicolons are the list boundaries.  Splitting every "and"
        # incorrectly turns one alternative such as "wartime damage and
        # military history" into two alternatives.  Only a final conjunction
        # without an Oxford comma is treated as a boundary.
        parts = [item.strip() for item in re.split(r"\s*,\s*|\s*;\s*", candidate)]
        if len(parts) >= 2 and re.search(r"\s+(?:and|or)\s+", parts[-1], re.I):
            tail = re.split(r"\s+(?:and|or)\s+", parts[-1], maxsplit=1, flags=re.I)
            if len(tail) == 2 and all(len(item.strip()) >= 2 for item in tail):
                parts[-1:] = tail
        parts = [
            re.sub(r"^(?:and|or)\s+", "", item.strip(), flags=re.I)
            for item in parts
        ]
        parts = [item.strip(" .?—-") for item in parts if len(item.strip(" .?—-")) >= 2]
        if 2 <= len(parts) <= 12:
            return list(dict.fromkeys(parts))
    return []


def _numeric_condition(question: str) -> Optional[str]:
    """Normalize common question thresholds into a reducer-friendly condition."""

    text = question.replace("≥", ">=").replace("≤", "<=")
    patterns = (
        (r"\b(?:at least|no less than)\s+([-+]?\d[\d,]*(?:\.\d+)?)\s*([\w²^ -]{0,20})", ">="),
        (r"\b(?:more than|over|above)\s+([-+]?\d[\d,]*(?:\.\d+)?)\s*([\w²^ -]{0,20})", ">"),
        (r"\b([-+]?\d[\d,]*(?:\.\d+)?)\s*([\w²^ -]{0,20}?)\s+or more\b", ">="),
        (r"\b(?:at most|no more than)\s+([-+]?\d[\d,]*(?:\.\d+)?)\s*([\w²^ -]{0,20})", "<="),
        (r"\b(?:less than|under|below)\s+([-+]?\d[\d,]*(?:\.\d+)?)\s*([\w²^ -]{0,20})", "<"),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            unit = re.sub(r"\s+", " ", match.group(2)).strip(" ,.?-")
            return f"value {operator} {match.group(1)}" + (f" {unit}" if unit else "")
    symbol = re.search(r"(>=|<=|>|<)\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return f"value {symbol.group(1)} {symbol.group(2)}" if symbol else None


def fallback_question_plan(question: Question) -> QuestionPlan:
    category = _category(question)
    template = CATEGORY_TEMPLATES[category]
    terms = [
        token for token in TOKEN_RE.findall(question.question)
        if len(token) >= 4 and _norm(token) not in {
            "which", "what", "where", "when", "whose", "does", "from", "with",
            "about", "book", "document", "entry", "entries", "profile", "profiles",
        }
    ]
    topics = terms[:8] if category in {"absence", "global_synthesis"} else []
    if category == "absence":
        topics = _explicit_alternatives(question.question) or topics
    return QuestionPlan(
        question_id=question.id,
        query_type=category,
        target_unit="document-defined entity",
        target_entities=terms[:8],
        required_fields=terms[:6] if category in {"aggregation", "superlative"} else [],
        candidate_topics=topics,
        claim_slots=terms[:6] if category == "contradiction" else [],
        relation_types=terms[:6] if category == "cross_section" else [],
        aliases={"question_terms": terms[:12]},
        condition=_numeric_condition(question.question),
        operations=list(template["operations"]),
        required_passes=list(template["required_passes"]),
        closure_rules=list(template["closure_rules"]),
    )


def normalize_plan_batch(value: Any, questions: Sequence[Question]) -> dict[str, Any]:
    raw_plans = value.get("plans", value.get("questions", [])) if isinstance(value, dict) else value
    if isinstance(raw_plans, dict):
        raw_plans = [dict(item, question_id=key) if isinstance(item, dict) else {} for key, item in raw_plans.items()]
    by_id = {
        str(item.get("question_id", item.get("id", ""))): item
        for item in raw_plans if isinstance(item, dict)
    } if isinstance(raw_plans, list) else {}
    plans = []
    for question in questions:
        raw = by_id.get(question.id)
        if raw is None:
            raise ValueError(f"planner omitted question {question.id}")
        category = _category(question)
        template = CATEGORY_TEMPLATES[category]
        aliases_raw = raw.get("aliases", {})
        if isinstance(aliases_raw, list):
            aliases = {"question_terms": _strings(aliases_raw)}
        elif isinstance(aliases_raw, dict):
            aliases = {str(key): _strings(item) for key, item in aliases_raw.items()}
        else:
            aliases = {}
        topics = _strings(raw.get("candidate_topics"))
        explicit_alternatives = _explicit_alternatives(question.question)
        if category == "absence" and explicit_alternatives:
            topics = explicit_alternatives
            for topic in explicit_alternatives:
                aliases.setdefault(topic, [topic])
        condition = None if raw.get("condition") in (None, "", "null") else str(raw.get("condition"))
        if condition:
            condition = condition.replace("≥", ">=").replace("≤", "<=")
        condition = condition or _numeric_condition(question.question)
        plans.append(
            {
                "question_id": question.id,
                "query_type": category,
                "target_unit": str(raw.get("target_unit") or "document-defined entity"),
                "target_entities": _strings(raw.get("target_entities")),
                "required_fields": _strings(raw.get("required_fields")),
                "candidate_topics": topics,
                "claim_slots": _strings(raw.get("claim_slots")),
                "relation_types": _strings(raw.get("relation_types")),
                "aliases": aliases,
                "condition": condition,
                # Fixed templates cannot be weakened by the model.
                "operations": list(dict.fromkeys(template["operations"] + _strings(raw.get("operations")))),
                "required_passes": list(template["required_passes"]),
                "closure_rules": list(dict.fromkeys(template["closure_rules"] + _strings(raw.get("closure_rules")))),
            }
        )
    return {"plans": plans}


def build_shared_requirements(plans: Sequence[QuestionPlan]) -> list[SharedRequirement]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    def add(kind: str, description: str, aliases: Sequence[str], consumer: str) -> None:
        canonical = _norm(description)
        key = (kind, canonical)
        record = merged.setdefault(
            key,
            {
                "key": f"{kind}:{stable_hash(canonical)[:12]}",
                "kind": kind,
                "description": description,
                "aliases": [],
                "consumer_question_ids": [],
            },
        )
        record["aliases"] = list(dict.fromkeys(record["aliases"] + [item for item in aliases if item]))[:30]
        if consumer not in record["consumer_question_ids"]:
            record["consumer_question_ids"].append(consumer)

    def focused_aliases(plan: QuestionPlan, description: str) -> list[str]:
        """Attach only aliases for this field instead of every alias in the plan."""

        description_tokens = set(_norm(description).split())
        selected = [description]
        for canonical, variants in plan.aliases.items():
            canonical_tokens = set(_norm(canonical).split())
            if description_tokens & canonical_tokens or _norm(canonical) in _norm(description):
                selected.extend([canonical, *variants])
        return list(dict.fromkeys(item for item in selected if item))[:16]

    for plan in plans:
        if "facts" in plan.required_passes:
            fields = plan.required_fields or plan.target_entities or [plan.target_unit]
            for field in fields:
                add("fact", field, focused_aliases(plan, field), plan.question_id)
        if plan.query_type == "contradiction":
            # One broad acquisition key can emit many atomic rows. Creating a
            # separate key for every planner phrase made the model duplicate
            # the same quote under five labels, exhausting the per-pack output
            # cap before the remote counterclaim appeared.
            parts = list(dict.fromkeys([
                plan.target_unit,
                *plan.target_entities,
                *plan.claim_slots,
                *plan.required_fields,
            ]))
            description = "all explicit sides of the requested contradiction: " + "; ".join(parts[:12])
            aliases = [
                "largest", "biggest", "highest", "lowest", "longest", "oldest",
                "the most", "second highest", "different unit", "reported figure",
                *parts,
            ]
            add("claim", description, list(dict.fromkeys(aliases)), plan.question_id)
        if plan.query_type == "aggregation":
            # A document that catalogues a collection usually states the
            # collection's own size somewhere in its front matter or summary.
            # Deriving the count by enumeration is the fallback, not the only
            # route: a census that misses entries silently undercounts, while
            # the document's own declaration is a single grounded quote.
            description = f"explicit stated total number of {plan.target_unit} the document covers"
            add(
                "claim",
                description,
                list(dict.fromkeys([
                    plan.target_unit,
                    f"total {plan.target_unit}",
                    f"number of {plan.target_unit}",
                    *plan.target_entities[:4],
                ])),
                plan.question_id,
            )
        if "claims" in plan.required_passes:
            if plan.query_type == "cross_section" and plan.claim_slots:
                description = (
                    "all explicit cross-section claims requested: "
                    + "; ".join(plan.claim_slots[:12])
                )
                aliases = list(dict.fromkeys([
                    *plan.claim_slots,
                    *(
                        alias for slot in plan.claim_slots
                        for alias in focused_aliases(plan, slot)
                    ),
                ]))
                add("claim", description, aliases, plan.question_id)
            elif plan.query_type != "contradiction":
                for slot in plan.claim_slots:
                    add("claim", slot, focused_aliases(plan, slot), plan.question_id)
            for relation in plan.relation_types:
                add("relation", relation, focused_aliases(plan, relation), plan.question_id)
            if not plan.claim_slots and not plan.relation_types:
                description = "explicit relation: " + "; ".join(plan.target_entities[:6])
                add("relation", description, focused_aliases(plan, description), plan.question_id)
        if "topics" in plan.required_passes:
            kind = "theme" if plan.query_type == "global_synthesis" else "topic"
            topics = plan.candidate_topics or plan.target_entities
            if plan.query_type == "global_synthesis":
                # Synthesis must acquire not only named subjects but also the
                # narrative relation/arc requested by the question.  Planner
                # alias keys encode those concepts without introducing any
                # document- or benchmark-specific answer.
                topics = list(dict.fromkeys(
                    [*topics, *plan.target_entities, *plan.aliases.keys()]
                ))
                description = (
                    "recurring narrative, causes, changes, outcomes and exceptions for: "
                    + "; ".join(
                        list(dict.fromkeys([plan.target_unit, *topics]))[:12]
                    )
                )
                aliases = list(dict.fromkeys([
                    *topics,
                    *(alias for values in plan.aliases.values() for alias in values),
                    "decline", "persistence", "recovery", "intervention", "exception",
                ]))
                add(kind, description, aliases, plan.question_id)
            else:
                for topic in topics:
                    add(kind, topic, focused_aliases(plan, topic), plan.question_id)
        if "needle" in plan.required_passes:
            description = "direct answer evidence: " + "; ".join(
                [*plan.target_entities, *plan.required_fields][:8]
            )
            needle_aliases = [description]
            for canonical, variants in plan.aliases.items():
                needle_aliases.extend([canonical, *variants])
            add("needle", description, needle_aliases, plan.question_id)
    return [SharedRequirement.model_validate(item) for item in merged.values()]


def requirements_for_pass(plan: SharedEvidencePlan, pass_name: str) -> list[SharedRequirement]:
    kinds = {
        "facts": {"fact"},
        "claims": {"claim", "relation"},
        "topics": {"topic", "theme"},
        "needle": {"needle"},
    }[pass_name]
    return [item for item in plan.requirements if item.kind in kinds]


def base_requirements_for_pass(
    plan: SharedEvidencePlan, pass_name: str
) -> list[SharedRequirement]:
    """Return the requirements assigned to the reusable full-document pass."""

    requirements = requirements_for_pass(plan, pass_name)
    if pass_name != "facts":
        return requirements
    analytical_ids = {
        item.question_id
        for item in plan.question_plans
        if item.query_type in {"aggregation", "superlative"}
    }
    if not analytical_ids:
        return requirements
    return [
        item for item in requirements
        if analytical_ids.intersection(item.consumer_question_ids)
    ]


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    request_hash TEXT PRIMARY KEY, model TEXT NOT NULL,
                    response_text TEXT NOT NULL, usage_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plans (
                    document_hash TEXT NOT NULL, questions_hash TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    PRIMARY KEY (document_hash, questions_hash)
                );
                CREATE TABLE IF NOT EXISTS scans (
                    document_hash TEXT NOT NULL, questions_hash TEXT NOT NULL,
                    pass_name TEXT NOT NULL, pack_id TEXT NOT NULL,
                    status TEXT NOT NULL, delta_json TEXT, error TEXT,
                    PRIMARY KEY (document_hash, questions_hash, pass_name, pack_id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    document_hash TEXT NOT NULL, questions_hash TEXT NOT NULL,
                    evidence_id TEXT NOT NULL, evidence_json TEXT NOT NULL,
                    PRIMARY KEY (document_hash, questions_hash, evidence_id)
                );
                CREATE TABLE IF NOT EXISTS coverage (
                    document_hash TEXT NOT NULL, questions_hash TEXT NOT NULL,
                    question_id TEXT NOT NULL, coverage_json TEXT NOT NULL,
                    PRIMARY KEY (document_hash, questions_hash, question_id)
                );
                CREATE TABLE IF NOT EXISTS answers (
                    document_hash TEXT NOT NULL, questions_hash TEXT NOT NULL,
                    question_id TEXT NOT NULL, answer_version TEXT NOT NULL,
                    generator TEXT NOT NULL, answer_json TEXT NOT NULL,
                    PRIMARY KEY (document_hash, questions_hash, question_id)
                );
                """
            )

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def cached_api(self, request_hash: str) -> Optional[tuple[str, dict[str, Any]]]:
        with self.lock:
            row = self.connection.execute(
                "SELECT response_text, usage_json FROM api_cache WHERE request_hash=?",
                (request_hash,),
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def put_api(self, request_hash: str, model: str, text: str, usage: dict[str, Any]) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO api_cache VALUES (?, ?, ?, ?)",
                (request_hash, model, text, json.dumps(usage, ensure_ascii=False)),
            )

    def get_plan(self, document_hash: str, q_hash: str) -> Optional[SharedEvidencePlan]:
        with self.lock:
            row = self.connection.execute(
                "SELECT plan_json FROM plans WHERE document_hash=? AND questions_hash=?",
                (document_hash, q_hash),
            ).fetchone()
        return SharedEvidencePlan.model_validate_json(row[0]) if row else None

    def put_plan(self, plan: SharedEvidencePlan) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO plans VALUES (?, ?, ?)",
                (plan.document_hash, plan.questions_hash, plan.model_dump_json()),
            )

    def get_scan(self, document_hash: str, q_hash: str, pass_name: str, pack_id: str) -> Optional[str]:
        with self.lock:
            row = self.connection.execute(
                "SELECT status FROM scans WHERE document_hash=? AND questions_hash=? AND pass_name=? AND pack_id=?",
                (document_hash, q_hash, pass_name, pack_id),
            ).fetchone()
        return str(row[0]) if row else None

    def put_scan(
        self,
        document_hash: str,
        q_hash: str,
        pass_name: str,
        pack_id: str,
        status: str,
        delta: Optional[ScanDelta],
        error: Optional[str],
    ) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO scans VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document_hash, q_hash, pass_name, pack_id, status,
                    delta.model_dump_json() if delta else None, error,
                ),
            )
            if delta:
                for item in delta.evidence:
                    existing = self.connection.execute(
                        "SELECT evidence_json FROM evidence WHERE document_hash=? AND questions_hash=? AND evidence_id=?",
                        (document_hash, q_hash, item.id),
                    ).fetchone()
                    if existing:
                        previous = GlobalEvidence.model_validate_json(existing[0])
                        item = item.model_copy(
                            update={
                                "consumer_question_ids": list(
                                    dict.fromkeys(previous.consumer_question_ids + item.consumer_question_ids)
                                ),
                                "confidence": max(previous.confidence, item.confidence),
                            }
                        )
                    self.connection.execute(
                        "INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?)",
                        (document_hash, q_hash, item.id, item.model_dump_json()),
                    )

    def statuses(self, document_hash: str, q_hash: str) -> dict[tuple[str, str], str]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT pass_name, pack_id, status FROM scans WHERE document_hash=? AND questions_hash=?",
                (document_hash, q_hash),
            ).fetchall()
        return {(row[0], row[1]): row[2] for row in rows}

    def evidence(self, document_hash: str, q_hash: str) -> list[GlobalEvidence]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT evidence_json FROM evidence WHERE document_hash=? AND questions_hash=?",
                (document_hash, q_hash),
            ).fetchall()
        return [GlobalEvidence.model_validate_json(row[0]) for row in rows]

    def put_coverage(self, document_hash: str, q_hash: str, item: CoverageEntry) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?)",
                (document_hash, q_hash, item.question_id, item.model_dump_json()),
            )

    def get_answers(self, document_hash: str, q_hash: str) -> dict[str, SubmissionAnswer]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT question_id, answer_json FROM answers WHERE document_hash=? AND questions_hash=? AND answer_version=? AND generator='llm'",
                (document_hash, q_hash, ANSWER_VERSION),
            ).fetchall()
        return {row[0]: SubmissionAnswer.model_validate_json(row[1]) for row in rows}

    def put_answer(self, document_hash: str, q_hash: str, answer: SubmissionAnswer) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, 'llm', ?)",
                (document_hash, q_hash, answer.id, ANSWER_VERSION, answer.model_dump_json()),
            )

    def clear_run(self, document_hash: str, q_hash: str) -> None:
        with self.lock, self.connection:
            for table in ("plans", "scans", "evidence", "coverage", "answers"):
                self.connection.execute(
                    f"DELETE FROM {table} WHERE document_hash=? AND questions_hash=?",
                    (document_hash, q_hash),
                )


class CallBudget:
    def __init__(self, settings: Settings, expected: int, answer_reserve: int = 0) -> None:
        self.settings = settings
        self.expected = max(expected, 1)
        self.answer_reserve = max(0, answer_reserve)
        self.started = time.monotonic()
        self.deadline = self.started + settings.total_time_budget_minutes * 60
        self.completed = 0
        self.failed = 0
        self.active = 0
        self.active_stages: dict[str, int] = defaultdict(int)
        self.active_started: list[float] = []
        self.attempts = 0
        self.durations: list[float] = []
        self.lock = asyncio.Lock()

    def add_expected(self, calls: int) -> None:
        """Keep heartbeat totals honest when recovery work is discovered at runtime."""

        self.expected += max(0, calls)

    async def begin(self, stage: str) -> tuple[float, str]:
        async with self.lock:
            if self.attempts >= self.settings.max_total_api_calls:
                raise BudgetExceeded("MAX_TOTAL_API_CALLS reached")
            remaining_calls = self.settings.max_total_api_calls - self.attempts
            # Full shared scans and narrow needle verification are mandatory:
            # reserving answer calls must never leave the evidence store half-built.
            optional_preanswer = (
                stage == "audit"
                or (stage.endswith("_repair") and stage != "answer_repair")
            )
            if optional_preanswer and remaining_calls <= self.answer_reserve:
                raise BudgetExceeded("final-answer API-call reserve reached")
            if time.monotonic() >= self.deadline:
                raise BudgetExceeded("time budget exhausted")
            reserve_boundary = self.deadline - self.settings.final_reserve_minutes * 60
            if optional_preanswer and time.monotonic() >= reserve_boundary:
                raise BudgetExceeded("final-answer time reserve reached")
            # Needle, answer, and answer-contract calls are the work protected
            # by the reserve. Release one slot only as that work actually
            # begins, so earlier optional repair/audit calls cannot borrow it.
            if stage in {"needle", "answer", "answer_repair"} and self.answer_reserve:
                self.answer_reserve -= 1
            self.attempts += 1
            self.active += 1
            self.active_stages[stage] += 1
            started = time.monotonic()
            self.active_started.append(started)
        return started, stage

    async def finish(self, handle: tuple[float, str], success: bool) -> None:
        started, stage = handle
        async with self.lock:
            self.active = max(0, self.active - 1)
            self.active_stages[stage] = max(0, self.active_stages[stage] - 1)
            if self.active_stages[stage] == 0:
                self.active_stages.pop(stage, None)
            try:
                self.active_started.remove(started)
            except ValueError:
                pass
            self.completed += int(success)
            self.failed += int(not success)
            self.durations.append(max(0.001, time.monotonic() - started))

    def status_text(self) -> str:
        now = time.monotonic()
        left = max(0.0, self.deadline - now)
        done = self.completed + self.failed
        if self.durations and done:
            throughput = done / max(now - self.started, 0.001)
            eta = max(self.expected - done, 0) / max(throughput, 1e-6)
            eta_text = format_duration(eta)
        else:
            eta_text = "calculating"
        active_detail = ""
        if self.active:
            stages = ", ".join(
                f"{stage}:{count}" for stage, count in sorted(self.active_stages.items())
            )
            oldest = now - min(self.active_started) if self.active_started else 0.0
            active_detail = f" [{stages}; oldest {format_duration(oldest)}]"
        return (
            f"API attempts {done}/{self.expected} "
            f"(active {self.active}{active_detail}, failed attempts {self.failed}), "
            f"ETA~{eta_text}, deadline left {format_duration(left)}"
        )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:02d}" if hours else f"{minutes:02d}:{remainder:02d}"


async def progress_heartbeat(budget: CallBudget) -> None:
    while True:
        await asyncio.sleep(15)
        LOGGER.info("progress heartbeat: %s", budget.status_text())


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        budget: CallBudget,
        *,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings
        self.store = store
        self.budget = budget
        self.use_cache = use_cache
        # Roles can intentionally use the same model. Keying this map by model
        # silently collapsed extraction concurrency to reasoning concurrency.
        self.extraction_semaphore = asyncio.Semaphore(settings.extraction_concurrency)
        self.reasoning_semaphore = asyncio.Semaphore(settings.reasoning_concurrency)
        self.model_failures: dict[str, int] = defaultdict(int)
        self.disabled_models: set[str] = set()
        self.json_mode_supported = settings.structured_json_mode
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise PipelineError("openai package is required") from exc
        self.client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.api_base)

    async def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        stage: str,
        max_tokens: int,
        bypass_cache: bool = False,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        request_hash = stable_hash(
            {
                "version": PIPELINE_VERSION,
                "model": model,
                "system": system,
                "user": user,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "json_mode": bool(json_mode and self.settings.structured_json_mode),
                "max_tokens": max_tokens,
            }
        )
        if self.use_cache and not bypass_cache:
            cached = self.store.cached_api(request_hash)
            if cached:
                return cached
        # Shared full-document passes are independent pack work even when a
        # stronger model is selected for semantic claims/topics. Route by role,
        # not model name: otherwise those 40 wide scans serialize behind the
        # single reasoning slot and dominate wall time on larger documents.
        # Planning and final composition retain their conservative semaphore.
        semaphore = (
            self.extraction_semaphore
            if stage in {"facts", "claims", "topics"}
            else self.reasoning_semaphore
        )
        last_error: Optional[Exception] = None
        attempt = 1
        rate_limit_attempt = 0
        max_rate_limit_attempts = self.settings.max_rate_limit_attempts
        while attempt <= self.settings.max_api_attempts:
            if model in self.disabled_models:
                raise PipelineError(
                    f"circuit breaker is open for {model}; use the configured fallback"
                )
            async with semaphore:
                attempt_handle = await self.budget.begin(stage)
                try:
                    if stage.startswith("planning"):
                        configured_timeout = self.settings.planning_timeout_seconds
                    elif model == "mistral-small-3-2" or stage.startswith(
                        ("answer", "audit", "needle")
                    ):
                        configured_timeout = self.settings.reasoning_timeout_seconds
                    else:
                        configured_timeout = self.settings.request_timeout_seconds
                    timeout = min(
                        configured_timeout,
                        max(1.0, self.budget.deadline - time.monotonic()),
                    )
                    request: dict[str, Any] = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": self.settings.temperature,
                        "top_p": self.settings.top_p,
                        "max_tokens": max_tokens,
                    }
                    if json_mode and self.json_mode_supported:
                        request["response_format"] = {"type": "json_object"}
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(**request),
                        timeout=timeout,
                    )
                    text = str(response.choices[0].message.content or "")
                    usage_object = getattr(response, "usage", None)
                    usage = (
                        usage_object.model_dump()
                        if usage_object is not None and hasattr(usage_object, "model_dump")
                        else {}
                    )
                    await self.budget.finish(attempt_handle, True)
                    self.model_failures[model] = 0
                    self.store.put_api(request_hash, model, text, usage)
                    return text, usage
                except Exception as exc:
                    await self.budget.finish(attempt_handle, False)
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    unsupported_json_mode = bool(
                        json_mode and status_code in {400, 404, 422}
                        and re.search(r"response[_ -]?format|json[_ -]?mode", str(exc), re.I)
                    )
                    if unsupported_json_mode:
                        self.json_mode_supported = False
                        LOGGER.warning(
                            "Broker rejected structured JSON mode; retrying with the "
                            "same strict JSON prompt and local validation"
                        )
                    rate_limited = (
                        status_code == 429
                        or "ratelimit" in type(exc).__name__.lower()
                        or bool(re.search(r"(?:\b429\b|rate[_ -]?limit)", str(exc), re.I))
                    )
                    # A 429 means that the service is healthy but asks us to
                    # slow down.  Treating it as a model failure opened the
                    # circuit breaker and degraded all remaining answers to 3B.
                    if not rate_limited and not unsupported_json_mode:
                        self.model_failures[model] += 1
                        if (
                            model == "mistral-small-3-2"
                            and self.model_failures[model]
                            >= self.settings.model_failure_threshold
                        ):
                            self.disabled_models.add(model)
                            LOGGER.warning(
                                "Circuit breaker opened for %s after %d failures; "
                                "subsequent eligible stages will use Ministral 3B",
                                model,
                                self.model_failures[model],
                            )
                    detail = str(exc).strip() or type(exc).__name__
                    if rate_limited:
                        rate_limit_attempt += 1
                        LOGGER.warning(
                            "%s call rate limited (429) on %s (rate limit retry %d/%d): %s",
                            stage,
                            model,
                            rate_limit_attempt,
                            max_rate_limit_attempts,
                            detail,
                        )
                    else:
                        LOGGER.warning(
                            "%s call failed with %s on attempt %d/%d: %s",
                            stage,
                            model,
                            attempt,
                            self.settings.max_api_attempts,
                            detail,
                        )
            if rate_limited:
                if rate_limit_attempt >= max_rate_limit_attempts:
                    break
                sleep_seconds = min(30.0, 2.0 * min(rate_limit_attempt, 5))
                await asyncio.sleep(sleep_seconds)
                continue
            else:
                if attempt < self.settings.max_api_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                attempt += 1
        raise PipelineError(f"{stage} failed with {model}: {last_error}")


def _balanced_json(text: str) -> Optional[str]:
    start = next((index for index, char in enumerate(text) if char in "[{"), None)
    if start is None:
        return None
    stack = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def tolerant_json(text: str) -> Any:
    text = text.strip().replace("“", '"').replace("”", '"')
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text, _strip_json_comments(text)]
    balanced = _balanced_json(text)
    if balanced and balanced != text:
        candidates.append(balanced)
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = re.sub(r":\s*,", ": null,", repaired)
        # Small models sometimes write two page candidates as `"page":172,178`
        # instead of choosing one integer. Keep the first grounded page; the
        # quote locator recalculates the exact page later in normalization.
        repaired = re.sub(
            r'("page"\s*:\s*-?\d+)\s*,\s*-?\d+(?=\s*[,}])', r"\1", repaired
        )
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            continue
    raise ValueError("model response is not complete valid JSON")


def _strip_json_comments(text: str) -> str:
    """Remove JavaScript-style comments without touching quoted URLs or text."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            index += 2
            while index + 1 < len(text) and text[index : index + 2] != "*/":
                index += 1
            index = min(len(text), index + 2)
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _recover_complete_evidence_items(text: str) -> Optional[dict[str, Any]]:
    """Recover only fully closed objects from a truncated scan or census JSON."""

    normalized = _strip_json_comments(text.replace("“", '"').replace("”", '"'))
    marker = re.search(r'"(?:evidence|items|records)"\s*:\s*\[', normalized)
    if not marker:
        return None
    field = re.search(r'"(evidence|items|records)"', marker.group(0)).group(1)
    decoder = json.JSONDecoder()
    cursor = marker.end()
    items = []
    while cursor < len(normalized):
        while cursor < len(normalized) and normalized[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(normalized) or normalized[cursor] == "]":
            break
        try:
            item, end = decoder.raw_decode(normalized, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            items.append(item)
        cursor = end
    if not items:
        return None
    return {
        ("records" if field == "records" else "evidence"): items,
        "uncertain_notes": [
            "[local partial recovery] model output ended before the scan JSON closed"
        ],
    }


async def call_structured(
    client: LLMClient,
    *,
    model: str,
    fallback_model: Optional[str],
    system: str,
    user: str,
    stage: str,
    max_tokens: int,
    response_model: Type[T],
    normalize: Callable[[Any], dict[str, Any]],
    bypass_cache: bool = False,
) -> T:
    errors = []
    models = list(dict.fromkeys(item for item in (model, fallback_model) if item))
    for selected_model in models:
        try:
            text, _ = await client.call(
                model=selected_model,
                system=system,
                user=user,
                stage=stage,
                max_tokens=max_tokens,
                bypass_cache=bypass_cache,
                json_mode=True,
            )
            try:
                return response_model.model_validate(normalize(tolerant_json(text)))
            except (ValueError, ValidationError) as first_error:
                # Scan output can be large and occasionally ends mid-object.
                # Recover only fully closed evidence locally; a second LLM call
                # costs as much as a fresh scan and can amplify transient failures.
                if response_model is ScanDelta:
                    partial = _recover_complete_evidence_items(text)
                    if partial:
                        return response_model.model_validate(normalize(partial))
                    raise PipelineError(
                        f"invalid scan response: {first_error}"
                    ) from first_error
                repair_system = (
                    "Repair the supplied response into complete JSON matching the requested schema. "
                    "Preserve supported content, add no facts, and return JSON only."
                )
                repair_user = f"SCHEMA:\n{json.dumps(response_model.model_json_schema())}\n\nRESPONSE:\n{text}"
                repaired, _ = await client.call(
                    model=selected_model,
                    system=repair_system,
                    user=repair_user,
                    stage=f"{stage}_repair",
                    max_tokens=max_tokens,
                    bypass_cache=bypass_cache,
                    json_mode=True,
                )
                try:
                    return response_model.model_validate(normalize(tolerant_json(repaired)))
                except (ValueError, ValidationError) as second_error:
                    raise PipelineError(
                        f"invalid structured response after repair: {second_error}; first: {first_error}"
                    ) from second_error
        except (PipelineError, BudgetExceeded) as exc:
            errors.append(f"{selected_model}: {exc}")
    raise PipelineError(f"{stage} failed across configured models: {' | '.join(errors)}")


async def compile_shared_plan(
    client: LLMClient,
    store: SQLiteStore,
    document: DocumentInfo,
    q_hash: str,
    questions: Sequence[Question],
    chunks: Sequence[DocumentChunk],
    force: bool,
) -> SharedEvidencePlan:
    if not force:
        cached = store.get_plan(document.sha256, q_hash)
        if cached:
            return cached
    summary = {
        "characters": len(document.text),
        "chunks": len(chunks),
        "pages": len(PAGE_RE.findall(document.text)),
        # Planning needs scale and structure, not document entity names. Keeping
        # headings out prevents the planner from treating source-specific names
        # as query aliases while extraction still sees the full source.
        "heading_paths": sum(bool(chunk.heading_path) for chunk in chunks),
        "primary_units": len(
            {unit for chunk in chunks for unit in chunk.primary_units}
        ),
    }
    batches = [
        questions[index : index + client.settings.plan_batch_size]
        for index in range(0, len(questions), client.settings.plan_batch_size)
    ]

    async def plan_batch(batch: Sequence[Question]) -> list[QuestionPlan]:
        templates = {
            _category(question): CATEGORY_TEMPLATES[_category(question)] for question in batch
        }
        try:
            result = await call_structured(
                client,
                model=client.settings.planning_model,
                fallback_model=client.settings.extraction_model,
                system=PLANNING_SYSTEM,
                user=planning_prompt(batch, templates, summary),
                stage="planning",
                max_tokens=client.settings.reasoning_max_tokens,
                response_model=PlanBatch,
                normalize=lambda value: normalize_plan_batch(value, batch),
            )
            return result.plans
        except (PipelineError, BudgetExceeded) as exc:
            LOGGER.warning(
                "Planning failed for %s; fixed safe plans preserve scan coverage: %s",
                [item.id for item in batch], exc,
            )
            return [fallback_question_plan(item) for item in batch]

    planned = await asyncio.gather(*(plan_batch(batch) for batch in batches))
    plans = [plan for group in planned for plan in group]
    shared = SharedEvidencePlan(
        document_hash=document.sha256,
        questions_hash=q_hash,
        question_plans=plans,
        requirements=build_shared_requirements(plans),
        enabled_passes=enabled_passes(questions),
    )
    store.put_plan(shared)
    return shared


def _whitespace_ground(text: str, quote: str) -> Optional[str]:
    if not quote.strip():
        return None
    pattern = r"\s+".join(re.escape(part) for part in quote.strip().split())
    match = re.search(pattern, text, re.I)
    return match.group(0) if match else None


def _quote_tokens(value: Any) -> list[str]:
    return [token for token in _norm(value).split() if token]


def _source_candidates(text: str) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    candidates: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        if 8 <= len(paragraph) <= 800:
            candidates.append(paragraph)
        if index + 1 < len(paragraphs):
            combined = f"{paragraph}\n{paragraphs[index + 1]}"
            if len(combined) <= 800:
                candidates.append(combined)
        candidates.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if 12 <= len(sentence.strip()) <= 600
        )
    return list(dict.fromkeys(candidates))


def _ground_source_quote(
    raw: dict[str, Any], text: str, quote: Optional[str]
) -> tuple[Optional[str], float]:
    """Relocate a lightly reformatted model quote to one exact source span."""

    exact = _whitespace_ground(text, quote or "")
    if exact:
        return exact, 1.0
    anchors = [
        raw.get("subject"), raw.get("predicate"), raw.get("value_raw", raw.get("value")),
        raw.get("unit_raw", raw.get("unit")), raw.get("topic"), raw.get("related_entity"),
    ]
    quote_tokens = set(_quote_tokens(quote))
    anchor_tokens = set(_quote_tokens(" ".join(str(item) for item in anchors if item)))
    best: tuple[float, Optional[str]] = (0.0, None)
    for candidate in _source_candidates(text):
        tokens = set(_quote_tokens(candidate))
        if not tokens:
            continue
        quote_recall = len(tokens & quote_tokens) / len(quote_tokens) if quote_tokens else 0.0
        anchor_recall = len(tokens & anchor_tokens) / len(anchor_tokens) if anchor_tokens else 0.0
        score = 0.75 * quote_recall + 0.25 * anchor_recall
        if score > best[0]:
            best = score, candidate
    if best[1] is None or best[0] < (0.70 if quote_tokens else 0.84):
        return None, best[0]
    exact = _whitespace_ground(text, best[1])
    return exact, best[0]


def _expand_quote_to_number(source_text: str, quote: str, value: Any) -> str:
    """Extend a grounded quote to an adjacent, exact numeric answer value."""

    expected = NUMBER_RE.search(str(value or ""))
    position = source_text.find(quote)
    if expected is None or position < 0:
        return quote
    digits = expected.group(0).replace(",", "")
    if digits in quote.replace(",", ""):
        return quote
    window_start = max(0, position - 240)
    window_end = min(len(source_text), position + len(quote) + 240)
    candidates = [
        match for match in NUMBER_RE.finditer(source_text[window_start:window_end])
        if match.group(0).replace(",", "") == digits
    ]
    if not candidates:
        return quote
    nearest = min(
        candidates,
        key=lambda match: min(
            abs(window_start + match.start() - position),
            abs(window_start + match.end() - (position + len(quote))),
        ),
    )
    number_start = window_start + nearest.start()
    number_end = window_start + nearest.end()
    start = min(position, number_start)
    end = max(position + len(quote), number_end)
    # Do not bridge a large unrelated section just because the same number
    # happens to occur elsewhere.
    return source_text[start:end].strip() if end - start <= 520 else quote


def _measurement_family(value: Any) -> Optional[str]:
    text = _norm(value)
    if any(token in text for token in ("area", "surface", "square kilomet", "sq km", "sq mile")):
        return "area"
    if any(token in text for token in ("highest", "elevation", "summit", "peak height", "altitude")):
        return "highest"
    if any(token in text for token in ("visitor", "attendance", "visits per year")):
        return "visitors"
    return None


_PLACEHOLDER_UNIT_RE = re.compile(r"^(?:section|chunk|pack)[_ ]?\d+$", re.I)
_COMPONENT_PREDICATE_PREFIXES = {
    "highest", "lowest", "top", "best", "first", "last", "record",
    "winner", "runner", "summit", "peak", "tallest", "largest", "smallest",
}


def _is_placeholder_unit(value: Any) -> bool:
    """Detect an internal bookkeeping id such as ``section_0001``.

    Pack metadata exposes these weak structural labels to the extraction
    model as context. A model occasionally echoes one back as a field value
    instead of the real profiled name; such a value must never seed an
    identity or grouping key.
    """

    text = str(value or "").strip()
    return bool(_PLACEHOLDER_UNIT_RE.match(text))


def _is_identity_predicate(predicate: Any) -> bool:
    """True when a predicate labels the profiled entity itself.

    A predicate ending in "name" only identifies the whole census unit when
    its remaining wording does not name one of the unit's own components
    (for example "highest_point_name" names a summit, not the profiled
    entity that the summit belongs to).
    """

    tokens = _norm(predicate).split()
    if not tokens:
        return False
    joined = " ".join(tokens)
    # Domain-neutral ways a model labels "this is the thing itself".
    if joined in {"identity", "title", "record name", "label", "designation"}:
        return True
    if tokens[0] == "is" and len(tokens) <= 4:
        # `is_in`, `is_a`, and `is_highest_point_of` are relations, not names.
        # Only explicit naming copulas identify the record itself.
        return bool(
            {"named", "called", "known", "labelled", "labeled", "identified"}
            & set(tokens[1:])
        )
    return tokens[-1] == "name" and tokens[0] not in _COMPONENT_PREDICATE_PREFIXES


def _looks_like_identity_value(value: Any) -> bool:
    """Reject internal ids and leaked field labels as a candidate entity name.

    Every real profiled identity (a park, country, species, or similar named
    subject) is copied verbatim from source text and therefore capitalized.
    A bare, lowercase, schema-shaped string such as ``area_km2`` or
    ``highest_point_name`` is almost always a requirement field label that
    leaked into the wrong JSON field, not an actual name.
    """

    text = str(value or "").strip()
    if not text or _is_placeholder_unit(text):
        return False
    if _measurement_family(text):
        return False
    return bool(re.search(r"[^\W\d_]", text)) and bool(re.search(r"[A-ZÀ-ÖØ-Þ]", text))


def mask_page_markers(text: str) -> str:
    """Blank out injected page markers, preserving every character offset.

    ``<!-- page-start-marker-261 -->`` is scaffolding this pipeline adds while
    reading a PDF. Its number sits immediately beside real prose, so a nearest-
    number search for a field like "area" can select the page number instead of
    the measurement. Replacing the marker with spaces of the same length keeps
    positions valid for quote alignment while removing the digits from view.
    """

    return PAGE_RE.sub(lambda match: " " * len(match.group(0)), text)


def _measurement_number(
    source_text: str,
    quote: str,
    value: Any,
    predicate: Any,
    requirement: SharedRequirement,
) -> tuple[Any, str, bool]:
    """Ensure a numeric value belongs to the requested measurement label."""

    if normalize_number(value) is None:
        return value, quote, True
    predicate_family = _measurement_family(predicate)
    requirement_family = _measurement_family(
        " ".join([requirement.description, *requirement.aliases])
    )
    # A measurement attached to a generic identity/context requirement is a
    # schema violation and must not become a comparison candidate.
    if predicate_family and not requirement_family:
        return value, quote, False
    expected = requirement_family or predicate_family
    family_cues = {
        "area": ("area", "surface", "square", "sq "),
        "highest": ("highest", "elevation", "altitude", "summit", "peak"),
        "visitors": ("visitors per year", "visitor", "attendance", "visits"),
    }
    quote_norm = _norm(quote)
    if expected and not any(cue in quote_norm for cue in family_cues[expected]):
        return value, quote, False
    source_position = source_text.find(quote)
    context_start = max(0, source_position - 140) if source_position >= 0 else 0
    context_end = min(len(source_text), source_position + len(quote) + 140) if source_position >= 0 else len(quote)
    context = source_text[context_start:context_end] if source_position >= 0 else quote
    # Page markers are scaffolding this pipeline injected, not document content.
    # Their page numbers sit next to real prose and read as plausible values, so
    # they are hidden from numeric candidate selection while keeping every offset
    # intact for quote positioning and page attribution.
    numbers = [
        match for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", mask_page_markers(context))
    ]
    if expected and numbers:
        # Prefer explicit labels over generic component mentions. A phrase such
        # as "peaks nearly 3000m; X (2914m) being the highest" must bind 2914,
        # not the earlier descriptive number.
        cues = family_cues[expected]
        lowered = context.casefold()
        quote_offset = source_position - context_start if source_position >= 0 else 0
        for cue_text in cues:
            cue_positions = [
                match.start() for match in re.finditer(re.escape(cue_text), lowered)
            ]
            in_quote = [
                position for position in cue_positions
                if quote_offset <= position <= quote_offset + len(quote)
            ]
            candidates = in_quote or cue_positions
            if not candidates:
                continue
            cue = min(
                candidates,
                key=lambda position: min(
                    min(abs(item.start() - position), abs(item.end() - position))
                    for item in numbers
                ),
            )
            competing_cues = {
                "area": ("highest", "elevation", "altitude", "visitor", "attendance"),
                "highest": ("area", "surface", "square", "visitor", "attendance"),
                "visitors": ("area", "surface", "square", "highest", "elevation", "altitude"),
            }[expected]

            def candidate_cost(item: re.Match[str]) -> float:
                distance = min(abs(item.start() - cue), abs(item.end() - cue))
                bridge = lowered[min(cue, item.start()) : max(cue, item.end())]
                # PDF statistic boxes interleave labels and values. A number
                # separated from this label by another field label belongs to
                # that intervening field, even when it is geometrically close.
                crossed_field = any(term in bridge for term in competing_cues)
                return distance + (10_000 if crossed_field else 0)

            nearest = min(numbers, key=candidate_cost)
            between = context[min(cue, nearest.start()) : max(cue, nearest.end())]
            # A quote spanning into the next physical page can put that page's
            # entry number next to the previous page's measurement label.
            if PAGE_RE.search(between):
                continue
            start = min(quote_offset, nearest.start())
            end = max(quote_offset + len(quote), nearest.end())
            return nearest.group(0), context[start:end].strip(), True
    expected_number = NUMBER_RE.search(str(value or ""))
    if expected_number and expected_number.group(0).replace(",", "") in mask_page_markers(
        quote
    ).replace(",", ""):
        return value, quote, True
    return value, quote, not expected


def _match_requirement(
    raw: dict[str, Any], requirements: Sequence[SharedRequirement]
) -> Optional[SharedRequirement]:
    key = str(raw.get("requirement_key", ""))
    exact = next((item for item in requirements if item.key == key), None)
    haystack = _norm(
        " ".join(
            str(raw.get(field, ""))
            for field in ("predicate", "topic", "subject", "related_entity", "source_quote")
        )
    )
    if exact:
        # Models occasionally repeat a valid key on every unrelated passage.
        # Claims, relations, and needles therefore require a semantic anchor too.
        if exact.kind in {"claim", "relation", "needle", "topic"}:
            stop = {
                "direct", "answer", "evidence", "claim", "relation", "explicit",
                "entity", "value", "field", "event", "type", "description",
                "park", "parks", "country", "countries", "source", "location",
                "name", "named", "record", "records", "item", "items",
            }
            anchors = {
                token
                for term in [exact.description, *exact.aliases]
                for token in _norm(term).split()
                if len(token) >= 4 and token not in stop
            }
            # Only the copied source quote may satisfy the semantic anchor.
            # Model-authored subject/topic labels are not evidence: accepting
            # them would allow an unrelated passage to be assigned a requested
            # topic that never occurs in the copied source.
            source_haystack = _norm(str(raw.get("source_quote", "")))

            def occurs(anchor: str) -> bool:
                if anchor in source_haystack:
                    return True
                # Conservative morphology for retreat/retreated,
                # visitor/visitors, tourist/tourism, etc.  Prefix matching is
                # limited to long stems to avoid broad short-word collisions.
                stem = anchor[:6]
                return len(anchor) >= 7 and any(
                    token.startswith(stem) for token in source_haystack.split()
                )

            if anchors and not any(occurs(token) for token in anchors):
                return None
        return exact
    if key:
        return None
    scored = []
    for requirement in requirements:
        terms = [requirement.description, *requirement.aliases]
        score = sum(1 for term in terms if len(_norm(term)) >= 3 and _norm(term) in haystack)
        if score:
            scored.append((score, requirement))
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _subject_occurs_near_quote(subject: Any, quote: str, source_text: str) -> bool:
    """Reject numeric row-to-entity joins that are not present in the source region."""

    normalized_subject = _norm(subject)
    if not normalized_subject:
        return False
    quote_position = source_text.find(quote)
    if quote_position < 0:
        return False
    start = max(0, quote_position - 2_000)
    end = min(len(source_text), quote_position + len(quote) + 800)
    context = _norm(source_text[start:end])
    if normalized_subject in context:
        return True
    # Headings sometimes omit generic suffixes used by the extractor.
    significant = [
        token for token in normalized_subject.split()
        if len(token) >= 4 and token not in {"national", "park", "reserve", "protected", "area"}
    ]
    return bool(significant) and sum(token in context for token in significant) >= min(2, len(significant))


_LENGTH_UNITS = {"km", "kilometer", "kilometers", "kilometre", "kilometres",
                 "ft", "feet", "foot", "mi", "mile", "miles"}


def normalize_number(value: Any, unit: Any = None, family: Optional[str] = None) -> Optional[float]:
    """The number a value carries, converted to the field's own scale.

    A statistics block sets a figure and its label on separate lines, so an
    extractor reading "290 / Area covered (sq km) / 82 / Length of longest
    trail" sometimes pairs the area with the trail's unit. Converting that
    kilometre reading to metres turns a 290 km² entry into 290,000 and hands it
    the largest-area answer outright. An area measured in a length unit is a
    mislabelled unit, not a different scale: keep the number as written.
    """

    if isinstance(value, bool):
        return None
    if family == "area" and _norm(unit) in _LENGTH_UNITS:
        unit = None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value or "")
        match = NUMBER_RE.search(text)
        if not match:
            if re.search(r"\b(?:a|one)\s+million\b", text, re.I):
                number = 1_000_000.0
            elif re.search(r"\b(?:a|one)\s+thousand\b", text, re.I):
                number = 1_000.0
            else:
                return None
        else:
            number = float(match.group(0).replace(",", ""))
        if re.search(r"\b(?:million|mn)\b", text, re.I):
            if number < 1_000_000:
                number *= 1_000_000
        elif re.search(r"\b(?:thousand|k)\b", text, re.I):
            if number < 1_000:
                number *= 1_000
    unit_text = _norm(unit)
    if unit_text in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
        return number * 1_000
    if unit_text in {"ft", "feet", "foot"}:
        return number * 0.3048
    if unit_text in {"sq miles", "square miles", "mi2", "sq mi"}:
        return number * 2.589988110336
    return number


def normalize_scan_delta(
    value: Any,
    *,
    pass_name: str,
    pack: ScanPack,
    requirements: Sequence[SharedRequirement],
    chunks_by_id: dict[str, DocumentChunk],
) -> dict[str, Any]:
    raw_items = value.get("evidence", value.get("items", [])) if isinstance(value, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []
    evidence = []
    notes = _strings(value.get("uncertain_notes", [])) if isinstance(value, dict) else []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        requirement = _match_requirement(raw, requirements)
        if requirement is None:
            notes.append("discarded evidence that matched no shared requirement")
            continue
        requested_chunk_id = str(raw.get("chunk_id", ""))
        quote = str(raw.get("source_quote", raw.get("quote", ""))).strip()
        grounded: list[tuple[float, DocumentChunk, str]] = []
        search_ids = ([requested_chunk_id] if requested_chunk_id in pack.chunk_ids else []) + [
            item for item in pack.chunk_ids if item != requested_chunk_id
        ]
        for chunk_id in search_ids:
            candidate, score = _ground_source_quote(raw, chunks_by_id[chunk_id].text, quote)
            if candidate:
                grounded.append((score, chunks_by_id[chunk_id], candidate))
        if not grounded:
            notes.append(f"discarded ungrounded evidence in {requested_chunk_id or pack.id}")
            continue
        _, grounded_chunk, grounded_quote = max(grounded, key=lambda item: item[0])
        kind = str(raw.get("kind", "uncertain")).casefold()
        if kind not in {"fact", "claim", "relation", "topic", "theme", "counterevidence", "uncertain"}:
            kind = "uncertain"
        topic_status = str(raw.get("topic_status", "")).casefold()
        if topic_status == "unrelated":
            continue
        if _is_identity_predicate(raw.get("predicate")):
            # An identity fact is shared backbone context for every question
            # consuming this requirement (for example every aggregation and
            # superlative question sharing one census). The extraction model
            # sometimes narrows consumer_question_ids to only the question it
            # was focused on; keeping that narrowing here would silently hide
            # a census unit's own name from a sibling question whenever one
            # of its other facts needs this identity as a grouping fallback.
            consumers = list(requirement.consumer_question_ids)
        else:
            consumers = [
                item for item in _strings(raw.get("consumer_question_ids"))
                if item in requirement.consumer_question_ids
            ] or list(requirement.consumer_question_ids)
        raw_value = raw.get("value_raw", raw.get("value"))
        placeholder = _norm(raw_value)
        if placeholder in {
            "not explicitly stated", "not stated", "not mentioned", "unknown",
            "none", "none specified", "not listed", "n a", "not applicable",
        }:
            continue
        raw_value, grounded_quote, numeric_grounded = _measurement_number(
            grounded_chunk.text, grounded_quote, raw_value, raw.get("predicate"), requirement
        )
        if not numeric_grounded:
            notes.append(f"discarded mismatched numeric measurement in {grounded_chunk.id}")
            continue
        predicate_norm = _norm(raw.get("predicate"))
        categorical_predicate = any(
            token in predicate_norm.split()
            for token in ("name", "country", "countries", "nation", "location", "located")
        ) or predicate_norm in {"is in", "belongs to"}
        if (
            normalize_number(raw_value, raw.get("unit_raw", raw.get("unit"))) is None
            and categorical_predicate
        ):
            value_tokens = [
                token for token in _norm(raw_value).split()
                if len(token) >= 3
                and token not in {"the", "national", "park", "marine", "country", "countries"}
            ]
            quote_tokens = set(_norm(grounded_quote).split())
            if value_tokens and not all(
                token in quote_tokens
                or (len(token) >= 7 and any(item.startswith(token[:6]) for item in quote_tokens))
                for token in value_tokens
            ):
                notes.append(f"discarded categorical value absent from quote in {grounded_chunk.id}")
                continue
        if (
            normalize_number(raw_value, raw.get("unit_raw", raw.get("unit"))) is not None
            and _measurement_family(" ".join([requirement.description, *requirement.aliases]))
            and not any(
                _subject_occurs_near_quote(candidate, grounded_quote, grounded_chunk.text)
                for candidate in (
                    raw.get("subject"), raw.get("primary_unit"), raw.get("related_entity")
                )
                if candidate
            )
        ):
            notes.append(f"discarded ungrounded numeric subject in {grounded_chunk.id}")
            continue
        normalized_value: Any = raw.get("value_normalized")
        # The requirement says what is being measured, so it is what decides
        # whether the unit the extractor attached is dimensionally possible.
        field_family = _measurement_family(
            " ".join([requirement.description, *requirement.aliases, _norm(raw.get("predicate"))])
        )
        numeric = normalize_number(
            raw_value, raw.get("unit_raw", raw.get("unit")), field_family
        )
        if numeric is not None and numeric < 1_000_000 and re.search(
            r"\b(?:million|mn)\b", grounded_quote, re.I
        ):
            numeric *= 1_000_000
        elif numeric is not None and numeric < 1_000 and re.search(
            r"\b(?:thousand)\b", grounded_quote, re.I
        ):
            numeric *= 1_000
        if pass_name == "needle" and numeric is not None:
            grounded_quote = _expand_quote_to_number(
                grounded_chunk.text, grounded_quote, raw_value
            )
            expected_number = NUMBER_RE.search(str(raw_value or ""))
            if (
                expected_number
                and expected_number.group(0).replace(",", "")
                not in grounded_quote.replace(",", "")
            ):
                notes.append(f"discarded needle number absent from quote in {grounded_chunk.id}")
                continue
        if numeric is not None:
            normalized_value = numeric
        elif normalized_value is None and raw_value is not None:
            normalized_value = str(raw_value)
        page = raw.get("page", grounded_chunk.page_start)
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = grounded_chunk.page_start
        if grounded_quote:
            position = grounded_chunk.text.find(grounded_quote)
            pages = PAGE_RE.findall(grounded_chunk.text[: max(position, 0)])
            if pages:
                page = int(pages[-1])
        confidence = raw.get("confidence", 0.5)
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        item = {
            "pass_name": pass_name,
            "pack_id": pack.id,
            "chunk_id": grounded_chunk.id,
            "primary_unit": str(raw.get("primary_unit")) if raw.get("primary_unit") else None,
            "kind": kind,
            "requirement_key": requirement.key,
            "consumer_question_ids": consumers,
            "subject": str(raw.get("subject")) if raw.get("subject") else None,
            "predicate": str(raw.get("predicate")) if raw.get("predicate") else None,
            "value_raw": str(raw_value) if raw_value is not None else None,
            "value_normalized": normalized_value,
            "unit_raw": str(raw.get("unit_raw", raw.get("unit"))) if raw.get("unit_raw", raw.get("unit")) else None,
            "unit_normalized": str(raw.get("unit_normalized")) if raw.get("unit_normalized") else None,
            "related_entity": str(raw.get("related_entity")) if raw.get("related_entity") else None,
            "relation_type": str(raw.get("relation_type")) if raw.get("relation_type") else None,
            "topic": str(raw.get("topic")) if raw.get("topic") else None,
            "topic_status": topic_status or None,
            "source_quote": grounded_quote,
            "page": page,
            "section": str(raw.get("section")) if raw.get("section") else (
                grounded_chunk.heading_path[-1] if grounded_chunk.heading_path else None
            ),
            "original_order": grounded_chunk.order,
            "confidence": confidence,
        }
        item["id"] = stable_hash(
            {
                key: item[key]
                for key in (
                    "pass_name", "chunk_id", "kind", "subject", "predicate",
                    "value_raw", "unit_raw", "related_entity", "topic", "source_quote",
                )
            }
        )[:24]
        evidence.append(item)
    return {
        "pass_name": pass_name,
        "pack_id": pack.id,
        "inspected_chunk_ids": list(pack.chunk_ids),
        "inspected_primary_units": list(pack.primary_units),
        "evidence": evidence,
        "uncertain_notes": list(dict.fromkeys(notes)),
    }


CENSUS_IDENTITY_KEY = "fact:census_record_name"


def census_identity_requirement(
    requirements: Sequence[SharedRequirement],
) -> Optional[SharedRequirement]:
    """The requirement under which a record's own name is stored.

    A record's name is shared backbone context: every question that counts,
    groups, or ranks records needs it, not just the question whose field
    happened to sit first in the list. Borrowing another requirement's key
    would also borrow its narrower consumer list and hide the name from the
    counting questions, so identity gets its own key spanning every consumer
    of the census.
    """

    consumers = list(dict.fromkeys(
        consumer for item in requirements for consumer in item.consumer_question_ids
    ))
    if not consumers:
        return None
    explicit = next(
        (item for item in requirements if _is_identity_predicate(item.description)), None
    )
    return SharedRequirement(
        key=CENSUS_IDENTITY_KEY,
        kind="fact",
        description=explicit.description if explicit else "record name",
        aliases=list(explicit.aliases) if explicit else [],
        consumer_question_ids=consumers,
    )


def census_to_scan_payload(
    value: Any,
    requirements: Sequence[SharedRequirement],
    identity_requirement: Optional[SharedRequirement] = None,
) -> dict[str, Any]:
    """Flatten a record census into the shared evidence shape.

    The census contract is narrow on purpose — a small model transcribing five
    fields per record does not skip records the way it skips them under the
    full evidence schema. Widening the rows back out here keeps grounding,
    numeric validation, and the evidence store completely unchanged.
    """

    raw_records = value.get("records", value.get("items", [])) if isinstance(value, dict) else []
    if not isinstance(raw_records, list):
        raw_records = []
    by_key = {item.key: item for item in requirements}
    identity_requirement = identity_requirement or census_identity_requirement(requirements)
    evidence: list[dict[str, Any]] = []
    # Carry any local-recovery marker through so a truncated census still marks
    # its pack partial and stays eligible for a fresh scan.
    notes: list[str] = _strings(value.get("uncertain_notes", [])) if isinstance(value, dict) else []
    for entry in raw_records:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("record_name", entry.get("name", ""))).strip()
        chunk_id = str(entry.get("chunk_id", ""))
        fields = entry.get("fields", [])
        if not isinstance(fields, list):
            fields = []
        if not _looks_like_identity_value(name):
            notes.append("discarded census record without a usable name")
            continue
        if identity_requirement is not None:
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "kind": "fact",
                    "requirement_key": identity_requirement.key,
                    "primary_unit": name,
                    "subject": name,
                    "predicate": "record name",
                    "value_raw": name,
                    "source_quote": str(entry.get("name_quote") or name),
                    "confidence": 0.9,
                }
            )
        for field in fields:
            if not isinstance(field, dict):
                continue
            requirement = by_key.get(str(field.get("key", "")))
            label = str(field.get("field", "")).strip()
            if requirement is None:
                notes.append(f"discarded census field with no matching requirement: {label}")
                continue
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "kind": "fact",
                    "requirement_key": requirement.key,
                    "primary_unit": name,
                    "subject": name,
                    "predicate": label or requirement.description,
                    "value_raw": field.get("value"),
                    "unit_raw": field.get("unit"),
                    "source_quote": field.get("quote", ""),
                    "confidence": 0.85,
                }
            )
    return {"evidence": evidence, "uncertain_notes": notes}


async def _scan_one(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    pass_name: str,
    pack: ScanPack,
    requirements: Sequence[SharedRequirement],
    chunks_by_id: dict[str, DocumentChunk],
    force: bool,
    audit_reason: Optional[str] = None,
    census: bool = False,
    census_identity_scope: Optional[Sequence[SharedRequirement]] = None,
    optional_repair: bool = False,
) -> bool:
    previous_status = store.get_scan(document_hash, q_hash, pass_name, pack.id)
    if not force and previous_status in {"success", "audited"}:
        return True
    model = client.settings.audit_model if (
        audit_reason or pass_name in {"claims", "topics"}
    ) else (
        client.settings.answer_model
        if pass_name == "needle" else client.settings.extraction_model
    )
    fallback = client.settings.extraction_model if model != client.settings.extraction_model else None
    # A gathered/transposed pack contains dense statistics from many records.
    # Asking it for categorical and narrative fields that cannot occur in that
    # slot creates long placeholder-like output and, worse, lets nearby names be
    # attached to the wrong record. Keep only numeric measurement requirements;
    # regular record census and question focus still cover every other field.
    effective_requirements = list(requirements)
    if census and pack.id.startswith("slot_"):
        measured = [
            item for item in requirements
            if _measurement_family(" ".join([item.description, *item.aliases]))
        ]
        if measured:
            effective_requirements = measured
    if census and not audit_reason:
        system = CENSUS_SYSTEM
        user = census_prompt(
            pack,
            effective_requirements,
            pack.expected_records or len(pack.record_ids) or len(pack.chunk_ids),
            transposed=pack.id.startswith("slot_"),
        )
    elif audit_reason:
        system = AUDIT_SYSTEM
        user = audit_prompt(pass_name, pack, effective_requirements, audit_reason)
    else:
        system = EXTRACTION_SYSTEMS[pass_name]
        user = extraction_prompt(pass_name, pack, effective_requirements)

    def normalize(value: Any) -> dict[str, Any]:
        matchable = list(effective_requirements)
        if census and not audit_reason:
            identity = census_identity_requirement(census_identity_scope or requirements)
            value = census_to_scan_payload(value, effective_requirements, identity)
            if identity is not None:
                matchable.append(identity)
        return normalize_scan_delta(
            value,
            pass_name=pass_name,
            pack=pack,
            requirements=matchable,
            chunks_by_id=chunks_by_id,
        )

    try:
        delta = await call_structured(
            client,
            model=model,
            fallback_model=fallback,
            system=system,
            user=user,
            stage=(
                "audit" if audit_reason
                else f"{pass_name}_repair" if optional_repair
                else pass_name
            ),
            max_tokens=(
                client.settings.reasoning_max_tokens
                if audit_reason or pass_name == "needle"
                else client.settings.census_max_tokens
                if census
                else client.settings.extraction_max_tokens
            ),
            response_model=ScanDelta,
            # A partial/error status means the cached response was already
            # proven unusable. Resume must make one genuinely fresh attempt.
            bypass_cache=previous_status in {"partial", "error"},
            normalize=normalize,
        )
    except (PipelineError, BudgetExceeded) as exc:
        # A supplementary audit must never erase the completion status of the
        # original full-pack extraction. The evidence store is append-only.
        if not (audit_reason and previous_status in {"success", "audited"}):
            store.put_scan(document_hash, q_hash, pass_name, pack.id, "error", None, str(exc))
        LOGGER.error("%s/%s failed: %s", pass_name, pack.id, exc)
        return False
    partial = any(
        note.startswith("[local partial recovery]") for note in delta.uncertain_notes
    )
    store.put_scan(
        document_hash, q_hash, pass_name, pack.id,
        ("partial" if partial else ("audited" if audit_reason else "success")), delta, None,
    )
    return True


async def run_shared_scans(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    shared_plan: SharedEvidencePlan,
    pass_packs: dict[str, list[ScanPack]],
    chunks: Sequence[DocumentChunk],
    force: bool,
    census_passes: Sequence[str] = ("facts",),
) -> None:
    chunks_by_id = {item.id: item for item in chunks}
    jobs = []
    for pass_name in shared_plan.enabled_passes:
        requirements = base_requirements_for_pass(shared_plan, pass_name)
        # The full facts pass is the census for all count/extreme questions.
        # Cross-section-only fields are recovered by their targeted relation
        # closure; mixing them into every census pack caused profile sampling.
        if not requirements:
            continue
        census = pass_name in census_passes
        for pack in pass_packs.get(pass_name, []):
            scan_pack = pack
            if pass_name == "facts" and not census:
                instruction = (
                    "CENSUS INSTRUCTION (not source text): Inspect every profiled "
                    "unit in this pack. Emit its exact identity plus every requested "
                    "country or numeric field once. Do not sample representative "
                    "units, and do not emit a per-profile value of 1 as a collection "
                    "total.\n\n"
                )
                scan_pack = pack.model_copy(
                    update={
                        "text": instruction + pack.text,
                        "character_count": len(instruction) + pack.character_count,
                    }
                )
            jobs.append(
                _scan_one(
                    client, store, document_hash, q_hash, pass_name, scan_pack,
                    requirements, chunks_by_id, force, census=census,
                )
            )
    if jobs:
        LOGGER.info("Starting %d shared extraction jobs", len(jobs))
        await asyncio.gather(*jobs)


def _focus_pack(
    plan: QuestionPlan,
    question: Question,
    pass_name: str,
    requirements: Sequence[SharedRequirement],
    chunks: Sequence[DocumentChunk],
    max_characters: int = 34_000,
) -> Optional[ScanPack]:
    """Build a compact, source-spanning lexical shadow pack for one question."""

    raw_terms = [question.question, *_plan_terms(plan)]
    for requirement in requirements:
        raw_terms.extend([requirement.description, *requirement.aliases])
    phrases: list[str] = []
    for term in raw_terms:
        normalized = re.sub(r"[_-]+", " ", str(term)).casefold().strip()
        if normalized:
            phrases.append(normalized)
            phrases.extend(
                token for token in re.findall(r"[\wÀ-ÖØ-öø-ÿ]{4,}", normalized)
                if token not in {
                    "which", "what", "where", "when", "with", "from", "book", "document",
                    "park", "parks", "name", "named", "field", "value", "answer", "evidence",
                }
            )
    phrases = list(dict.fromkeys(phrases))
    snippets: list[tuple[int, int, DocumentChunk, str]] = []
    for chunk in chunks:
        lowered = chunk.text.casefold()
        positions: set[int] = set()
        for term in phrases:
            if len(term) < 4:
                continue
            for match in list(re.finditer(re.escape(term), lowered))[:12]:
                positions.add(match.start())
        if not positions:
            continue
        windows: list[tuple[float, int]] = []
        for position in sorted(positions):
            start = max(0, position - 280)
            end = min(len(chunk.text), position + 820)
            snippet_lower = chunk.text[start:end].casefold()
            score = sum(
                1.0 + min(len(term.split()), 5) + min(len(term) / 24.0, 3.0)
                for term in phrases
                if len(term) >= 4 and term in snippet_lower
            )
            windows.append((score, position))
        selected_positions: list[int] = []
        for _, position in sorted(windows, key=lambda item: (-item[0], item[1])):
            if any(abs(position - previous) < 650 for previous in selected_positions):
                continue
            selected_positions.append(position)
            if len(selected_positions) == 2:
                break
        for position in selected_positions:
            start = max(0, position - 280)
            end = min(len(chunk.text), position + 820)
            snippet = chunk.text[start:end].strip()
            if chunk.record_ids and start > 700:
                opening = chunk.text[:600].strip()
                snippet = (
                    "[RECORD OPENING — copied source context for ownership]\n"
                    f"{opening}\n\n[RELEVANT SOURCE WINDOW]\n{snippet}"
                )
            snippet_lower = snippet.casefold()
            score = sum(
                1.0 + min(len(term.split()), 5) + min(len(term) / 24.0, 3.0)
                for term in phrases
                if len(term) >= 4 and term in snippet_lower
            )
            snippets.append((chunk.order, -score, chunk, snippet))
    if not snippets:
        return None
    # Preserve at least one candidate from every matching source region before
    # adding second snippets, then restore document order for the model.
    snippets.sort(key=lambda item: (item[1], item[0]))
    target = (
        f"TARGET QUESTION (retrieval instruction, not source text): {question.question}\n"
        "Extract only evidence that directly bears on this exact question.\n\n"
    )
    source_budget = max(1, max_characters - len(target))
    chosen: list[tuple[int, DocumentChunk, str]] = []
    size = 0
    seen_chunks: set[str] = set()
    seen_rendered: set[str] = set()
    for require_new_chunk in (True, False):
        for order, _, chunk, snippet in snippets:
            if require_new_chunk and chunk.id in seen_chunks:
                continue
            if not require_new_chunk and chunk.id not in seen_chunks:
                continue
            rendered = f"<<<CHUNK {chunk.id}>>>\n{snippet}\n<<<END CHUNK {chunk.id}>>>"
            if rendered in seen_rendered:
                continue
            separator = 2 if chosen else 0
            if size + separator + len(rendered) > source_budget:
                continue
            chosen.append((order, chunk, rendered))
            seen_chunks.add(chunk.id)
            seen_rendered.add(rendered)
            size += separator + len(rendered)
    chosen.sort(key=lambda item: item[0])
    text = target + "\n\n".join(item[2] for item in chosen)
    selected_chunks = list(dict.fromkeys(item[1].id for item in chosen))
    chunk_map = {item.id: item for item in chunks}
    pages = [
        page for chunk_id in selected_chunks
        for page in (chunk_map[chunk_id].page_start, chunk_map[chunk_id].page_end)
        if page is not None
    ]
    return ScanPack(
        id=f"focus_{plan.question_id}_{pass_name}",
        order=0,
        chunk_ids=selected_chunks,
        primary_units=list(dict.fromkeys(
            unit for chunk_id in selected_chunks for unit in chunk_map[chunk_id].primary_units
        )),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        text=text,
        character_count=len(text),
    )


def _ranked_full_source_packs(
    plan: QuestionPlan,
    question: Question,
    pass_name: str,
    requirements: Sequence[SharedRequirement],
    chunks: Sequence[DocumentChunk],
    *,
    candidate_chunks: int = 2,
    chunks_per_pack: int = 2,
) -> list[ScanPack]:
    """Retrieve several complete source regions for multi-evidence joins.

    Snippet shadows work for a single mention, but contradiction and
    cross-section questions often need two remote passages plus nearby table or
    heading context.  This returns ranked full chunks in more than one pack so
    one early false lead cannot displace the second side of the join.
    """

    terms = [question.question, *_plan_terms(plan)]
    for requirement in requirements:
        terms.extend([requirement.description, *requirement.aliases])
    ranked = sorted(
        ((lexical_score(chunk.text, terms), chunk) for chunk in chunks),
        key=lambda item: (item[0], -item[1].order),
        reverse=True,
    )
    selected = [chunk for score, chunk in ranked[:candidate_chunks] if score > 0]
    packs: list[ScanPack] = []
    for index in range(0, len(selected), chunks_per_pack):
        group = selected[index : index + chunks_per_pack]
        if not group:
            continue
        group = sorted(group, key=lambda item: item.order)
        source = _pack_text(group)
        instruction = (
            f"TARGET QUESTION (retrieval instruction, not source text): {question.question}\n"
            "Find only directly sourced evidence for every requested side, item, "
            "status, quantity, date, participant, and qualifier. Preserve distinct "
            "claims rather than resolving them.\n\n"
        )
        pages = [
            page for chunk in group
            for page in (chunk.page_start, chunk.page_end)
            if page is not None
        ]
        packs.append(
            ScanPack(
                id=f"focus_{plan.question_id}_{pass_name}_{index // chunks_per_pack + 1}",
                order=index // chunks_per_pack,
                chunk_ids=[item.id for item in group],
                primary_units=list(dict.fromkeys(
                    unit for item in group for unit in item.primary_units
                )),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                text=instruction + source,
                character_count=len(instruction) + len(source),
            )
        )
    return packs


def _ranking_focus_pack(
    plan: QuestionPlan,
    question: Question,
    pass_name: str,
    chunks: Sequence[DocumentChunk],
    max_characters: int = 34_000,
) -> Optional[ScanPack]:
    """Gather every compact ranking-claim window into one source-spanning view."""

    rank_pattern = re.compile(
        r"\b(?:largest|biggest|highest|lowest|smallest|tallest|widest|longest|"
        r"deepest|oldest|greatest|most|rank\w*|first|second|third|fourth|fifth|"
        r"sixth|seventh|eighth|ninth|tenth)\w*\b",
        re.IGNORECASE,
    )
    terms = [question.question, *_plan_terms(plan)]
    candidates: list[tuple[float, int, DocumentChunk, str]] = []
    for chunk in chunks:
        matches = list(rank_pattern.finditer(chunk.text))
        if not matches:
            continue
        # One window per source region prevents verbose entries from crowding
        # out a remote counterclaim.
        best: Optional[tuple[float, str]] = None
        for match in matches:
            start = max(0, match.start() - 230)
            end = min(len(chunk.text), match.end() + 360)
            snippet = chunk.text[start:end].strip()
            if chunk.record_ids and start > 700:
                snippet = (
                    "[RECORD OPENING — copied source context for ownership]\n"
                    f"{chunk.text[:600].strip()}\n\n"
                    "[RANKING SOURCE WINDOW]\n"
                    f"{snippet}"
                )
            score = lexical_score(snippet, terms) + 6.0
            if best is None or score > best[0]:
                best = (score, snippet)
        if best:
            candidates.append((best[0], chunk.order, chunk, best[1]))
    if not candidates:
        return None
    instruction = (
        f"TARGET QUESTION (retrieval instruction, not source text): {question.question}\n"
        "Extract exact ranking claims, their scopes, entities, figures and units.\n\n"
    )
    selected: list[tuple[float, int, DocumentChunk, str]] = []
    used = len(instruction)
    for candidate in sorted(candidates, key=lambda row: (-row[0], row[1])):
        addition = len(candidate[3]) + 80
        if selected and used + addition > max_characters:
            continue
        selected.append(candidate)
        used += addition
    selected.sort(key=lambda row: row[1])
    text = instruction + "\n\n".join(
        f"<<<CHUNK {chunk.id}>>>\n{snippet}"
        for _, _, chunk, snippet in selected
    )
    pages = [
        page for _, _, chunk, _ in selected
        for page in (chunk.page_start, chunk.page_end) if page is not None
    ]
    return ScanPack(
        id=f"focus_{plan.question_id}_{pass_name}_ranking",
        order=0,
        chunk_ids=[chunk.id for _, _, chunk, _ in selected],
        primary_units=list(dict.fromkeys(
            unit for _, _, chunk, _ in selected for unit in chunk.primary_units
        )),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        text=text,
        character_count=len(text),
    )


async def run_question_focus_closures(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    questions: Sequence[Question],
    shared_plan: SharedEvidencePlan,
    chunks: Sequence[DocumentChunk],
    force: bool,
) -> None:
    """Re-extract compact question-specific candidates after the shared pass."""

    question_map = {item.id: item for item in questions}
    chunks_by_id = {item.id: item for item in chunks}
    jobs = []
    for plan in shared_plan.question_plans:
        if plan.query_type in {"needle", "superlative"}:
            continue
        for pass_name in (name for name in plan.required_passes if name != "needle"):
            requirements = [
                item for item in requirements_for_pass(shared_plan, pass_name)
                if plan.question_id in item.consumer_question_ids
            ]
            if plan.query_type in {"contradiction", "cross_section"}:
                vocabulary = _norm(" ".join([
                    plan.target_unit, *_plan_terms(plan),
                    question_map[plan.question_id].question,
                ]))
                ranking_oriented = plan.query_type == "contradiction" and any(
                    cue in vocabulary for cue in (
                        "rank", "largest", "biggest", "highest", "lowest",
                        "smallest", "tallest", "widest",
                    )
                )
                compact = (
                    _ranking_focus_pack(
                        plan, question_map[plan.question_id], pass_name, chunks,
                        max_characters=34_000,
                    )
                    if ranking_oriented else
                    _focus_pack(
                        plan, question_map[plan.question_id], pass_name, requirements,
                        chunks, max_characters=34_000,
                    )
                )
                if compact:
                    compact = compact.model_copy(update={
                        "id": f"focus_{plan.question_id}_{pass_name}_shadow"
                    })
                # Contradictions and cross-section joins often need a second
                # side that ranks below the top three lexically-nearest
                # chunks (the "other side" rarely echoes the question's own
                # wording as closely as the side it conflicts with). Using
                # every ranked context pack, not only the first, keeps that
                # second side from being silently dropped.
                ranked_context = _ranked_full_source_packs(
                    plan, question_map[plan.question_id], pass_name,
                    requirements, chunks,
                )
                context_packs = [
                    item.model_copy(update={
                        "id": f"focus_{plan.question_id}_{pass_name}_context_{index}"
                    })
                    for index, item in enumerate(ranked_context)
                ]
                focus_packs = [item for item in (compact, *context_packs) if item]
            elif plan.query_type in {"aggregation", "global_synthesis"}:
                compact = _focus_pack(
                    plan, question_map[plan.question_id], pass_name, requirements,
                    chunks, max_characters=34_000,
                )
                focus_packs = [compact] if compact else []
            else:
                compact = _focus_pack(
                    plan, question_map[plan.question_id], pass_name, requirements,
                    chunks, max_characters=22_000,
                )
                focus_packs = [compact] if compact else []
            for pack in focus_packs:
                if requirements:
                    jobs.append((plan, pass_name, pack, requirements))
    jobs = jobs[: client.settings.max_focus_calls]
    if jobs:
        LOGGER.info("Starting %d question-focused closure jobs", len(jobs))
        await asyncio.gather(*(
            _scan_one(
                client, store, document_hash, q_hash, pass_name, pack,
                requirements, chunks_by_id, force,
                (
                    "Verify only evidence that directly answers this exact question: "
                    + question_map[plan.question_id].question
                    if plan.query_type in {
                        "contradiction", "cross_section", "aggregation", "global_synthesis"
                    }
                    else None
                ),
            )
            for plan, pass_name, pack, requirements in jobs
        ))


def _plan_terms(plan: QuestionPlan) -> list[str]:
    return list(
        dict.fromkeys(
            plan.target_entities
            + plan.required_fields
            + plan.candidate_topics
            + plan.claim_slots
            + plan.relation_types
            + list(plan.aliases)
            + [item for values in plan.aliases.values() for item in values]
        )
    )


def lexical_score(text: str, terms: Sequence[str]) -> float:
    normalized = _norm(text)
    tokens = set(normalized.split())
    score = 0.0
    for term in terms:
        term_norm = _norm(term)
        if not term_norm:
            continue
        if term_norm in normalized:
            score += 5.0 + min(len(term_norm.split()), 4)
        score += sum(1.0 for token in term_norm.split() if token in tokens and len(token) >= 4)
    return score


def evidence_for_question(
    evidence: Sequence[GlobalEvidence], plan: QuestionPlan
) -> list[GlobalEvidence]:
    return [item for item in evidence if plan.question_id in item.consumer_question_ids]


def trusted_evidence_for_question(
    evidence: Sequence[GlobalEvidence], plan: QuestionPlan
) -> list[GlobalEvidence]:
    """Prefer exhaustive shared evidence over supplementary focus candidates."""

    relevant = evidence_for_question(evidence, plan)
    relevant = [
        item for item in relevant
        if not item.pack_id.startswith("focus_")
        or item.pack_id.startswith(f"focus_{plan.question_id}_")
    ]
    # Focus closure supplements exhaustive shared evidence. A short focus hit
    # must never erase a stronger remote quote from the full-document pass.
    return relevant


def semantic_evidence_for_question(
    evidence: Sequence[GlobalEvidence], plan: QuestionPlan
) -> list[GlobalEvidence]:
    """Route compatible shared facts across pass/question boundaries."""

    direct = trusted_evidence_for_question(evidence, plan)
    requested_families = {
        family
        for term in plan.required_fields + plan.target_entities + plan.claim_slots
        for family in [_measurement_family(term)]
        if family
    }
    # A categorical attribute the question groups by — a country, a status, a
    # category — is shared backbone context in the same way a measurement is.
    # Whichever channel happened to record it, the question that counts by that
    # attribute has to see it: an attribute stored under one question's
    # requirement is otherwise invisible to every other question, and a count
    # taken without it silently reports an empty group.
    #
    # Only a question that actually groups by the attribute gets this. The
    # evidence budget is fixed, so routing categorical rows everywhere would
    # displace relation and topic evidence used by other query types.
    requested_attributes = {
        _norm(term)
        for term in plan.required_fields + plan.target_entities
        if _norm(term) and not _measurement_family(term)
    } if plan.query_type == "aggregation" else set()
    if not requested_families and not requested_attributes:
        return direct

    def usable(item: GlobalEvidence) -> bool:
        return (
            item.kind != "uncertain"
            and bool(item.source_quote)
            and (
                not item.pack_id.startswith("focus_")
                or item.pack_id.startswith(f"focus_{plan.question_id}_")
            )
        )

    def matches_attribute(item: GlobalEvidence) -> bool:
        if not requested_attributes or isinstance(item.value_normalized, (int, float)):
            return False
        predicate = _norm(item.predicate)
        if not predicate:
            return False
        return any(
            predicate == attribute
            or attribute in predicate.split()
            or predicate in attribute.split()
            for attribute in requested_attributes
        )

    compatible = [
        item for item in evidence
        if usable(item)
        and (
            _measurement_family(
                " ".join(str(value or "") for value in (item.predicate, item.source_quote))
            ) in requested_families
            or matches_attribute(item)
        )
    ]
    return list({item.id: item for item in [*direct, *compatible]}.values())


async def run_selective_audits(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    shared_plan: SharedEvidencePlan,
    pass_packs: dict[str, list[ScanPack]],
    chunks: Sequence[DocumentChunk],
) -> None:
    statuses = store.statuses(document_hash, q_hash)
    evidence = store.evidence(document_hash, q_hash)
    pack_by_id = {
        item.id: item for packs in pass_packs.values() for item in packs
    }
    chunks_by_id = {item.id: item for item in chunks}
    candidates: list[tuple[float, str, ScanPack, str]] = []
    for pass_name in shared_plan.enabled_passes:
        for pack in pass_packs.get(pass_name, []):
            status = statuses.get((pass_name, pack.id))
            if status not in {"success", "audited"}:
                candidates.append((10_000.0, pass_name, pack, f"recover {status or 'missing'} scan"))
    # Requirement-level lexical shadows catch silent omissions even when the
    # pack response itself was valid JSON. This is essential for global extrema:
    # one omitted measurement can otherwise change the winner.
    definite_keys = {
        (item.pass_name, item.pack_id, item.requirement_key)
        for item in evidence
        if item.kind != "uncertain" and item.source_quote and item.requirement_key
    }
    for pass_name in shared_plan.enabled_passes:
        for requirement in base_requirements_for_pass(shared_plan, pass_name):
            terms = [requirement.description, *requirement.aliases]
            for pack in pass_packs.get(pass_name, []):
                score = lexical_score(pack.text, terms)
                if score > 0 and (pass_name, pack.id, requirement.key) not in definite_keys:
                    critical_bonus = 300.0 if _measurement_family(requirement.description) else 0.0
                    candidates.append(
                        (
                            9_500.0 + critical_bonus + min(score, 400.0),
                            pass_name,
                            pack,
                            f"requirement shadow for {requirement.description}",
                        )
                    )
    # A pack that returned valid JSON can still silently skip whole profiled
    # units when it holds several of them (a multi-entity census pack). A
    # chunk that contributed zero fact evidence to an otherwise-completed
    # census pack is a document-agnostic red flag for that failure mode.
    census_requirements = base_requirements_for_pass(shared_plan, "facts")
    if census_requirements:
        fact_chunk_ids = {
            item.chunk_id for item in evidence
            if item.pass_name == "facts" and item.kind != "uncertain" and item.source_quote
        }
        for pack in pass_packs.get("facts", []):
            if statuses.get(("facts", pack.id)) not in {"success", "audited"}:
                continue
            empty_chunks = [
                chunk_id for chunk_id in pack.chunk_ids if chunk_id not in fact_chunk_ids
            ]
            if empty_chunks:
                candidates.append(
                    (
                        9_800.0 + min(len(empty_chunks), 100.0),
                        "facts",
                        pack,
                        "this pack produced no census facts for "
                        f"{len(empty_chunks)} of its source region(s); re-inspect and "
                        "extract every remaining profiled unit and requested field",
                    )
                )
    for plan in shared_plan.question_plans:
        definite = [item for item in evidence_for_question(evidence, plan) if item.kind != "uncertain"]
        uncertain = [item for item in evidence_for_question(evidence, plan) if item.kind == "uncertain"]
        regular = [name for name in plan.required_passes if name != "needle"]
        if uncertain:
            uncertain_packs = {item.pack_id for item in uncertain}
            for pass_name in regular:
                for pack_id in uncertain_packs:
                    if pack_id in pack_by_id:
                        candidates.append((9_000.0, pass_name, pack_by_id[pack_id], f"resolve uncertainty for {plan.question_id}"))
        if not definite:
            for pass_name in regular:
                requirements = requirements_for_pass(shared_plan, pass_name)
                relevant_terms = _plan_terms(plan)
                ranked = sorted(
                    (
                        (lexical_score(pack.text, relevant_terms), pack)
                        for pack in pass_packs.get(pass_name, [])
                    ),
                    key=lambda item: item[0], reverse=True,
                )
                if ranked and ranked[0][0] > 0 and any(plan.question_id in req.consumer_question_ids for req in requirements):
                    candidates.append((ranked[0][0], pass_name, ranked[0][1], f"lexical shadow for {plan.question_id}"))
    unique = []
    seen = set()
    for score, pass_name, pack, reason in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = (pass_name, pack.id)
        if key in seen:
            continue
        seen.add(key)
        unique.append((pass_name, pack, reason))
        if len(unique) >= client.settings.max_audit_calls:
            break
    await asyncio.gather(
        *(
            _scan_one(
                client, store, document_hash, q_hash, pass_name, pack,
                base_requirements_for_pass(shared_plan, pass_name), chunks_by_id,
                True, reason,
            )
            for pass_name, pack, reason in unique
        )
    )


async def recover_incomplete_base_scans(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    shared_plan: SharedEvidencePlan,
    pass_packs: dict[str, list[ScanPack]],
    chunks: Sequence[DocumentChunk],
    census_passes: Sequence[str] = ("facts",),
) -> None:
    """Retry every incomplete full-document scan before optional focus work."""

    statuses = store.statuses(document_hash, q_hash)
    chunks_by_id = {item.id: item for item in chunks}
    jobs = [
        (pass_name, pack)
        for pass_name in shared_plan.enabled_passes
        for pack in pass_packs.get(pass_name, [])
        if statuses.get((pass_name, pack.id)) not in {"success", "audited"}
    ]
    # Recovery is bounded. A widespread outage is handled by the base-scan
    # success gate instead of spending the complete API budget retrying it.
    jobs = jobs[: client.settings.max_repair_calls]
    if jobs:
        LOGGER.warning("Recovering %d incomplete base scans before closure", len(jobs))
        client.budget.add_expected(len(jobs))
        await asyncio.gather(*(
            # A retried census keeps the census contract: the interrupted pack
            # failed because the wide schema was too much output, so falling
            # back to it here would reproduce the same truncation.
            _scan_one(
                client, store, document_hash, q_hash, pass_name, pack,
                base_requirements_for_pass(shared_plan, pass_name), chunks_by_id, True,
                census=pass_name in census_passes,
            )
            if pass_name in census_passes else
            _scan_one(
                client, store, document_hash, q_hash, pass_name, pack,
                base_requirements_for_pass(shared_plan, pass_name), chunks_by_id, True,
                "complete an interrupted full-document scan; return fewer, fully supported atomic items",
            )
            for pass_name, pack in jobs
        ))


def record_container_names(
    evidence: Sequence[GlobalEvidence], chunks: Sequence[DocumentChunk]
) -> dict[str, str]:
    """The entry each chunk belongs to, keyed by chunk id.

    A measurement often names a component rather than the entry that owns it —
    a summit, a river, a subsidiary — and an answer that reports the component
    where the entry was asked for scores nothing. The census already recorded
    each entry's own name, and the record layout already says which entry a
    chunk sits in, so the container is structural rather than a label the
    extraction model chose.
    """

    chunk_records = {item.id: item.record_ids for item in chunks}
    text_by_chunk = {item.id: item.text for item in chunks}
    # Phase-aligned records begin at their own opening. If several spelling
    # variants were extracted, the earliest grounded identity is therefore the
    # title that owns every fact and observation in the record. Position is a
    # structural signal and is more reliable than model frequency.
    earliest: dict[str, tuple[int, str]] = {}
    for item in evidence:
        if item.pass_name != "facts" or not _is_identity_predicate(item.predicate):
            continue
        name = str(item.value_raw or "").strip()
        if not _looks_like_identity_value(name):
            continue
        body = text_by_chunk.get(item.chunk_id, "")
        position = body.find(name)
        if position < 0:
            continue
        for record_id in chunk_records.get(item.chunk_id, []):
            current = earliest.get(record_id)
            if current is None or position < current[0]:
                earliest[record_id] = (position, name)
    names = {record_id: value[1] for record_id, value in earliest.items()}
    return {
        chunk.id: names[record_id]
        for chunk in chunks
        for record_id in chunk.record_ids
        if record_id in names
    }


def _record_chunks(chunks: Sequence[DocumentChunk]) -> dict[str, list[DocumentChunk]]:
    """Every chunk belonging to each record, in source order.

    A record can span more than one chunk. Keeping only one of them would
    re-scan a fragment during repair and permanently lose whatever sits in the
    rest of the record.
    """

    grouped: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        for record_id in chunk.record_ids:
            grouped[record_id].append(chunk)
    return grouped


def census_gaps(
    evidence: Sequence[GlobalEvidence],
    census_packs: Sequence[ScanPack],
    chunks: Sequence[DocumentChunk],
    attempted: Sequence[str] = (),
) -> tuple[list[str], int]:
    """Records that produced no census fact, and the record denominator.

    A detected record structure supplies the one thing a coverage check
    normally lacks: a denominator. Without it, "this pack returned some facts"
    is the only available signal, and a pack that censused one of its three
    records looks complete. With it, an omission is exact and repairable.

    A record that has already been re-scanned and still yielded nothing is not
    an omission — it is a span with nothing censusable in it, such as a table
    of contents or an index. Those leave the denominator so coverage can
    actually reach closure, and they are never repaired twice.
    """

    record_ids = list(dict.fromkeys(
        record_id for pack in census_packs for record_id in pack.record_ids
    ))
    if not record_ids:
        return [], 0
    chunk_records = {item.id: item.record_ids for item in chunks}
    covered: set[str] = set()
    for item in evidence:
        if item.pass_name != "facts" or item.kind == "uncertain" or not item.source_quote:
            continue
        covered.update(chunk_records.get(item.chunk_id, []))
    exhausted = {
        record_id for record_id in attempted
        if record_id in record_ids and record_id not in covered
    }
    denominator = [item for item in record_ids if item not in exhausted]
    missing = [item for item in denominator if item not in covered]
    return missing, len(denominator)


_GENERIC_FIELD_TOKENS = {
    "per", "year", "years", "name", "names", "total", "number", "count",
    "value", "unit", "units", "the", "and", "for", "with", "from", "data",
    "field", "fields", "record", "records", "item", "items", "profile",
    "profiles", "entry", "entries", "info", "information", "detail", "details",
}


def _field_is_plausible(
    body: str,
    requirement: SharedRequirement,
    observed_values: Optional[Sequence[str]] = None,
) -> bool:
    """Does this record's own text look like it actually states the field?

    Matching on any shared token is useless: "visitors_per_year" contains
    "year", which appears in nearly every entry. Even the distinctive token is
    not enough on its own — most entries mention "visitors" in prose while only
    a few report a visitor figure. For a measured field the wording therefore
    has to sit near a number, which is what a stated measurement looks like and
    what a passing mention does not.
    """

    distinctive = {
        token
        for term in [requirement.description, *requirement.aliases]
        for token in _norm(term).split()
        if len(token) >= 4 and token not in _GENERIC_FIELD_TOKENS
    }
    if not distinctive:
        return False
    haystack = _norm(body)
    # A categorical field is stated by its VALUE, not its label: an entry that
    # sits in a country says the country's name and never the word "country".
    # Judging such a field by its label would mean never chasing it, so the
    # values the field already takes elsewhere in the census are the test.
    for value in observed_values or ():
        candidate = _norm(value)
        if len(candidate) >= 4 and re.search(rf"\b{re.escape(candidate)}\b", haystack):
            return True
    positions = [
        match.start()
        for token in distinctive
        for match in re.finditer(re.escape(token), haystack)
    ]
    if not positions:
        return False
    if not _measurement_family(" ".join([requirement.description, *requirement.aliases])):
        return True
    digits = [match.start() for match in re.finditer(r"\d", haystack)]
    if not digits:
        return False
    return any(
        min(abs(digit - position) for digit in digits) <= 80
        for position in positions
    )


def census_field_gaps(
    evidence: Sequence[GlobalEvidence],
    census_packs: Sequence[ScanPack],
    chunks: Sequence[DocumentChunk],
    requirements: Sequence[SharedRequirement],
    attempted: Sequence[str] = (),
) -> dict[str, list[SharedRequirement]]:
    """Requested fields a record plausibly states but never yielded.

    Record-level coverage is too weak a gate: a record that returns only its
    own name counts as covered while its measurements are silently missing,
    which is exactly how a "how many report X" question loses members. A field
    is only chased when the record's own text shows the field's wording, so a
    record that genuinely lacks the field is never re-scanned forever.
    """

    if not requirements:
        return {}
    record_ids = list(dict.fromkeys(
        record_id for pack in census_packs for record_id in pack.record_ids
    ))
    exhausted = set(attempted)
    text_by_record: dict[str, str] = defaultdict(str)
    for chunk in chunks:
        for record_id in chunk.record_ids:
            text_by_record[record_id] += "\n" + chunk.text
    chunk_records = {item.id: item.record_ids for item in chunks}
    seen: set[tuple[str, str]] = set()
    # Every value a field already takes, so a categorical field can be chased by
    # value rather than by a label the source never writes.
    values: dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        if item.pass_name != "facts" or item.kind == "uncertain" or not item.source_quote:
            continue
        if not item.requirement_key:
            continue
        for record_id in chunk_records.get(item.chunk_id, []):
            seen.add((record_id, item.requirement_key))
        if item.value_raw and not isinstance(item.value_normalized, (int, float)):
            text = str(item.value_raw).strip()
            if 3 <= len(text) <= 40:
                values[item.requirement_key].add(text)
    # Only fields that recur across a meaningful share of the detected records
    # are part of the repeated schema. A one-off attribute may be lexically
    # plausible in many entries, but repairing it record by record wastes the
    # budget and asks the model to invent values that the source never states.
    support = Counter(requirement_key for _, requirement_key in seen)
    minimum_support = (
        1 if len(record_ids) < 5
        else max(3, math.ceil(len(record_ids) * 0.20))
    )
    repeated_requirements = list(requirements) if len(record_ids) < 5 else [
        requirement for requirement in requirements
        # Numeric fields can be intentionally sparse (for example only a few
        # records report attendance). Their nearby label+number is already a
        # strong source-side plausibility gate, so prevalence must not suppress
        # precisely the rare values needed by extrema and aggregation queries.
        if _measurement_family(" ".join([requirement.description, *requirement.aliases]))
        or support.get(requirement.key, 0) >= minimum_support
    ]
    gaps: dict[str, list[SharedRequirement]] = {}
    for record_id in record_ids:
        if record_id in exhausted:
            continue
        body = text_by_record.get(record_id, "")
        if not body:
            continue
        missing = [
            requirement for requirement in repeated_requirements
            if (record_id, requirement.key) not in seen
            and _field_is_plausible(body, requirement, sorted(values.get(requirement.key, ())))
        ]
        if missing:
            gaps[record_id] = missing
    return gaps


async def repair_census_gaps(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    shared_plan: SharedEvidencePlan,
    census_packs: Sequence[ScanPack],
    chunks: Sequence[DocumentChunk],
) -> list[str]:
    """Re-census records that produced nothing, then records missing a field.

    Returns the record ids attempted, so a record that stays empty is counted
    as structurally empty rather than as an unresolved omission.
    """

    requirements = base_requirements_for_pass(shared_plan, "facts")
    if not requirements:
        return []
    evidence = store.evidence(document_hash, q_hash)
    missing, total = census_gaps(evidence, census_packs, chunks)
    field_gaps = census_field_gaps(evidence, census_packs, chunks, requirements)
    if not missing and not field_gaps:
        if total:
            LOGGER.info("Census closure: %d/%d records complete", total, total)
        return []
    LOGGER.warning(
        "Census closure: %d/%d records covered; %d record(s) empty, "
        "%d record(s) missing a requested field",
        total - len(missing), total, len(missing), len(field_gaps),
    )
    chunks_by_id = {item.id: item for item in chunks}
    by_record = _record_chunks(chunks)
    # Keep repair proportional to the census size. The hard setting remains an
    # outage safety ceiling, while a clean large document does not automatically
    # spend dozens of calls chasing sparse optional fields.
    budget = min(
        client.settings.max_repair_calls,
        max(len(missing), 8, math.ceil(total * 0.20)),
    )
    # An empty record loses everything, so it outranks a record that is only
    # missing one field. Field gaps are then taken worst-first.
    ordered: list[tuple[str, Sequence[SharedRequirement]]] = [
        (record_id, requirements) for record_id in missing
    ]
    ordered.extend(
        sorted(
            ((record_id, fields) for record_id, fields in field_gaps.items()
             if record_id not in set(missing)),
            key=lambda entry: -len(entry[1]),
        )
    )
    # Repair *fields*, not calls. The old one-record-per-call closure detected
    # dozens of genuine omissions but could revisit only the first twelve. A
    # repair response is much smaller than the original census response because
    # it transcribes only missing columns, so several whole records fit safely
    # in one pack. This covers roughly three times as many holes without adding
    # latency or API cost.
    measurement_priority = {
        requirement.key
        for requirement in requirements
        if _measurement_family(
            " ".join([requirement.description, *requirement.aliases])
        )
    }
    categorical_priority = {
        requirement.key
        for requirement in requirements
        if any(
            token in _norm(" ".join([requirement.description, *requirement.aliases])).split()
            for token in ("country", "category", "type", "region", "group")
        )
    }

    aggregation_question_ids = {
        plan.question_id
        for plan in shared_plan.question_plans
        if plan.query_type == "aggregation"
    }
    aggregation_priority = {
        requirement.key
        for requirement in requirements
        if aggregation_question_ids.intersection(
            requirement.consumer_question_ids
        )
    }

    def repair_priority(
        entry: tuple[str, Sequence[SharedRequirement]]
    ) -> tuple[int, int, int, int]:
        record_id, wanted = entry
        keys = {item.key for item in wanted}
        group = by_record.get(record_id) or []
        order = group[0].order if group else 10**9
        return (
            -int(bool(keys & aggregation_priority)),
            -int(bool(keys & (measurement_priority | categorical_priority))),
            -len(wanted),
            order,
        )

    ordered.sort(key=repair_priority)
    batches: list[list[tuple[str, Sequence[SharedRequirement], list[DocumentChunk]]]] = []
    current: list[tuple[str, Sequence[SharedRequirement], list[DocumentChunk]]] = []
    current_size = 0
    # At most four records keeps the small model in transcription mode even
    # when the records happen to be very short. PACK_MAX_CHARS remains the hard
    # input boundary for long records and non-catalogue fallback chunks.
    for record_id, wanted in ordered:
        group = by_record.get(record_id) or []
        if not group:
            continue
        size = len(_pack_text(group)) + 80
        if current and (
            len(current) >= 4
            or current_size + size > client.settings.pack_max_chars
        ):
            batches.append(current)
            current = []
            current_size = 0
        current.append((record_id, wanted, group))
        current_size += size
    if current:
        batches.append(current)

    jobs = []
    attempted: list[str] = []
    for batch_index, batch in enumerate(batches[:budget], 1):
        record_ids = [record_id for record_id, _, _ in batch]
        group = sorted(
            {chunk.id: chunk for _, _, chunks_ in batch for chunk in chunks_}.values(),
            key=lambda item: item.order,
        )
        wanted_by_key = {
            requirement.key: requirement
            for _, wanted, _ in batch for requirement in wanted
        }
        attempted.extend(record_ids)
        pages = [
            page for item in group
            for page in (item.page_start, item.page_end) if page is not None
        ]
        pack = ScanPack(
            id=(
                f"repair_batch_{batch_index:02d}_"
                f"{stable_hash(record_ids)[:10]}"
            ),
            order=group[0].order,
            chunk_ids=[item.id for item in group],
            primary_units=list(dict.fromkeys(
                unit for item in group for unit in item.primary_units
            )),
            record_ids=record_ids,
            expected_records=len(record_ids),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            text=_pack_text(group),
            character_count=sum(len(item.text) for item in group),
        )
        jobs.append(
            _scan_one(
                client, store, document_hash, q_hash, "facts", pack,
                list(wanted_by_key.values()), chunks_by_id, True, census=True,
                census_identity_scope=requirements,
                optional_repair=True,
            )
        )
    if jobs:
        client.budget.add_expected(len(jobs))
        await asyncio.gather(*jobs)
    return attempted


async def run_needles(
    client: LLMClient,
    store: SQLiteStore,
    document_hash: str,
    q_hash: str,
    shared_plan: SharedEvidencePlan,
    questions: Sequence[Question],
    chunks: Sequence[DocumentChunk],
    force: bool,
) -> None:
    chunks_by_id = {item.id: item for item in chunks}
    plans = [item for item in shared_plan.question_plans if item.query_type == "needle"]
    question_map = {item.id: item for item in questions}
    jobs = []
    for plan in plans[: client.settings.max_needle_calls]:
        pack_id = f"needle_{plan.question_id}"
        if not force and store.get_scan(document_hash, q_hash, "needle", pack_id) in {"success", "audited"}:
            continue
        requirements = [
            item for item in requirements_for_pass(shared_plan, "needle")
            if plan.question_id in item.consumer_question_ids
        ]
        question = question_map[plan.question_id]
        retrieval_terms = [question.question, *_plan_terms(plan)]
        ranked = sorted(
            ((lexical_score(chunk.text, retrieval_terms), chunk) for chunk in chunks),
            key=lambda item: (item[0], -item[1].order), reverse=True,
        )
        selected = [item[1] for item in ranked[:3] if item[0] > 0]
        if not selected:
            continue
        text = _pack_text(selected)
        pages = [page for item in selected for page in (item.page_start, item.page_end) if page]
        pack = ScanPack(
            id=pack_id, order=0, chunk_ids=[item.id for item in selected],
            primary_units=list(dict.fromkeys(
                unit for item in selected for unit in item.primary_units
            )),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            text=text, character_count=len(text),
        )
        jobs.append(
            _scan_one(
                client, store, document_hash, q_hash, "needle", pack,
                requirements, chunks_by_id, True,
            )
        )
    if jobs:
        await asyncio.gather(*jobs)


_CATEGORICAL_SPLIT_RE = re.compile(
    r"\s*(?:,|;|/|&|\band\b|\bund\b|\bet\b|\bor\b)\s*", re.IGNORECASE
)


def _categorical_members(value: Any) -> list[str]:
    """Every group a categorical value places its entity in.

    A record can legitimately belong to more than one group — an entry that
    straddles a border states both. Keeping the compound string as its own
    bucket would leave that entry out of each real group, so the count for
    every group it belongs to comes out short. The full string is kept as well,
    so an answer can still quote the value exactly as written.
    """

    text = str(value or "").strip()
    if not text:
        return []
    parts = [part.strip(" .") for part in _CATEGORICAL_SPLIT_RE.split(text)]
    parts = [part for part in parts if len(part) >= 2]
    if len(parts) <= 1:
        return [text]
    return list(dict.fromkeys([text, *parts]))


_TOTAL_CUES = (
    "top", "total", "all", "profiles", "profiled", "covers", "covered",
    "features", "featured", "includes", "included", "selection", "chosen",
    "choose", "best", "guide to",
)

_MEASUREMENT_CONTEXT_CUES = (
    "area", "elevation", "altitude", "height", "highest", "lowest",
    "visitors", "attendance", "population", "years", "year", "distance",
    "length", "width", "depth", "metres", "meters", "kilometres",
    "kilometers", "square", "hectares", "feet",
)


def _stated_totals(
    records: Sequence[dict[str, Any]], plan: QuestionPlan
) -> list[dict[str, Any]]:
    """Counts the document states about the collection the question asks about.

    Only a quote that pairs a plain integer with the population's own wording
    and a collection cue qualifies. That keeps a per-entry measurement — an
    area, an elevation, a visitor count — from being read as a collection size.
    """

    unit_tokens = {
        token
        for term in [plan.target_unit, *plan.target_entities]
        for token in _norm(term).split()
        if len(token) >= 4
    }
    if not unit_tokens:
        return []
    found = []
    for item in records:
        value = item.get("value_normalized")
        # A "total" of one is a stray claim about a single entry, never the size
        # of a collection the question is asking about.
        if not isinstance(value, (int, float)) or value != int(value) or not 2 <= value <= 100_000:
            continue
        if item.get("unit"):
            continue
        quote = _norm(item.get("source_quote"))
        if not quote:
            continue
        quote_tokens = set(quote.split())
        if not unit_tokens & quote_tokens:
            continue
        number_text = str(int(value))
        number_match = re.search(
            rf"(?<!\d){re.escape(number_text)}(?!\d)", quote.replace(",", "")
        )
        if not number_match:
            continue
        # Collection wording must be local to the number. A long quote can
        # contain both a catalogue introduction and an unrelated elevation;
        # accepting cues anywhere in that quote turned the elevation into the
        # collection size.
        local = quote[
            max(0, number_match.start() - 90): number_match.end() + 90
        ]
        if not any(cue in local for cue in _TOTAL_CUES):
            continue
        if not (unit_tokens & set(local.split())):
            continue
        # A unitless extractor row can still have come from a visibly measured
        # field. Reject that row when the number's immediate context is a
        # measurement rather than a count of the requested population.
        if any(cue in local for cue in _MEASUREMENT_CONTEXT_CUES) and not any(
            cue in local for cue in ("total", "profiles", "profiled", "top", "selection")
        ):
            continue
        found.append(
            {
                "value": int(value),
                "predicate": item.get("predicate"),
                "source_quote": item.get("source_quote"),
                "page": item.get("page"),
            }
        )
    found.sort(key=lambda entry: -entry["value"])
    return found[:5]


def merge_entity_aliases(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Map every name an entry was recorded under to one canonical name.

    An entry reaches the store under several names: its full title, a short
    form, the region it sits in, the summit it is known for. Counted naively
    each alias becomes another member, so a population of eight is reported as
    ten. Two names carrying the *identical* value for the same measured field
    are the same entry — the document does not hand two entries the same
    measurement by coincidence — and that is a signal no vocabulary is needed
    to read.
    """

    # Question-focus and transposed views deliberately trade context for recall.
    # They are useful answer evidence, but a single mistaken ownership there
    # must not create a global alias edge that changes every count.
    reliable_records = [
        item for item in records
        if not str(item.get("pack_id") or "").startswith(("slot_", "focus_"))
    ] or list(records)
    groups: dict[tuple[str, float], set[str]] = defaultdict(set)
    for item in reliable_records:
        value = item.get("value_normalized")
        family = _measurement_family(str(item.get("predicate") or ""))
        entity = item.get("subject") or item.get("primary_unit")
        if not (family and entity and isinstance(value, (int, float))):
            continue
        if _looks_like_identity_value(entity):
            groups[(family, float(value))].add(str(entity).strip())

    parent: dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    known = sorted({
        str(item.get("subject") or item.get("primary_unit") or "").strip()
        for item in records
        if _looks_like_identity_value(item.get("subject") or item.get("primary_unit"))
    })

    # One shared figure can be coincidence — two entries can genuinely report
    # the same area. A merge therefore needs corroboration: either the names
    # overlap (a short form, or a component named inside the fuller name), or
    # the two agree on more than one measured field.
    agreements: dict[tuple[str, str], int] = defaultdict(int)
    for names in groups.values():
        ordered = sorted(names)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                agreements[(left, right)] += 1

    # Every member of a catalogue carries the same type words ("National
    # Park"), so those tokens say nothing about whether two names are the same
    # entry and are removed before the names are compared.
    recurring_phrases = _recurring_name_suffixes(known)
    shared_vocabulary = {
        token
        for phrase in recurring_phrases
        for token in phrase.split()
    }

    def distinctive_tokens(name: str) -> set[str]:
        return {
            token for token in _norm(name).split()
            if len(token) >= 4 and token not in shared_vocabulary
        }

    def names_overlap(left: str, right: str) -> bool:
        a, b = _norm(left), _norm(right)
        if a.startswith(b + " ") or b.startswith(a + " ") or a == b:
            return True
        return bool(distinctive_tokens(left) & distinctive_tokens(right))

    # Agreement on two independent measured fields is what links an entry known
    # by two unrelated words — a park and the valley it lies in — where no
    # comparison of the names themselves could. Source-region co-occurrence was
    # tried as a third signal and rejected: a detected record boundary can put
    # two entries in one chunk, which merged two genuinely different entries.
    for (left, right), shared in agreements.items():
        if shared >= 2 or names_overlap(left, right):
            union(left, right)

    # A component-name field is an explicit ownership edge. If another row was
    # accidentally stored under that component's name, fold it back into the
    # containing record instead of counting the component as a second member.
    # The predicate test is learned from schema wording, not from domain nouns.
    by_normalized_name = {_norm(name): name for name in known}
    for item in reliable_records:
        owner = item.get("subject") or item.get("primary_unit")
        component = item.get("value_raw")
        predicate = _norm(item.get("predicate"))
        if (
            not _looks_like_identity_value(owner)
            or not _looks_like_identity_value(component)
            or isinstance(item.get("value_normalized"), (int, float))
            or not ({"name", "title", "label", "component"} & set(predicate.split()))
        ):
            continue
        component_name = by_normalized_name.get(_norm(component))
        owner_name = by_normalized_name.get(_norm(owner))
        if component_name and owner_name and component_name != owner_name:
            union(owner_name, component_name)

    # Also fold a short form into the longer name that contains it, which
    # catches aliases whose measurements happened to land in different rows.
    normalized = {name: _norm(name) for name in known}
    for index, left in enumerate(known):
        for right in known[index + 1 :]:
            if normalized[left] and normalized[left] == normalized[right]:
                union(left, right)
    for short in known:
        for long in known:
            if short is long or len(normalized[short]) < 4:
                continue
            if len(normalized[long]) > len(normalized[short]) and (
                normalized[long].startswith(normalized[short] + " ")
            ):
                union(long, short)
    # An entry recorded once with a fuller title and once with a shorter one
    # shares no measurement at all when each name carried different fields, so
    # the rules above never see them. Names that nest on their distinctive
    # words are the same entry: the shorter is the longer with a qualifier
    # dropped, not a second entry that merely starts the same way.
    for short in known:
        for long in known:
            if short is long:
                continue
            short_tokens, long_tokens = distinctive_tokens(short), distinctive_tokens(long)
            if not short_tokens or short_tokens == long_tokens:
                continue
            # Every distinctive word of the shorter name appears in the longer,
            # and the longer adds at most one qualifier — "Foo" and "Foo Lakes"
            # merge, "Foo" and "Foo Valley Coastal Reserve" do not.
            if short_tokens < long_tokens and len(long_tokens - short_tokens) <= 1:
                union(long, short)

    # The name an answer should use is the one the document keeps repeating,
    # not the longest string in the group: a one-off variant that trails a
    # descriptive phrase is longer than the real title and would otherwise win.
    frequency: Counter = Counter()
    for item in reliable_records:
        entity = item.get("subject") or item.get("primary_unit")
        if _looks_like_identity_value(entity):
            frequency[str(entity).strip()] += 1

    canonical: dict[str, str] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for name in set(parent) | set(known):
        members[find(name)].append(name)
    for root, names in members.items():
        best = max(names, key=lambda item: (frequency.get(item, 0), len(_norm(item)), item))
        for name in names:
            canonical[name] = best
    return canonical


def _recurring_name_suffixes(names: Sequence[Any], minimum: int = 2) -> list[str]:
    """Trailing type words most of the observed names share.

    A catalogue's members often carry their category in the name — companies
    may end in "Inc", statutes in "Act", and reports in "Report". Learning that
    tail from the names actually extracted keeps deduplication working on any
    domain, where a fixed list of nouns would fit exactly one.
    """

    cleaned = [_norm(item) for item in names if _looks_like_identity_value(item)]
    cleaned = [item for item in cleaned if len(item.split()) >= 2]
    if len(cleaned) < minimum:
        return []
    counts: Counter = Counter()
    for item in cleaned:
        tokens = item.split()
        for size in (1, 2, 3):
            if len(tokens) > size:
                counts[" ".join(tokens[-size:])] += 1
    # A tail has to be shared by a real share of the population, not by a
    # coincidental pair inside a large collection.
    threshold = max(minimum, len(cleaned) // 4)
    return [
        phrase for phrase, count in counts.items()
        if count >= threshold and count >= len(cleaned) * 0.5
    ]


_SUPERLATIVE_RE = re.compile(
    r"\b(largest|biggest|highest|longest|deepest|oldest|smallest|greatest|"
    r"tallest|widest|most)\b",
    re.IGNORECASE,
)


_NEGATED_SUPERLATIVE_RE = re.compile(
    r"\b(?:not|isn't|is not|no longer|may not be|rather than)\b[^.]{0,40}$",
    re.IGNORECASE,
)


def _superlative_conflicts(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Superlative claims the document's own numbers contradict.

    A statement like "the largest X in region R" is checked against
    the other entries **in that same named group** — never against a larger
    entry somewhere else, which would manufacture a conflict out of nothing.
    The group comes from the categorical facts the census already collected, so
    the scope is the document's own attribution rather than a guess from
    overlapping words. This is the one contradiction the evidence store can
    settle by arithmetic: the claim and the larger figure sit hundreds of pages
    apart, so no single scan holds both, but the fact table holds both.
    """

    measured: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_of: dict[str, set[str]] = defaultdict(set)
    for item in records:
        entity = item.get("subject") or item.get("primary_unit")
        if not entity:
            continue
        value = item.get("value_normalized")
        family = _measurement_family(str(item.get("predicate") or ""))
        if isinstance(value, (int, float)) and family:
            measured[family].append({**item, "_entity": str(entity)})
        elif item.get("value_raw") and not isinstance(value, (int, float)):
            for member in _categorical_members(item["value_raw"]):
                group_of[_norm(entity)].add(_norm(member))

    conflicts = []
    for item in records:
        quote = str(item.get("source_quote") or "")
        match = _SUPERLATIVE_RE.search(quote)
        if not match:
            continue
        # "may not be the region's largest" asserts the opposite; treating it as a
        # superlative claim would invent a contradiction the document denies.
        if _NEGATED_SUPERLATIVE_RE.search(quote[: match.start()]):
            continue
        claimed = item.get("subject") or item.get("primary_unit")
        if not _looks_like_identity_value(claimed):
            continue
        family = _measurement_family(quote) or _measurement_family(
            str(item.get("predicate") or "")
        )
        peers = measured.get(family or "", [])
        own = next(
            (
                peer for peer in peers
                if _norm(peer["_entity"]) == _norm(claimed)
                and isinstance(peer.get("value_normalized"), (int, float))
            ),
            None,
        )
        if own is None:
            continue
        # The scope is whichever group the claim names and the claimant belongs
        # to; without one, the claim cannot be checked and is left alone.
        claimed_groups = group_of.get(_norm(claimed), set())
        quote_norm = _norm(quote)
        scopes = {
            group for group in claimed_groups
            if len(group) >= 4 and group in quote_norm
        }
        if not scopes:
            continue
        bigger = [
            peer for peer in peers
            if _norm(peer["_entity"]) != _norm(claimed)
            and float(peer["value_normalized"]) > float(own["value_normalized"])
            and scopes & group_of.get(_norm(peer["_entity"]), set())
        ]
        if not bigger:
            continue
        rival = max(bigger, key=lambda peer: float(peer["value_normalized"]))
        conflicts.append(
            {
                "claim": {
                    "entity": str(claimed),
                    "value": own.get("value_raw"),
                    "value_normalized": own.get("value_normalized"),
                    "unit": own.get("unit"),
                    "source_quote": item.get("source_quote"),
                    "page": item.get("page"),
                },
                "counterexample": {
                    "entity": rival["_entity"],
                    "value": rival.get("value_raw"),
                    "value_normalized": rival.get("value_normalized"),
                    "unit": rival.get("unit"),
                    "source_quote": rival.get("source_quote"),
                    "page": rival.get("page"),
                },
                "measurement": family,
            }
        )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in conflicts:
        unique.setdefault(
            (_norm(entry["claim"]["entity"]), _norm(entry["counterexample"]["entity"])),
            entry,
        )
    return list(unique.values())[:6]


_UNIT_ALIASES = (
    ("mi2", ("square mile", "squared mile", "sq mile", "sq mi", "mi2", "mi²")),
    ("km2", ("square kilometre", "square kilometer", "squared kilometre",
             "squared kilometer", "sq km", "km2", "km²")),
    ("ha", ("hectare", " ha")),
    ("acre", ("acre",)),
    ("ft", ("feet", "foot", " ft")),
    ("m", ("metre", "meter", " m ")),
)


def _canonical_unit(value: Any) -> Optional[str]:
    """Collapse a written unit to the dimension it measures in.

    "square kilometres", "sq km" and "square kilometers (lake area only)" are
    one unit written three ways; comparing the raw strings would report three
    different units and bury the entry that genuinely uses another system.
    Anything unrecognisable is dropped rather than guessed — an extractor's
    stray "covered" or "none" is not a unit.
    """

    text = f" {_norm(value)} "
    if not text.strip():
        return None
    for canonical, needles in _UNIT_ALIASES:
        for needle in needles:
            normalized = _norm(needle)
            if normalized and re.search(
                rf"(?<!\w){re.escape(normalized)}(?:s)?(?!\w)", text
            ):
                return canonical
    return None


def _group_rankings(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every measured entity ranked inside each group the document assigns it.

    A ranking claim ("the largest in R") is only checkable against the other
    members of R. Handing the writer the sorted table makes the check
    arithmetic instead of a guess, and makes an incorrect pairing visible.
    """

    group_of: dict[str, set[str]] = defaultdict(set)
    for item in records:
        entity = item.get("subject") or item.get("primary_unit")
        value = item.get("value_normalized")
        if not (entity and item.get("value_raw")) or isinstance(value, (int, float)):
            continue
        if item.get("kind") and item.get("kind") != "fact":
            continue
        predicate = _norm(item.get("predicate"))
        if _is_identity_predicate(predicate) or any(
            token in predicate.split()
            for token in ("rank", "claim", "largest", "highest", "lowest", "smallest")
        ):
            continue
        # A unit is not a group. Only a categorical attribute places an entity
        # among peers; measuring everything in square kilometres does not.
        if _canonical_unit(item["value_raw"]) or _measurement_family(str(item.get("predicate") or "")):
            continue
        for member in _categorical_members(item["value_raw"]):
            if 2 <= len(member) <= 40 and not _canonical_unit(member):
                group_of[_norm(entity)].add(member)

    tables: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, str]] = set()
    for item in records:
        entity = item.get("subject") or item.get("primary_unit")
        value = item.get("value_normalized")
        family = _measurement_family(str(item.get("predicate") or ""))
        if not (entity and family and isinstance(value, (int, float))):
            continue
        for group in group_of.get(_norm(entity), set()):
            marker = (family, _norm(group), _norm(entity))
            if marker in seen:
                continue
            seen.add(marker)
            tables[family][group].append(
                {
                    "entity": str(entity),
                    "value": float(value),
                    "value_raw": item.get("value_raw"),
                    "unit": item.get("unit"),
                    "source_quote": item.get("source_quote"),
                    "page": item.get("page"),
                    "chunk_id": item.get("chunk_id"),
                }
            )
    ranked: dict[str, Any] = {}
    for family, groups in tables.items():
        rows = {
            group: sorted(members, key=lambda row: -row["value"])[:6]
            for group, members in groups.items()
            if len(members) >= 2
        }
        if rows:
            ranked[family] = rows
    return ranked


def _ranking_claims(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every superlative assertion found, so the writer works from real claims."""

    claims = []
    for item in records:
        quote = str(item.get("source_quote") or "")
        match = _SUPERLATIVE_RE.search(quote)
        if not match:
            continue
        entity = item.get("subject") or item.get("primary_unit")
        claims.append(
            {
                "entity": str(entity) if entity else None,
                "superlative": match.group(0).casefold(),
                "negated": bool(_NEGATED_SUPERLATIVE_RE.search(quote[: match.start()])),
                "source_quote": quote[:300],
                "page": item.get("page"),
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for claim in claims:
        unique.setdefault(_norm(claim["source_quote"])[:120], claim)
    return sorted(
        unique.values(),
        key=lambda item: (
            item.get("page") is None,
            item.get("page") if item.get("page") is not None else 10**9,
        ),
    )[:40]


def _ranking_claim_measurements(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Exact measured figures embedded in ranking claims and counterclaims."""

    rank_cue = re.compile(
        r"\b(?:rank\w*|largest|biggest|highest|lowest|smallest|tallest|widest|"
        r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\w*\b",
        re.IGNORECASE,
    )
    measured = re.compile(
        r"(?<!\w)(\d[\d,]*(?:\.\d+)?)\s*"
        r"(km\s*[2²]|sq\.?\s*(?:km|mi)|square\s+(?:kilometres?|miles?)|"
        r"metres?|meters?|m|feet|ft|hectares?|ha)\b",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in records:
        quote = str(item.get("source_quote") or "")
        rank_matches = list(rank_cue.finditer(quote))
        if not rank_matches:
            continue
        entity = item.get("subject") or item.get("primary_unit")
        for match in measured.finditer(quote):
            if min(
                min(abs(match.start() - rank.start()), abs(match.end() - rank.end()))
                for rank in rank_matches
            ) > 140:
                continue
            grounded_entity = entity if _norm(entity) and _norm(entity) in _norm(quote) else None
            if grounded_entity is None:
                component = re.search(
                    rf"([A-ZÀ-Þ][\wÀ-ɏ'\-’]+(?:\s+[A-ZÀ-Þ][\wÀ-ɏ'\-’]+){{0,4}})"
                    rf"\s*\(\s*{re.escape(match.group(1))}\s*{re.escape(match.group(2))}",
                    quote,
                )
                if component:
                    grounded_entity = component.group(1)
            marker = (
                _norm(grounded_entity), match.group(1).replace(",", ""),
                _norm(match.group(2)),
            )
            if marker in seen:
                continue
            seen.add(marker)
            rows.append(
                {
                    "entity": grounded_entity,
                    "value": match.group(1),
                    "unit": match.group(2),
                    "source_quote": quote[:420],
                    "page": item.get("page"),
                    "chunk_id": item.get("chunk_id"),
                }
            )
    unique_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("value") or "").replace(",", ""),
            _norm(row.get("unit")),
            str(row.get("chunk_id") or ""),
        )
        previous = unique_rows.get(key)
        if previous is None or (not previous.get("entity") and row.get("entity")):
            unique_rows[key] = row
    return list(unique_rows.values())[:16]


_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_SCOPED_ORDINAL_RE = re.compile(
    r"(?P<scope>[A-ZÀ-Þ][\wÀ-ɏ'’.-]*(?:\s+[A-ZÀ-Þ][\wÀ-ɏ'’.-]*){0,3})"
    r"[’']s\s+(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth)\s*-?\s*(?P<direction>highest|lowest|largest|"
    r"smallest|longest|shortest|oldest|youngest)",
    re.IGNORECASE,
)
_RANKING_MEASUREMENT_RE = re.compile(
    r"(?<!\w)(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>km\s*[2²]|sq\.?\s*(?:km|mi)|square\s+(?:kilometres?|miles?)|"
    r"metres?|meters?|m|feet|ft|hectares?|ha)\b",
    re.IGNORECASE,
)


def _source_ranking_claim_records(
    chunks: Sequence[DocumentChunk],
) -> list[dict[str, Any]]:
    """Source-grounded superlative windows missed by stochastic extraction.

    These rows do not assert a parsed answer. They only retain literal source
    windows for the existing scope/entity/arithmetic join. This makes a failed
    LLM extraction recoverable without encoding any document vocabulary.
    """

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        for match in _SUPERLATIVE_RE.finditer(chunk.text):
            start = max(0, match.start() - 260)
            end = min(len(chunk.text), match.end() + 300)
            # Prefer paragraph/sentence boundaries while retaining enough
            # preceding context to name the entity described by the claim.
            boundary = max(
                chunk.text.rfind("\n\n", start, match.start()),
                chunk.text.rfind(". ", start, match.start()),
            )
            if boundary >= start:
                start = boundary + (2 if chunk.text[boundary:boundary + 2] in {"\n\n", ". "} else 1)
            following = [
                position for position in (
                    chunk.text.find(". ", match.end(), end),
                    chunk.text.find("\n\n", match.end(), end),
                ) if position >= 0
            ]
            if following:
                end = min(following) + 1
            quote = chunk.text[start:end].strip()
            marker = (_norm(quote), chunk.order)
            if not quote or marker in seen:
                continue
            seen.add(marker)
            rows.append({
                "source_quote": quote[:620],
                "page": chunk.page_start,
                "chunk_id": chunk.id,
            })
    # This is a local, source-grounded ledger rather than an LLM prompt. Keep a
    # generous document-scale ceiling so a three-times-larger input does not
    # lose late claims merely because the original document happened to fit an
    # early fixed slice.
    return rows[:1_200]


def _source_ordinal_ranking_conflicts(
    question: Question,
    plan: QuestionPlan,
    chunks: Sequence[DocumentChunk],
) -> list[dict[str, Any]]:
    """Find numerical rank inversions directly in literal source sentences.

    If an item called second-highest has a smaller measured value than an item
    called sixth-highest in the same stated scope, the contradiction is pure
    arithmetic. Extracting this relation deterministically removes run-to-run
    pairing variation while remaining domain- and dataset-independent.
    """

    named_terms = list(dict.fromkeys([
        *plan.target_entities,
        *plan.aliases.keys(),
        *(alias for values in plan.aliases.values() for alias in values),
        *re.findall(
            r"\b[A-ZÀ-Þ][\wÀ-ɏ'’.-]+(?:\s+[A-ZÀ-Þ][\wÀ-ɏ'’.-]+){0,4}",
            question.question,
        ),
    ]))
    requested_families = {
        family
        for term in [
            question.question, plan.target_unit, *plan.required_fields,
            *plan.claim_slots, *plan.target_entities,
        ]
        for family in [_measurement_family(term)]
        if family
    }
    generic_names = {
        "which", "what", "where", "when", "the", "book", "claim", "rank",
        "mountain", "mountains", "park", "parks", "figure", "figures",
    }

    def named_entity(quote: str, measurement: re.Match[str]) -> Optional[str]:
        value = re.escape(measurement.group("value"))
        unit = re.escape(measurement.group("unit"))
        component = re.search(
            rf"([A-ZÀ-Þ][\wÀ-ɏ'’.-]+(?:\s+[A-ZÀ-Þ][\wÀ-ɏ'’.-]+){{0,4}})"
            rf"\s*\(\s*{value}\s*{unit}",
            quote,
        )
        if component:
            candidate = component.group(1).strip()
            # Drop a leading prose verb captured by the permissive proper-name
            # pattern while keeping multiword formal names intact.
            words = candidate.split()
            while len(words) > 1 and words[0].casefold() in {
                "to", "the", "at", "on", "of", "summit"
            }:
                words.pop(0)
            return " ".join(words)
        candidates = []
        quote_norm = _norm(quote)
        for term in named_terms:
            normalized = _norm(term)
            if (
                not normalized
                or normalized in generic_names
                or all(token in generic_names for token in normalized.split())
            ):
                continue
            position = quote_norm.rfind(normalized)
            if position >= 0:
                candidates.append((position, len(normalized), str(term)))
        return max(candidates, default=(0, 0, None))[2]

    statements: list[dict[str, Any]] = []
    for chunk in chunks:
        for rank in _SCOPED_ORDINAL_RE.finditer(chunk.text):
            start = max(0, rank.start() - 480)
            end = min(len(chunk.text), rank.end() + 260)
            quote = chunk.text[start:end].strip()
            relative_rank_start = rank.start() - start
            measures = list(_RANKING_MEASUREMENT_RE.finditer(quote))
            if not measures:
                continue
            measurement = min(
                measures,
                key=lambda item: min(
                    abs(item.start() - relative_rank_start),
                    abs(item.end() - relative_rank_start),
                ),
            )
            if min(
                abs(measurement.start() - relative_rank_start),
                abs(measurement.end() - relative_rank_start),
            ) > 190:
                continue
            statements.append({
                "entity": named_entity(quote, measurement),
                "scope": rank.group("scope"),
                "ordinal": _ORDINAL_WORDS[rank.group("ordinal").casefold()],
                "direction": rank.group("direction").casefold(),
                "value": measurement.group("value"),
                "numeric_value": float(measurement.group("value").replace(",", "")),
                "unit": measurement.group("unit"),
                "source_quote": quote[:760],
                "page": chunk.page_start,
                "chunk_id": chunk.id,
            })

    def same_scope(left: str, right: str) -> bool:
        left_tokens = {_norm(token)[:5] for token in _norm(left).split() if len(token) >= 4}
        right_tokens = {_norm(token)[:5] for token in _norm(right).split() if len(token) >= 4}
        return bool(left_tokens & right_tokens)

    conflicts: list[dict[str, Any]] = []
    terms = [question.question, *_plan_terms(plan)]
    for left_index, left in enumerate(statements):
        for right in statements[left_index + 1:]:
            if (
                left["direction"] != right["direction"]
                or not same_scope(str(left["scope"]), str(right["scope"]))
                or _canonical_unit(left["unit"]) != _canonical_unit(right["unit"])
            ):
                continue
            higher_rank, lower_rank = sorted(
                (left, right), key=lambda item: int(item["ordinal"])
            )
            direction = higher_rank["direction"]
            family = _measurement_family(direction) or direction
            if requested_families and family not in requested_families:
                continue
            inversion = (
                higher_rank["numeric_value"] < lower_rank["numeric_value"]
                if direction in {"highest", "largest", "longest", "oldest"}
                else higher_rank["numeric_value"] > lower_rank["numeric_value"]
            )
            if not inversion:
                continue
            score = lexical_score(
                f"{higher_rank['source_quote']} {lower_rank['source_quote']}", terms
            )
            if score <= 0:
                continue
            conflicts.append({
                "measurement": family,
                "scope": higher_rank["scope"],
                "claim": {
                    key: higher_rank.get(key) for key in (
                        "entity", "value", "unit", "source_quote", "page", "chunk_id"
                    )
                },
                "counterexample": {
                    key: lower_rank.get(key) for key in (
                        "entity", "value", "unit", "source_quote", "page", "chunk_id"
                    )
                },
                "rank_claim": higher_rank["ordinal"],
                "counterexample_rank": lower_rank["ordinal"],
                "relevance": score,
            })
    conflicts.sort(key=lambda item: float(item.get("relevance", 0)), reverse=True)
    return conflicts[:6]


def _claim_ranking_conflicts(
    records: Sequence[dict[str, Any]], rankings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join a scoped superlative claim to its independently ranked peers."""

    conflicts: list[dict[str, Any]] = []
    for item in records:
        quote = str(item.get("source_quote") or "")
        match = _SUPERLATIVE_RE.search(quote)
        if not match or _NEGATED_SUPERLATIVE_RE.search(quote[: match.start()]):
            continue
        superlative = match.group(1).casefold()
        family_for_claim = {
            "largest": "area", "biggest": "area", "widest": "area",
            "highest": "highest", "tallest": "highest",
        }.get(superlative)
        if family_for_claim is None:
            continue
        quote_norm = _norm(quote)
        for family, groups in rankings.items():
            if family != family_for_claim:
                continue
            for group, members in groups.items():
                group_norm = _norm(group)
                if (
                    len(members) < 2
                    or group_norm not in quote_norm
                    or group_norm in {superlative, family_for_claim}
                ):
                    continue
                group_position = quote_norm.find(group_norm)
                claim_position = quote_norm.find(superlative)
                if abs(group_position - claim_position) > 90:
                    continue
                token_sets = [
                    {
                        token for token in _norm(member.get("entity")).split()
                        if len(token) >= 4
                    }
                    for member in members
                ]
                # A shortened alias may omit the collection's type suffix, so
                # strict intersection lets generic words from the first row
                # (for example the shared entity type) masquerade as the
                # claimant. Treat vocabulary carried by a clear majority as
                # shared and identify the claimant on its distinctive tokens.
                token_frequency = Counter(
                    token for tokens in token_sets for token in tokens
                )
                common_tokens = {
                    token for token, count in token_frequency.items()
                    if count >= max(2, math.ceil(len(token_sets) * 0.60))
                }
                if set(group_norm.split()) and set(group_norm.split()) <= common_tokens:
                    continue
                # The superlative must describe the same kind of entity as the
                # ranked table. A quote about a record's "largest glacier" must
                # not be tested against the areas of national parks merely
                # because "national park" occurs later in the sentence.
                immediate_claim_tail = set(
                    quote_norm[claim_position:].split()[:4]
                )
                if common_tokens and not (common_tokens & immediate_claim_tail):
                    continue
                claimant = None
                for member, tokens in zip(members, token_sets):
                    entity_norm = _norm(member.get("entity"))
                    positions = [
                        quote_norm.find(entity_norm) if entity_norm else -1,
                        *(
                            quote_norm.find(token)
                            for token in tokens - common_tokens
                            if token in quote_norm.split()
                        ),
                    ]
                    if any(position >= 0 and abs(position - claim_position) <= 75 for position in positions):
                        claimant = member
                        break
                if claimant is None:
                    continue
                winner = max(members, key=lambda row: float(row["value"]))
                if _norm(winner.get("entity")) == _norm(claimant.get("entity")):
                    continue
                if float(winner["value"]) <= float(claimant["value"]):
                    continue
                conflicts.append(
                    {
                        "measurement": family,
                        "scope": group,
                        "claim": {
                            "entity": claimant.get("entity"),
                            "value": claimant.get("value_raw"),
                            "value_normalized": claimant.get("value"),
                            "unit": claimant.get("unit"),
                            "source_quote": quote,
                            "page": item.get("page"),
                            "chunk_id": item.get("chunk_id"),
                        },
                        "counterexample": {
                            "entity": winner.get("entity"),
                            "value": winner.get("value_raw"),
                            "value_normalized": winner.get("value"),
                            "unit": winner.get("unit"),
                            "source_quote": winner.get("source_quote"),
                            "page": winner.get("page"),
                            "chunk_id": winner.get("chunk_id"),
                        },
                    }
                )
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for conflict in conflicts:
        marker = (
            _norm(conflict["scope"]),
            _norm(conflict["claim"]["entity"]),
            _norm(conflict["counterexample"]["entity"]),
        )
        unique.setdefault(marker, conflict)
    return list(unique.values())[:8]


def _unit_outliers(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entries reporting a field in a unit the rest of the collection does not.

    "One entry is reported in different units from all the others" is a
    question about the shape of the collection, which only the assembled fact
    table can see — each entry on its own looks perfectly normal.
    """

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        family = _measurement_family(str(item.get("predicate") or ""))
        entity = item.get("subject") or item.get("primary_unit")
        canonical = _canonical_unit(item.get("unit"))
        if not (
            family and entity and canonical
            and isinstance(item.get("value_normalized"), (int, float))
        ):
            continue
        rows[family].append({**item, "_entity": str(entity), "_unit": canonical})
    outliers = []
    for family, items in rows.items():
        counts = Counter(item["_unit"] for item in items)
        if len(counts) < 2:
            continue
        majority, majority_count = counts.most_common(1)[0]
        if majority_count < 3:
            continue
        for unit, count in counts.items():
            if unit == majority or count > max(1, majority_count // 10):
                continue
            for item in items:
                if item["_unit"] != unit:
                    continue
                outliers.append(
                    {
                        "measurement": family,
                        "entity": item["_entity"],
                        "unit": item.get("unit"),
                        "canonical_unit": unit,
                        "value": item.get("value_raw"),
                        "majority_unit": majority,
                        "majority_count": majority_count,
                        "source_quote": item.get("source_quote"),
                        "page": item.get("page"),
                    }
                )
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in outliers:
        unique.setdefault(
            (_norm(item["measurement"]), _norm(item["entity"]), _norm(item["unit"])),
            item,
        )
    return list(unique.values())[:6]


_STRONG_TRANSBOUNDARY_RE = re.compile(
    r"\b(?:cross(?:es|ed|ing)?\s+(?:into|across|the\s+border)|straddl\w*|"
    r"extend\w*\s+across|span\w*\s+(?:across|between)|split\s+between|"
    r"shared\s+(?:between|among|by)|twin\w*\s+with|pair\w*\s+with|"
    r"(?:sit|lie)\w*\s+(?:directly\s+)?on\s+(?:the\s+)?border\s+between|"
    r"continu\w*.{0,80}\bacross\b|transnational)\b",
    re.IGNORECASE,
)
_FORMAL_STATUS_RE = re.compile(
    r"\b(?:inscrib\w*|designat\w*|declar\w*|placed\s+on\s+(?:the\s+)?"
    r"(?:[\w-]+\s+){0,4}list|nominat\w*|tentative\s+list|"
    r"(?:[\w-]+\s+){0,3}(?:status|reserve))\b",
    re.IGNORECASE,
)
_DATED_FIRST_RE = re.compile(
    r"\bfirst\b[\s\S]{0,220}\b(?:1[0-9]{3}|20[0-9]{2}|[0-9]{1,2}\s+"
    r"[A-Za-z]{3,9}\s+1[0-9]{3})\b|\b(?:1[0-9]{3}|20[0-9]{2})\b"
    r"[\s\S]{0,220}\bfirst\b",
    re.IGNORECASE,
)


def _compact_record(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "kind", "pack_id", "subject", "primary_unit", "predicate", "value_raw", "value_normalized",
        "unit", "related_entity", "relation_type", "topic", "topic_status",
        "chunk_id", "page", "section", "source_quote", "order",
    )
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "", [])}


def _cross_section_candidates(
    plan: QuestionPlan,
    records: Sequence[dict[str, Any]],
    aliases: Optional[dict[str, str]] = None,
    known_entities: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build a small, relation-precise ledger for cross-entry answers."""

    vocabulary = _norm(" ".join([
        plan.target_unit, *plan.target_entities, *plan.required_fields,
        *plan.candidate_topics, *plan.claim_slots, *plan.relation_types,
        *list(plan.aliases), *(value for values in plan.aliases.values() for value in values),
    ]))
    terms = _plan_terms(plan)
    strict_border = "border" in vocabulary and any(
        cue in vocabulary for cue in ("across", "paired", "twin", "shared", "cross")
    )
    formal_status = any(
        cue in vocabulary for cue in (
            "designation", "status", "inscribed", "nominated", "tentative", "list"
        )
    )
    dated_first = "first" in vocabulary and any(
        cue in vocabulary for cue in ("date", "dated", "when", "year")
    )
    climbing_first = dated_first and any(
        cue in vocabulary for cue in ("climb", "ascent", "mountaineer")
    )

    identity_names = {
        str(item.get("value_raw")).strip()
        for item in records
        if _is_identity_predicate(item.get("predicate"))
        and _looks_like_identity_value(item.get("value_raw"))
    }
    identity_names.update(str(item).strip() for item in known_entities if str(item).strip())
    identity_norms = {
        _norm((aliases or {}).get(name, name)) for name in identity_names
    }
    target_type_tokens = {
        token.rstrip("s")
        for token in _norm(plan.target_unit).split()
        if len(token) >= 4
        and token not in {
            "document", "defined", "entity", "entities", "status", "statuses",
            "relation", "relations", "across", "their", "with", "dates", "first",
        }
    }

    def belongs_to_known_record(item: dict[str, Any]) -> bool:
        candidates = (
            item.get("primary_unit"), item.get("subject"), item.get("related_entity")
        )
        for candidate in candidates:
            if not _looks_like_identity_value(candidate):
                continue
            canonical = (aliases or {}).get(str(candidate).strip(), str(candidate))
            normalized = _norm(canonical)
            if normalized in identity_norms:
                return True
            candidate_tokens = {token.rstrip("s") for token in normalized.split()}
            if target_type_tokens and not (target_type_tokens & candidate_tokens):
                continue
            if any(
                normalized and (normalized in known or known in normalized)
                for known in identity_norms
                if len(known) >= 5
            ):
                return True
        return False

    def is_known_record_name(candidate: Any) -> bool:
        if not _looks_like_identity_value(candidate):
            return False
        canonical = (aliases or {}).get(str(candidate).strip(), str(candidate))
        normalized = _norm(canonical)
        if normalized in identity_norms:
            return True
        return any(
            normalized and len(normalized) >= 5
            and (normalized in known or known in normalized)
            for known in identity_norms
        )

    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in records:
        quote = str(item.get("source_quote") or "")
        if not quote:
            continue
        if strict_border:
            relation_match = _STRONG_TRANSBOUNDARY_RE.search(quote)
            if relation_match is None:
                continue
            if identity_norms and not belongs_to_known_record(item):
                continue
            before = _norm(quote[max(0, relation_match.start() - 120) : relation_match.start()])
            named_target = any(
                token in before.split()
                for candidate in (
                    item.get("primary_unit"), item.get("subject"), item.get("related_entity")
                )
                if is_known_record_name(candidate)
                for token in _norm((aliases or {}).get(str(candidate or ""), candidate)).split()
                if len(token) >= 5
            )
            generic_target = re.search(
                r"\b(?:the\s+)?(?:park|reserve|site|region|area|territory|entry)\b",
                before,
            )
            if not named_target and generic_target is None:
                continue
        if formal_status and not _FORMAL_STATUS_RE.search(quote):
            continue
        if formal_status and identity_norms and not belongs_to_known_record(item):
            continue
        if dated_first and not _DATED_FIRST_RE.search(quote):
            continue
        if climbing_first and not re.search(
            r"\b(?:climb\w*|ascent|mountaineer\w*)\b", quote, re.IGNORECASE
        ):
            continue
        searchable = " ".join(str(item.get(key) or "") for key in (
            "subject", "primary_unit", "predicate", "value_raw", "related_entity",
            "relation_type", "topic", "source_quote",
        ))
        score = lexical_score(searchable, terms)
        if score <= 0:
            continue
        if item.get("kind") == "relation":
            score += 4.0
        if item.get("pack_id", "").startswith(f"focus_{plan.question_id}_"):
            score += 8.0
        # When two extractions reuse the same quote but disagree about its
        # owner, prefer the more specific known record name. This remains a
        # tie-breaker only; it cannot make an irrelevant quote relevant.
        known_candidates = [
            str(candidate)
            for candidate in (item.get("primary_unit"), item.get("subject"))
            if is_known_record_name(candidate)
        ]
        if known_candidates:
            score += min(1.0, max(len(_norm(name).split()) for name in known_candidates) * 0.2)
        ranked.append((score, item))

    chosen: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    per_chunk: Counter = Counter()
    for _, item in sorted(ranked, key=lambda pair: (-pair[0], pair[1].get("order", 0))):
        quote_key = _norm(item.get("source_quote"))
        chunk_id = str(item.get("chunk_id") or "")
        if not quote_key or quote_key in seen_quotes or per_chunk[chunk_id] >= 3:
            continue
        seen_quotes.add(quote_key)
        per_chunk[chunk_id] += 1
        chosen.append(_compact_record(item))
        if len(chosen) >= 32:
            break
    return sorted(chosen, key=lambda item: item.get("order", 0))


def _synthesis_candidates(
    plan: QuestionPlan, records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep strong examples from distinct regions instead of a noisy dump."""

    terms = _plan_terms(plan)
    vocabulary = _norm(" ".join(terms))
    change_oriented = any(
        cue in vocabulary for cue in (
            "recover", "declin", "increase", "decrease", "threat", "extinct",
            "reintroduc", "conservation", "change over time",
        )
    )
    def arc_for(quote: str) -> str:
        if re.search(r"\b(?:recover\w*|reintroduc\w*|rebuil\w*|rebound\w*|return\w*|thrive\w*)\b", quote, re.I):
            return "recovery"
        if re.search(r"\b(?:declin\w*|extinct\w*|threaten\w*|endanger\w*|poach\w*|loss|fewer|fragile)\b", quote, re.I):
            return "threat_or_decline"
        if re.search(r"\b(?:protect\w*|conserv\w*|ban\w*|manage\w*|monitor\w*|breeding|reserve)\b", quote, re.I):
            return "intervention"
        return "persistence_or_context"

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for item in records:
        if item.get("topic_status") == "passing_mention":
            continue
        quote = str(item.get("source_quote") or "")
        searchable = " ".join(str(item.get(key) or "") for key in (
            "subject", "primary_unit", "predicate", "value_raw", "topic", "source_quote",
        ))
        score = lexical_score(searchable, terms)
        if change_oriented and re.search(
            r"\b(?:recover\w*|reintroduc\w*|rebuil\w*|return\w*|declin\w*|"
            r"extinct\w*|threaten\w*|increase\w*|rose|grew|from\s+[^.]{0,40}\s+to)\b",
            quote,
            re.IGNORECASE,
        ):
            score += 8.0
        if item.get("pack_id", "").startswith(f"focus_{plan.question_id}_"):
            score += 8.0
        if score > 0:
            ranked.append((score, arc_for(quote), item))

    chosen: list[dict[str, Any]] = []
    per_chunk: Counter = Counter()
    seen_quotes: set[str] = set()
    ordered = sorted(ranked, key=lambda row: (-row[0], row[2].get("order", 0)))
    # Seed the set with the strongest example from each observed story stage,
    # then fill by relevance. This prevents a long run of threat mentions from
    # displacing the intervention or recovery that gives the synthesis its arc.
    seeded: list[tuple[float, str, dict[str, Any]]] = []
    for arc in ("threat_or_decline", "intervention", "recovery", "persistence_or_context"):
        best = next((row for row in ordered if row[1] == arc), None)
        if best:
            seeded.append(best)
    traversal = seeded + [row for row in ordered if row not in seeded]
    for _, arc, item in traversal:
        chunk_id = str(item.get("chunk_id") or "")
        quote_key = _norm(item.get("source_quote"))
        if not quote_key or quote_key in seen_quotes or per_chunk[chunk_id] >= 2:
            continue
        seen_quotes.add(quote_key)
        per_chunk[chunk_id] += 1
        compact = _compact_record(item)
        compact["story_arc"] = arc
        chosen.append(compact)
        if len(chosen) >= 36:
            break
    return sorted(chosen, key=lambda item: item.get("order", 0))


def reduce_question(
    plan: QuestionPlan,
    evidence: Sequence[GlobalEvidence],
    aliases: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Calculate only over LLM-extracted, source-grounded evidence."""

    relevant = [
        item for item in semantic_evidence_for_question(evidence, plan)
        if item.kind != "uncertain" and item.source_quote
    ]
    unique: dict[tuple[str, ...], GlobalEvidence] = {}
    for item in relevant:
        key = (
            _norm(item.subject), _norm(item.predicate), _norm(item.value_raw),
            _norm(item.unit_raw), _norm(item.related_entity), item.chunk_id,
            _norm(item.source_quote),
        )
        unique.setdefault(key, item)
    relevant = list(unique.values())

    def record(item: GlobalEvidence) -> dict[str, Any]:
        # The stored figure is re-derived rather than trusted whenever its unit
        # cannot belong to the field it measures. An area recorded in kilometres
        # was scaled to metres when it was written, and a store built before
        # this check still holds the inflated number — which would win the
        # largest-area comparison outright.
        unit = item.unit_normalized or item.unit_raw
        family = _measurement_family(f"{item.predicate or ''} {item.requirement_key or ''}")
        mislabelled = family == "area" and _norm(unit) in _LENGTH_UNITS
        normalized = (
            float(item.value_normalized)
            if isinstance(item.value_normalized, (int, float)) and not mislabelled
            else normalize_number(item.value_raw, unit, family)
        )
        return {
            "pass_name": item.pass_name,
            "kind": item.kind,
            "pack_id": item.pack_id,
            "subject": item.subject,
            "primary_unit": item.primary_unit,
            "requirement_key": item.requirement_key,
            "predicate": item.predicate,
            "value_raw": item.value_raw,
            "value_normalized": normalized if normalized is not None else item.value_normalized,
            "unit": item.unit_normalized or item.unit_raw,
            "related_entity": item.related_entity,
            "relation_type": item.relation_type,
            "topic": item.topic,
            "topic_status": item.topic_status,
            "chunk_id": item.chunk_id,
            "page": item.page,
            "section": item.section,
            "source_quote": item.source_quote,
            "order": item.original_order,
        }

    records = [record(item) for item in relevant]
    known_entities = [
        str(item.value_raw).strip()
        for item in evidence
        if item.kind != "uncertain"
        and item.source_quote
        and _is_identity_predicate(item.predicate)
        and _looks_like_identity_value(item.value_raw)
    ]
    result: dict[str, Any] = {
        "evidence_record_count": len(records),
        "source_region_count": len({item.chunk_id for item in relevant}),
    }
    if plan.query_type == "aggregation":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # "Alpha" and "Alpha National Park" are one entity, so the population's
        # own type words are stripped before grouping. The phrases come from the
        # plan's own vocabulary and from the recurring tail of the observed
        # names — never from a fixed list, which would only ever fit one domain.
        removable_phrases = [
            _norm(item) for item in [*plan.target_entities, plan.target_unit]
            if 1 <= len(_norm(item).split()) <= 3 and not _measurement_family(item)
        ]
        removable_phrases.extend(
            _recurring_name_suffixes([
                *(item.get("value_raw") for item in records
                  if _is_identity_predicate(item.get("predicate"))),
                *(item.get("primary_unit") for item in records),
                *(item.get("subject") for item in records),
            ])
        )
        removable_phrases = sorted(
            {phrase for phrase in removable_phrases if phrase}, key=len, reverse=True
        )

        # Fold the aliases one entry was recorded under before any grouping, so
        # a single entry cannot be counted once per name it happens to carry.
        # The map is built from the whole store when supplied: this question's
        # own slice often lacks the second measurement that links two names.
        aliases = aliases or merge_entity_aliases(records)

        def entity_key(value: Any) -> str:
            canonical = aliases.get(str(value or "").strip(), value)
            normalized = _norm(canonical)
            for phrase in removable_phrases:
                normalized = re.sub(rf"\b{re.escape(phrase)}\b", " ", normalized)
            return re.sub(r"\s+", " ", normalized).strip() or _norm(canonical)

        def _is_component_leak(candidate: Any, item: dict[str, Any]) -> bool:
            """A measurement's related_entity names a component (a summit,
            river, ...), not the profiled unit itself. When primary_unit or
            subject repeats that same component name on a measurement
            record, the model mislabeled the component as the container
            rather than describing a genuinely separate profiled unit."""

            if not _measurement_family(item["predicate"]):
                return False
            related = item.get("related_entity")
            return bool(related) and _norm(candidate) == _norm(related)

        # A profiled unit's own identity fact and its measurement facts can
        # land in different (but adjacent, source-ordered) chunks, so linking
        # by chunk_id alone misses cross-chunk profiles. When a record's own
        # subject/primary_unit is unusable (a leaked internal id or field
        # label), fall back to the nearest identity established earlier in
        # source order: profile facts almost always follow their own unit's
        # identity line before the next unit's identity appears.
        identity_by_order: list[tuple[int, str]] = []
        for item in records:
            name = None
            if _is_identity_predicate(item["predicate"]) and _looks_like_identity_value(
                item.get("value_raw")
            ):
                name = str(item["value_raw"])
            elif _looks_like_identity_value(item.get("primary_unit")) and not _is_component_leak(
                item.get("primary_unit"), item
            ):
                name = str(item["primary_unit"])
            if name:
                identity_by_order.append((item["order"], name))
        identity_by_order.sort(key=lambda pair: pair[0])
        identity_orders = [pair[0] for pair in identity_by_order]
        known_identity_norms = {_norm(name) for name in known_entities}
        target_identity_tokens = {
            token for token in _norm(plan.target_unit).split() if len(token) >= 4
        }

        def _nearest_preceding_identity(order: int, max_distance: int = 1) -> Optional[str]:
            index = bisect.bisect_right(identity_orders, order) - 1
            if index < 0:
                return None
            candidate_order, name = identity_by_order[index]
            # A record with no usable subject/primary_unit at all (rather
            # than a corrupted one) may belong to a unit whose own identity
            # fact was never extracted; bound the lookback so it is not
            # silently folded into an unrelated, more distant neighbor.
            return name if order - candidate_order <= max_distance else None

        def record_entity(item: dict[str, Any]) -> str:
            # An identity predicate's own value_raw is the exact profiled
            # name. Otherwise prefer whichever of primary_unit/subject looks
            # like a real, capitalized identity rather than a leaked internal
            # id or a copy of the requirement's own field label (for example
            # "area_km2" or "highest_point_name" showing up as the subject).
            def canonical(value: Any) -> str:
                return aliases.get(str(value or "").strip(), str(value))

            if _is_identity_predicate(item["predicate"]) and _looks_like_identity_value(
                item.get("value_raw")
            ):
                return canonical(item["value_raw"])
            related = item.get("related_entity")
            if (
                _measurement_family(item.get("predicate"))
                and _looks_like_identity_value(related)
                and (
                    _norm(related) in known_identity_norms
                    or target_identity_tokens & set(_norm(related).split())
                )
            ):
                return canonical(related)
            for candidate in (item.get("primary_unit"), item.get("subject")):
                if _looks_like_identity_value(candidate) and not _is_component_leak(candidate, item):
                    return canonical(candidate)
            if _looks_like_identity_value(related) and not _measurement_family(item["predicate"]):
                return canonical(related)
            linked = _nearest_preceding_identity(item["order"])
            if linked:
                return aliases.get(linked.strip(), linked)
            return str(item["subject"] or item["primary_unit"] or item["related_entity"] or item["chunk_id"])

        condition = plan.condition or ""
        has_numeric_condition = bool(
            re.search(r"(>=|<=|>|<|==|=)\s*[-+]?\d", condition)
        )
        identity_records = [
            item for item in records if _is_identity_predicate(item["predicate"])
        ]
        trusted_identity_records = [
            item for item in identity_records
            if str(item.get("pack_id") or "").startswith(("census_", "repair_"))
        ]
        aggregation_families = {
            family
            for term in plan.required_fields + plan.target_entities
            for family in [_measurement_family(term)]
            if family
        }
        numeric_records = [
            item for item in records
            if isinstance(item["value_normalized"], (int, float))
            and (
                not aggregation_families
                or _measurement_family(
                    " ".join(str(value or "") for value in (item["predicate"], item["source_quote"]))
                ) in aggregation_families
            )
        ]
        # The same exact measurement can be emitted by the base census, a
        # gathered slot, and a focus scan under different owners. It is one
        # source observation, not several population members. Keep the row
        # whose owner is independently known and whose pack retains full record
        # context.
        numeric_by_observation: dict[tuple[str, str, float], dict[str, Any]] = {}

        def ownership_quality(item: dict[str, Any]) -> tuple[int, int, int]:
            entity = record_entity(item)
            pack_id = str(item.get("pack_id") or "")
            return (
                1 if _norm(entity) in known_identity_norms else 0,
                1 if pack_id.startswith(("census_", "repair_")) else 0,
                0 if pack_id.startswith("slot_") else 1,
            )

        for item in numeric_records:
            key = (
                str(item.get("chunk_id") or ""),
                str(_measurement_family(item.get("predicate")) or ""),
                float(item["value_normalized"]),
            )
            previous = numeric_by_observation.get(key)
            if previous is None or ownership_quality(item) > ownership_quality(previous):
                numeric_by_observation[key] = item
        numeric_records = list(numeric_by_observation.values())
        population_records = numeric_records if has_numeric_condition else (
            trusted_identity_records or identity_records or records
        )
        for item in population_records:
            key = entity_key(record_entity(item))
            groups[key].append(item)
        qualifying = set(groups)
        result.update(
            {
                "entity_count": len(groups),
                "entities": [
                    record_entity(groups[key][0])
                    for key in sorted(groups)
                ],
            }
        )
        # Attach categorical attributes only to entities established by an
        # identity row; a stray country mention must not create a new entity.
        attribute_records = [item for item in records if not isinstance(item["value_normalized"], (int, float))]
        for item in attribute_records:
            key = entity_key(record_entity(item))
            if key in groups and item not in groups[key]:
                groups[key].append(item)
        # A base identity row may be absent while a source-grounded claim puts
        # an entity in the requested group explicitly. Admit that edge only
        # when entity, group, and target-unit wording co-occur in the quote.
        for item in attribute_records:
            entity = record_entity(item)
            quote_norm = _norm(item.get("source_quote"))
            value_norm = _norm(item.get("value_raw"))
            key = entity_key(entity)
            if (
                entity and value_norm and value_norm in quote_norm
                and _norm(item.get("subject")) in quote_norm
                and target_identity_tokens & set(quote_norm.split())
            ):
                if item not in groups[key]:
                    groups[key].append(item)
        # One attribute arrives under several predicate labels: a census row
        # says "country", a ranking claim says "largest_in_country". Bucketing
        # by the raw label splits the attribute, and a count taken over one
        # bucket misses every entry recorded under the others.
        requested_fields = [
            field for field in plan.required_fields + plan.target_entities
            if _norm(field) and not _measurement_family(field)
        ]

        def attribute_key(predicate: Any) -> str:
            tokens = set(_norm(predicate).split())
            for field in requested_fields:
                field_tokens = {token for token in _norm(field).split() if len(token) >= 4}
                if field_tokens and field_tokens <= tokens:
                    return _norm(field)
            return str(predicate)

        categorical: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for key, values in groups.items():
            entity = record_entity(values[0])
            for item in values:
                if item["predicate"] and item["value_raw"] and not isinstance(item["value_normalized"], (int, float)):
                    # An entry that spans two groups states both ("R and S").
                    # Counting the compound string as its own bucket
                    # drops that entry from each real group it belongs to, so a
                    # per-group count comes out one short.
                    for part in _categorical_members(item["value_raw"]):
                        if _norm(part) not in _norm(item.get("source_quote")):
                            continue
                        if _norm(part) == _norm(entity):
                            continue
                        categorical[attribute_key(item["predicate"])][part].add(str(entity))
        result["categorical_groups"] = {
            predicate: {
                value: {"count": len(entities), "entities": sorted(entities)}
                for value, entities in values.items()
            }
            for predicate, values in categorical.items()
        }
        # Surface only group values explicitly named by the question plan.
        # This joins equivalent labels such as `country` and `is_in_country`
        # without routing every categorical value into the answer prompt.
        requested_vocabulary = {
            _norm(term)
            for term in [
                *plan.target_entities,
                *plan.required_fields,
                *plan.aliases.keys(),
                *(alias for aliases_ in plan.aliases.values() for alias in aliases_),
            ]
            if _norm(term)
        }
        requested_groups: dict[str, set[str]] = defaultdict(set)
        requested_sources: dict[str, set[str]] = defaultdict(set)
        for predicate, values in categorical.items():
            for value, entities in values.items():
                normalized_value = _norm(value)
                if (
                    not normalized_value
                    or normalized_value == _norm(plan.target_unit)
                    or not any(
                    normalized_value == term
                    or (
                        len(normalized_value.split()) > 1
                        and normalized_value in term
                    )
                    for term in requested_vocabulary
                    )
                ):
                    continue
                requested_groups[value].update(entities)
                requested_sources[value].add(predicate)
        # Also recover an explicitly grounded group edge whose base identity
        # row was the field omitted by extraction. This is not a loose mention:
        # owner, group value and requested unit must share one exact quote.
        for item in records:
            if not item.get("predicate") or not item.get("value_raw"):
                continue
            entity = record_entity(item)
            quote_norm = _norm(item.get("source_quote"))
            if (
                not entity or _norm(item.get("subject")) not in quote_norm
                or not (target_identity_tokens & set(quote_norm.split()))
            ):
                continue
            for value in _categorical_members(item.get("value_raw")):
                normalized_value = _norm(value)
                if (
                    normalized_value in requested_vocabulary
                    and normalized_value != _norm(plan.target_unit)
                    and normalized_value in quote_norm
                    and normalized_value != _norm(entity)
                ):
                    requested_groups[value].add(entity)
                    requested_sources[value].add(str(item.get("predicate")))

        def deduplicate_group_members(entities: Iterable[str]) -> list[str]:
            """Collapse short/full names with the same population identity.

            The main population table is keyed by ``entity_key`` but requested
            categorical groups used to return their raw display strings. That
            let one entry appear twice as, for example, a short heading and a
            full formal title. Prefer the most informative grounded spelling
            while retaining the same key used by the population count.
            """

            representatives: dict[str, str] = {}
            for raw in entities:
                canonical = aliases.get(str(raw).strip(), str(raw))
                key = entity_key(canonical)
                previous = representatives.get(key)
                if previous is None or (
                    len(_norm(canonical).split()), len(canonical)
                ) > (
                    len(_norm(previous).split()), len(previous)
                ):
                    representatives[key] = canonical
            return sorted(representatives.values(), key=_norm)

        result["requested_categorical_groups"] = {
            value: {
                "count": len(deduplicate_group_members(entities)),
                "entities": deduplicate_group_members(entities),
                "source_predicates": sorted(requested_sources[value]),
            }
            for value, entities in requested_groups.items()
        }
        result["explicit_population_totals"] = [
            item for item in records
            if "population" in _norm(item["predicate"])
            and isinstance(item["value_normalized"], (int, float))
        ]
        # A count the document states about its own collection. It is reported
        # beside the derived count rather than replacing it: the enumeration is
        # what names the members, while the stated total is what the document
        # claims about its own size, and a census that missed entries will
        # undercount without any local signal that it did.
        result["stated_collection_totals"] = _stated_totals(records, plan)
        match = re.search(r"(>=|<=|>|<|==|=)\s*([-+]?\d[\d,]*(?:\.\d+)?)", condition)
        qualifying_records: list[dict[str, Any]] = []
        if match:
            operator, raw_threshold = match.groups()
            threshold = float(raw_threshold.replace(",", ""))
            compare = {
                ">=": lambda value: value >= threshold,
                "<=": lambda value: value <= threshold,
                ">": lambda value: value > threshold,
                "<": lambda value: value < threshold,
                "=": lambda value: value == threshold,
                "==": lambda value: value == threshold,
            }[operator]

            def matches_requested_measurement(item: dict[str, Any]) -> bool:
                if not plan.required_fields:
                    return True
                observed_family = _measurement_family(
                    " ".join(str(value or "") for value in (
                        item.get("predicate"), item.get("source_quote")
                    ))
                )
                required_families = {
                    family
                    for field in plan.required_fields
                    for family in [_measurement_family(field)]
                    if family
                }
                if required_families and observed_family in required_families:
                    return True
                return any(
                    _norm(field) in _norm(item["predicate"])
                    or _norm(item["predicate"]) in _norm(field)
                    for field in plan.required_fields
                )
            qualifying = {
                key for key, values in groups.items()
                if any(
                    isinstance(item["value_normalized"], (int, float))
                    and matches_requested_measurement(item)
                    and compare(float(item["value_normalized"]))
                    for item in values
                )
            }
            for key in sorted(qualifying):
                candidates = [
                    item for item in groups[key]
                    if isinstance(item["value_normalized"], (int, float))
                    and matches_requested_measurement(item)
                    and compare(float(item["value_normalized"]))
                ]
                if not candidates:
                    continue
                chosen = (
                    min(candidates, key=lambda item: float(item["value_normalized"]))
                    if operator in {"<=", "<"}
                    else max(candidates, key=lambda item: float(item["value_normalized"]))
                )
                qualifying_records.append(
                    {
                        "entity": record_entity(chosen),
                        "predicate": chosen.get("predicate"),
                        "value_raw": chosen.get("value_raw"),
                        "value_normalized": chosen.get("value_normalized"),
                        "unit": chosen.get("unit"),
                        "related_entity": chosen.get("related_entity"),
                        "source_quote": chosen.get("source_quote"),
                        "page": chosen.get("page"),
                        "chunk_id": chosen.get("chunk_id"),
                    }
                )
        result.update(
            {
                "condition": plan.condition,
                "qualifying_count": len(qualifying),
                "qualifying_entities": [
                    record_entity(groups[key][0])
                    for key in sorted(qualifying)
                ],
                "qualifying_records": qualifying_records,
            }
        )
    elif plan.query_type == "superlative":
        requested_families = {
            family
            for field in plan.required_fields + plan.target_entities
            for family in [_measurement_family(field)]
            if family
        }
        numeric = [
            item for item in records
            if isinstance(item["value_normalized"], (int, float))
            and (item["subject"] or item["related_entity"] or item["primary_unit"])
            and (
                not requested_families
                or _measurement_family(item["predicate"]) in requested_families
            )
        ]
        if numeric:
            result["maximum"] = max(numeric, key=lambda item: float(item["value_normalized"]))
            result["minimum"] = min(numeric, key=lambda item: float(item["value_normalized"]))
            result["candidate_count"] = len(numeric)
    elif plan.query_type == "absence":
        matrix = {}
        positive_controls: dict[str, list[dict[str, Any]]] = {}
        for topic in plan.candidate_topics:
            statuses: dict[str, int] = defaultdict(int)
            matched_rows: list[dict[str, Any]] = []
            topic_terms = [topic, *plan.aliases.get(topic, [])]
            concept_tokens = {
                token
                for term in topic_terms
                for token in _norm(term).split()
                if len(token) >= 4
                and token not in {"subject", "topic", "legacy", "status", "designation"}
            }
            for item in records:
                label = _norm(item["topic"] or item["predicate"])
                matches = any(
                    _norm(term) in label
                    or (label and label in _norm(term))
                    for term in topic_terms
                    if _norm(term)
                )
                quote_tokens = set(_norm(item.get("source_quote")).split())
                grounded_tokens = {
                    token for token in concept_tokens
                    if token in quote_tokens
                    or (
                        len(token) >= 6
                        and any(
                            source.startswith(token[:6]) or token.startswith(source[:6])
                            for source in quote_tokens if len(source) >= 6
                        )
                    )
                }
                # normalize_scan_delta already required this exact requirement
                # (including its aliases) to occur in the grounded quote. Do
                # not require the canonical wording a second time here: a
                # source may say "tourists/crowds" for the planned alias
                # "tourist impact" without ever saying "visitor numbers".
                if matches and grounded_tokens:
                    status = str(item["topic_status"] or "substantive")
                    statuses[status] += 1
                    if status == "substantive":
                        matched_rows.append(item)
            matrix[topic] = dict(statuses)
            controls: list[dict[str, Any]] = []
            seen_regions: set[str] = set()
            for item in sorted(matched_rows, key=lambda row: row.get("order", 0)):
                region = str(item.get("chunk_id") or "")
                if region in seen_regions:
                    continue
                seen_regions.add(region)
                controls.append(_compact_record(item))
                if len(controls) == 2:
                    break
            if controls:
                positive_controls[topic] = controls
        result["topic_matrix"] = matrix
        result["topic_positive_controls"] = positive_controls
        result["topics_without_substantive_evidence"] = [
            topic for topic, statuses in matrix.items()
            if statuses.get("substantive", 0) == 0
        ]
    elif plan.query_type == "contradiction":
        terms = _plan_terms(plan)
        ranked_records = sorted(
            records,
            key=lambda item: (
                item.get("pack_id", "").startswith(f"focus_{plan.question_id}_"),
                lexical_score(str(item.get("source_quote") or ""), terms),
            ),
            reverse=True,
        )
        claim_records = [
            item for item in ranked_records
            if lexical_score(
                " ".join(str(item.get(key) or "") for key in (
                    "subject", "predicate", "value_raw", "related_entity", "source_quote"
                )),
                terms,
            ) > 0
        ][:40] or ranked_records[:24]
        slots: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in claim_records:
            slots[f"{_norm(item['subject'])}|{_norm(item['predicate'])}"].append(
                _compact_record(item)
            )
        result["claim_slots"] = dict(slots)
        result["conflict_candidates"] = [
            {"slot": slot, "claims": claims}
            for slot, claims in slots.items()
            if len({
                _norm(item.get("value_raw") or item.get("related_entity"))
                for item in claims
            }) > 1
        ]
        result["superlative_conflicts"] = _superlative_conflicts(records)
        result["unit_outliers"] = _unit_outliers(records)
        # The ranked table the claim has to be checked against. Without it the
        # writer has only loose evidence rows and may assemble unrelated peers.
        result["ranking_claims"] = _ranking_claims(records)
        generic_rank_terms = {
            "rank", "ranking", "claim", "largest", "biggest", "highest", "lowest",
            "smallest", "tallest", "widest", "figure", "figures", "mountain",
            "mountains", "peak", "peaks", "hill", "hills", "elevation",
            "contradictory", "contradiction",
        }
        distinctive_terms = {
            token
            for term in terms
            for token in _norm(term).split()
            if len(token) >= 5 and token not in generic_rank_terms
        }
        measurement_source = [
            item for item in ranked_records
            if not distinctive_terms
            or any(
                source.startswith(term[:4]) or term.startswith(source[:4])
                for term in distinctive_terms
                for source in _norm(item.get("source_quote")).split()
                if len(source) >= 5
            )
        ]
        result["ranking_claim_measurements"] = _ranking_claim_measurements(
            measurement_source or claim_records
        )
    elif plan.query_type == "cross_section":
        result["cross_section_candidates"] = _cross_section_candidates(
            plan, records, aliases, known_entities
        )
    elif plan.query_type == "global_synthesis":
        result["synthesis_candidates"] = _synthesis_candidates(plan, records)
    else:
        result["ordered_records"] = sorted(records, key=lambda item: item["order"])
    return result


CENSUS_QUERY_TYPES = {"aggregation", "superlative", "absence"}


def coverage_for_question(
    plan: QuestionPlan,
    pass_packs: dict[str, list[ScanPack]],
    statuses: dict[tuple[str, str], str],
    evidence: Sequence[GlobalEvidence],
    census: tuple[int, int] = (0, 0),
) -> CoverageEntry:
    regular = [name for name in plan.required_passes if name != "needle"]
    required = {
        (name, pack.id) for name in regular for pack in pass_packs.get(name, [])
    }
    if "needle" in plan.required_passes:
        required.add(("needle", f"needle_{plan.question_id}"))
    completed = {item for item in required if statuses.get(item) in {"success", "audited"}}
    relevant = semantic_evidence_for_question(evidence, plan)
    definite = [item for item in relevant if item.kind != "uncertain" and item.source_quote]
    uncertain = [item for item in relevant if item.kind == "uncertain"]
    definite_requirement_keys = {item.requirement_key for item in definite if item.requirement_key}
    blocking_uncertain = [
        item for item in uncertain
        if not item.requirement_key or item.requirement_key not in definite_requirement_keys
    ]
    failures = []
    if len(completed) != len(required):
        failures.append(f"incomplete required scans: {len(required) - len(completed)}")
    if blocking_uncertain:
        failures.append(f"unresolved requirement uncertainty: {len(blocking_uncertain)}")
    if not definite:
        failures.append("no verified evidence")
    if plan.query_type == "superlative" and not any(
        normalize_number(item.value_normalized or item.value_raw, item.unit_normalized or item.unit_raw) is not None
        for item in definite
    ):
        failures.append("no comparable numeric evidence")
    if plan.query_type == "contradiction" and len({item.chunk_id for item in definite}) < 2:
        failures.append("both claim regions were not verified")
    if plan.query_type == "cross_section" and not any(item.kind in {"claim", "relation"} for item in definite):
        failures.append("no explicit relation evidence")
    if plan.query_type == "global_synthesis" and len({item.chunk_id for item in definite}) < 2:
        failures.append("fewer than two source regions support synthesis")
    covered_records, total_records = census
    # A count, an extreme, or an absence claim is only as trustworthy as the
    # fraction of the record population it actually saw. Report the shortfall
    # rather than letting a partial census read as a complete one.
    if (
        total_records
        and plan.query_type in CENSUS_QUERY_TYPES
        and covered_records < total_records
    ):
        failures.append(
            f"census incomplete: {covered_records}/{total_records} records produced evidence"
        )
    return CoverageEntry(
        question_id=plan.question_id,
        required_passes=plan.required_passes,
        completed_packs=len(completed),
        required_packs=len(required),
        evidence_items=len(definite),
        uncertain_items=len(blocking_uncertain),
        records_with_evidence=covered_records,
        total_records=total_records,
        shadow_candidates=[],
        closure_passed=not failures,
        failures=failures,
    )


def evidence_limit_for(plan: QuestionPlan, settings: Settings, total_records: int) -> int:
    """Never cap a census question below the size of the population it counts.

    "Name every entry that qualifies" cannot be answered from fewer evidence
    items than there are entries, so a fixed cap silently truncates the answer
    before the writer ever sees the full census.
    """

    if plan.query_type in {"aggregation", "superlative"} and total_records:
        return max(settings.evidence_per_question, total_records + 20)
    category_caps = {
        "needle": 16,
        "superlative": 28,
        "absence": 48,
        "contradiction": 44,
        "cross_section": 48,
        "global_synthesis": 56,
    }
    return min(settings.evidence_per_question, category_caps.get(plan.query_type, 48))


def select_evidence(
    question: Question,
    plan: QuestionPlan,
    evidence: Sequence[GlobalEvidence],
    limit: int,
    result: Optional[dict[str, Any]] = None,
    records_by_chunk: Optional[dict[str, list[str]]] = None,
) -> list[GlobalEvidence]:
    direct = [
        item for item in semantic_evidence_for_question(evidence, plan)
        if item.kind != "uncertain" and item.source_quote
    ]
    focus_chunks: set[str] = set()
    focus_subjects: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("chunk_id"):
                focus_chunks.add(str(value["chunk_id"]))
            if value.get("subject"):
                focus_subjects.add(_norm(value["subject"]))
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    # Context expansion follows only answer-ready rows. Expanding every
    # intermediate table (especially claim_slots or ordered records) turns a
    # compact answer prompt back into a document dump and pushes the decisive
    # quote out of attention.
    for key in (
        "maximum", "minimum", "qualifying_records", "superlative_conflicts",
        "unit_outliers", "claim_ranking_conflicts", "cross_section_candidates",
        "synthesis_candidates",
    ):
        collect((result or {}).get(key))
    contextual = [
        item for item in evidence
        if item.kind != "uncertain" and item.source_quote
        and (
            item.chunk_id in focus_chunks
            or (_norm(item.subject) and _norm(item.subject) in focus_subjects)
        )
    ]
    if plan.query_type == "needle":
        # A needle quote often gives the value but relies on its surrounding
        # profile for park/country context (for example, a named tree in a fact
        # box).  Admit only evidence from the exact verified source chunks;
        # subject-wide expansion could import an unrelated remote mention.
        contextual = [item for item in contextual if item.chunk_id in focus_chunks]
    relevant = list({item.id: item for item in [*direct, *contextual]}.values())
    direct_ids = {item.id for item in direct}
    terms = _plan_terms(plan) + TOKEN_RE.findall(question.question)
    def relevance(item: GlobalEvidence) -> float:
        return lexical_score(
            " ".join(str(value or "") for value in (
                item.subject, item.predicate, item.value_raw, item.related_entity,
                item.topic, item.source_quote,
            )),
            terms,
        )

    ranked = sorted(
        relevant,
        key=lambda item: (
            item.pack_id.startswith(f"focus_{plan.question_id}_"),
            item.chunk_id in focus_chunks,
            relevance(item),
            item.id in direct_ids,
            item.confidence,
        ),
        reverse=True,
    )
    # For a census question the unit of diversity is the record, not the chunk:
    # one entry can span several chunks, and reserving a slot per chunk would
    # spend the budget on repeated views of the same entry while other entries
    # go unrepresented.
    census = plan.query_type in {"aggregation", "superlative"} and bool(records_by_chunk)

    def bucket(item: GlobalEvidence) -> str:
        if census:
            ids = (records_by_chunk or {}).get(item.chunk_id) or []
            if ids:
                return ids[0]
        return item.chunk_id

    if not census:
        strong = [
            item for item in ranked
            if relevance(item) > 0
            or item.chunk_id in focus_chunks
            or item.pack_id.startswith(f"focus_{plan.question_id}_")
        ]
        if strong:
            ranked = strong

    selected = []
    seen = set()
    covered: set[str] = set()
    # First preserve source diversity, then fill with the strongest remaining evidence.
    for diversity_key in ("unit", "all"):
        for item in ranked:
            exact = (item.chunk_id, _norm(item.source_quote), _norm(item.predicate), _norm(item.value_raw))
            if exact in seen:
                continue
            if diversity_key == "unit" and bucket(item) in covered:
                continue
            seen.add(exact)
            covered.add(bucket(item))
            selected.append(item)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: item.original_order)


def normalize_answer_batch(value: Any, expected_ids: Sequence[str]) -> dict[str, Any]:
    raw = value.get("answers", value.get("results", [])) if isinstance(value, dict) else value
    if not isinstance(raw, list):
        raise ValueError("answer response must contain a list")
    answers = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id", item.get("question_id", "")))
        raw_answer = item.get("answer", item.get("text", ""))
        if isinstance(raw_answer, (dict, list)):
            raise ValueError(f"answer for {identifier!r} must be a natural-language string")
        answer = str(raw_answer).strip()
        if identifier in expected_ids and answer:
            answers.append(
                {
                    "id": identifier,
                    "answer": answer,
                    "evidence": _strings(item.get("evidence", item.get("sources", []))),
                    "error": None,
                }
            )
    returned = {item["id"] for item in answers}
    missing = set(expected_ids) - returned
    if missing:
        raise ValueError(f"answer model omitted IDs: {sorted(missing)}")
    return {"answers": answers}


def normalize_direct_answer(text: str, question_id: str) -> str:
    """Accept prose directly while remaining compatible with cached JSON replies."""

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json|markdown)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        parsed = tolerant_json(candidate)
        normalized = normalize_answer_batch(parsed, [question_id])
        return normalized["answers"][0]["answer"]
    except (ValueError, ValidationError, KeyError, IndexError):
        pass
    # Older brokers sometimes place literal newlines inside a JSON answer
    # string. Recover that field locally instead of spending another API call.
    if candidate.lstrip().startswith("{"):
        match = re.search(
            r'"answer"\s*:\s*"(.*?)"\s*,\s*"(?:evidence|sources)"',
            candidate,
            re.S,
        )
        if match:
            candidate = match.group(1).replace('\\"', '"').strip()
        else:
            raise ValueError("answer response contained malformed structured output")
    candidate = re.sub(r"^\s*(?:FINAL\s+)?ANSWER\s*:\s*", "", candidate, flags=re.I).strip()
    if not candidate:
        raise ValueError("answer response was empty")
    return candidate


def answer_contract_issues(
    answer: str,
    plan: QuestionPlan,
    result: dict[str, Any],
) -> list[str]:
    """Find omissions that deterministic answer-ready data can prove."""

    normalized = _norm(answer)
    issues: list[str] = []

    def mentions(value: Any) -> bool:
        target = _norm(value)
        if not target:
            return True
        if target in normalized:
            return True
        tokens = {
            token for token in target.split()
            if len(token) >= 4 and token not in {"national", "park", "reserve", "site"}
        }
        return bool(tokens) and tokens <= set(normalized.split())

    if plan.query_type == "aggregation":
        for group, data in result.get("requested_categorical_groups", {}).items():
            entities = data.get("entities", [])
            if not mentions(group) or any(not mentions(entity) for entity in entities):
                issues.append(
                    f"State the verified group {group} count ({data.get('count')}) and "
                    "all members: " + "; ".join(str(entity) for entity in entities)
                )
        if result.get("qualifying_records"):
            count = result.get("qualifying_count")
            if isinstance(count, (int, float)) and not re.search(
                rf"(?<!\d){re.escape(str(int(count)))}(?!\d)", answer
            ):
                issues.append(f"State the verified qualifying count: {int(count)}")
            rows = result.get("qualifying_records", [])
            if len(rows) <= 25:
                missing = [
                    str(row.get("entity")) for row in rows
                    if row.get("entity") and not mentions(row.get("entity"))
                ]
                if missing:
                    issues.append("Include every verified qualifying entity: " + "; ".join(missing))
    elif plan.query_type == "superlative" and result.get("maximum"):
        winner = result["maximum"]
        entity = winner.get("subject") or winner.get("primary_unit") or winner.get("related_entity")
        if entity and not mentions(entity):
            issues.append(f"Name the verified maximum entity: {entity}")
        value = winner.get("value_raw")
        if value and _norm(value) not in normalized:
            issues.append(f"State the verified maximum value exactly: {value}")
    elif plan.query_type == "absence":
        absent = result.get("topics_without_substantive_evidence", [])
        if len(absent) == 1 and not mentions(absent[0]):
            issues.append(f"Name the sole verified absent alternative: {absent[0]}")
        for topic, rows in result.get("topic_positive_controls", {}).items():
            named = [
                row.get("subject") or row.get("primary_unit")
                for row in rows
                if row.get("subject") or row.get("primary_unit")
            ]
            if not mentions(topic) or (named and not any(mentions(item) for item in named)):
                issues.append(
                    f"Confirm the present alternative {topic} with a concrete verified "
                    f"example: {named[0] if named else rows[0].get('source_quote')}"
                )
    elif plan.query_type == "contradiction":
        contradiction_vocabulary = _norm(" ".join([
            plan.target_unit, *_plan_terms(plan)
        ]))
        unit_oriented = any(
            cue in contradiction_vocabulary
            for cue in ("unit", "reported in different", "measurement basis")
        )
        ranking_oriented = any(
            cue in contradiction_vocabulary
            for cue in ("rank", "largest", "biggest", "highest", "lowest", "tallest")
        )
        if result.get("claim_ranking_conflicts"):
            # A second pass is intentional here even when all strings occur:
            # prose can contain both figures yet reverse the counterexample.
            conflict = result["claim_ranking_conflicts"][0]
            claim = conflict.get("claim", {})
            counter = conflict.get("counterexample", {})
            issues.append(
                "State the directional arithmetic explicitly: the document makes the "
                f"claim about {claim.get('entity')} ({claim.get('value')}), but "
                f"{counter.get('entity')} ({counter.get('value')}) is strictly larger"
            )
        if unit_oriented and result.get("unit_outliers"):
            outlier = result["unit_outliers"][0]
            required = [outlier.get("entity"), outlier.get("value"), outlier.get("unit")]
            if any(value and not mentions(value) for value in required):
                issues.append(
                    "Identify the unit outlier with its exact entity, value and unit, "
                    f"then contrast it with the majority unit {outlier.get('majority_unit')}: "
                    f"{outlier.get('entity')} = {outlier.get('value')} {outlier.get('unit')}"
                )
        measurement_rows = result.get("ranking_claim_measurements", []) if ranking_oriented else []
        missing_measurements = [
            row for row in measurement_rows
            if (row.get("entity") and not mentions(row.get("entity")))
            or (row.get("value") and _norm(row.get("value")) not in normalized)
        ]
        if missing_measurements:
            issues.append(
                "Include the exact measured figures attached to both ranking claims: "
                + "; ".join(
                    f"{row.get('entity')}: {row.get('value')} {row.get('unit')}"
                    for row in missing_measurements[:6]
                )
            )
    elif plan.query_type == "cross_section":
        vocabulary = _norm(" ".join([
            plan.target_unit, *_plan_terms(plan)
        ]))
        if "first" in vocabulary and any(
            cue in vocabulary for cue in ("date", "year", "when")
        ):
            years = sorted({
                match.group(0)
                for row in result.get("cross_section_candidates", [])
                for match in re.finditer(
                    r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                    str(row.get("source_quote") or ""),
                )
            })
            missing = [year for year in years if year not in answer]
            if missing:
                issues.append("Include every verified event date: " + ", ".join(missing))
    return issues


def write_submission(path: Path, submission: Submission) -> None:
    ids = [item.id for item in submission.answers]
    if len(ids) != len(set(ids)):
        raise PipelineError("Duplicate IDs in submission")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(submission.model_dump(mode="json"), ensure_ascii=False, indent=2).encode("utf-8")
    if len(payload) >= 32 * 1024 * 1024:
        raise PipelineError("Submission exceeds 32 MB")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise PipelineError(f"Could not write submission: {exc}") from exc


def build_report(
    document: DocumentInfo,
    q_hash: str,
    questions: Sequence[Question],
    chunks: Sequence[DocumentChunk],
    packs: Sequence[ScanPack],
    passes: Sequence[str],
    settings: Settings,
    census_packs: Sequence[ScanPack] = (),
    layout: Optional[RecordLayout] = None,
) -> RunReport:
    planning = math.ceil(len(questions) / settings.plan_batch_size)
    census_packs = list(census_packs) or list(packs)
    extraction = sum(
        len(census_packs if name == "facts" else packs) for name in passes
    )
    focus = min(
        settings.max_focus_calls,
        sum(
            (
                # One lexical-shadow pack plus one bounded full-source context
                # pack for joins; one lexical closure for aggregate/synthesis.
                2 * len([
                    name for name in CATEGORY_TEMPLATES[_category(item)]["required_passes"]
                    if name != "needle"
                ])
                if _category(item) in {"contradiction", "cross_section"}
                else len([
                    name for name in CATEGORY_TEMPLATES[_category(item)]["required_passes"]
                    if name != "needle"
                ])
                if _category(item) in {"aggregation", "global_synthesis"}
                else 1 if _category(item) == "absence" else 0
            )
            for item in questions
        ),
    )
    needles = min(sum(_category(item) == "needle" for item in questions), settings.max_needle_calls)
    answers = math.ceil(len(questions) / settings.answer_batch_size)
    health_probe = 1
    base_calls = (
        health_probe + planning + extraction + focus + settings.max_audit_calls
        + needles + answers
    )
    # Contract repair is elastic. Preserve mandatory extraction and one answer
    # per question first, then use only the total-call headroom that remains.
    answer_repairs = min(
        settings.max_answer_repair_calls,
        answers,
        max(0, settings.max_total_api_calls - base_calls),
    )
    worst = base_calls + answer_repairs
    return RunReport(
        method="llm_first_shared",
        document_hash=document.sha256,
        questions_hash=q_hash,
        chunks=len(chunks),
        packs=len(packs),
        census_packs=len(census_packs),
        records=layout.count if layout else 0,
        record_structure=layout.reason if layout else "not detected",
        passes=list(passes),
        estimated_calls={
            "llm_health_probe": health_probe,
            "llm_planning": planning,
            "llm_shared_extraction": extraction,
            "llm_question_focus_closure": focus,
            "llm_selective_audit_max": settings.max_audit_calls,
            "llm_census_repair_max": settings.max_repair_calls,
            "llm_needle_max": needles,
            "llm_final_answer_calls": answers,
            "llm_answer_repair_max": answer_repairs,
            "cold_start_without_repairs": worst,
            "hard_cap_including_repairs_and_retries": settings.max_total_api_calls,
            "deterministic_answer_shortcuts": 0,
        },
        budget={
            "time_minutes": settings.total_time_budget_minutes,
            "final_reserve_minutes": settings.final_reserve_minutes,
            "extraction_concurrency": settings.extraction_concurrency,
            "reasoning_concurrency": settings.reasoning_concurrency,
            "largest_pack_characters": max((item.character_count for item in packs), default=0),
            "largest_census_pack_characters": max(
                (item.character_count for item in census_packs), default=0
            ),
            "census_records_per_pack": settings.census_records_per_pack,
            "planning_model": settings.planning_model,
            "extraction_model": settings.extraction_model,
            "audit_model": settings.audit_model,
            "answer_model": settings.answer_model,
        },
    )


async def run_llm_first_pipeline(
    *,
    settings: Settings,
    document_path: Path,
    questions_path: Path,
    output_path: Path,
    state_db_path: Path,
    question_id: Optional[str] = None,
    category: Optional[str] = None,
    force: bool = False,
) -> tuple[Submission, RunReport]:
    settings.validate(require_key=True)
    document = normalize_document(document_path)
    questions = [
        item for item in load_questions(questions_path)
        if (question_id is None or item.id == question_id)
        and (category is None or _category(item) == category)
    ]
    if not questions:
        raise PipelineError("No questions matched the supplied filters")
    q_hash = questions_hash(questions)
    layout = detect_document_records(document)
    LOGGER.info("Record structure: %s", layout.reason)
    chunks = chunk_document(document, settings, layout)
    passes = enabled_passes(questions)
    # Facts use the separate record census below. Counting that channel here
    # as if it also consumed every wide pack makes a large document fail the
    # budget check before the census has a chance to widen its grouping.
    wide_pass_count = max(1, sum(name != "facts" for name in passes))
    packs = make_scan_packs(chunks, wide_pass_count, settings)
    # The claims and topics channels have already claimed their share of the
    # extraction budget; the census sizes itself against what is left.
    reserved = len(packs) * len([name for name in passes if name != "facts"])
    # Reserve gathered-view calls before sizing the per-record census. Without
    # this, a larger catalogue can fit each channel independently yet exceed
    # MAX_EXTRACTION_CALLS when the two fact views are combined.
    transposed_packs = make_transposed_packs(document, chunks, layout, settings)
    census_packs = make_census_packs(
        chunks, layout, settings, reserved + len(transposed_packs)
    ) + transposed_packs
    pass_packs = packs_by_pass(passes, packs, census_packs)
    census_passes = ("facts",) if layout.detected else ()
    records_by_chunk = {item.id: item.record_ids for item in chunks}
    report = build_report(
        document, q_hash, questions, chunks, packs, passes, settings, census_packs, layout
    )
    if report.estimated_calls["cold_start_without_repairs"] > settings.max_total_api_calls:
        LOGGER.warning(
            "Cold-start plan is %d calls against MAX_TOTAL_API_CALLS=%d. Starting anyway: "
            "a partial submission beats none, and answers hold their own reserve.",
            report.estimated_calls["cold_start_without_repairs"], settings.max_total_api_calls,
        )
    store = SQLiteStore(state_db_path)
    heartbeat: Optional[asyncio.Task[Any]] = None
    try:
        cached_answers = {} if force else store.get_answers(document.sha256, q_hash)
        if len(cached_answers) == len(questions):
            submission = Submission(
                team=settings.team_name,
                notes="LLM-first Shared-Pass MELT; answers restored from an exact-run LLM cache.",
                answers=[cached_answers[item.id] for item in questions],
            )
            write_submission(output_path, submission)
            return submission, report

        statuses = store.statuses(document.sha256, q_hash)
        pending_extraction = sum(
            1 for pass_name in passes for pack in pass_packs.get(pass_name, [])
            if force or statuses.get((pass_name, pack.id)) not in {"success", "audited"}
        )
        pending_planning = report.estimated_calls["llm_planning"] if force or not store.get_plan(document.sha256, q_hash) else 0
        pending_answers = math.ceil((len(questions) - len(cached_answers)) / settings.answer_batch_size)
        expected = min(
            settings.max_total_api_calls,
            pending_planning + pending_extraction + settings.max_audit_calls
            + report.estimated_calls["llm_question_focus_closure"]
            + report.estimated_calls["llm_needle_max"] + pending_answers
            + min(report.estimated_calls["llm_answer_repair_max"], pending_answers),
        )
        budget = CallBudget(
            settings,
            expected,
            answer_reserve=min(
                settings.max_total_api_calls,
                pending_answers
                + report.estimated_calls["llm_needle_max"]
                + min(report.estimated_calls["llm_answer_repair_max"], pending_answers),
            ),
        )
        client = LLMClient(settings, store, budget, use_cache=not force)
        heartbeat = asyncio.create_task(progress_heartbeat(budget))
        LOGGER.info("LLM-first cold path started: %s", budget.status_text())

        # A sequential, tiny probe prevents a temporary broker outage from
        # fanning out into dozens of guaranteed timeouts and consuming the run.
        budget.add_expected(1)
        try:
            health_text, _ = await client.call(
                model=settings.extraction_model,
                system="You are an API availability probe. Reply with exactly OK.",
                user="Reply with exactly OK.",
                stage="health",
                max_tokens=4,
                bypass_cache=True,
            )
            if "ok" not in health_text.casefold():
                raise PipelineError("API health probe returned an unexpected response")
        except (PipelineError, BudgetExceeded) as exc:
            raise PipelineError(
                "LLM API health check failed; no document work was started and the previous "
                f"submission was preserved. Retry when the broker is responsive: {exc}"
            ) from exc

        # Destructive cache replacement is safe only after the broker has
        # demonstrated that a new run can actually begin.
        if force:
            store.clear_run(document.sha256, q_hash)

        answers_by_id = dict(cached_answers)
        progress_path = output_path.with_suffix(".partial.json")
        write_submission(
            progress_path,
            Submission(
                team=settings.team_name,
                notes="LLM-first Shared-Pass MELT run in progress.",
                answers=[
                    answers_by_id.get(
                        item.id,
                        SubmissionAnswer(
                            id=item.id,
                            answer="Pending LLM evidence analysis.",
                            evidence=[],
                            error="pending",
                        ),
                    )
                    for item in questions
                ],
            ),
        )

        shared_plan = await compile_shared_plan(
            client, store, document, q_hash, questions, chunks, force
        )
        await run_shared_scans(
            client, store, document.sha256, q_hash, shared_plan, pass_packs, chunks,
            force, census_passes,
        )
        await recover_incomplete_base_scans(
            client, store, document.sha256, q_hash, shared_plan, pass_packs, chunks,
            census_passes,
        )
        # A record that produced nothing is an exact, cheap repair target. It
        # runs before any small-model audit so the audit budget is spent on
        # genuine ambiguity rather than on plugging census holes.
        repaired_records: list[str] = []
        if census_passes:
            repaired_records = await repair_census_gaps(
                client, store, document.sha256, q_hash, shared_plan, census_packs, chunks
            )
        base_statuses = store.statuses(document.sha256, q_hash)
        required_base = [
            (name, pack.id) for name in passes for pack in pass_packs.get(name, [])
        ]
        completed_base = sum(
            base_statuses.get(item) in {"success", "audited"} for item in required_base
        )
        if completed_base < max(1, math.ceil(len(required_base) * 0.75)):
            incomplete = len(required_base) - completed_base
            raise PipelineError(
                "Base extraction remained incomplete: "
                f"{completed_base}/{len(required_base)} scans completed ({incomplete} partial/error). "
                "The run stopped before focus/audit/answer calls; resume to retry the remaining scans."
            )
        await run_question_focus_closures(
            client, store, document.sha256, q_hash, questions, shared_plan, chunks, force
        )
        # Narrow needle scans are required evidence, so complete them before
        # spending any remaining budget on optional audits.
        await run_needles(
            client, store, document.sha256, q_hash, shared_plan, questions, chunks, force
        )
        await run_selective_audits(
            client, store, document.sha256, q_hash, shared_plan, pass_packs, chunks
        )

        evidence = store.evidence(document.sha256, q_hash)
        # Store rows retain the extractor's literal fields for auditability.
        # The reduction view adds one deterministic structural correction: all
        # evidence inside a phase-aligned record receives that record's own
        # container identity. Facts are calculations over the container, while
        # claims/topics retain their atomic subject and use primary_unit for
        # provenance. This prevents a summit, neighbouring country or following
        # entry from becoming the owner of a value without discarding its quote.
        container_by_chunk = record_container_names(evidence, chunks)
        evidence = [
            item.model_copy(update={
                "primary_unit": container_by_chunk[item.chunk_id],
                **(
                    {"subject": container_by_chunk[item.chunk_id]}
                    if item.pass_name == "facts" and item.kind == "fact"
                    else {}
                ),
            })
            if item.chunk_id in container_by_chunk else item
            for item in evidence
        ]
        statuses = store.statuses(document.sha256, q_hash)
        all_missing, all_records = census_gaps(evidence, census_packs, chunks)
        # Groups live on the census facts, which a contradiction question never
        # consumes, so the ranking table is built from the whole store once and
        # handed to the questions that need to check a ranking claim.
        global_records = [
            {
                "kind": item.kind, "pass_name": item.pass_name,
                "pack_id": item.pack_id,
                "subject": item.subject, "primary_unit": item.primary_unit,
                "predicate": item.predicate, "value_raw": item.value_raw,
                "value_normalized": item.value_normalized,
                "unit": item.unit_normalized or item.unit_raw,
                "chunk_id": item.chunk_id, "source_quote": item.source_quote,
                "page": item.page, "section": item.section,
            }
            for item in evidence
            if item.kind != "uncertain" and item.source_quote
        ]
        entity_aliases = merge_entity_aliases(global_records)
        canonical_global_records = [
            {
                **record,
                "subject": entity_aliases.get(
                    str(record.get("subject") or ""), record.get("subject")
                ),
                "primary_unit": entity_aliases.get(
                    str(record.get("primary_unit") or ""), record.get("primary_unit")
                ),
            }
            for record in global_records
        ]
        global_rankings = _group_rankings(canonical_global_records)
        source_ranking_claims = _source_ranking_claim_records(chunks)
        ranking_record_pool = [*canonical_global_records, *source_ranking_claims]
        global_claim_conflicts = _claim_ranking_conflicts(
            ranking_record_pool, global_rankings
        )
        global_ranking_claims = _ranking_claims(ranking_record_pool)
        missing_records, total_records = census_gaps(
            evidence, census_packs, chunks, repaired_records
        )
        census_state = (total_records - len(missing_records), total_records)
        if total_records:
            LOGGER.info(
                "Census denominator: %d/%d records carry evidence "
                "(%d repaired, %d span(s) held nothing censusable and left the "
                "denominator of %d)",
                total_records - len(missing_records), total_records,
                len(repaired_records) - len(all_missing),
                all_records - total_records, all_records,
            )
        plan_map = {item.question_id: item for item in shared_plan.question_plans}
        results: dict[str, dict[str, Any]] = {}
        coverage: dict[str, CoverageEntry] = {}
        selected: dict[str, list[GlobalEvidence]] = {}
        for question in questions:
            plan = plan_map[question.id]
            results[question.id] = reduce_question(plan, evidence, entity_aliases)
            plan_vocabulary = _norm(" ".join([
                plan.target_unit, *_plan_terms(plan), question.question
            ]))
            ranking_question = plan.query_type == "contradiction" and any(
                cue in plan_vocabulary for cue in (
                    "rank", "largest", "biggest", "highest", "tallest", "widest"
                )
            )
            if ranking_question and global_rankings:
                claim_terms = [*_plan_terms(plan), *TOKEN_RE.findall(question.question)]
                scored_claims = sorted(
                    (
                        (lexical_score(str(claim.get("source_quote") or ""), claim_terms), claim)
                        for claim in global_ranking_claims
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                relevant_claims = [
                    claim for score, claim in scored_claims[:12] if score > 0
                ]
                results[question.id]["global_ranking_claims"] = relevant_claims
                claim_text = _norm(" ".join(
                    str(claim.get("source_quote") or "") for claim in relevant_claims
                ))
                scoped_rankings = {
                    family: {
                        group: members for group, members in groups.items()
                        if _norm(group) in claim_text
                    }
                    for family, groups in global_rankings.items()
                }
                scoped_rankings = {
                    family: groups for family, groups in scoped_rankings.items() if groups
                }
                if scoped_rankings:
                    results[question.id]["group_rankings"] = scoped_rankings
                requested_families = {
                    family
                    for term in [plan.target_unit, *_plan_terms(plan)]
                    for family in [_measurement_family(term)]
                    if family
                }
                conflicts = [
                    conflict for conflict in global_claim_conflicts
                    if not requested_families
                    or conflict.get("measurement") in requested_families
                ]
                # The claim quote must also be relevant to this question; this
                # prevents a valid contradiction elsewhere in the document
                # from becoming the answer merely because it shares a unit.
                conflicts = [
                    conflict for conflict in conflicts
                    if lexical_score(
                        str(conflict.get("claim", {}).get("source_quote") or ""),
                        claim_terms,
                    ) > 0
                ]
                if conflicts:
                    results[question.id]["claim_ranking_conflicts"] = conflicts[:4]
                    results[question.id]["ranking_claim_measurements"] = [
                        conflicts[0][side] for side in ("claim", "counterexample")
                    ]
            if ranking_question:
                ordinal_conflicts = _source_ordinal_ranking_conflicts(
                    question, plan, chunks
                )
                if ordinal_conflicts:
                    # A literal ordinal/value inversion is stronger than an
                    # LLM-authored pairing. Put it first so both the answer
                    # writer and the directional contract use the verified
                    # pair, retaining any other scoped conflicts afterwards.
                    existing = results[question.id].get(
                        "claim_ranking_conflicts", []
                    )
                    seen_conflicts: set[tuple[str, str, str, str]] = set()
                    merged_conflicts = []
                    for conflict in [*ordinal_conflicts, *existing]:
                        marker = (
                            _norm(conflict.get("claim", {}).get("entity")),
                            str(conflict.get("claim", {}).get("value") or ""),
                            _norm(conflict.get("counterexample", {}).get("entity")),
                            str(conflict.get("counterexample", {}).get("value") or ""),
                        )
                        if marker in seen_conflicts:
                            continue
                        seen_conflicts.add(marker)
                        merged_conflicts.append(conflict)
                    results[question.id]["claim_ranking_conflicts"] = merged_conflicts[:4]
                    decisive = ordinal_conflicts[0]
                    results[question.id]["ranking_claim_measurements"] = [
                        decisive[side] for side in ("claim", "counterexample")
                    ]
            if ranking_question and results[question.id].get("claim_ranking_conflicts"):
                # Once a scoped pair has survived structural ownership and
                # arithmetic, broad candidate ledgers only invite the writer
                # to assemble a second, unrelated contradiction. Keep the
                # decisive pair and its two measurements as the complete
                # answer contract.
                for noisy_key in (
                    "ranking_claims", "global_ranking_claims", "group_rankings",
                    "claim_slots", "conflict_candidates", "superlative_conflicts",
                ):
                    results[question.id].pop(noisy_key, None)
            coverage[question.id] = coverage_for_question(
                plan, pass_packs, statuses, evidence, census_state
            )
            store.put_coverage(document.sha256, q_hash, coverage[question.id])
            selected[question.id] = select_evidence(
                question, plan, evidence,
                evidence_limit_for(plan, settings, total_records),
                results[question.id],
                records_by_chunk,
            )

        pending_questions = [item for item in questions if item.id not in answers_by_id]
        answer_repairs_used = 0
        answer_repair_limit = report.estimated_calls["llm_answer_repair_max"]
        answer_repair_lock = asyncio.Lock()

        async def reserve_answer_repair() -> bool:
            nonlocal answer_repairs_used
            async with answer_repair_lock:
                if answer_repairs_used >= answer_repair_limit:
                    return False
                answer_repairs_used += 1
                budget.add_expected(1)
                return True

        async def answer_one(question: Question) -> tuple[Question, Optional[SubmissionAnswer]]:
            errors = []
            for model in dict.fromkeys((settings.answer_model, settings.extraction_model)):
                try:
                    text, _ = await client.call(
                        model=model,
                        system=ANSWER_SYSTEM,
                        user=single_answer_prompt(
                            question,
                            plan_map[question.id],
                            results[question.id],
                            selected[question.id],
                            coverage[question.id],
                        ),
                        stage="answer",
                        max_tokens=min(settings.reasoning_max_tokens, 3200),
                    )
                    draft = normalize_direct_answer(text, question.id)
                    issues = answer_contract_issues(
                        draft, plan_map[question.id], results[question.id]
                    )
                    if issues and await reserve_answer_repair():
                        try:
                            revised_text, _ = await client.call(
                                model=settings.answer_model,
                                system=ANSWER_SYSTEM,
                                user=answer_repair_prompt(
                                    question,
                                    plan_map[question.id],
                                    results[question.id],
                                    selected[question.id],
                                    coverage[question.id],
                                    draft,
                                    issues,
                                ),
                                stage="answer_repair",
                                max_tokens=min(settings.reasoning_max_tokens, 3200),
                            )
                            revised = normalize_direct_answer(revised_text, question.id)
                            if revised:
                                draft = revised
                        except (PipelineError, BudgetExceeded, ValueError) as exc:
                            LOGGER.warning(
                                "Answer repair skipped for %s: %s", question.id, exc
                            )
                    return question, SubmissionAnswer(
                        id=question.id,
                        answer=draft,
                        evidence=[],
                    )
                except (PipelineError, BudgetExceeded, ValueError) as exc:
                    errors.append(f"{model}: {exc}")
            LOGGER.error("Final LLM answer failed for %s: %s", question.id, " | ".join(errors))
            return question, None

        tasks = [asyncio.create_task(answer_one(question)) for question in pending_questions]
        for task in asyncio.as_completed(tasks):
            question, returned = await task
            answer = returned or SubmissionAnswer(
                id=question.id,
                answer="The LLM answer stage did not complete.",
                evidence=[],
                error="LLM final-answer generation failed",
            )
            if not answer.evidence:
                answer = answer.model_copy(
                    update={
                        "evidence": list(
                            dict.fromkeys(
                                f"page {item.page} / {item.section or item.chunk_id}"
                                for item in selected[question.id]
                            )
                        )[:12]
                    }
                )
            if not coverage[question.id].closure_passed and not answer.error:
                LOGGER.warning(
                    "Coverage warning for %s (answer remains submittable): %s",
                    question.id, "; ".join(coverage[question.id].failures),
                )
            answers_by_id[question.id] = answer
            if not answer.error:
                store.put_answer(document.sha256, q_hash, answer)
            write_submission(
                progress_path,
                Submission(
                    team=settings.team_name,
                    notes="LLM-first Shared-Pass MELT with grounded evidence and coverage checks.",
                    answers=[
                        answers_by_id.get(
                            item.id,
                            SubmissionAnswer(
                                id=item.id,
                                answer="Pending LLM answer generation.",
                                evidence=[],
                                error="pending",
                            ),
                        )
                        for item in questions
                    ],
                ),
            )

        for question in questions:
            answers_by_id.setdefault(
                question.id,
                SubmissionAnswer(
                    id=question.id,
                    answer="The run ended before the LLM answer stage completed.",
                    evidence=[],
                    error="time or API-call budget exhausted",
                ),
            )
        submission = Submission(
            team=settings.team_name,
            notes="LLM-first Shared-Pass MELT with grounded evidence and coverage checks.",
            answers=[answers_by_id[item.id] for item in questions],
        )
        write_submission(output_path, submission)
        LOGGER.info("Execution complete: %s", budget.status_text())
        return submission, report
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        store.close()


def dry_run_report(
    *,
    settings: Settings,
    document_path: Path,
    questions_path: Path,
    question_id: Optional[str] = None,
    category: Optional[str] = None,
) -> RunReport:
    settings.validate(require_key=False)
    document = normalize_document(document_path)
    questions = [
        item for item in load_questions(questions_path)
        if (question_id is None or item.id == question_id)
        and (category is None or _category(item) == category)
    ]
    if not questions:
        raise PipelineError("No questions matched the supplied filters")
    layout = detect_document_records(document)
    chunks = chunk_document(document, settings, layout)
    passes = enabled_passes(questions)
    wide_pass_count = max(1, sum(name != "facts" for name in passes))
    packs = make_scan_packs(chunks, wide_pass_count, settings)
    # The claims and topics channels have already claimed their share of the
    # extraction budget; the census sizes itself against what is left.
    reserved = len(packs) * len([name for name in passes if name != "facts"])
    transposed_packs = make_transposed_packs(document, chunks, layout, settings)
    census_packs = make_census_packs(
        chunks, layout, settings, reserved + len(transposed_packs)
    ) + transposed_packs
    report = build_report(
        document, questions_hash(questions), questions, chunks, packs, passes, settings,
        census_packs, layout,
    )
    if report.estimated_calls["cold_start_without_repairs"] > settings.max_total_api_calls:
        LOGGER.warning(
            "Cold-start plan is %d calls against MAX_TOTAL_API_CALLS=%d: the run will "
            "stop at the cap or the deadline with part of the document unread.",
            report.estimated_calls["cold_start_without_repairs"], settings.max_total_api_calls,
        )
    return report
