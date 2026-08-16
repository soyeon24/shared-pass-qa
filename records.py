"""Generic detection of repeating record structure in a long document.

Many long documents are catalogues: a profile, entry, chapter, or record is
repeated tens of times with the same internal shape. When that is true, the
record — not an arbitrary character window — is the correct unit for a census
extraction, and the record count is the correct denominator for a coverage
check.

The detector is deliberately document-agnostic. It never looks for a
particular phrase. It looks for *any* short block whose normalized text recurs
many times at a regular spacing, and it requires several independent markers to
agree on the same period before it will claim the document is periodic. A
document with no such structure simply yields no records and the pipeline falls
back to sequential chunking.
"""

from __future__ import annotations

import bisect
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence


# A marker must repeat often enough to be a record delimiter rather than a
# coincidence, but not so often that it is ordinary prose punctuation.
MIN_OCCURRENCES = 12
MAX_OCCURRENCES = 400
# Coefficient of variation of the gaps between consecutive occurrences. A true
# per-record marker repeats at a near-constant spacing; prose repeats erratically.
MAX_GAP_VARIATION = 0.45
# A marker that only covers the front third of a document describes a section,
# not the document's record structure.
MIN_DOCUMENT_SPAN = 0.55
# Markers whose counts differ by more than this fraction are not describing the
# same record structure and must not be merged into one consensus.
COUNT_AGREEMENT = 0.15
# How many independent markers must agree before the document is called periodic.
MIN_AGREEING_MARKERS = 2

_BLOCK_RE = re.compile(r"[^\n][^\n]*(?:\n(?=\n)|\Z)")
_STRIP_RE = re.compile(r"[^\w ]+", re.UNICODE)


@dataclass(frozen=True)
class RecordMarker:
    """One candidate per-record delimiter and how regular it looks."""

    label: str
    offsets: tuple[int, ...]
    gap_variation: float
    document_span: float

    @property
    def count(self) -> int:
        return len(self.offsets)

    @property
    def score(self) -> float:
        """Lower is better: regular spacing over as much of the document as possible."""

        return self.gap_variation + (1.0 - self.document_span)


@dataclass(frozen=True)
class DocumentRecord:
    """One detected record: a half-open character span of the document."""

    id: str
    order: int
    start: int
    end: int
    label: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RecordSlot:
    """One repeated section, gathered across every record.

    A catalogue repeats the same sections in every entry — a statistics block,
    a things-to-do block, a wildlife block. Slicing the document by position
    puts one entry's sections together; slicing it by *slot* puts the same
    section from every entry together. For a question about a field that every
    entry reports, the slot view is the one that fits in a single read.
    """

    label: str
    pieces: tuple[tuple[int, int], ...]  # (start, end) spans in source order
    digit_density: float

    @property
    def count(self) -> int:
        return len(self.pieces)

    @property
    def characters(self) -> int:
        return sum(end - start for start, end in self.pieces)


@dataclass(frozen=True)
class RecordLayout:
    """The detected record structure, or an explicitly empty fallback."""

    records: tuple[DocumentRecord, ...]
    markers: tuple[RecordMarker, ...]
    detected: bool
    reason: str

    @property
    def count(self) -> int:
        return len(self.records)

    def record_at(self, offset: int) -> Optional[DocumentRecord]:
        for record in self.records:
            if record.start <= offset < record.end:
                return record
        return None

    def boundaries(self) -> list[int]:
        """Every record start offset, in order. Empty when nothing was detected."""

        return [record.start for record in self.records]

    def slots(self, text: str) -> list[RecordSlot]:
        """Each repeated section gathered across every record, densest first.

        A slot runs from one marker to whichever marker comes next, so the
        slots partition the periodic part of the document. Ranking by digit
        density surfaces the statistics block without naming it: the section
        that carries an entry's figures is the one made mostly of numbers.
        """

        if not self.detected or not self.markers:
            return []
        # Marker detection works on whole blocks, but the same heading is
        # sometimes run together with the preceding sentence instead of
        # standing alone. Those entries are invisible to a block scan, and an
        # entry missing from the gathered view is missing from every count
        # taken over it — so once the label is known it is searched for
        # everywhere, standalone or inline.
        # Searching for the label text finds every occurrence, including the
        # entries where the heading is run together with the preceding sentence
        # and so never formed a block of its own. Block offsets are not merged
        # in: they point a few characters earlier (before a "## " prefix), which
        # would split each of those sections into a stub plus the real body and
        # make one entry look like two.
        occurrences = sorted(
            (match.start(), marker.label)
            for marker in self.markers
            for match in re.finditer(_label_pattern(marker.label), text)
        )
        spans: dict[str, list[tuple[int, int]]] = {}
        for index, (start, label) in enumerate(occurrences):
            end = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(text)
            if end > start:
                spans.setdefault(label, []).append((start, end))
        found = []
        for label, pieces in spans.items():
            body = "".join(text[start:end] for start, end in pieces)
            if not body:
                continue
            digits = sum(character.isdigit() for character in body)
            found.append(
                RecordSlot(
                    label=label,
                    pieces=tuple(pieces),
                    digit_density=digits / len(body),
                )
            )
        return sorted(found, key=lambda slot: -slot.digit_density)


def _label_pattern(label: str) -> re.Pattern[str]:
    """Match a normalized marker label back in the raw text, inline or not."""

    words = [word for word in label.split() if word]
    if not words:
        return re.compile(r"(?!x)x")
    body = r"[\s\W]{0,4}".join(re.escape(word) for word in words)
    return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE)


def _normalize_marker(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = text.lstrip("#").strip()
    text = _STRIP_RE.sub(" ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _candidate_markers(text: str) -> dict[str, list[int]]:
    """Group short blocks by normalized text, keeping their source offsets."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for match in _BLOCK_RE.finditer(text):
        label = _normalize_marker(match.group(0))
        # Long blocks are body prose; single tokens are page numbers and OCR
        # debris. Neither delimits a record.
        if 3 <= len(label) <= 48 and not label.isdigit():
            grouped[label].append(match.start())
    return grouped


def _score_marker(label: str, offsets: Sequence[int], length: int) -> Optional[RecordMarker]:
    if not MIN_OCCURRENCES <= len(offsets) <= MAX_OCCURRENCES or length <= 0:
        return None
    gaps = [offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)]
    if len(gaps) < 5:
        return None
    mean_gap = statistics.mean(gaps)
    if mean_gap <= 0:
        return None
    variation = statistics.pstdev(gaps) / mean_gap
    span = (offsets[-1] - offsets[0]) / length
    if variation > MAX_GAP_VARIATION or span < MIN_DOCUMENT_SPAN:
        return None
    return RecordMarker(
        label=label,
        offsets=tuple(offsets),
        gap_variation=variation,
        document_span=span,
    )


def find_markers(text: str) -> list[RecordMarker]:
    """Every candidate per-record marker, best (most regular) first."""

    markers = []
    for label, offsets in _candidate_markers(text).items():
        marker = _score_marker(label, offsets, len(text))
        if marker is not None:
            markers.append(marker)
    return sorted(markers, key=lambda item: item.score)


def _agreeing(markers: Sequence[RecordMarker]) -> list[RecordMarker]:
    """Markers whose occurrence counts corroborate the best marker's count.

    One regular marker could be an artifact. Several markers independently
    repeating the same number of times is what makes a record count credible.
    """

    if not markers:
        return []
    best = markers[0]
    tolerance = max(1, round(best.count * COUNT_AGREEMENT))
    return [item for item in markers if abs(item.count - best.count) <= tolerance]


def _cycles(markers: Sequence[RecordMarker]) -> list[list[int]]:
    """Group every agreeing marker occurrence into one list per repetition cycle.

    Walking the merged occurrences in source order, a repeated label means the
    previous cycle has ended. This tolerates markers that skip a cycle or vary
    in internal order, which is normal for text recovered from a PDF.
    """

    merged = sorted(
        (offset, marker.label) for marker in markers for offset in marker.offsets
    )
    cycles: list[list[int]] = []
    current: list[int] = []
    seen: set[str] = set()
    for offset, label in merged:
        if label in seen:
            cycles.append(current)
            current = []
            seen = set()
        current.append(offset)
        seen.add(label)
    if current:
        cycles.append(current)
    return cycles


def _block_starts(text: str) -> list[int]:
    return [match.start() for match in _BLOCK_RE.finditer(text)]


def _snap_back(offset: int, anchors: Sequence[int], floor: int) -> int:
    """Move a cut back to the nearest structural anchor at or before it.

    An entry's own title and opening prose sit ahead of its first internal
    marker, with nothing marking where they begin. Physical page starts are
    real structure and a catalogue entry almost always opens on one, so the
    last page start before the cycle recovers the entry's own heading. The
    floor keeps the cut inside the gap between the two cycles.
    """

    if not anchors:
        return offset
    index = bisect.bisect_right(anchors, offset) - 1
    if index < 0:
        return offset
    candidate = anchors[index]
    return candidate if candidate > floor else offset


def _page_opening_score(
    text: str,
    start: int,
    end: int,
    marker_labels: set[str],
) -> float:
    """How much a physical page resembles the opening of a repeated record.

    Repetition discovers the period but an internal repeated heading discovers
    the wrong *phase*: cutting on it puts the tail of one entry beside the head
    of the next. Record openings instead tend to have a short proper title near
    the top, while internal pages begin with prose or one of the repeated
    section labels. The score uses only typography learned from this document.
    """

    body = text[start:end]
    body = re.sub(r"^<!--\s*page-start-marker-\d+\s*-->\s*", "", body)
    head = body[:1_400]
    lines = [re.sub(r"\s+", " ", line).strip(" #") for line in head.splitlines()]
    lines = [line for line in lines if line]
    score = 0.0
    first_label = ""
    for index, line in enumerate(lines[:18]):
        normalized = _normalize_marker(line)
        if not normalized or normalized.isdigit():
            continue
        if not first_label:
            first_label = normalized
        if normalized in marker_labels:
            continue
        if "|" in line or len(line) > 120 or line[-1:] in ".!?":
            continue
        words = re.findall(r"[^\W\d_]+", line, flags=re.UNICODE)
        if not 1 <= len(words) <= 14:
            continue
        capitalized = sum(
            word[:1].isupper() or word.isupper() for word in words
        ) / len(words)
        if capitalized < 0.45:
            continue
        line_score = 3.0 + min(3.0, len(words) / 2)
        if re.match(r"^#{1,6}\s+", body.lstrip()):
            line_score += 2.0
        if index <= 5:
            line_score += 2.0
        score = max(score, line_score)
    # An internal slot is identified by what the page *opens* with. A repeated
    # caption later under a genuine title must not erase that title's signal;
    # many otherwise unrelated catalogues repeat the same one-line deck below
    # every record title.
    if first_label in marker_labels:
        score -= 10.0
    return score


def _opening_small_integers(text: str, start: int, end: int) -> set[int]:
    """Short integer tokens printed near a physical page opening.

    Some repeating documents number their logical entries. The sequence is a
    useful phase signal, but page-marker numbers, years in prose, measurements,
    and list contents must not be mistaken for it. Restricting the observation
    to short display-like lines near the top supplies a generic signal; callers
    only trust it when the same arithmetic progression spans most cycles.
    """

    body = text[start:end]
    body = re.sub(r"^<!--\s*page-start-marker-\d+\s*-->\s*", "", body)
    values: set[int] = set()
    for raw in body.splitlines()[:18]:
        line = re.sub(r"\s+", " ", raw).strip(" #")
        if not line or len(line) > 48 or line[-1:] in ".!?":
            continue
        for token in re.findall(r"(?<!\w)\d{1,3}(?!\w)", line):
            values.add(int(token))
    return values


def _phase_aligned_page_records(
    text: str,
    anchor: RecordMarker,
    markers: Sequence[RecordMarker],
    page_offsets: Sequence[int],
) -> list[DocumentRecord]:
    """Align a repeated internal cycle to the best opening page in each cycle.

    The best repeating marker supplies one occurrence per logical record. For
    every occurrence, search only the preceding cycle for a title-like page.
    This retains recursive record chunking while preventing adjacent records
    from being mixed merely because the most regular marker sits halfway
    through each record.
    """

    if len(anchor.offsets) < MIN_OCCURRENCES or not page_offsets:
        return []
    gaps = [
        right - left for left, right in zip(anchor.offsets, anchor.offsets[1:])
        if right > left
    ]
    if not gaps:
        return []
    typical_gap = int(statistics.median(gaps))
    marker_labels = {marker.label for marker in markers}
    pages = sorted(set(page_offsets))
    cycle_pages: list[list[int]] = []
    sequence_support: Counter[int] = Counter()
    # Learn, rather than assume, an optional entry-number sequence. For cycle
    # i, a displayed value v votes for base v-i. Ordinary repeated list numbers
    # do not agree on one base; a genuine record ordinal does.
    for index, occurrence in enumerate(anchor.offsets):
        lower = (
            anchor.offsets[index - 1]
            if index > 0 else max(0, occurrence - int(typical_gap * 1.15))
        )
        candidates = [page for page in pages if lower < page < occurrence]
        cycle_pages.append(candidates)
        bases: set[int] = set()
        for page in candidates:
            next_index = bisect.bisect_right(pages, page)
            page_end = pages[next_index] if next_index < len(pages) else occurrence
            bases.update(
                value - index
                for value in _opening_small_integers(
                    text, page, min(page_end, occurrence)
                )
            )
        sequence_support.update(bases)
    ordinal_base: Optional[int] = None
    if sequence_support:
        candidate_base, support = sequence_support.most_common(1)[0]
        if support >= max(6, round(len(anchor.offsets) * 0.55)):
            ordinal_base = candidate_base

    starts: list[int] = []
    for index, occurrence in enumerate(anchor.offsets):
        lower = (
            anchor.offsets[index - 1]
            if index > 0 else max(0, occurrence - int(typical_gap * 1.15))
        )
        candidates = cycle_pages[index]
        if not candidates:
            return []
        scored = []
        span = max(1, occurrence - lower)
        for page in candidates:
            next_index = bisect.bisect_right(pages, page)
            page_end = pages[next_index] if next_index < len(pages) else occurrence
            page_end = min(page_end, occurrence)
            phase_penalty = 4.0 * ((page - lower) / span)
            ordinal_bonus = 0.0
            if ordinal_base is not None and ordinal_base + index in _opening_small_integers(
                text, page, page_end
            ):
                ordinal_bonus = 10.0
            scored.append((
                _page_opening_score(
                    text, page, page_end, marker_labels
                ) - phase_penalty + ordinal_bonus,
                page,
            ))
        best_score, best_page = max(scored, key=lambda item: (item[0], item[1]))
        if best_score < 3.0 or (starts and best_page <= starts[-1]):
            return []
        starts.append(best_page)
    # The first printed entry occasionally omits the ordinal that is present
    # on all later entries (for example because the number was placed on an
    # unextractable image). Once a sequence and the later phase are known, use
    # their median spacing to select the nearest title-like opening instead of
    # accidentally absorbing front matter into record zero.
    if ordinal_base is not None and len(starts) >= 7:
        first_page_index = bisect.bisect_right(pages, starts[0])
        first_page_end = (
            pages[first_page_index] if first_page_index < len(pages) else anchor.offsets[0]
        )
        first_values = _opening_small_integers(
            text, starts[0], min(first_page_end, anchor.offsets[0])
        )
        if ordinal_base not in first_values:
            later_gaps = [
                right - left for left, right in zip(starts[1:], starts[2:])
                if right > left
            ]
            expected = starts[1] - int(statistics.median(later_gaps))
            title_candidates: list[tuple[int, int]] = []
            for page in cycle_pages[0]:
                next_index = bisect.bisect_right(pages, page)
                page_end = (
                    pages[next_index]
                    if next_index < len(pages) else anchor.offsets[0]
                )
                if _page_opening_score(
                    text, page, min(page_end, anchor.offsets[0]), marker_labels
                ) >= 3.0:
                    title_candidates.append((abs(page - expected), page))
            if title_candidates:
                starts[0] = min(title_candidates)[1]
    lengths = [right - left for left, right in zip(starts, starts[1:])]
    if len(lengths) >= 5:
        mean = statistics.mean(lengths)
        if mean <= 0 or statistics.pstdev(lengths) / mean > 0.55:
            return []
    final_end = len(text)
    post_limit = min(len(text), starts[-1] + int(typical_gap * 1.5))
    for page in (item for item in pages if anchor.offsets[-1] < item < post_limit):
        next_index = bisect.bisect_right(pages, page)
        page_end = pages[next_index] if next_index < len(pages) else post_limit
        if _page_opening_score(text, page, page_end, marker_labels) >= 6.0:
            final_end = page
            break
    return [
        DocumentRecord(
            id=f"record_{index:04d}",
            order=index,
            start=start,
            end=starts[index + 1] if index + 1 < len(starts) else final_end,
            label=anchor.label,
        )
        for index, start in enumerate(starts)
    ]


def detect_records(text: str, page_offsets: Sequence[int] = ()) -> RecordLayout:
    """Segment the document into repeating records, or report that it is not periodic."""

    if not text:
        return RecordLayout((), (), False, "empty document")
    markers = find_markers(text)
    if not markers:
        return RecordLayout((), (), False, "no marker repeats at a regular spacing")
    agreeing = _agreeing(markers)
    if len(agreeing) < MIN_AGREEING_MARKERS:
        return RecordLayout(
            (), tuple(markers[:5]), False,
            f"only {len(agreeing)} marker(s) agree on a record count; need {MIN_AGREEING_MARKERS}",
        )

    anchor = min(agreeing, key=lambda item: (item.score, -item.count))
    aligned = _phase_aligned_page_records(text, anchor, markers, page_offsets)
    if aligned:
        return RecordLayout(
            records=tuple(aligned),
            markers=tuple(agreeing[:5]),
            detected=True,
            reason=(
                f"{len(aligned)} records phase-aligned from {anchor.count} "
                f"repetitions; {len(agreeing)} markers corroborate the period "
                f"(anchor {anchor.label!r})"
            ),
        )

    # A record boundary belongs *between* two cycles, not on a marker: the
    # entry's own title and opening prose precede its first internal marker,
    # so cutting on a marker would attach that opening to the previous record.
    cycles = [cycle for cycle in _cycles(agreeing) if cycle]
    if len(cycles) < MIN_OCCURRENCES:
        return RecordLayout((), tuple(markers[:5]), False, "too few repetition cycles")
    anchors = sorted(page_offsets) or _block_starts(text)
    bounds = [0]
    for previous, following in zip(cycles, cycles[1:]):
        cut = _snap_back(min(following), anchors, floor=max(previous))
        if cut > bounds[-1]:
            bounds.append(cut)
    records = []
    for order, start in enumerate(bounds):
        end = bounds[order + 1] if order + 1 < len(bounds) else len(text)
        if end - start <= 0:
            continue
        records.append(
            DocumentRecord(
                id=f"record_{len(records):04d}",
                order=len(records),
                start=start,
                end=end,
                label=anchor.label,
            )
        )
    if len(records) < MIN_OCCURRENCES:
        return RecordLayout((), tuple(markers[:5]), False, "too few usable record spans")
    return RecordLayout(
        records=tuple(records),
        markers=tuple(agreeing[:5]),
        detected=True,
        reason=(
            f"{len(records)} records from {len(cycles)} repetition cycles; "
            f"{len(agreeing)} markers agree on the count (anchor {anchor.label!r})"
        ),
    )
