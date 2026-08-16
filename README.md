# LLM-First Shared-Pass MELT-QA

> **AICOSS Summer School 2026 — Long-Context Challenge (DASU Capstone), 1st place.**
> This repository is the exact pipeline behind the winning submission (internally `v11_54`).
> `results/final_submission.json` is the submitted answer set.
>
> The competition document and question set are third-party material and are not
> redistributed here; `input/` is gitignored. Supply your own document and questions
> as described under [Usage](#usage).

An LLM-first system for grounded question answering over long documents. Every cold
run uses language models to plan the questions, extract evidence from the complete
document, selectively verify uncertain evidence, and write the final answers.
Python never answers directly from the raw document.

## Method

```text
Document + Questions
        │
        ├── LLM Joint Planning
        │      └── question-specific evidence requirements
        │
        ├── Shared Evidence Schema
        │      └── reusable facts, claims, relations, topics, and themes
        │
        ├── Structure-Aware Document Packs
        │      └── complete-document shared extraction with Ministral 3B
        │
        ├── Grounded Global Evidence Store
        │      └── exact quotes, pages, sections, entities, values, and consumers
        │
        ├── Python Verification and Reduction
        │      └── quote grounding, units, count, max/min, deduplication, coverage
        │
        ├── Selective LLM Audit
        │      └── failed packs, uncertainty, and strong lexical shadows only
        │
        └── LLM Final Answering
               └── verified evidence + deterministic calculations only
```

### LLM responsibilities

- Compile all questions into semantic evidence plans
- Interpret unfamiliar document structures and terminology
- Extract atomic evidence from every required document pack
- Preserve claims, relations, negations, qualifiers, and thematic observations
- Recheck unresolved evidence with raw source context
- Compose every final answer from verified evidence

### Python responsibilities

- Preserve document structure, offsets, physical pages, and source order
- Schedule bounded concurrent model calls
- Ground every quote back to the supplied source
- Store evidence monotonically in SQLite
- Normalize units and perform deterministic calculations
- Measure coverage and prevent unsupported closure
- Cache exact LLM requests and resume interrupted runs
- Write a valid submission atomically

The runtime has no local preflight answer generator, document-specific entity list,
question-ID allowlist, or deterministic raw-document answer shortcut. A dry run
reports `deterministic_answer_shortcuts: 0` explicitly.

## Repeating-Record Detection

Long documents are frequently catalogues: an entry, profile, or record repeated tens
of times with the same internal shape. `records.py` recovers that structure without
looking for any particular phrase. It scores every short block that recurs 12–400
times, keeps those whose spacing is regular and whose occurrences span most of the
document, and declares the document periodic only when at least two independent
markers agree on the same count. Cuts are placed between repetition cycles and
snapped back to a physical page start, so an entry's own title stays with its data.

A document with no periodic structure yields no records, and packing falls back to
sequential chunking unchanged.

## Shared Evidence Extraction

Questions are planned jointly before scanning begins. Their required fields, topics,
claim slots, relations, aliases, and closure rules are merged into a
`SharedEvidencePlan`. The document is then inspected through up to three reusable
channels:

| Pass | Pack shape | Extracted evidence |
|---|---|---|
| Facts | one record per call | per-record census: identities, countries, measurements, units, dates, and named events |
| Claims | wide packs | atomic assertions, conflicts, rankings, relations, and qualifiers |
| Topics | wide packs | substantive discussion, passing mentions, negations, themes, and exceptions |

The census runs on a deliberately narrow contract — one record per call, five fields
per row. A small model asked to enumerate an unknown number of units inside a large
pack writes a few good rows and stops; given one record and a fixed field list it
transcribes. The rows are widened back into the normal evidence shape afterwards, so
grounding, numeric validation, and the evidence store are unchanged.

Each pack is read once per enabled channel, regardless of how many questions consume
its evidence. Independent packs run concurrently without reducing document coverage.

## Evidence Grounding

Every usable evidence item must include a quote that can be located in one of the
pack's source chunks. Whitespace differences may be normalized, and a quote attached
to an adjacent chunk can be relocated when the source match is unique. Unsupported,
invented, placeholder, or requirement-mismatched evidence is rejected before it can
enter the evidence store.

The Global Evidence Store preserves:

- Consumer question IDs
- Pass, pack, chunk, and semantic source unit
- Subject, predicate, raw value, normalized value, and unit
- Related entity, relation type, topic, and topic status
- Exact source quote, page, section, order, and confidence

Evidence is append-only. Repeated extraction is idempotent, while conflicting claims
remain separate until verification.

## Coverage and Recovery

Coverage is evaluated per question against every required pass and document pack.
Additional closure checks require comparable numeric evidence, both sides of a
contradiction, explicit relation evidence, multiple synthesis regions, or a direct
needle quote as appropriate.

When records are detected they supply the one thing a coverage check normally lacks:
a denominator. Without it, "this pack returned some facts" is the only signal, and a
pack that censused one of its three records reads as complete. With it, an omission
is exact — the records that produced no census fact are named, re-scanned on the
cheap model before any Small-model audit runs, and any remaining shortfall is
reported to the answer writer as `census incomplete: n/N records`.

Counting and superlative questions also raise their own evidence cap to the record
count, because "name every entry that qualifies" cannot be answered from fewer
evidence items than there are entries.

Only failed packs, requirement-level lexical shadows, grounded uncertainty, or
strong question shadows are sent to the audit model. Multi-source contradiction
and cross-section questions combine a document-spanning lexical shadow with full
ranked source context. Planning JSON may receive a
bounded repair. Structured stages request JSON mode from compatible brokers; local
validation also removes out-of-schema comments and recovers fully closed evidence
objects from a truncated response. A recovered pack remains `partial` and eligible
for a fresh scan. Final answers are generated independently as direct prose, so one malformed or
long answer cannot invalidate other questions in a JSON batch.

## Time and Call Safety

- Concurrent extraction and reasoning semaphores
- Hard total-call and extraction-call caps
- Separate final-answer call reserve
- Separate final-answer time reserve
- Rate-limit-aware serialized reasoning, backoff, retry limits, and model fallback paths
- Progress heartbeat with active calls, failures, ETA, and deadline
- Pack-level SQLite persistence and exact-run resume

The claims and topics passes use 34K-character packs. The facts census starts at
one record per call, then groups whole records only when a larger document would
otherwise exceed the extraction-call ceiling. Six extraction calls run
concurrently. Reasoning defaults to one call at a time because broker rate limits
are more reliable than bursty Small-model traffic. Increasing extraction
concurrency changes wall-clock time, not document coverage or total token usage.

The call cap is set by what the time budget allows rather than by spend. Census gap
repair is attempted for recurring fields and for any numeric field whose own label
and number co-occur in a record, and is capped proportionally to the census. Up to six final answers may
receive one contract-repair call when deterministic result data proves that a
required count, entity, value, or comparison direction was omitted.

## Supported Inputs

- UTF-8 `.txt` and `.md`
- Searchable `.pdf` with an embedded text layer
- YAML question lists

```yaml
- id: q01
  question: Which profile reports the largest area, and how large is it?
```

Only `id` and `question` are loaded. Any extra YAML keys and YAML comments are
discarded before prompts, hashes, plans, caches, or answers are built. Query type,
fields, relations, and completeness requirements are inferred from the question.

Image-only or extremely sparse PDFs are rejected before any API request. Apply OCR
before processing them.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

Enter a valid API key in `.env` before an actual run:

```dotenv
COURSE_API_KEY=your_api_key_here
```

`--dry-run` does not require a key and never calls an API.

## Usage

Inspect document structure and the cold-start call budget:

```bash
python3 main.py \
  --document /path/to/document.pdf \
  --questions /path/to/questions.yaml \
  --dry-run
```

Run the LLM-first pipeline:

```bash
python3 main.py \
  --document /path/to/document.pdf \
  --questions /path/to/questions.yaml \
  --output ./output/submission.json \
  --state-db ./output/state-v8_4.db \
  --parallel-profile balanced \
  --resume
```

Use `--force` to bypass exact-run caches and perform a new LLM analysis.

### Concurrency profiles

| Profile | Extraction | Reasoning | Use case |
|---|---:|---:|---|
| `safe` | 4 | 1 | Frequent timeouts or rate limits |
| `balanced` | 6 | 1 | Default |
| `aggressive` | 8 | 2 | Broker capacity has been verified |

## Model Roles

| Stage | Default model | Fallback |
|---|---|---|
| Joint planning | `mistral-small-3-2` | `ministral-3b-2512` |
| Shared extraction | `ministral-3b-2512` | — |
| Selective audit | `mistral-small-3-2` | `ministral-3b-2512` |
| Final answers | `mistral-small-3-2` | `ministral-3b-2512` |

## Testing

```bash
python3 -m unittest discover -s tests -v
```

The offline suite verifies that a cold run invokes planning, extraction, and final
answer models; reducers accept only supplied evidence; unsupported quotes are
rejected; evaluation rubrics never enter prompts; PDF pages are preserved; and no
deterministic answer shortcut exists in the runtime.

## Project Structure

```text
main.py           CLI and run orchestration
config.py         Models, budgets, concurrency, and environment settings
records.py        Repeating-record detection and segmentation
pipeline.py       Document processing, LLM calls, evidence, reducers, and coverage
prompts.py        Strict planning, census, extraction, audit, and answer contracts
schemas.py        Pydantic data contracts
requirements.txt  Python dependencies
tests/            Offline unit and integration tests
```
