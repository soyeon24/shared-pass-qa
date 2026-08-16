# LLM-First Shared-Pass MELT-QA

## Goal

Answer questions over a previously unseen long document accurately within a fixed
time budget. The system must remain general: do not encode known source answers,
question IDs, document-specific entities, or rubric key points.

## Core Idea

Plan all questions first, merge their evidence requirements, and scan the document
through shared passes instead of rereading it once per question.

```text
Questions → Joint LLM planning → Shared evidence requirements
Document  → Repeating-record detection → record-aligned packs
          → Shared LLM extraction (per-record census + wide reading passes)
          → Grounded global evidence store
          → Python reduction, record-denominator coverage, cheap gap repair
          → Selective Small-model verification
          → Small-model final answers
```

The three shared evidence channels are:

- `facts`: exhaustive per-record census of identities, countries, measurements and units
- `claims`: assertions, contradictions, relations, rankings and qualifiers
- `topics`: substantive mentions, absence controls, themes and exceptions

## Record Structure

Many long documents are catalogues of repeating entries. `records.py` detects that
shape without naming any phrase: it scores short blocks that recur 12–400 times at a
regular spacing over most of the document, and requires at least two independent
markers to agree on the count before claiming the document is periodic. A document
with no such structure yields no records and everything falls back to sequential
chunking.

When records are detected they become the census unit:

- a chunk never straddles two records, and `record_ids` carries the assignment
- the `facts` channel runs on record-aligned census packs, one record per call,
  widening the grouping only when the record count would not fit the extraction
  budget left over after the other channels
- a record's own name is stored under its own identity requirement spanning every
  consumer of the census, because counting questions group by that name and must
  not be excluded by whichever field happened to sit first
- the record count is the coverage denominator, so an omitted entry is exact
  rather than inferred from "this pack returned something"
- a record that produced no census fact is re-scanned once on the cheap model,
  from all of its chunks, before any Small-model audit spends its budget
- a record that is still empty after that attempt is a span with nothing
  censusable in it (a contents page, an index) and leaves the denominator, so
  coverage can reach closure instead of re-scanning it on every run
- coverage is per field, not only per record. A record that returned nothing but
  its own name is not censused, and a missing measurement is chased whenever the
  record's own text shows that field's wording beside a number — a passing prose
  mention of the word is not enough, or every entry would be re-scanned forever

Heading depth is only the fallback unit signal. Text recovered from a PDF routinely
flattens every heading to one level, which mints hundreds of spurious "units" out of
per-entry sidebar titles.

## Accuracy Strategy

- Every evidence item needs an exact quote grounded in its source chunk.
- Model-authored labels are not evidence by themselves.
- Counts, comparisons, unit conversion and deduplication are performed in Python.
- Aggregation and superlative questions use the full facts census, not sampled retrieval.
- A count has two independent routes and aggregation uses both: the census
  enumerates the members, and the claims channel looks for a total the document
  states about its own collection. The enumeration is only as complete as the
  scan that produced it, so where the document declares its own size that figure
  is the total — while the enumeration still supplies the names.
- The census contract is deliberately narrow — one record per call, five fields per
  row — because a small model asked to enumerate an unknown number of units inside a
  large pack writes a few good rows and stops.
- A census question is never capped below the record count: "name every entry that
  qualifies" cannot be answered from fewer evidence items than there are entries.
- Answers commit to the best-supported result rather than returning a range, a
  disjunction, or commentary about what extraction may have missed.
- Contradiction and cross-section questions combine a document-wide lexical shadow
  with complete high-ranking source context.
- Absence requires both full-document closure and positive evidence for alternatives
  that are present.
- Only unresolved or high-risk evidence is rechecked with the stronger model.
- The final writer receives verified evidence, reducer output, coverage and a
  question-derived completeness checklist.

## Runtime Strategy

- Ministral 3B performs parallel shared extraction, the per-record census, and all
  census gap repair. It costs a fifth of the Small model on input and a fifteenth on
  output, so extraction breadth is bounded by the time budget rather than by spend.
- `mistral-small-3-2` performs planning, verification and final writing. The cloud
  currently routes this legacy model ID to the newer Mistral Small model, so keep
  the existing ID.
- Small-model requests are serialized to avoid rate-limit bursts.
- HTTP 429 responses use backoff and must not open the model circuit breaker.
- SQLite provides exact-request caching, append-only evidence and safe resume.
- Sparse field-gap repair is prevalence-gated and proportional, so an optional
  field seen in a few records cannot trigger a near-document-wide rescan.
- At most six answers receive one extra contract-repair call, and only when the
  deterministic reducer proves a required count, entity, value, or direction was
  omitted from the first draft.

## Scale and the Deadline

A document several times larger than the practice one must still produce a
submission inside the hour.

- A document larger than the call cap is not an error. Every guard that used to
  refuse such a document now warns and proceeds: the total-call cap and the
  deadline are the real limits, and reading part of a document beats refusing
  to start. The census keeps at least half the extraction budget.
- Throughput, not the call cap, decides whether the document is read in time.
  The broker sustains 48 concurrent extraction calls without error.
- The audit and answer phases scale with the question count, not the document,
  so a larger document lengthens extraction alone: 187 calls becomes 647 at
  five times the size, not five times 187.

## Documents With No Repeating Structure

Record detection declines on documents that are not catalogues. The fallback
still censuses, because narrowness — not record structure — is what makes a
census exhaustive: without records the unit is a fixed window, as narrow as the
leftover extraction budget affords.

## Combining Runs

Answers and evidence pool on different rules, both measured on the leaderboard:

- More evidence stores raise the score. The same three systems scored 77.1% on
  stores accumulated over many runs and 64.8% on single fresh ones.
- More candidate answers lower it, and a weak draft lowers it hardest. Asked to
  adjudicate one answer saying six and one saying three, the writer returns
  five. Never offer a partial run's answers as a draft; offer its evidence.
- A question answered by naming many things from all over the document gets its
  instances collected from the evidence first, then written. Told to notice the
  details itself, the writer merges drafts and stops. Contradiction questions
  are excluded: handed thirty instances the writer surveys them instead of
  choosing the one conflicting pair, and its score fell.

## Unit Dimensions

A statistics block sets a figure and its label on separate lines, so an
extractor pairs an area with a neighbouring row's unit often enough to matter.
An area measured in a length unit is a mislabelled unit, not a different scale:
converting it scaled a 290 km² entry to 290,000 and handed it the largest-area
answer. The check runs when evidence is written and again when it is read, so a
store built before the fix is corrected on the way out.

## Non-Negotiable Constraints

- Do not add deterministic raw-document answer shortcuts.
- Load only question IDs and question text. Discard every extra YAML field before
  planning, hashing, caching, prompting, or answering.
- Do not hardcode known questions, answers, names, counts or expected examples.
- Do not weaken quote grounding, full-pass coverage or numeric validation for speed.
- Preserve existing evidence when retrying or auditing.
- Keep the legacy Small model ID unless the API contract explicitly requires a new ID.
- Record detection must stay phrase-agnostic. It may score repetition and spacing;
  it may never look for a particular marker, and the private document may have no
  repeating structure at all.

## Main Files

- `main.py`: CLI, concurrency profiles and orchestration
- `records.py`: repeating-record detection and record segmentation
- `pipeline.py`: document processing, API calls, evidence, reducers and coverage
- `prompts.py`: planning, census, extraction, audit and answer contracts
- `schemas.py`: strict Pydantic data models
- `config.py`: models, budgets, timeouts and concurrency
- `synthesize.py`: adjudicates several runs' answers over their pooled evidence
- `tests/test_v4.py`: offline regression and anti-hardcoding tests
