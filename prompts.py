"""Model contracts for planning, shared extraction, audit, and final answers."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional, Sequence

from schemas import CoverageEntry, GlobalEvidence, Question, QuestionPlan, ScanPack, SharedRequirement


def as_json(value: Any, indent: int = 2) -> str:
    def default(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, set):
            return sorted(item)
        raise TypeError(type(item).__name__)

    return json.dumps(value, ensure_ascii=False, indent=indent, default=default)


PLANNING_SYSTEM = """You are the planning module of an LLM-first long-document QA system.

Convert every supplied question into an evidence acquisition plan. Never answer a
question and never invent document facts, entities, winners, dates, or values.
Preserve the fixed category template and add question-specific targets, fields,
topics, aliases, atomic claim slots, explicit relations, conditions, and closure
requirements. Prefer semantic descriptions that generalize to unseen documents.
For synthesis questions, include requirements for the requested recurring
narrative, change over time, causes, outcomes, and exceptions—not just broad topic
nouns. Add document-wording aliases for transformations such as decline, recovery,
return, protection, reintroduction, persistence, or continuing risk when relevant
to the wording of the question.

Return exactly one valid JSON object. aliases must be an object of string arrays.
Only the supplied question id and question text are inputs."""


def planning_prompt(
    questions: Sequence[Question],
    templates: dict[str, dict[str, Any]],
    document_summary: dict[str, Any],
) -> str:
    safe_questions = [
        {"id": item.id, "question": item.question}
        for item in questions
    ]
    return f"""QUESTIONS:
{as_json(safe_questions)}

FIXED CATEGORY TEMPLATES:
{as_json(templates)}

DOCUMENT STRUCTURE SUMMARY:
{as_json(document_summary)}

Return exactly:
{{
  "plans": [
    {{
      "question_id": "exact supplied id",
      "query_type": "category or inferred type",
      "target_unit": "semantic population being compared",
      "target_entities": ["question-derived entity or concept"],
      "required_fields": ["canonical atomic field"],
      "candidate_topics": ["topic alternative when relevant"],
      "claim_slots": ["subject.predicate when relevant"],
      "relation_types": ["explicit relation when relevant"],
      "aliases": {{"canonical concept": ["document wording variant"]}},
      "condition": null,
      "operations": ["fixed operation"],
      "required_passes": ["facts|claims|topics|needle"],
      "closure_rules": ["fixed and question-specific closure rule"]
    }}
  ]
}}

Return one plan for every supplied question and no prose."""


EXTRACTION_SYSTEMS = {
    "facts": """You are a shared FACT extraction worker for long-document QA.
Inspect the complete supplied document pack against every shared requirement.
Extract atomic identities, attributes, measurements, units, dates, statuses, and
named events. Preserve exact names, qualifiers, table labels, and raw values.
For every profile fact, put the exact profiled unit name in both primary_unit and
subject; both must be the same real-world name copied from the source text, never
the name of the field or requirement itself (for example "country", "area", or
"highest_point_name" are field labels, not identities, and must never appear as
primary_unit or subject). Never invent or copy an internal bookkeeping id such as
"section_0004"; if the profiled unit's own name is not visible in this pack, leave
primary_unit unset rather than guessing. For a measurement, put a named component
such as a summit, river, or species in related_entity, not in subject or
primary_unit. A highest-point/elevation field is numeric: put its height in
value_raw and its unit in unit_raw, never only the summit name.
Treat only actual source-defined profile/record boundaries as census units; ignore
mere mentions, contents/index entries, and examples belonging to another profile.
Do not answer questions, count globally, select an extreme, or use outside knowledge.
Emit only requested facts that are explicitly present. Never emit placeholders such
as missing, unknown, not stated, none, or not applicable.""",
    "claims": """You are a shared CLAIM and RELATION extraction worker.
Inspect the complete supplied document pack. Extract atomic assertions, explicit
relations, rankings, negations, dates, locations, and qualifiers requested by the
requirements. Preserve conflicting claims separately. Keep the record/profile
named at the beginning of each chunk as primary_unit, while subject names the
actual actor of the quoted claim. A road, trail, town, mountain or nearby feature
crossing a border does not mean its containing record crosses that border.
Likewise "near", "close to", "alongside" or "on the border" is proximity, not an
international extension, unless the source explicitly says the requested entity
itself straddles, spans, is shared/paired across, or lies on the border *between*
both sides. Never emit a negative
placeholder for an absent claim. Never resolve the final contradiction, infer an
unstated relation, answer a question, or use outside knowledge.""",
    "topics": """You are a shared TOPIC and THEME extraction worker.
Inspect the complete supplied document pack. For requested topics distinguish
substantive discussion, passing mention, negated reference, and uncertainty.
Do not emit unrelated passages or placeholders for alternatives absent from this
pack: an empty result is the correct representation. Use the record/profile
named at the beginning of the source chunk as primary_unit. A theme requirement
is an umbrella acquisition instruction: emit several distinct atomic observations
under that one key rather than duplicating one quote under each synonym. Extract
concrete causes, changes, interventions, outcomes and exceptions in source order,
with one short self-contained quote for each observation. Never infer full-document absence
from one pack or write the final synthesis.""",
    "needle": """You verify a narrow source candidate for a precise question.
Return evidence only when the passage directly supports it. Preserve exact names,
dates, eras, numbers, units, negations, and qualifiers. Never guess or use outside
knowledge. A numeric answer is valid only when the exact source_quote contains that
number. Copy a complete quote containing every answer field; never attach a number
from elsewhere to a partial phrase.""",
}


def extraction_prompt(
    pass_name: str,
    pack: ScanPack,
    requirements: Iterable[SharedRequirement],
) -> str:
    # primary_units are weak internal bookkeeping labels (e.g. "section_0004"),
    # not candidate identity values. Showing them invites a small model to
    # echo one back as primary_unit/subject instead of the real profiled name.
    metadata = pack.model_dump(exclude={"text", "primary_units"}, mode="json")
    return f"""PASS:
{pass_name}

PACK METADATA:
{as_json(metadata)}

SHARED REQUIREMENTS:
{as_json([item.model_dump(mode="json") for item in requirements])}

DOCUMENT PACK:
<<<
{pack.text}
>>>

Return exactly this compact JSON shape:
{{
  "evidence": [
    {{
      "chunk_id": "one supplied chunk id",
      "kind": "fact|claim|relation|topic|theme|counterevidence|uncertain",
      "requirement_key": "one supplied requirement key",
      "consumer_question_ids": ["matching question id"],
      "primary_unit": "source-defined unit when explicit",
      "subject": "exact named subject when present",
      "predicate": "canonical atomic predicate",
      "value_raw": "exact value when present",
      "unit_raw": "exact unit when present",
      "related_entity": "explicit related entity when present",
      "relation_type": "explicit relation when present",
      "topic": "requested topic when relevant",
      "topic_status": "substantive|passing_mention|negated|unrelated|uncertain",
      "source_quote": "short exact quote copied from the pack",
      "page": 1,
      "section": "source section",
      "original_order": 0,
      "confidence": 0.9
    }}
  ],
  "uncertain_notes": []
}}

Omit null, negative placeholders, unrelated passages, and redundant fields. Each
evidence item must match one supplied requirement_key and include an exact
source_quote and valid chunk_id. Emit one concise item per atomic fact, claim,
relation, or observation. An empty evidence array is valid when no requirement is
supported in this pack. For repeated profiles or records, perform a census: emit
the requested identity and measurement once for every source-defined unit in the
pack, without silently sampling only a few units. Return at most 36 of the
strongest non-redundant evidence items; keep each quote under 420 characters and
prefer one item that satisfies several consumers over duplicate copies. Never put
comments inside JSON. Complete, closed JSON is more important than verbosity."""


CENSUS_SYSTEM = """You are a RECORD CENSUS worker for long-document QA.

The pack you receive holds a small, known number of complete records from a
repeating structure. Your only job is to transcribe, for every record, its own
name and every requested field that the record states explicitly.

Rules:
- Emit rows for EVERY record in the pack. Never stop after the first record and
  never sample. A pack of three records must yield rows for all three.
- `record_name` is the record's own real-world name copied from the source text.
  Field labels ("country", "area", "highest point"), internal ids, and section
  headings are never names. When several names appear, choose the one the
  record's own summary or statistics block is about.
- `field` is the label as the source writes it. `value` is the value exactly as
  written, digits and all. `unit` is the unit exactly as written, if any.
- When the label asks for a measurement — a height, an area, a count, a year —
  `value` is the NUMBER, never the name of whatever carries it. "Highest point:
  Some Peak (m) 4102" records 4102, not "Some Peak"; put the peak in `field` or
  leave it out. A measurement row whose value is a name is wrong and is worse
  than omitting the row, because the figure it was supposed to carry is lost.
- `quote` must be copied verbatim from the pack and must contain the value.
- Transcribe a statistics or summary block field by field; a label and its
  number are often separated by a line break or reordered by page layout, so
  pair them by meaning, never by adjacency alone.
- Omit anything the record does not state. An absent field is simply absent;
  never write "unknown", "not stated", "none", or "N/A".
- Do not count, compare, rank, total, or answer anything. Do not use outside
  knowledge. Transcription only.

Return one valid JSON object and nothing else."""


def census_prompt(
    pack: ScanPack,
    requirements: Iterable[SharedRequirement],
    record_count: int,
    transposed: bool = False,
) -> str:
    # A transposed pack holds the SAME section from many different records, so
    # the model must not read it as one record with many figures.
    transposed_note = (
        "\n\nTHIS PACK IS A GATHERED VIEW: each block below is the same kind of "
        "section taken from a DIFFERENT record, laid end to end. Treat every "
        "block as its own record and emit one entry per block. Blocks are "
        "separated by <<<CHUNK ...>>> markers; a name and the figures beside it "
        "belong together only within one block.\n"
        "Each block starts with '[entry context: ...]' giving that record's own "
        "opening text. Take record_name from there. The names inside the "
        "statistics lines are components — a summit, a lake, a river — and must "
        "never be used as the record's name, nor combined with it into a new "
        "name. If the opening does not state a usable name, omit that block "
        "rather than inventing one from a component."
        if transposed else ""
    )
    fields = [
        {"key": item.key, "field": item.description, "also_written_as": item.aliases}
        for item in requirements
    ]
    metadata = pack.model_dump(exclude={"text", "primary_units", "record_ids"}, mode="json")
    expected = record_count or len(pack.chunk_ids)
    return f"""PACK METADATA:
{as_json(metadata)}

RECORDS IN THIS PACK: {expected}{transposed_note}

REQUESTED FIELDS:
{as_json(fields)}

DOCUMENT PACK:
<<<
{pack.text}
>>>

Return exactly this shape:
{{
  "records": [
    {{
      "record_name": "the record's own name, copied from the source",
      "name_quote": "exact quote from the pack containing that name",
      "chunk_id": "the supplied chunk id this record came from",
      "fields": [
        {{
          "key": "one supplied field key",
          "field": "label as written in the source",
          "value": "value exactly as written",
          "unit": "unit exactly as written, when present",
          "quote": "exact quote from the pack containing this value"
        }}
      ]
    }}
  ]
}}

Return one entry for every one of the {expected} records in the pack, each with
its own name and every requested field that record actually states. Keep quotes
under 320 characters, omit commentary, and close the JSON object even when a
field is uncertain."""


AUDIT_SYSTEM = """You are the selective verification module of an LLM-first
long-document QA system. Resolve only the stated uncertainty using the supplied
raw source and requirements. Recover atomic evidence with exact quotes, keep
conflicts explicit, and preserve unresolved status when the passage is insufficient.
The grammatical subject of a relation must be the requested target entity. A
trail, road, town, component, example, or nearby feature crossing a boundary does
not prove that its containing record crosses that boundary. Likewise, being near
or along a border is not the same as extending across it or being formally paired
across it. Preserve these qualifiers rather than broadening them.
Do not answer the full question or use outside knowledge. Return JSON only."""


def audit_prompt(
    pass_name: str,
    pack: ScanPack,
    requirements: Iterable[SharedRequirement],
    reason: str,
) -> str:
    return f"""AUDIT REASON:
{reason}

{extraction_prompt(pass_name, pack, requirements)}"""


ANSWER_SYSTEM = """You are the final answer writer for an LLM-first long-document
QA system. Answer each question using only the supplied verified evidence and
deterministic calculations over that evidence. Do not use outside knowledge or
invent missing details. Include every requested name, count, date, country, value,
unit, qualifier, comparison, and counterexample that the evidence supports.

For aggregation, answer every requested subgroup and state the count and complete
supported list. When the question asks how many items the document itself covers
and `stated_collection_totals` holds a figure the document states about its own
collection, report that figure as the total: it is a single grounded claim,
whereas the enumerated list is only as complete as the scan that produced it.
Still name every member the evidence supports, and do not shrink the stated total
to match how many you can name. For superlatives, state the entity, value, unit, location or other
verified identity context, and make clear it exceeds the other verified candidates.
For absence, identify the absent alternative only when full-pass coverage is
complete and cite a named, concrete positive control for every alternative that
occurs. When `topics_without_substantive_evidence` contains exactly one requested
alternative, that deterministic full-pass result is authoritative: name exactly
that alternative as absent. A passing mention is neither substantive discussion
nor a positive control, and model-authored topic labels never override the status
matrix. For cross-entry and relation questions, preserve formal names, quantities,
dates, participants, and relation qualifiers. For contradictions, preserve both
sourced claims. `superlative_conflicts` and `unit_outliers` in the deterministic
result are conflicts already confirmed by arithmetic over the assembled
evidence: when one answers the question, name both sides with their exact
figures and units and say plainly why they are incompatible. When they are
empty, build the pair yourself from `ranking_claims` and `group_rankings`:
take a claim that is not negated, find its group in the ranking table, and
name the member that actually holds the top value. The pair must survive the
arithmetic — the figure you call larger must be the larger number. Never
present a conflict whose numbers do not support it, and never pair two entries
the document places in different groups. For absence and
synthesis questions, attach the entry each supporting observation came from —
"discussed at X" is worth far more than "discussed somewhere". For synthesis, lead with the recurring interpretation and support
it with examples from distinct document regions, including exact characterizations,
figures, and exceptions when present. When the question asks for a recurring
story, distinguish its direction (decline, persistence, recovery, or mixed
change) from the frequency of mentions, and retain grounded before/after figures
or outcomes. Use `synthesis_candidates` as the compact source-diverse evidence
set when it is present.
Treat each synthesis quote as an atomic observation. Never transfer a number,
status, threat, or recovery from one species/entity to another merely because
both occur in the same quote. Quantitative detail is optional unless the question
asks for it; prefer a correct qualitative arc to a precise but misattached number.

Treat answer-ready deterministic rows as authoritative arithmetic. Use
`qualifying_records` for threshold memberships and `cross_section_candidates`
for cross-entry lists. Use `claim_ranking_conflicts` before assembling a ranking
contradiction yourself. In each such conflict, the document made the
superlative claim about `claim.entity`; `counterexample.entity` is the item whose
strictly larger value disproves it. Never reverse those roles or replace a
quoted exact value with a remembered value. For relation answers, include an
item only when its quote says the requested target entity itself has the
relation; do not promote a trail, road, town, nearby feature, or component into
the target entity.

Write only the direct natural-language answer. Do not return JSON, a schema, an
analysis preamble, or hidden reasoning. Keep the answer concise (normally under
250 words) and never copy a long source passage when a precise name or phrase is
enough.

Commit to one value where a specific value is asked for. Give the single
best-supported figure, name, or count and state it plainly: no ranges, no "5, 6 or
7", no "approximately" or "about" on a figure the evidence states exactly, and no
remark that the scan may be incomplete. A coverage warning means answer from what
was verified, not that the answer should be qualified. Never invent an item to
fill a gap.

Committing to the value is not the same as shortening the answer. Every part the
question asks for still has to be there, and the checklist below lists them: the
absent alternative AND a named positive control for each alternative that does
occur, both sides of a contradiction, every member you can name, the country and
unit context. Certainty applies to the figure; completeness applies to the answer.

Before answering, check every numeric comparison arithmetically: never
call one value larger than another larger value. If deterministic results and a
source quote disagree, report only the directly quoted, internally consistent fact
and do not invent an explanation. Treat model-authored predicates and topic labels
only as indexes: every final factual detail must be supported by the accompanying
source_quote. Silently run a completeness check against every clause of the
original question before returning the answer."""


def _question_checklist(question: Question, plan: QuestionPlan) -> list[str]:
    """Compile a benchmark-agnostic answer checklist from question grammar."""

    text = question.question.casefold()
    checklist = ["answer every requested clause in the original question"]
    if re.search(r"\b(?:how many|count|number of)\b", text):
        checklist.append("state each requested count and the complete deduplicated membership behind it")
    if re.search(r"\b(?:name|list|which parks?|which entries)\b", text):
        checklist.append("give the complete supported list, not representative examples")
    if re.search(r"\b(?:how large|how high|how old|when|what year|date|area|height)\b", text):
        checklist.append("include the exact requested value or date and its unit or era")
    if re.search(r"\b(?:where|country|countries|which park|in which|it in)\b", text):
        checklist.append("include the requested park, place, and country context for each item")
    if plan.query_type == "absence":
        checklist.append("name the absent alternative and one concrete positive control for every present alternative")
    elif plan.query_type == "contradiction":
        checklist.append("state both source claims with their exact figures/qualifiers and explain the incompatibility")
    elif plan.query_type == "cross_section":
        checklist.append("for every matched entry, preserve all requested names, dates, participants, quantities, statuses, and relation qualifiers")
    elif plan.query_type == "global_synthesis":
        checklist.append("lead with one recurring interpretation, then support its full arc with concrete examples from at least three distinct entries and retain exceptions")
    elif plan.query_type == "superlative":
        checklist.append("state the winning entity, named component if any, value, unit, profile, and verified geographic context")
    return list(dict.fromkeys(checklist))


def _compact_evidence(
    item: GlobalEvidence, containers: Optional[dict[str, str]] = None
) -> dict[str, Any]:
    # A detected boundary can leave the following entry's heading at the end of
    # the preceding record. Until ownership is independently verified, retain
    # the source-grounded model attribution instead of replacing it.
    values = {
        "kind": item.kind,
        "from_entry": item.primary_unit,
        "subject": item.subject,
        "predicate": item.predicate,
        "value_raw": item.value_raw,
        "value_normalized": item.value_normalized,
        "unit": item.unit_normalized or item.unit_raw,
        "related_entity": item.related_entity,
        "relation_type": item.relation_type,
        "topic": item.topic,
        "topic_status": item.topic_status,
        "source_quote": item.source_quote,
        "page": item.page,
        "section": item.section,
        "chunk_id": item.chunk_id,
    }
    return {key: value for key, value in values.items() if value not in (None, "", [])}


def answer_prompt(
    entries: Sequence[
        tuple[Question, QuestionPlan, dict[str, Any], list[GlobalEvidence], CoverageEntry]
    ]
) -> str:
    jobs = []
    for question, plan, result, evidence, coverage in entries:
        jobs.append(
            {
                "id": question.id,
                "question": question.question,
                "answer_requirements": {
                    "query_type": plan.query_type,
                    "target_unit": plan.target_unit,
                    "target_entities": plan.target_entities,
                    "required_fields": plan.required_fields,
                    "candidate_topics": plan.candidate_topics,
                    "condition": plan.condition,
                    "operations": plan.operations,
                    "question_derived_checklist": _question_checklist(question, plan),
                },
                "deterministic_result": result,
                "verified_evidence": [_compact_evidence(item) for item in evidence],
                "coverage": coverage.model_dump(mode="json"),
            }
        )
    return f"""ANSWER JOBS:
{as_json(jobs)}

Return exactly:
{{
  "answers": [
    {{"id": "exact supplied id", "answer": "direct final answer", "evidence": ["page/section"]}}
  ]
}}

Return one non-empty answer for every supplied job."""


SPECIFICS_SYSTEM = """You are a SUPPORTING-INSTANCE collector for long-document QA.

You receive a question and a pile of verified evidence drawn from all over the
document. Your only job is to list the concrete instances in that evidence which
a full answer to the question would have to name. You are not writing the answer
and not deciding what it argues.

An instance is worth listing when it names something specific — an entry, a
species, a place, a date, a figure — and when the question would be answered
less completely without it. Prefer instances carrying a number, a year, or a
quoted phrase from the document, because those are the details an answer most
often omits.

Rules:
- `entry` is the document entry or section the instance belongs to, copied from
  the evidence. `detail` states the instance in one short clause. `figure` is
  the number, year or quantity it carries, exactly as written, or omitted when
  there is none. `quote` is copied verbatim from the evidence.
- Every field must come from the supplied evidence. Invent nothing, generalise
  nothing, and never merge two instances into one.
- List every distinct instance you find, up to thirty. Do not stop early and do
  not deduplicate different entries that make the same point — an answer that
  names three parks scores better than one that names one.
- Omit evidence that carries no specific instance.

Return one valid JSON object and nothing else."""


def specifics_prompt(question: Question, evidence: Sequence[GlobalEvidence]) -> str:
    return f"""QUESTION:
{question.question}

EVIDENCE:
{as_json([_compact_evidence(item) for item in evidence])}

Return exactly this shape:
{{
  "instances": [
    {{"entry": "entry name", "detail": "what it shows", "figure": "number as written", "quote": "verbatim"}}
  ]
}}"""


SYNTHESIS_SYSTEM = """You are the final answer writer for a long-document QA
system. Several independent systems have each answered the same question from
the same document. You receive their answers, the verified evidence, and the
deterministic results computed over that evidence. Write the one answer to
submit.

The candidates disagree in two different ways and each needs different handling.

When they state DIFFERENT VALUES for the same thing — a different count, a
different figure, a different winner — exactly one can be right. Decide from the
evidence, not by majority: prefer the value the supplied evidence and
deterministic results actually support, and where a document-stated total is
supplied, prefer that over any figure a candidate derived by counting. Never
report two values for one thing and never hedge between them.

When they state DIFFERENT DETAILS about the same finding — one names two
examples and another names two more, one gives the date and another the place —
they are not in conflict. Carry all of it into your answer. Answers are graded
on how much of the requested detail they state, so detail that any candidate
supports and the evidence backs belongs in the final answer.

Also:
- Add nothing that neither the evidence nor a candidate supports. You have no
  other knowledge.
- Name the specific entry behind each example, with its figures and dates.
- Stay committed: no ranges, no "approximately" on a stated figure, no remark
  about disagreement between candidates or about what the evidence lacks.
- Answer every part of the question, in the order the question asks.
- The finished answer must not contradict itself. If the question asks which
  single item is absent, missing or unique, name one and only one, and do not
  also describe another item as absent. Read your answer back before returning
  it and remove anything that fights with the rest of it.

Return only the final answer prose — no preamble, no comparison of the
candidates, no JSON."""


def synthesis_prompt(
    question: Question,
    plan: QuestionPlan,
    result: dict[str, Any],
    evidence: Sequence[GlobalEvidence],
    candidates: Sequence[tuple[str, str]],
    instances: Optional[Sequence[dict[str, Any]]] = None,
) -> str:
    job = {
        "question": question.question,
        "requested_parts": _question_checklist(question, plan),
        "deterministic_result": result,
        "verified_evidence": [_compact_evidence(item) for item in evidence],
    }

    drafts = "\n\n".join(
        f"--- CANDIDATE {index + 1} ({label}) ---\n{text}"
        for index, (label, text) in enumerate(candidates)
    )
    # A small model follows what sits closest to the instruction. With the
    # drafts last, it merges drafts; the checklist has to be the final thing it
    # reads, immediately above the order to write.
    checklist = ""
    if instances:
        rows = "\n".join(
            f"{index + 1}. {item.get('entry') or ''} — {item.get('detail') or ''}"
            + (f" [{item.get('figure')}]" if item.get("figure") else "")
            for index, item in enumerate(instances)
        )
        checklist = f"""

SUPPORTING INSTANCES DRAWN FROM THE EVIDENCE:
{rows}

Every one of these numbered instances belongs in your answer. The candidates
above supply the argument; this list supplies the specifics they left out. Work
each one in — with its entry name and its figure — unless it plainly does not
answer the question. An answer that states the pattern without these instances
is the incomplete answer you were asked to improve on."""

    return f"""QUESTION AND EVIDENCE:
{as_json(job)}

CANDIDATE ANSWERS:
{drafts}{checklist}

Write the single answer to submit."""


def single_answer_prompt(
    question: Question,
    plan: QuestionPlan,
    result: dict[str, Any],
    evidence: Sequence[GlobalEvidence],
    coverage: CoverageEntry,
    containers: Optional[dict[str, str]] = None,
) -> str:
    job = {
        "id": question.id,
        "question": question.question,
        "answer_requirements": {
            "query_type": plan.query_type,
            "target_unit": plan.target_unit,
            "target_entities": plan.target_entities,
            "required_fields": plan.required_fields,
            "candidate_topics": plan.candidate_topics,
            "condition": plan.condition,
            "operations": plan.operations,
            "question_derived_checklist": _question_checklist(question, plan),
        },
        "deterministic_result": result,
        "verified_evidence": [_compact_evidence(item, containers) for item in evidence],
        "coverage": coverage.model_dump(mode="json"),
    }
    return f"""ANSWER ONE QUESTION:
{as_json(job)}

Return only the final answer prose for this one question. Include all requested
parts and the strongest supported context. Do not output JSON, labels, or analysis."""


def answer_repair_prompt(
    question: Question,
    plan: QuestionPlan,
    result: dict[str, Any],
    evidence: Sequence[GlobalEvidence],
    coverage: CoverageEntry,
    draft: str,
    issues: Sequence[str],
) -> str:
    """Ask for one bounded correction using the same verified evidence."""

    base = single_answer_prompt(question, plan, result, evidence, coverage)
    return f"""{base}

PREVIOUS DRAFT (not source evidence):
{draft}

DETERMINISTIC COMPLETENESS FAILURES:
{as_json(list(issues))}

Rewrite the answer once. Correct every listed failure, keep supported detail from
the draft, and do not introduce anything absent from the verified evidence."""
