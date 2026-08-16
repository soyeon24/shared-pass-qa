"""Combine several systems' submissions into one, adjudicated on the evidence.

Measured on the practice set, no single configuration answers any question
better than every other one: picking the best answer per question across three
of our own systems scores 81.2% against 71.7% for the best single system. The
disagreements are informative rather than noise — different pipelines surface
different parts of the same document — so the final answer is written from all
of them together, with the evidence store available to settle the cases where
they contradict each other outright.

    python synthesize.py --candidates a.json b.json c.json \\
        --state-db output/state.db --output output/submission.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from config import PROJECT_DIR, load_settings
from pipeline import (
    CallBudget,
    LLMClient,
    PipelineError,
    SQLiteStore,
    coverage_for_question,
    CENSUS_IDENTITY_KEY,
    evidence_limit_for,
    load_questions,
    merge_entity_aliases,
    normalize_direct_answer,
    questions_hash,
    reduce_question,
    tolerant_json,
    select_evidence,
    write_submission,
)
from prompts import (SPECIFICS_SYSTEM, SYNTHESIS_SYSTEM, specifics_prompt,
                     synthesis_prompt)
from schemas import SharedEvidencePlan, Submission, SubmissionAnswer

LOGGER = logging.getLogger("llm_first_melt.synthesis")

# Questions answered by naming many things from all over the document. Their
# drafts agree on the argument and differ on how much supporting detail each
# bothered to state, so merging them reproduces the thinnest one.
#
# Cross-section questions are excluded despite also spanning the document: they
# compare two named things, and a checklist of thirty instances pulls the answer
# away from the pair it was asked about. Measured, it cost that category
# fifteen points while giving synthesis twenty.
# Only the questions whose score is a count of distinct things named. A
# contradiction question is graded on pinning down one conflicting pair, and
# measured with the collector on, its score fell: handed seventeen instances the
# writer surveys them instead of choosing, which is the opposite of the job.
BREADTH_QUERY_TYPES = {"global_synthesis"}

# Opt-in, and measured to cost more than it wins on an answer that is already
# good: run over a finished 87.2% submission it fell to 64.2%, with counting
# questions the worst hit. The misses it targets are real — a runner-up figure
# to contrast against, two climbers' names — but handed a checklist the writer
# rebuilds the answer around it and loses the focus that scored. Reach for this
# only when the draft is visibly thin, never as a finishing pass.
ENRICH_QUERY_TYPES = {
    "global_synthesis", "cross_section", "absence", "superlative",
    "aggregation", "contradiction",
}

# Measured on a prompt this stage actually built: the broker counted 266,545
# tokens for 955,332 characters, so this text runs about 3.58 characters per
# token. Budget at 3.3 to stay wrong in the safe direction, and reserve the
# reply plus the system prompt.
CHARS_PER_TOKEN = 3.3
RESERVED_TOKENS = 5_000
MODEL_CONTEXT_TOKENS = {
    "mistral-small-3-2": 262_144,
    "ministral-3b-2512": 131_072,
}
# A model we have no figure for gets the smaller window rather than a guess.
DEFAULT_CONTEXT_TOKENS = 131_072


def prompt_char_budget(model: str) -> int:
    """How much prompt this model can take, in characters.

    Trimming to a single shared budget sized for the fallback throws away most
    of what the reasoning model would have read: the questions that carry a
    document-wide reducer output are exactly the ones that need it, and they
    fit the larger window untouched.
    """

    tokens = MODEL_CONTEXT_TOKENS.get(model, DEFAULT_CONTEXT_TOKENS)
    return int(max(tokens - RESERVED_TOKENS, 1) * CHARS_PER_TOKEN)


def read_candidates(paths: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """Answers keyed by question id, each tagged with the file it came from."""

    by_question: dict[str, list[tuple[str, str]]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"Could not read candidate {path}: {exc}") from exc
        answers = payload.get("answers")
        if not isinstance(answers, list):
            raise PipelineError(f"{path} has no answers list")
        # The label tells the adjudicator which system wrote which draft, so it
        # has to be distinct per file. A version laid out as <version>/output/
        # is named by its version; anything else is named by its own filename.
        label = path.parent.parent.name if path.parent.name == "output" else path.stem
        for item in answers:
            text = str(item.get("answer", "")).strip()
            if item.get("id") and text:
                by_question.setdefault(str(item["id"]), []).append((label, text))
    return by_question


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.validate(require_key=True)
    questions = load_questions(args.questions.resolve())
    q_hash = questions_hash(questions)
    candidates = read_candidates([item.resolve() for item in args.candidates])

    stores = [SQLiteStore(item.resolve()) for item in args.state_db]
    store = stores[0]
    try:
        shared_plan = None
        merged: dict[str, object] = {}
        for handle in stores:
            row = handle.connection.execute(
                "select document_hash, plan_json from plans limit 1"
            ).fetchone()
            if not row:
                continue
            if shared_plan is None:
                shared_plan = SharedEvidencePlan.model_validate_json(row[1])
            # The questions hash covers the pipeline version as well as the
            # questions, so a store written by an earlier build of this same
            # system is locked out even though it answers the identical set.
            # Opening that gate is an operator's call, not a default: the
            # caller is asserting these stores read the same document for the
            # same questions, which nothing in the file itself can prove.
            hashes = [q_hash]
            if args.pool_across_versions:
                hashes = [
                    value for (value,) in handle.connection.execute(
                        "select distinct questions_hash from evidence"
                    )
                ] or [q_hash]
            for item in [it for value in hashes for it in handle.evidence(row[0], value)]:
                merged.setdefault(item.id, item)
        if shared_plan is None:
            raise PipelineError("no supplied state DB holds a plan; run the pipeline first")
        plan_map = {item.question_id: item for item in shared_plan.question_plans}
        evidence = list(merged.values())
        if not evidence:
            raise PipelineError(
                "the supplied state DBs hold no evidence for this question set; the "
                "candidates and the stores must come from the same document and questions"
            )
        # The pipeline never caps a counting question below the number of
        # entries it counts, and the adjudicator must not either: it decides
        # between drafts that each saw the whole census, so handing it a
        # truncated one makes the fuller draft look like an overreach. The
        # population size is not passed in, but the census stamps each entry's
        # own name, so the store knows it.
        total_records = len({
            item.subject.strip().lower()
            for item in evidence
            if item.requirement_key == CENSUS_IDENTITY_KEY
            and item.subject and item.subject.strip()
        })
        LOGGER.info(
            "Evidence pooled from %d store(s): %d items, %d distinct census entries",
            len(stores), len(evidence), total_records,
        )
        aliases = merge_entity_aliases([
            {
                "subject": item.subject, "primary_unit": item.primary_unit,
                "predicate": item.predicate, "value_raw": item.value_raw,
                "value_normalized": item.value_normalized,
                "unit": item.unit_normalized or item.unit_raw,
                "chunk_id": item.chunk_id,
            }
            for item in evidence if item.kind != "uncertain" and item.source_quote
        ])

        budget = CallBudget(settings, expected=len(questions), answer_reserve=0)
        client = LLMClient(settings, store, budget, use_cache=False)

        def fit_prompt(question, plan, result, selected, drafts, limit, instances=None) -> str:
            """Shrink the prompt until it fits the model that will read it.

            Pooling several stores — or reading a much larger document — makes
            the reducer output for a contradiction or cross-section question
            grow without bound, and a prompt that overflows the context is not
            a degraded answer but no answer at all. Trimming more than the
            model requires costs real points, so the limit is that model's own
            window and nothing tighter.
            """

            text = synthesis_prompt(question, plan, result, selected, drafts, instances)
            if len(text) <= limit:
                return text

            # A contradiction or cross-section question carries a document-wide
            # lexical shadow in the reducer output, and pooling stores makes it
            # the bulk of the prompt — far larger than the quoted evidence.
            # Shrink that first, or halving the evidence five times destroys
            # the grounding while the prompt barely moves.
            trimmed = dict(result)
            while len(text) > limit and trimmed:
                widest = max(trimmed, key=lambda name: len(json.dumps(trimmed[name], default=str)))
                value = trimmed[widest]
                if isinstance(value, list) and len(value) > 4:
                    trimmed[widest] = value[: len(value) // 2]
                else:
                    trimmed.pop(widest)
                text = synthesis_prompt(question, plan, trimmed, selected, drafts, instances)

            kept = list(selected)
            while len(text) > limit and len(kept) > 1:
                kept = kept[: max(1, len(kept) // 2)]
                text = synthesis_prompt(question, plan, trimmed, kept, drafts, instances)
            LOGGER.warning(
                "%s: trimmed synthesis prompt to fit the context "
                "(evidence %d -> %d, reducer keys %d -> %d)",
                question.id, len(selected), len(kept), len(result), len(trimmed),
            )
            return text

        LOGGER.info(
            "Synthesising %d questions from %d candidate file(s)",
            len(questions), len(args.candidates),
        )

        async def one(question) -> SubmissionAnswer:
            drafts = candidates.get(question.id, [])
            if not drafts:
                return SubmissionAnswer(
                    id=question.id, answer="", evidence=[],
                    error="no candidate answer supplied",
                )
            plan = plan_map.get(question.id)
            if plan is None:
                return SubmissionAnswer(id=question.id, answer=drafts[0][1], evidence=[])
            # One draft leaves nothing to adjudicate, but the evidence still
            # holds supporting detail the draft never stated — the runner-up
            # figure a superlative should be contrasted against, the person a
            # first-ascent question names, the second example a synthesis
            # question needs. Enriching is a different job from choosing, so it
            # runs on the wider set of question types.
            enrich_only = len(drafts) == 1
            if enrich_only and not (args.enrich and plan.query_type in ENRICH_QUERY_TYPES):
                return SubmissionAnswer(id=question.id, answer=drafts[0][1], evidence=[])
            result = reduce_question(plan, evidence, aliases)
            selected = select_evidence(
                question, plan, evidence,
                evidence_limit_for(plan, settings, 0), result,
            )
            instances = None
            wants_instances = (
                plan.query_type in ENRICH_QUERY_TYPES if enrich_only
                else plan.query_type in BREADTH_QUERY_TYPES
            )
            if wants_instances and selected:
                # Told to notice the details itself, the writer merges drafts
                # and stops. Pulling the instances out first turns judgement
                # into transcription, the same move that made the per-record
                # census exhaustive.
                try:
                    raw, _ = await client.call(
                        model=settings.extraction_model,
                        system=SPECIFICS_SYSTEM,
                        user=specifics_prompt(question, selected),
                        stage="answer",
                        max_tokens=min(settings.reasoning_max_tokens, 3000),
                    )
                    found = tolerant_json(raw) or {}
                    instances = [
                        item for item in (found.get("instances") or [])
                        if isinstance(item, dict) and item.get("detail")
                    ][:30]
                    LOGGER.info("%s: collected %d supporting instances",
                                question.id, len(instances))
                except Exception as exc:  # noqa: BLE001 - the answer works without them
                    LOGGER.warning("%s: instance collection failed (%s)", question.id, exc)
            errors = []
            for model in dict.fromkeys((settings.answer_model, settings.extraction_model)):
                try:
                    text, _ = await client.call(
                        model=model,
                        system=SYNTHESIS_SYSTEM,
                        user=fit_prompt(question, plan, result, selected, drafts,
                                        prompt_char_budget(model), instances),
                        stage="answer",
                        max_tokens=min(settings.reasoning_max_tokens, 3200),
                    )
                    combined = normalize_direct_answer(text, question.id)
                    # A synthesis shorter than every candidate has dropped
                    # content instead of combining it; the best candidate is
                    # then the safer answer.
                    longest = max(drafts, key=lambda item: len(item[1]))[1]
                    if len(combined) < len(longest) * 0.6:
                        LOGGER.warning(
                            "%s: synthesis was shorter than its inputs; keeping the "
                            "fullest candidate", question.id,
                        )
                        return SubmissionAnswer(id=question.id, answer=longest, evidence=[])
                    return SubmissionAnswer(id=question.id, answer=combined, evidence=[])
                except Exception as exc:  # noqa: BLE001 - fall through to the next model
                    errors.append(f"{model}: {exc}")
            LOGGER.error("%s: synthesis failed (%s)", question.id, " | ".join(errors))
            longest = max(drafts, key=lambda item: len(item[1]))[1]
            return SubmissionAnswer(id=question.id, answer=longest, evidence=[])

        answers = await asyncio.gather(*(one(item) for item in questions))
        submission = Submission(
            team=settings.team_name,
            notes="Cross-system synthesis over independent runs, adjudicated on the shared evidence store.",
            answers=list(answers),
        )
        write_submission(args.output.resolve(), submission)
        failed = [item for item in submission.answers if item.error]
        print(
            f"Wrote {len(submission.answers)} synthesised answers to {args.output.resolve()} "
            f"({len(failed)} without candidates).",
            file=sys.stderr,
        )
        return 1 if failed else 0
    finally:
        for handle in stores:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--questions", type=Path, default=PROJECT_DIR / "input" / "questions.yaml")
    # One store per candidate system: adjudicating a disagreement using only
    # one side's evidence resolves it toward that side by construction.
    parser.add_argument("--state-db", type=Path, nargs="+",
                        default=[PROJECT_DIR / "output" / "state-v8.db"])
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "output" / "submission.json")
    parser.add_argument("--enrich", action="store_true",
                        help="With a single candidate, add supporting detail the "
                             "evidence holds and the draft omitted, instead of "
                             "returning it unchanged")
    parser.add_argument("--pool-across-versions", action="store_true",
                        help="Take evidence from stores written by other builds of this "
                             "system. Only pass this when you know every store read the "
                             "same document for the same questions")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    try:
        return asyncio.run(run(args))
    except (PipelineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
