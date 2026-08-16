"""Grade a submission locally with the challenge's own formula.

    question score = max(0, covered / total_key_points - 0.25 * forbidden_hits)
    quality        = mean over EVERY question in the set

The judge model is ours, not theirs, so the two will disagree at the margin.
What this is for is catching regressions, which are large, and reading a run on
a question set the leaderboard does not accept.

    python grade.py --questions ../other_questions.yaml --answers output/sub.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from config import PROJECT_DIR, load_settings
from pipeline import CallBudget, LLMClient, SQLiteStore, tolerant_json

JUDGE_SYSTEM = """You grade one answer against a checklist.

For each KEY POINT decide whether the answer states it. It counts only if the
answer asserts it — not if it merely mentions the topic, and not if it lists the
right thing among alternatives without committing to it.

For each FORBIDDEN CLAIM decide whether the answer makes it.

Judge only what the answer says. Do not use your own knowledge of the subject to
fill gaps, and give no credit for something the answer implies but never states.

Reply with JSON and nothing else:
{"key_points": [true, false, ...], "forbidden": [false, ...]}
one boolean per item, in the order given."""


def judge_prompt(answer: str, key_points: list[str], forbidden: list[str]) -> str:
    return f"""--- ANSWER ---
{answer or "(no answer given)"}
--- END ANSWER ---

KEY POINTS:
{chr(10).join(f"{i + 1}. {k}" for i, k in enumerate(key_points))}

FORBIDDEN CLAIMS:
{chr(10).join(f"{i + 1}. {f}" for i, f in enumerate(forbidden)) or "(none)"}"""


async def main_async(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.validate(require_key=True)
    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    submission = json.loads(args.answers.read_text(encoding="utf-8"))
    answers = {item["id"]: item.get("answer", "") for item in submission.get("answers", [])}

    store = SQLiteStore(Path(args.cache).resolve())
    budget = CallBudget(settings, expected=len(questions))
    client = LLMClient(settings, store, budget, use_cache=True)

    async def grade_one(question: dict) -> tuple[str, float, int, int, list[str], list[str]]:
        keys = list(question.get("key_points") or [])
        forb = list(question.get("forbidden") or [])
        if not keys:
            return question["id"], 0.0, 0, 0, [], []
        text, _ = await client.call(
            model=settings.answer_model,
            system=JUDGE_SYSTEM,
            user=judge_prompt(answers.get(question["id"], ""), keys, forb),
            stage="answer",
            max_tokens=400,
        )
        verdict = tolerant_json(text)
        got = [bool(x) for x in (verdict.get("key_points") or [])][: len(keys)]
        hit = [bool(x) for x in (verdict.get("forbidden") or [])][: len(forb)]
        score = max(0.0, sum(got) / len(keys) - 0.25 * sum(hit))
        missed = [k for k, ok in zip(keys, got + [False] * len(keys)) if not ok]
        tripped = [f for f, ok in zip(forb, hit + [False] * len(forb)) if ok]
        return question["id"], score, sum(got), sum(hit), missed, tripped

    try:
        results = await asyncio.gather(*(grade_one(item) for item in questions))
    finally:
        store.close()

    by_category: dict[str, list[float]] = {}
    lookup = {item["id"]: item for item in questions}
    print(f"\n{'q':<6}{'score':>7}  {'keys':>6}  category")
    print("-" * 46)
    for qid, score, got, hit, missed, tripped in results:
        question = lookup[qid]
        category = str(question.get("category") or "?")
        by_category.setdefault(category, []).append(score)
        total = len(question.get("key_points") or [])
        print(f"{qid:<6}{score * 100:>6.0f}%  {got:>2}/{total:<3}  {category}")
        for item in missed:
            print(f"        missed: {item[:78]}")
        for item in tripped:
            print(f"        TRIPPED: {item[:78]}")

    quality = sum(score for _, score, *_ in results) / len(questions)
    print("-" * 46)
    for category, scores in sorted(by_category.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        mean = 100 * sum(scores) / len(scores)
        print(f"  {category:<18}{mean:>6.1f}%  {'#' * int(mean / 4)}")
    print(f"\n  QUALITY  {quality * 100:.1f}%   (mean over all {len(questions)} questions)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=PROJECT_DIR / "input" / "questions.yaml")
    parser.add_argument("--answers", type=Path, default=PROJECT_DIR / "output" / "submission.json")
    parser.add_argument("--cache", default=str(PROJECT_DIR / "output" / "grade-cache.db"))
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except Exception as exc:  # noqa: BLE001 - a grading failure is not a pipeline failure
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
