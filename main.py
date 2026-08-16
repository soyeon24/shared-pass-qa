"""CLI for the LLM-first Shared-Pass MELT pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from config import PROJECT_DIR, load_settings
from pipeline import BudgetExceeded, PipelineError, dry_run_report, run_llm_first_pipeline


DEFAULT_DOCUMENT = PROJECT_DIR / "input" / "document.txt"
DEFAULT_QUESTIONS = PROJECT_DIR / "input" / "questions.yaml"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "submission.json"
DEFAULT_STATE = PROJECT_DIR / "output" / "state-v8_4.db"
PARALLEL_PROFILES = {
    "safe": (4, 1),
    "balanced": (6, 1),
    "aggressive": (8, 2),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-first shared-evidence pipeline for long-document QA"
    )
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Accepted for clarity; runs always resume safely")
    parser.add_argument("--force", action="store_true", help="Bypass exact-run caches and rebuild derived state")
    parser.add_argument("--question-id")
    parser.add_argument("--category")
    parser.add_argument("--time-budget-minutes", type=float)
    parser.add_argument("--max-total-calls", type=int)
    parser.add_argument("--parallel-profile", choices=sorted(PARALLEL_PROFILES), default="balanced")
    parser.add_argument("--extraction-concurrency", type=int)
    parser.add_argument("--reasoning-concurrency", type=int)
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def main() -> int:
    args = arguments()
    log_path = args.output.with_suffix(".log")
    if not args.dry_run:
        configure_logging(log_path)
    extraction, reasoning = PARALLEL_PROFILES[args.parallel_profile]
    try:
        settings = load_settings().with_overrides(
            time_budget_minutes=args.time_budget_minutes,
            extraction_concurrency=(
                args.extraction_concurrency
                if args.extraction_concurrency is not None else extraction
            ),
            reasoning_concurrency=(
                args.reasoning_concurrency
                if args.reasoning_concurrency is not None else reasoning
            ),
            max_total_api_calls=args.max_total_calls,
        )
        if args.dry_run:
            report = dry_run_report(
                settings=settings,
                document_path=args.document.resolve(),
                questions_path=args.questions.resolve(),
                question_id=args.question_id,
                category=args.category,
            )
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0
        submission, report = asyncio.run(
            run_llm_first_pipeline(
                settings=settings,
                document_path=args.document.resolve(),
                questions_path=args.questions.resolve(),
                output_path=args.output.resolve(),
                state_db_path=args.state_db.resolve(),
                question_id=args.question_id,
                category=args.category,
                force=args.force,
            )
        )
        hard_failures = [
            item for item in submission.answers
            if item.error and not item.error.startswith("coverage warning:")
        ]
        warnings = [
            item for item in submission.answers
            if item.error and item.error.startswith("coverage warning:")
        ]
        print(
            f"Wrote {len(submission.answers)} LLM-generated answers to {args.output.resolve()} "
            f"({report.estimated_calls['llm_shared_extraction']} planned shared extraction calls).",
            file=sys.stderr,
        )
        print(
            f"Answer status: {len(submission.answers) - len(hard_failures)} usable, "
            f"{len(warnings)} coverage warnings, {len(hard_failures)} hard failures.",
            file=sys.stderr,
        )
        return 1 if hard_failures else 0
    except (PipelineError, BudgetExceeded, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: rerun the same command to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
