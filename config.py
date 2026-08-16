"""Configuration for the LLM-first Shared-Pass MELT pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_API_BASE = "http://80.151.131.52:9839/v1"


def _integer(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _number(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_base: str
    team_name: str
    planning_model: str
    extraction_model: str
    audit_model: str
    answer_model: str
    temperature: float
    request_timeout_seconds: float
    planning_timeout_seconds: float
    reasoning_timeout_seconds: float
    max_api_attempts: int
    model_failure_threshold: int
    plan_batch_size: int
    answer_batch_size: int
    chunk_target_chars: int
    chunk_max_chars: int
    pack_target_chars: int
    pack_max_chars: int
    census_records_per_pack: int
    transposed_records_per_pack: int
    extraction_concurrency: int
    reasoning_concurrency: int
    total_time_budget_minutes: float
    final_reserve_minutes: float
    max_total_api_calls: int
    max_extraction_calls: int
    max_focus_calls: int
    max_audit_calls: int
    max_repair_calls: int
    max_needle_calls: int
    extraction_max_tokens: int
    census_max_tokens: int
    reasoning_max_tokens: int
    evidence_per_question: int
    max_answer_repair_calls: int = 6
    max_rate_limit_attempts: int = 20
    # Temperature zero makes repeated decoding as stable as the broker allows.
    # Keep top_p at 1 rather than constraining two sampling controls at once.
    top_p: float = 1.0
    structured_json_mode: bool = True

    def with_overrides(
        self,
        *,
        time_budget_minutes: Optional[float] = None,
        extraction_concurrency: Optional[int] = None,
        reasoning_concurrency: Optional[int] = None,
        max_total_api_calls: Optional[int] = None,
    ) -> "Settings":
        return replace(
            self,
            total_time_budget_minutes=(
                self.total_time_budget_minutes
                if time_budget_minutes is None
                else time_budget_minutes
            ),
            extraction_concurrency=(
                self.extraction_concurrency
                if extraction_concurrency is None
                else extraction_concurrency
            ),
            reasoning_concurrency=(
                self.reasoning_concurrency
                if reasoning_concurrency is None
                else reasoning_concurrency
            ),
            max_total_api_calls=(
                self.max_total_api_calls
                if max_total_api_calls is None
                else max_total_api_calls
            ),
        )

    def validate(self, require_key: bool) -> None:
        if require_key and not self.api_key:
            raise ValueError("COURSE_API_KEY is required for an LLM-first run")
        allowed = {"mistral-small-3-2", "ministral-3b-2512"}
        for name, model in {
            "PLANNING_MODEL": self.planning_model,
            "EXTRACTION_MODEL": self.extraction_model,
            "AUDIT_MODEL": self.audit_model,
            "ANSWER_MODEL": self.answer_model,
        }.items():
            if model not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {model!r}")
        if not 0 < self.chunk_target_chars <= self.chunk_max_chars:
            raise ValueError("CHUNK_TARGET_CHARS must not exceed CHUNK_MAX_CHARS")
        if not self.chunk_target_chars <= self.pack_target_chars <= self.pack_max_chars:
            raise ValueError("Pack sizes must satisfy CHUNK_TARGET <= PACK_TARGET <= PACK_MAX")
        if not 0 < self.final_reserve_minutes < self.total_time_budget_minutes:
            raise ValueError("FINAL_RESERVE_MINUTES must be inside the total budget")
        if not 0 < self.top_p <= 1:
            raise ValueError("COURSE_TOP_P must be in (0, 1]")
        for name, value in {
            "PLAN_BATCH_SIZE": self.plan_batch_size,
            "ANSWER_BATCH_SIZE": self.answer_batch_size,
            "CENSUS_RECORDS_PER_PACK": self.census_records_per_pack,
            "TRANSPOSED_RECORDS_PER_PACK": self.transposed_records_per_pack,
            "EXTRACTION_CONCURRENCY": self.extraction_concurrency,
            "REASONING_CONCURRENCY": self.reasoning_concurrency,
            "MAX_TOTAL_API_CALLS": self.max_total_api_calls,
            "MAX_EXTRACTION_CALLS": self.max_extraction_calls,
            "MAX_FOCUS_CALLS": self.max_focus_calls,
            "MAX_REPAIR_CALLS": self.max_repair_calls,
            "MAX_ANSWER_REPAIR_CALLS": self.max_answer_repair_calls,
            "MAX_RATE_LIMIT_ATTEMPTS": self.max_rate_limit_attempts,
            "MAX_API_ATTEMPTS": self.max_api_attempts,
            "MODEL_FAILURE_THRESHOLD": self.model_failure_threshold,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")


def load_settings(env_path: Optional[Path] = None) -> Settings:
    load_dotenv(env_path or PROJECT_DIR / ".env", override=False)
    return Settings(
        api_key=os.getenv("COURSE_API_KEY", "").strip(),
        api_base=os.getenv("COURSE_API_BASE", DEFAULT_API_BASE).rstrip("/"),
        team_name=os.getenv("TEAM_NAME", "yourteam").strip() or "yourteam",
        planning_model=os.getenv("PLANNING_MODEL", "mistral-small-3-2").strip(),
        extraction_model=os.getenv("EXTRACTION_MODEL", "ministral-3b-2512").strip(),
        audit_model=os.getenv("AUDIT_MODEL", "mistral-small-3-2").strip(),
        answer_model=os.getenv("ANSWER_MODEL", "mistral-small-3-2").strip(),
        temperature=_number("COURSE_TEMPERATURE", 0.0),
        request_timeout_seconds=_number("COURSE_REQUEST_TIMEOUT", 120.0),
        planning_timeout_seconds=_number("PLANNING_TIMEOUT_SECONDS", 60.0),
        reasoning_timeout_seconds=_number("REASONING_TIMEOUT_SECONDS", 75.0),
        max_api_attempts=_integer("MAX_API_ATTEMPTS", 2),
        model_failure_threshold=_integer("MODEL_FAILURE_THRESHOLD", 2),
        plan_batch_size=_integer("PLAN_BATCH_SIZE", 8),
        answer_batch_size=_integer("ANSWER_BATCH_SIZE", 1),
        chunk_target_chars=_integer("CHUNK_TARGET_CHARS", 12_000),
        chunk_max_chars=_integer("CHUNK_MAX_CHARS", 13_000),
        pack_target_chars=_integer("PACK_TARGET_CHARS", 34_000),
        pack_max_chars=_integer("PACK_MAX_CHARS", 34_000),
        # A census pack holds whole records. One record per call is the unit of
        # work at which a small model transcribes instead of sampling, and the
        # cheap model makes the extra calls affordable.
        census_records_per_pack=_integer("CENSUS_RECORDS_PER_PACK", 1),
        # How many gathered sections one transposed call transcribes. Large
        # enough that the count is visible in a single response, small
        # enough that the response closes inside the output ceiling.
        transposed_records_per_pack=_integer("TRANSPOSED_RECORDS_PER_PACK", 15),
        extraction_concurrency=_integer("EXTRACTION_CONCURRENCY", 6),
        reasoning_concurrency=_integer("REASONING_CONCURRENCY", 1),
        total_time_budget_minutes=_number("TIME_BUDGET_MINUTES", 50.0),
        final_reserve_minutes=_number("FINAL_RESERVE_MINUTES", 8.0),
        # Cheap extraction and repair run on ministral-3b, which costs a fifth
        # of the small model on input and a fifteenth on output, so the call
        # cap is set by what the time budget allows rather than by spend.
        max_total_api_calls=_integer("MAX_TOTAL_API_CALLS", 320),
        # Three-times-larger catalogues need more cheap extraction calls even
        # though the overall 320-call ceiling is unchanged. Small documents do
        # not consume this allowance merely because it is available.
        max_extraction_calls=_integer("MAX_EXTRACTION_CALLS", 250),
        max_focus_calls=_integer("MAX_FOCUS_CALLS", 32),
        max_audit_calls=_integer("MAX_AUDIT_CALLS", 12),
        max_repair_calls=_integer("MAX_REPAIR_CALLS", 60),
        max_needle_calls=_integer("MAX_NEEDLE_CALLS", 4),
        extraction_max_tokens=_integer("EXTRACTION_MAX_TOKENS", 9000),
        # The census schema is five fields per row, so a pack of whole records
        # closes its JSON well inside this ceiling.
        census_max_tokens=_integer("CENSUS_MAX_TOKENS", 4000),
        reasoning_max_tokens=_integer("REASONING_MAX_TOKENS", 5000),
        evidence_per_question=_integer("EVIDENCE_PER_QUESTION", 48),
        max_answer_repair_calls=_integer("MAX_ANSWER_REPAIR_CALLS", 6),
        max_rate_limit_attempts=_integer("MAX_RATE_LIMIT_ATTEMPTS", 20),
        top_p=_number("COURSE_TOP_P", 1.0),
        structured_json_mode=os.getenv("STRUCTURED_JSON_MODE", "1").strip().casefold()
        not in {"0", "false", "no", "off"},
    )
