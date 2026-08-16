"""Strict data contracts for the LLM-first Shared-Pass MELT pipeline."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Question(StrictModel):
    id: str
    question: str

    @field_validator("id", "question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class DocumentChunk(StrictModel):
    id: str
    order: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    heading_path: list[str] = Field(default_factory=list)
    primary_units: list[str] = Field(default_factory=list)
    # Ids of the detected repeating records this chunk covers, and whether the
    # chunk holds those records whole. Empty when the document is not periodic.
    record_ids: list[str] = Field(default_factory=list)
    text: str
    source_start: int
    source_end: int


class ScanPack(StrictModel):
    id: str
    order: int
    chunk_ids: list[str]
    primary_units: list[str] = Field(default_factory=list)
    record_ids: list[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    character_count: int
    # Logical rows expected from this pack. Transposed packs intentionally have
    # no record_ids because several logical records may share one source chunk.
    expected_records: int = 0


class QuestionPlan(StrictModel):
    question_id: str
    query_type: str
    target_unit: str
    target_entities: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    candidate_topics: list[str] = Field(default_factory=list)
    claim_slots: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    condition: Optional[str] = None
    operations: list[str] = Field(default_factory=list)
    required_passes: list[str] = Field(default_factory=list)
    closure_rules: list[str] = Field(default_factory=list)


class PlanBatch(StrictModel):
    plans: list[QuestionPlan]


RequirementKind = Literal["fact", "claim", "relation", "topic", "theme", "needle"]


class SharedRequirement(StrictModel):
    key: str
    kind: RequirementKind
    description: str
    aliases: list[str] = Field(default_factory=list)
    consumer_question_ids: list[str] = Field(default_factory=list)


class SharedEvidencePlan(StrictModel):
    document_hash: str
    questions_hash: str
    question_plans: list[QuestionPlan]
    requirements: list[SharedRequirement]
    enabled_passes: list[str]


EvidenceKind = Literal[
    "fact",
    "claim",
    "relation",
    "topic",
    "theme",
    "counterevidence",
    "uncertain",
]


class GlobalEvidence(StrictModel):
    id: str
    pass_name: str
    pack_id: str
    chunk_id: str
    primary_unit: Optional[str] = None
    kind: EvidenceKind
    requirement_key: Optional[str] = None
    consumer_question_ids: list[str] = Field(default_factory=list)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value_raw: Optional[str] = None
    value_normalized: Optional[Union[str, int, float, bool]] = None
    unit_raw: Optional[str] = None
    unit_normalized: Optional[str] = None
    related_entity: Optional[str] = None
    relation_type: Optional[str] = None
    topic: Optional[str] = None
    topic_status: Optional[str] = None
    source_quote: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    original_order: int = 0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ScanDelta(StrictModel):
    pass_name: str
    pack_id: str
    inspected_chunk_ids: list[str]
    inspected_primary_units: list[str] = Field(default_factory=list)
    evidence: list[GlobalEvidence] = Field(default_factory=list)
    uncertain_notes: list[str] = Field(default_factory=list)


class CoverageEntry(StrictModel):
    question_id: str
    required_passes: list[str]
    completed_packs: int
    required_packs: int
    evidence_items: int
    uncertain_items: int
    # Census closure against the detected record structure. Both stay 0 when the
    # document is not periodic, and the ratio is then not a closure condition.
    records_with_evidence: int = 0
    total_records: int = 0
    shadow_candidates: list[str] = Field(default_factory=list)
    closure_passed: bool
    failures: list[str] = Field(default_factory=list)


class SubmissionAnswer(StrictModel):
    id: str
    answer: str
    evidence: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class AnswerBatch(StrictModel):
    answers: list[SubmissionAnswer]


class Submission(StrictModel):
    team: str
    notes: str
    answers: list[SubmissionAnswer]


class RunReport(StrictModel):
    method: str
    document_hash: str
    questions_hash: str
    chunks: int
    packs: int
    census_packs: int = 0
    records: int = 0
    record_structure: str = "not detected"
    passes: list[str]
    estimated_calls: dict[str, Any]
    budget: dict[str, Any]
