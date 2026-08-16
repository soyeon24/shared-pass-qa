from __future__ import annotations

import asyncio
import inspect
import json
import re
import sqlite3
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import Settings
from pipeline import (
    BudgetExceeded,
    CallBudget,
    DocumentInfo,
    PipelineError,
    build_shared_requirements,
    answer_contract_issues,
    census_field_gaps,
    census_gaps,
    census_identity_requirement,
    census_to_scan_payload,
    chunk_document,
    call_structured,
    coverage_for_question,
    dry_run_report,
    enabled_passes,
    evidence_limit_for,
    load_questions,
    make_census_packs,
    mask_page_markers,
    normalize_document,
    normalize_answer_batch,
    normalize_direct_answer,
    normalize_number,
    normalize_plan_batch,
    normalize_scan_delta,
    questions_hash,
    reduce_question,
    run_llm_first_pipeline,
    select_evidence,
    tolerant_json,
    _explicit_alternatives,
    _category,
    _claim_ranking_conflicts,
    _canonical_unit,
    _cross_section_candidates,
    _field_is_plausible,
    _focus_pack,
    _measurement_number,
    _recover_complete_evidence_items,
    _record_chunks,
    _source_ordinal_ranking_conflicts,
    _source_ranking_claim_records,
    _stated_totals,
    trusted_evidence_for_question,
)
from records import detect_records
from prompts import ANSWER_SYSTEM, planning_prompt
from schemas import (
    DocumentChunk,
    GlobalEvidence,
    Question,
    QuestionPlan,
    ScanPack,
    ScanDelta,
    SharedRequirement,
)


def settings(api_key: str = "test-key") -> Settings:
    return Settings(
        api_key=api_key,
        api_base="http://example.invalid/v1",
        team_name="test-team",
        planning_model="ministral-3b-2512",
        extraction_model="ministral-3b-2512",
        audit_model="ministral-3b-2512",
        answer_model="ministral-3b-2512",
        temperature=0.0,
        request_timeout_seconds=2.0,
        planning_timeout_seconds=1.0,
        reasoning_timeout_seconds=1.0,
        max_api_attempts=1,
        model_failure_threshold=1,
        plan_batch_size=8,
        answer_batch_size=1,
        chunk_target_chars=300,
        chunk_max_chars=500,
        pack_target_chars=800,
        pack_max_chars=1200,
        census_records_per_pack=1,
        transposed_records_per_pack=15,
        extraction_concurrency=4,
        reasoning_concurrency=2,
        total_time_budget_minutes=5.0,
        final_reserve_minutes=1.0,
        max_total_api_calls=30,
        max_extraction_calls=18,
        max_focus_calls=4,
        max_audit_calls=2,
        max_repair_calls=2,
        max_needle_calls=2,
        extraction_max_tokens=2000,
        census_max_tokens=1200,
        reasoning_max_tokens=1200,
        evidence_per_question=12,
    )


def plan(category: str = "aggregation") -> QuestionPlan:
    return QuestionPlan(
        question_id="q1",
        query_type=category,
        target_unit="basin",
        target_entities=["basin area"],
        required_fields=["area"],
        candidate_topics=[],
        claim_slots=[],
        relation_types=[],
        aliases={"area": ["area", "covers"]},
        condition=">= 100" if category == "aggregation" else None,
        operations=["count"],
        required_passes=["facts"],
        closure_rules=["all packs inspected"],
    )


class LLMFirstTests(unittest.TestCase):
    def test_runtime_has_no_deterministic_answer_shortcut(self) -> None:
        source = Path(inspect.getfile(run_llm_first_pipeline)).read_text(encoding="utf-8")
        self.assertNotIn("local_preflight", source)
        self.assertNotIn("deterministic_submission_answer", source)
        self.assertNotIn("extract_profile_records", source)
        self.assertNotRegex(source, r"\bd\d{2}\b")

    def test_question_loader_discards_all_evaluation_metadata_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.yaml"
            second_path = root / "second.yaml"
            first_path.write_text(
                """# category: secret-comment-category
- id: q1
  category: deliberately_wrong
  question: How many basins qualify?
  key_points: [secret expected answer]
  forbidden: [secret forbidden answer]
""",
                encoding="utf-8",
            )
            second_path.write_text(
                """- id: q1
  category: another_wrong_value
  question: How many basins qualify?
  key_points: [different]
  forbidden: [different]
""",
                encoding="utf-8",
            )
            first = load_questions(first_path)
            second = load_questions(second_path)
        self.assertEqual(first[0].model_dump(), {
            "id": "q1", "question": "How many basins qualify?"
        })
        prompt_text = planning_prompt(first, {"aggregation": {}}, {"characters": 10})
        for secret in (
            "secret expected answer", "secret forbidden answer",
            "secret-comment-category", "deliberately_wrong",
        ):
            self.assertNotIn(secret, prompt_text)
        self.assertEqual(questions_hash(first), questions_hash(second))

    def test_question_type_is_inferred_from_wording_not_yaml_labels(self) -> None:
        examples = {
            "aggregation": "Which peaks are 3,000 metres or more?",
            "superlative": "Which entry reports the largest area?",
            "absence": "Which option is never substantively discussed?",
            "contradiction": "One is called the largest, but another has a larger figure.",
            "cross_section": "Compare the first section with Box 2: how far apart are they?",
            "global_synthesis": "Taking the document as a whole, what recurring story appears?",
            "needle": "In what year did the event begin?",
        }
        for expected, wording in examples.items():
            with self.subTest(expected=expected):
                self.assertEqual(_category(Question(id=expected, question=wording)), expected)

    def test_reducer_uses_only_supplied_llm_evidence(self) -> None:
        item = GlobalEvidence(
            id="e1", pass_name="facts", pack_id="p1", chunk_id="c1",
            kind="fact", requirement_key="fact:area",
            consumer_question_ids=["q1"], subject="Alpha Basin",
            predicate="area", value_raw="120", value_normalized=120,
            unit_raw="sq km", source_quote="Alpha Basin has an area of 120 sq km.",
            original_order=1, confidence=1.0,
        )
        result = reduce_question(plan(), [item])
        self.assertEqual(result["qualifying_count"], 1)
        self.assertEqual(result["qualifying_entities"], ["Alpha Basin"])
        self.assertEqual(
            list(inspect.signature(reduce_question).parameters),
            ["plan", "evidence", "aliases"]
        )

    def test_identity_census_accepts_canonical_name_predicate_and_deduplicates_suffix(self) -> None:
        census_plan = plan().model_copy(update={
            "target_unit": "park", "target_entities": ["park"],
            "required_fields": ["park_name"], "condition": None,
        })
        items = [
            GlobalEvidence(
                id="e-name-1", pass_name="facts", pack_id="p1", chunk_id="c1",
                kind="fact", requirement_key="fact:name", consumer_question_ids=["q1"],
                primary_unit="Alpha Park", subject="Alpha Park", predicate="name",
                value_raw="Alpha National Park", value_normalized="Alpha National Park",
                source_quote="Alpha National Park", confidence=1.0,
            ),
            GlobalEvidence(
                id="e-name-2", pass_name="facts", pack_id="p2", chunk_id="c2",
                kind="fact", requirement_key="fact:name", consumer_question_ids=["q1"],
                primary_unit="Alpha National Park", subject="Alpha National Park",
                predicate="park_name", value_raw="Alpha Park", value_normalized="Alpha Park",
                source_quote="Alpha Park", confidence=1.0,
            ),
        ]
        result = reduce_question(census_plan, items)
        self.assertEqual(result["entity_count"], 1)

    def test_superlative_reuses_compatible_measurement_from_another_question(self) -> None:
        query = plan("superlative").model_copy(update={
            "required_fields": ["area_sqkm"], "target_entities": ["park", "area"],
        })
        item = GlobalEvidence(
            id="e-cross", pass_name="claims", pack_id="p1", chunk_id="c1",
            kind="claim", requirement_key="claim:park_area",
            consumer_question_ids=["another_question"], subject="Large Park",
            predicate="area", value_raw="13,600", value_normalized=13600,
            unit_raw="sq km", source_quote="13,600 Area covered (sq km)",
            original_order=1, confidence=1.0,
        )
        self.assertEqual(reduce_question(query, [item])["maximum"]["subject"], "Large Park")

    def test_focus_evidence_supplements_shared_evidence(self) -> None:
        query = plan("contradiction")
        shared = GlobalEvidence(
            id="shared", pass_name="claims", pack_id="pack_001", chunk_id="c1",
            kind="claim", consumer_question_ids=["q1"], subject="Alpha",
            predicate="claim", source_quote="Alpha makes the first claim.",
        )
        focused = shared.model_copy(update={
            "id": "focused", "pack_id": "focus_q1_claims_shadow", "chunk_id": "c2",
            "subject": "Beta", "source_quote": "Beta supplies the second claim.",
        })
        selected = trusted_evidence_for_question([shared, focused], query)
        self.assertEqual({item.id for item in selected}, {"shared", "focused"})

    def test_focus_pack_keeps_a_stronger_later_occurrence(self) -> None:
        query = plan("contradiction").model_copy(update={
            "target_unit": "reserve", "target_entities": ["largest reserve", "country"],
            "aliases": {"largest reserve": ["largest protected reserve"]},
        })
        question = Question(
            id="q1",
            question="One reserve is called the largest in its country, but another is larger.",
        )
        text = (
            "The largest boulder is beside the trail. " + "filler " * 140
            + "Alpha Reserve is Freedonia's largest protected reserve."
        )
        chunk = DocumentChunk(
            id="c1", order=0, text=text, source_start=0, source_end=len(text)
        )
        requirement = SharedRequirement(
            key="claim:rank", kind="claim", description="largest reserve in a country",
            aliases=["largest protected reserve"], consumer_question_ids=["q1"],
        )
        pack = _focus_pack(query, question, "claims", [requirement], [chunk], 5_000)
        self.assertIsNotNone(pack)
        self.assertIn("Alpha Reserve is Freedonia's largest protected reserve", pack.text)

    def test_scoped_claim_is_joined_to_independent_group_ranking(self) -> None:
        records = [{
            "source_quote": "Alpha Reserve dominates Freedonia's largest reserve.",
            "page": 3, "chunk_id": "claim-chunk",
        }]
        rankings = {
            "area": {
                "Freedonia": [
                    {"entity": "Beta Reserve", "value": 120.0, "value_raw": "120",
                     "unit": "sq km", "source_quote": "Beta: 120 sq km",
                     "page": 9, "chunk_id": "beta-chunk"},
                    {"entity": "Alpha Reserve", "value": 100.0, "value_raw": "100",
                     "unit": "sq km", "source_quote": "Alpha: 100 sq km",
                     "page": 4, "chunk_id": "alpha-chunk"},
                ]
            }
        }
        conflicts = _claim_ranking_conflicts(records, rankings)
        self.assertEqual(conflicts[0]["claim"]["entity"], "Alpha Reserve")
        self.assertEqual(conflicts[0]["counterexample"]["entity"], "Beta Reserve")

    def test_literal_source_superlative_can_supply_a_missed_claim(self) -> None:
        text = (
            "The plateau of Alpha Reserve dominates Freedonia's largest reserve."
        )
        chunk = DocumentChunk(
            id="c1", order=0, text=text, source_start=0, source_end=len(text)
        )
        rankings = {
            "area": {"Freedonia": [
                {"entity": "Beta Reserve", "value": 120.0, "value_raw": "120",
                 "unit": "sq km", "source_quote": "Beta: 120 sq km"},
                {"entity": "Alpha Reserve", "value": 100.0, "value_raw": "100",
                 "unit": "sq km", "source_quote": "Alpha: 100 sq km"},
            ]}
        }
        source_rows = _source_ranking_claim_records([chunk])
        conflicts = _claim_ranking_conflicts(source_rows, rankings)
        self.assertEqual(conflicts[0]["claim"]["entity"], "Alpha Reserve")
        self.assertEqual(conflicts[0]["counterexample"]["entity"], "Beta Reserve")

    def test_short_alias_does_not_make_type_words_the_claimant(self) -> None:
        records = [{
            "source_quote": (
                "The plateau of Alpha dominates Freedonia's largest reserve."
            ),
            "page": 4,
            "chunk_id": "claim",
        }]
        rankings = {
            "area": {"Freedonia": [
                {"entity": "Beta Reserve", "value": 120.0, "value_raw": "120",
                 "unit": "sq km", "source_quote": "Beta: 120 sq km"},
                {"entity": "Alpha Reserve", "value": 100.0, "value_raw": "100",
                 "unit": "sq km", "source_quote": "Alpha: 100 sq km"},
                {"entity": "Gamma", "value": 80.0, "value_raw": "80",
                 "unit": "sq km", "source_quote": "Gamma: 80 sq km"},
            ]}
        }
        conflict = _claim_ranking_conflicts(records, rankings)[0]
        self.assertEqual(conflict["claim"]["entity"], "Alpha Reserve")
        self.assertEqual(conflict["counterexample"]["entity"], "Beta Reserve")

    def test_ordinal_rank_inversion_is_paired_by_scope_and_arithmetic(self) -> None:
        first = (
            "Alpha Peak is well known. At 1,080m, Freedonia's second-highest "
            "peak attracts many climbers."
        )
        second = (
            "Walk to the summit of Beta Peak (1,240m), Freedonia's sixthhighest "
            "mountain, for a wide view."
        )
        chunks = [
            DocumentChunk(id="c1", order=0, text=first, source_start=0, source_end=len(first)),
            DocumentChunk(id="c2", order=1, text=second, source_start=100, source_end=100 + len(second)),
        ]
        query = plan("contradiction").model_copy(update={
            "target_unit": "Freedonian mountain rank",
            "target_entities": ["Alpha Peak", "British-style rank"],
            "required_fields": ["rank", "elevation"],
        })
        question = Question(
            id="q1",
            question="What contradicts Alpha Peak's rank among Freedonian mountains?",
        )
        conflicts = _source_ordinal_ranking_conflicts(question, query, chunks)
        self.assertEqual(conflicts[0]["claim"]["entity"], "Alpha Peak")
        self.assertEqual(conflicts[0]["claim"]["value"], "1,080")
        self.assertEqual(conflicts[0]["counterexample"]["entity"], "Beta Peak")
        self.assertEqual(conflicts[0]["counterexample"]["value"], "1,240")

    def test_ordinal_height_conflict_does_not_answer_an_area_question(self) -> None:
        first = "At 1,080m, Freedonia's second-highest peak is Alpha Peak."
        second = "Beta Peak (1,240m) is Freedonia's sixthhighest mountain."
        chunks = [
            DocumentChunk(id="c1", order=0, text=first, source_start=0, source_end=len(first)),
            DocumentChunk(id="c2", order=1, text=second, source_start=100, source_end=100 + len(second)),
        ]
        query = plan("contradiction").model_copy(update={
            "target_unit": "reserve area",
            "target_entities": ["largest reserve in a country"],
            "required_fields": ["area", "country"],
        })
        question = Question(
            id="q1",
            question="Which largest-in-its-country reserve is contradicted by a larger area?",
        )
        self.assertEqual(
            _source_ordinal_ranking_conflicts(question, query, chunks), []
        )

    def test_component_superlative_is_not_joined_to_container_ranking(self) -> None:
        records = [{
            "source_quote": (
                "Alpha Reserve contains Freedonia's largest glacier and is a popular reserve."
            ),
            "page": 3, "chunk_id": "claim-chunk",
        }]
        rankings = {
            "area": {"Freedonia": [
                {"entity": "Beta Reserve", "value": 120.0, "value_raw": "120",
                 "unit": "sq km", "source_quote": "Beta: 120 sq km"},
                {"entity": "Alpha Reserve", "value": 100.0, "value_raw": "100",
                 "unit": "sq km", "source_quote": "Alpha: 100 sq km"},
            ]}
        }
        self.assertEqual(_claim_ranking_conflicts(records, rankings), [])

    def test_cross_section_rejects_component_crossing_and_keeps_target(self) -> None:
        query = plan("cross_section").model_copy(update={
            "target_unit": "reserves extending across an international border",
            "target_entities": ["reserve", "border"],
            "relation_types": ["extends across", "paired across"],
        })
        base = {
            "kind": "relation", "pack_id": "p", "predicate": "border relation",
            "value_raw": None, "value_normalized": None, "unit": None,
            "relation_type": "crosses", "topic": None, "topic_status": None,
            "page": 1, "section": None, "order": 1,
        }
        records = [
            {**base, "subject": "Alpha Reserve", "primary_unit": "Alpha Reserve",
             "related_entity": "Freedonia", "chunk_id": "c1",
             "source_quote": "Alpha Reserve is shared between Freedonia and Sylvania."},
            {**base, "subject": "Old Trail", "primary_unit": "Alpha Reserve",
             "related_entity": "Sylvania", "chunk_id": "c2",
             "source_quote": "The old trail crosses into Sylvania."},
        ]
        found = _cross_section_candidates(
            query, records, {}, ["Alpha Reserve", "Beta Reserve"]
        )
        self.assertEqual(len(found), 1)
        self.assertIn("Alpha Reserve", found[0]["source_quote"])

    def test_answer_contract_requires_directional_ranking_repair(self) -> None:
        query = plan("contradiction")
        result = {"claim_ranking_conflicts": [{
            "claim": {"entity": "Alpha Reserve", "value": "100"},
            "counterexample": {"entity": "Beta Reserve", "value": "120"},
        }]}
        issues = answer_contract_issues(
            "Alpha Reserve is larger than Beta Reserve.", query, result
        )
        self.assertTrue(any("Beta Reserve" in issue and "strictly larger" in issue for issue in issues))

    def test_needle_number_must_appear_in_the_grounded_quote(self) -> None:
        text = "Starting as a small art gallery, the hotel later expanded."
        chunk = DocumentChunk(id="c1", order=0, text=text, source_start=0, source_end=len(text))
        pack = ScanPack(id="needle_q1", order=0, chunk_ids=["c1"], text=text, character_count=len(text))
        requirement = SharedRequirement(
            key="needle:origin", kind="needle", description="hotel origin year",
            aliases=["hotel", "origin"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [{"chunk_id": "c1", "kind": "fact",
                "requirement_key": "needle:origin", "consumer_question_ids": ["q1"],
                "subject": "hotel", "predicate": "origin year", "value_raw": "2005",
                "source_quote": text}]},
            pass_name="needle", pack=pack, requirements=[requirement], chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])
        self.assertTrue(any("needle number" in note for note in normalized["uncertain_notes"]))

    def test_ungrounded_model_quote_is_discarded(self) -> None:
        chunk = DocumentChunk(
            id="c1", order=0, text="Alpha Basin has an area of 120 sq km.",
            source_start=0, source_end=39,
        )
        pack = ScanPack(
            id="p1", order=0, chunk_ids=["c1"], text=chunk.text,
            character_count=len(chunk.text),
        )
        requirement = SharedRequirement(
            key="fact:area", kind="fact", description="area",
            aliases=["area"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {
                "evidence": [{
                    "chunk_id": "c1", "kind": "fact",
                    "requirement_key": "fact:area",
                    "consumer_question_ids": ["q1"],
                    "subject": "Alpha Basin", "predicate": "area",
                    "value_raw": "999", "source_quote": "invented quote",
                }]
            },
            pass_name="facts", pack=pack, requirements=[requirement],
            chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])
        self.assertTrue(any("ungrounded" in note for note in normalized["uncertain_notes"]))

    def test_unknown_requirement_and_placeholder_are_discarded(self) -> None:
        chunk = DocumentChunk(
            id="c1", order=0, text="Alpha Basin has an area of 120 sq km.",
            source_start=0, source_end=39,
        )
        pack = ScanPack(id="p1", order=0, chunk_ids=["c1"], text=chunk.text, character_count=len(chunk.text))
        requirement = SharedRequirement(
            key="fact:area", kind="fact", description="area", aliases=["area"],
            consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [
                {"chunk_id": "c1", "kind": "fact", "requirement_key": "fact:other",
                 "consumer_question_ids": ["q1"], "predicate": "population",
                 "value_raw": "10", "source_quote": chunk.text},
                {"chunk_id": "c1", "kind": "fact", "requirement_key": "fact:area",
                 "consumer_question_ids": ["q1"], "predicate": "area",
                 "value_raw": "not explicitly stated", "source_quote": chunk.text},
            ]},
            pass_name="facts", pack=pack, requirements=[requirement], chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])

    def test_table_measurement_uses_number_nearest_the_label(self) -> None:
        text = "Example Park\n1309 Highest point: Ben Macdui (m) 278 Annual visitors (thousand)"
        chunk = DocumentChunk(id="c1", order=0, text=text, source_start=0, source_end=len(text))
        pack = ScanPack(id="p1", order=0, chunk_ids=["c1"], text=text, character_count=len(text))
        requirement = SharedRequirement(
            key="fact:highest", kind="fact", description="highest point elevation",
            aliases=["highest point"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [{"chunk_id": "c1", "kind": "fact",
                "requirement_key": "fact:highest", "consumer_question_ids": ["q1"],
                "subject": "Example Park", "predicate": "highest point",
                "value_raw": "278", "unit_raw": "m", "source_quote": text}]},
            pass_name="facts", pack=pack, requirements=[requirement], chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"][0]["value_raw"], "1309")

    def test_numeric_measurement_rejects_subject_not_present_near_source_row(self) -> None:
        text = "Actual Park\n1309 Highest point: Ben Macdui (m)"
        chunk = DocumentChunk(id="c1", order=0, text=text, source_start=0, source_end=len(text))
        pack = ScanPack(id="p1", order=0, chunk_ids=["c1"], text=text, character_count=len(text))
        requirement = SharedRequirement(
            key="fact:highest", kind="fact", description="highest point elevation",
            aliases=["highest point"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [{"chunk_id": "c1", "kind": "fact",
                "requirement_key": "fact:highest", "consumer_question_ids": ["q1"],
                "subject": "Different Park", "predicate": "highest point",
                "value_raw": "1309", "unit_raw": "m", "source_quote": text}]},
            pass_name="facts", pack=pack, requirements=[requirement], chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])
        self.assertTrue(any("numeric subject" in note for note in normalized["uncertain_notes"]))

    def test_absence_alternatives_and_numeric_threshold_are_normalized(self) -> None:
        questions = [
            Question(id="a", question=(
                "Of these threats — poaching, wartime damage, glacier retreat, and visitor pressure — "
                "which is never raised?"
            )),
            Question(id="b", question="Which peaks are 3,000 metres or more?"),
        ]
        raw = {"plans": [
            {"question_id": "a", "candidate_topics": ["threats"], "aliases": {}},
            {"question_id": "b", "required_fields": ["height"], "aliases": {}},
        ]}
        plans = normalize_plan_batch(raw, questions)["plans"]
        self.assertEqual(
            plans[0]["candidate_topics"],
            ["poaching", "wartime damage", "glacier retreat", "visitor pressure"],
        )
        self.assertEqual(plans[1]["condition"], "value >= 3,000 metres")

    def test_compound_alternative_is_not_split_on_internal_and(self) -> None:
        question = (
            "Of these four threats — poaching, wartime damage and military history, "
            "glacier retreat, and pressure from visitor numbers — which is absent?"
        )
        self.assertEqual(
            _explicit_alternatives(question),
            [
                "poaching",
                "wartime damage and military history",
                "glacier retreat",
                "pressure from visitor numbers",
            ],
        )

    def test_aggregation_does_not_create_a_fake_population_total_requirement(self) -> None:
        aggregation = plan().model_copy(update={
            "condition": None,
            "required_fields": ["park_name", "country"],
        })
        descriptions = [item.description for item in build_shared_requirements([aggregation])]
        self.assertNotIn("explicit stated size of the complete target population", descriptions)

    def test_topic_label_cannot_invent_a_mention_missing_from_quote(self) -> None:
        text = "Several polar explorers set off from the settlement in the early twentieth century."
        chunk = DocumentChunk(id="c1", order=0, text=text, source_start=0, source_end=len(text))
        pack = ScanPack(id="p1", order=0, chunk_ids=["c1"], text=text, character_count=len(text))
        requirement = SharedRequirement(
            key="topic:poaching", kind="topic", description="poaching",
            aliases=["illegal hunting", "poachers"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [{
                "chunk_id": "c1", "kind": "topic", "requirement_key": "topic:poaching",
                "consumer_question_ids": ["q1"], "topic": "poaching",
                "topic_status": "substantive", "source_quote": text,
            }]},
            pass_name="topics", pack=pack, requirements=[requirement],
            chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])

    def test_categorical_fact_value_must_occur_in_its_quote(self) -> None:
        text = "Alpha National Park — Freedonia"
        chunk = DocumentChunk(id="c1", order=0, text=text, source_start=0, source_end=len(text))
        pack = ScanPack(id="p1", order=0, chunk_ids=["c1"], text=text, character_count=len(text))
        requirement = SharedRequirement(
            key="fact:country", kind="fact", description="country",
            aliases=["country"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [{
                "chunk_id": "c1", "kind": "fact", "requirement_key": "fact:country",
                "consumer_question_ids": ["q1"], "primary_unit": "Alpha National Park",
                "subject": "Alpha National Park", "predicate": "country",
                "value_raw": "Sylvania", "source_quote": text,
            }]},
            pass_name="facts", pack=pack, requirements=[requirement],
            chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])

    def test_absence_combined_topic_accepts_a_constituent_alias(self) -> None:
        absence_plan = plan("absence").model_copy(update={
            "candidate_topics": ["poaching", "wartime damage and military history"],
            "aliases": {
                "poaching": ["illegal hunting"],
                "wartime damage and military history": ["military history", "war damage"],
            },
            "required_passes": ["topics"],
        })
        item = GlobalEvidence(
            id="e-war", pass_name="topics", pack_id="p1", chunk_id="c1",
            kind="topic", requirement_key="topic:war", consumer_question_ids=["q1"],
            topic="military history", topic_status="substantive",
            source_quote="Traces of many military routes remain.", confidence=1.0,
        )
        result = reduce_question(absence_plan, [item])
        self.assertEqual(
            result["topic_matrix"]["wartime damage and military history"],
            {"substantive": 1},
        )
        self.assertIn("poaching", result["topics_without_substantive_evidence"])

    def test_answer_objects_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "natural-language string"):
            normalize_answer_batch(
                {"answers": [{"id": "q1", "answer": {"count": 2}}]}, ["q1"]
            )

    def test_multiline_malformed_json_answer_is_recovered_locally(self) -> None:
        text = '''```json
{"answers":[{"id":"q1","answer":"First line.
Second line.","evidence":[]}]}
```'''
        self.assertEqual(
            normalize_direct_answer(text, "q1"), "First line.\nSecond line."
        )

    def test_normalized_measurement_is_not_converted_twice(self) -> None:
        superlative = plan("superlative").model_copy(
            update={"required_fields": ["area"], "target_entities": ["area"]}
        )
        normalized_area = normalize_number("1151", "square miles")
        item = GlobalEvidence(
            id="e-area", pass_name="facts", pack_id="p1", chunk_id="c1",
            kind="fact", requirement_key="fact:area", consumer_question_ids=["q1"],
            subject="Example Park", predicate="area", value_raw="1151",
            value_normalized=normalized_area, unit_raw="square miles",
            source_quote="1151 Area covered (square miles)", confidence=1.0,
        )
        result = reduce_question(superlative, [item])
        self.assertAlmostEqual(result["maximum"]["value_normalized"], normalized_area)

    def test_unrelated_claim_cannot_reuse_an_exact_requirement_key(self) -> None:
        text = "Alpha Basin has an area of 120 sq km."
        chunk = DocumentChunk(id="c1", order=0, text=text, source_start=0, source_end=len(text))
        pack = ScanPack(id="p1", order=0, chunk_ids=["c1"], text=text, character_count=len(text))
        requirement = SharedRequirement(
            key="claim:summit", kind="claim", description="Alpha Summit height ranking",
            aliases=["Alpha Summit", "height rank"], consumer_question_ids=["q1"],
        )
        normalized = normalize_scan_delta(
            {"evidence": [{"chunk_id": "c1", "kind": "claim",
                "requirement_key": "claim:summit", "consumer_question_ids": ["q1"],
                "predicate": "height_rank", "value_raw": "120",
                "source_quote": text}]},
            pass_name="claims", pack=pack, requirements=[requirement], chunks_by_id={"c1": chunk},
        )
        self.assertEqual(normalized["evidence"], [])

    def test_needle_requirements_remain_question_specific(self) -> None:
        first = plan("needle").model_copy(update={
            "question_id": "q1", "target_entities": ["founding year"],
            "required_fields": ["year"], "aliases": {"Archive Hall": ["archive venue"]},
            "required_passes": ["needle"],
        })
        second = first.model_copy(update={
            "question_id": "q2", "target_entities": ["oldest tree"],
            "aliases": {"oldest tree": ["ancient tree"]},
        })
        requirements = build_shared_requirements([first, second])
        self.assertEqual(len(requirements), 2)
        self.assertEqual(
            {tuple(item.consumer_question_ids) for item in requirements}, {("q1",), ("q2",)}
        )

    def test_tolerant_json_repairs_fences_and_trailing_commas(self) -> None:
        value = tolerant_json('```json\n{"items": [1, 2,],}\n```')
        self.assertEqual(value, {"items": [1, 2]})

    def test_tolerant_json_removes_comments_but_preserves_url_strings(self) -> None:
        value = tolerant_json(
            '{\n// model commentary\n"url":"https://example.test/a//b",'
            '"items":[1, /* uncertain */ 2]\n}'
        )
        self.assertEqual(value["url"], "https://example.test/a//b")
        self.assertEqual(value["items"], [1, 2])

    def test_tolerant_json_collapses_ambiguous_bare_page_pair(self) -> None:
        value = tolerant_json('{"evidence":[{"page":172,178,"kind":"fact"}]}')
        self.assertEqual(value["evidence"][0]["page"], 172)

    def test_repeating_records_are_detected_without_naming_any_marker(self) -> None:
        entries = []
        for index in range(24):
            entries.append(
                f"Entry {index} Reserve\n\nIntroductory prose for entry {index}.\n\n"
                f"## Key figures\n\n{100 + index}\n\nArea covered (sq km)\n\n"
                f"## Visit this\n\nSomething to do at entry {index}.\n\n"
                f"## Species\n\nWildlife seen at entry {index}.\n"
            )
        layout = detect_records("\n\n".join(entries))
        self.assertTrue(layout.detected)
        self.assertGreaterEqual(layout.count, 20)
        self.assertGreaterEqual(len(layout.markers), 2)
        # The detector must corroborate, not trust one regular marker.
        source = Path(inspect.getfile(detect_records)).read_text(encoding="utf-8")
        for token in ("key figures", "visit this", "species", "park in numbers"):
            self.assertNotIn(token, source.casefold())

    def test_internal_repetition_is_phase_aligned_to_record_openings(self) -> None:
        pages = [
            "<!-- page-start-marker-1 -->\n\n## Front matter\n\nPreface."
        ]
        page_number = 2
        for index in range(24):
            pages.extend([
                f"<!-- page-start-marker-{page_number} -->\n\n"
                f"## Entry {index} Reserve\n\nRegion {index}\n\nOpening summary.",
                f"<!-- page-start-marker-{page_number + 1} -->\n\n"
                f"## Key figures\n\n{100 + index}\n\nArea covered.",
                f"<!-- page-start-marker-{page_number + 2} -->\n\n"
                "## Species\n\nA substantive wildlife paragraph.",
                f"<!-- page-start-marker-{page_number + 3} -->\n\n"
                "## Visit this\n\nClosing itinerary.",
            ])
            page_number += 4
        pages.append(
            f"<!-- page-start-marker-{page_number} -->\n\n## Appendix\n\nReference index."
        )
        text = "\n\n".join(pages)
        offsets = [match.start() for match in re.finditer(
            r"<!--\s*page-start-marker-\d+\s*-->", text
        )]
        layout = detect_records(text, offsets)
        self.assertTrue(layout.detected)
        self.assertEqual(layout.count, 24)
        for index, record in enumerate(layout.records):
            opening = text[record.start:record.start + 220]
            self.assertIn(f"Entry {index} Reserve", opening)
        self.assertNotIn("Appendix", text[layout.records[-1].start:layout.records[-1].end])

    def test_semantic_shared_requirements_are_consolidated_per_question(self) -> None:
        synthesis = plan("global_synthesis").model_copy(update={
            "required_passes": ["topics"],
            "candidate_topics": ["decline", "intervention", "recovery"],
            "target_entities": ["species conservation", "recurring story"],
            "aliases": {"recovery": ["return", "reintroduction"]},
        })
        contradiction = plan("contradiction").model_copy(update={
            "required_passes": ["claims"],
            "claim_slots": ["first ranking claim", "second measured figure", "scope"],
            "relation_types": [],
        })
        requirements = build_shared_requirements([synthesis, contradiction])
        self.assertEqual(sum(item.kind == "theme" for item in requirements), 1)
        self.assertEqual(sum(item.kind == "claim" for item in requirements), 1)

    def test_absence_accepts_a_grounded_alias_wording(self) -> None:
        query = plan("absence").model_copy(update={
            "required_passes": ["topics"],
            "candidate_topics": ["pressure from visitor numbers", "illegal dumping"],
            "aliases": {
                "pressure from visitor numbers": ["tourist impact", "overtourism"],
                "illegal dumping": ["waste disposal"],
            },
        })
        item = GlobalEvidence(
            id="tourists", pass_name="topics", pack_id="p", chunk_id="c",
            kind="topic", requirement_key="topic:visitors",
            consumer_question_ids=["q1"], subject="Alpha Reserve",
            topic="pressure from visitor numbers", topic_status="substantive",
            source_quote="The reserve is crushed with tourists throughout summer.",
        )
        result = reduce_question(query, [item])
        self.assertNotIn(
            "pressure from visitor numbers",
            result["topics_without_substantive_evidence"],
        )
        self.assertIn("illegal dumping", result["topics_without_substantive_evidence"])

    def test_prose_without_periodic_structure_yields_no_records(self) -> None:
        prose = "\n\n".join(
            f"Paragraph {index} of continuous discussion with no repeating layout."
            for index in range(200)
        )
        layout = detect_records(prose)
        self.assertFalse(layout.detected)
        self.assertEqual(layout.count, 0)

    def test_chunks_never_straddle_two_records(self) -> None:
        entries = []
        for index in range(24):
            entries.append(
                f"Entry {index} Reserve\n\nProse for entry {index}.\n\n"
                f"## Key figures\n\n{100 + index}\n\nArea covered (sq km)\n\n"
                f"## Visit this\n\nSomething at entry {index}.\n\n"
                f"## Species\n\nWildlife at entry {index}.\n"
            )
        text = "\n\n".join(entries)
        document = DocumentInfo(path=Path("memory.txt"), text=text, sha256="hash")
        layout = detect_records(text)
        chunks = chunk_document(document, settings(), layout)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.record_ids), 1)
        packs = make_census_packs(chunks, layout, settings())
        # Every record is censused exactly once, and no record is split across
        # two packs; the pack count itself follows the extraction budget.
        assigned = [record_id for pack in packs for record_id in pack.record_ids]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {item.id for item in layout.records})

    def test_census_rows_become_grounded_evidence_per_record(self) -> None:
        text = "Alpha Reserve\n\nPark in numbers\n\n120\n\nArea covered (sq km)"
        chunk = DocumentChunk(
            id="c1", order=0, text=text, record_ids=["record_0000"],
            source_start=0, source_end=len(text),
        )
        pack = ScanPack(
            id="census_000", order=0, chunk_ids=["c1"], record_ids=["record_0000"],
            text=text, character_count=len(text),
        )
        requirement = SharedRequirement(
            key="fact:area", kind="fact", description="area",
            aliases=["area covered"], consumer_question_ids=["q1"],
        )
        payload = census_to_scan_payload(
            {"records": [{
                "record_name": "Alpha Reserve", "chunk_id": "c1",
                "fields": [{"key": "fact:area", "field": "Area covered",
                            "value": "120", "unit": "sq km",
                            "quote": "120\n\nArea covered (sq km)"}],
            }]},
            [requirement],
        )
        normalized = normalize_scan_delta(
            payload, pass_name="facts", pack=pack,
            requirements=[requirement], chunks_by_id={"c1": chunk},
        )
        areas = [item for item in normalized["evidence"] if item["value_normalized"] == 120]
        self.assertEqual(len(areas), 1)
        self.assertEqual(areas[0]["subject"], "Alpha Reserve")
        self.assertIn("120", areas[0]["source_quote"])

    def test_aggregation_asks_for_the_documents_own_stated_total(self) -> None:
        # A catalogue usually declares its own size in prose the per-record
        # census never reads, and an enumeration that missed entries has no
        # local signal that it undercounted.
        aggregation = plan().model_copy(update={"target_unit": "national park"})
        requirements = build_shared_requirements([aggregation])
        stated = [
            item for item in requirements
            if item.kind == "claim" and "stated total" in item.description
        ]
        self.assertEqual(len(stated), 1)
        self.assertIn(aggregation.question_id, stated[0].consumer_question_ids)
        # The claims channel has to run even if no other category enables it.
        self.assertIn(
            "claims",
            enabled_passes([Question(id="q1", question="How many parks?")]),
        )

    def test_stated_total_accepts_a_declared_count_and_rejects_a_measurement(self) -> None:
        aggregation = plan().model_copy(
            update={"target_unit": "national park", "target_entities": ["national park"]}
        )
        base = {
            "subject": None, "primary_unit": None, "requirement_key": "claim:x",
            "predicate": "profiles", "unit": None, "related_entity": None,
            "relation_type": None, "topic": None, "topic_status": None,
            "chunk_id": "c1", "page": 5, "section": None, "order": 0,
        }
        declared = {
            **base, "value_raw": "60", "value_normalized": 60,
            "source_quote": "Setting out to choose Europe's top 60 national parks was no easy task.",
        }
        self.assertEqual(
            [item["value"] for item in _stated_totals([declared], aggregation)], [60]
        )
        # A single park's area is a per-entry measurement, not a collection size.
        measurement = {
            **base, "value_raw": "13,600", "value_normalized": 13600, "unit": "sq km",
            "predicate": "area", "source_quote": "13,600 Area covered (sq km)",
        }
        self.assertEqual(_stated_totals([measurement], aggregation), [])

    def test_field_gap_needs_the_wording_beside_a_number(self) -> None:
        # Nearly every entry says "visitors" in prose; only a few report a
        # visitor figure. Chasing the word alone would re-scan the whole
        # document forever.
        requirement = SharedRequirement(
            key="fact:visitors", kind="fact", description="visitors_per_year",
            aliases=["visitors"], consumer_question_ids=["q1"],
        )
        prose = "Most visitors take small bites from the edges of the park."
        stated = "Park in numbers\n\n15 million\n\nVisitors per year"
        self.assertFalse(_field_is_plausible(prose, requirement))
        self.assertTrue(_field_is_plausible(stated, requirement))

    def test_rare_numeric_field_is_repaired_without_prevalence_support(self) -> None:
        requirement = SharedRequirement(
            key="fact:visitors", kind="fact", description="visitors_per_year",
            aliases=["visitors"], consumer_question_ids=["q1"],
        )
        chunks = []
        packs = []
        evidence = []
        for index in range(10):
            record_id = f"record_{index:04d}"
            text = (
                "Park in numbers\n\n15 million\n\nVisitors per year"
                if index == 9 else "Ordinary narrative with no measured attendance."
            )
            chunk_id = f"c{index}"
            chunks.append(DocumentChunk(
                id=chunk_id, order=index, text=text, record_ids=[record_id],
                source_start=0, source_end=len(text),
            ))
            packs.append(ScanPack(
                id=f"census_{index}", order=index, chunk_ids=[chunk_id],
                record_ids=[record_id], text=text, character_count=len(text),
            ))
            evidence.append(GlobalEvidence(
                id=f"e{index}", pass_name="facts", pack_id=f"census_{index}",
                chunk_id=chunk_id, kind="fact",
                requirement_key="fact:census_record_name",
                consumer_question_ids=["q1"], subject=f"Entry {index}",
                predicate="record name", value_raw=f"Entry {index}",
                source_quote=f"Entry {index}", confidence=1.0,
            ))
        gaps = census_field_gaps(evidence, packs, chunks, [requirement])
        self.assertEqual(list(gaps), ["record_0009"])

    def test_a_record_with_only_its_name_is_not_a_complete_census(self) -> None:
        text = "Alpha Reserve\n\nPark in numbers\n\n120\n\nArea covered (sq km)"
        chunks = [
            DocumentChunk(id="c0", order=0, text=text, record_ids=["record_0000"],
                          source_start=0, source_end=len(text))
        ]
        packs = [
            ScanPack(id="census_000", order=0, chunk_ids=["c0"],
                     record_ids=["record_0000"], text=text, character_count=len(text))
        ]
        requirements = [
            SharedRequirement(key="fact:area", kind="fact", description="area covered",
                              aliases=["area"], consumer_question_ids=["q1"])
        ]
        name_only = [
            GlobalEvidence(
                id="e1", pass_name="facts", pack_id="census_000", chunk_id="c0",
                kind="fact", requirement_key="fact:census_record_name",
                consumer_question_ids=["q1"], subject="Alpha Reserve",
                predicate="record name", value_raw="Alpha Reserve",
                source_quote="Alpha Reserve", confidence=1.0,
            )
        ]
        # Record-level coverage is satisfied — the record returned something.
        self.assertEqual(census_gaps(name_only, packs, chunks), ([], 1))
        # Field-level coverage is not: the area the record states is missing.
        gaps = census_field_gaps(name_only, packs, chunks, requirements)
        self.assertEqual(list(gaps), ["record_0000"])
        self.assertEqual([item.key for item in gaps["record_0000"]], ["fact:area"])

    def test_page_marker_digits_are_never_taken_as_a_measurement(self) -> None:
        # The marker is scaffolding this pipeline injects while reading a PDF.
        # It sits beside real prose, so a nearest-number search for "area" once
        # selected the page number and overwrote a correct extraction.
        quote = (
            "<!-- page-start-marker-261 -->\n\nEurope's largest protected area, "
            "Vatnajokull National Park is a stunning sample platter. "
            "Sprawling over 13,600 sq km, the park takes up eastern Iceland."
        )
        requirement = SharedRequirement(
            key="fact:area", kind="fact", description="area covered",
            aliases=["area"], consumer_question_ids=["q1"],
        )
        value, _, grounded = _measurement_number(quote, quote, "13,600", "area_km2", requirement)
        self.assertTrue(grounded)
        self.assertEqual(str(value).replace(",", ""), "13600")

    def test_measurement_pairing_does_not_cross_another_field_label(self) -> None:
        quote = "2170 Area covered (sq km)\n\nHighest point (m) 1085"
        requirement = SharedRequirement(
            key="fact:highest", kind="fact", description="highest point metres",
            aliases=["highest point"], consumer_question_ids=["q1"],
        )
        value, _, grounded = _measurement_number(
            quote, quote, "2170", "highest_point_metres", requirement
        )
        self.assertTrue(grounded)
        self.assertEqual(value, "1085")

    def test_masking_page_markers_preserves_offsets(self) -> None:
        text = "before <!-- page-start-marker-7 --> after"
        masked = mask_page_markers(text)
        self.assertEqual(len(masked), len(text))
        self.assertNotIn("7", masked)
        self.assertIn("before", masked)
        self.assertIn("after", masked)
        self.assertEqual(masked.index("after"), text.index("after"))

    def test_record_name_reaches_every_census_consumer(self) -> None:
        # A record's name is what counting questions group by. Borrowing another
        # requirement's key would borrow its narrower consumer list and hide the
        # name from every question except the one that owns that field.
        requirements = [
            SharedRequirement(key="fact:area", kind="fact", description="area",
                              aliases=["area covered"], consumer_question_ids=["d04"]),
            SharedRequirement(key="fact:country", kind="fact", description="country",
                              aliases=[], consumer_question_ids=["d01", "d03"]),
        ]
        text = "Alpha Reserve\n\nPark in numbers\n\n120\n\nArea covered (sq km)"
        chunk = DocumentChunk(id="c1", order=0, text=text, record_ids=["record_0000"],
                              source_start=0, source_end=len(text))
        pack = ScanPack(id="census_000", order=0, chunk_ids=["c1"],
                        record_ids=["record_0000"], text=text, character_count=len(text))
        payload = census_to_scan_payload(
            {"records": [{"record_name": "Alpha Reserve", "name_quote": "Alpha Reserve",
                          "chunk_id": "c1", "fields": []}]},
            requirements,
        )
        identity = census_identity_requirement(requirements)
        normalized = normalize_scan_delta(
            payload, pass_name="facts", pack=pack,
            requirements=[*requirements, identity], chunks_by_id={"c1": chunk},
        )
        rows = [item for item in normalized["evidence"] if item["predicate"] == "record name"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["consumer_question_ids"]), {"d01", "d03", "d04"})

    def test_census_pack_count_cannot_exceed_the_extraction_budget(self) -> None:
        entries = [
            f"Entry {index}\n\nProse {index}.\n\n## Key figures\n\n{index}\n\n"
            f"Area (sq km)\n\n## Visit this\n\nThing {index}.\n\n"
            f"## Species\n\nWildlife {index}.\n"
            for index in range(120)
        ]
        text = "\n\n".join(entries)
        document = DocumentInfo(path=Path("memory.txt"), text=text, sha256="hash")
        layout = detect_records(text)
        configured = replace(settings(), max_extraction_calls=20)
        chunks = chunk_document(document, configured, layout)
        packs = make_census_packs(chunks, layout, configured)
        self.assertGreater(layout.count, configured.max_extraction_calls)
        self.assertLessEqual(len(packs), configured.max_extraction_calls)
        self.assertEqual(
            sum(len(pack.record_ids) for pack in packs if pack.record_ids), layout.count
        )
        self.assertLessEqual(
            max(pack.character_count for pack in packs), configured.pack_max_chars
        )
        # A budget too small even for the widest grouping is a large document,
        # not a broken configuration: the census proceeds, still covering every
        # record and still inside the pack limit, and the deadline decides how
        # much of it gets read. Refusing here would return no submission at all.
        squeezed = make_census_packs(
            chunks, layout, replace(configured, max_extraction_calls=18)
        )
        self.assertTrue(squeezed)
        self.assertEqual(
            sum(len(pack.record_ids) for pack in squeezed if pack.record_ids), layout.count
        )
        self.assertLessEqual(
            max(pack.character_count for pack in squeezed), configured.pack_max_chars
        )

    def test_a_span_holding_nothing_censusable_leaves_the_denominator(self) -> None:
        chunks = [
            DocumentChunk(id=f"c{index}", order=index, text="body",
                          record_ids=[f"record_000{index}"], source_start=0, source_end=4)
            for index in range(3)
        ]
        packs = [
            ScanPack(id=f"census_00{index}", order=index, chunk_ids=[f"c{index}"],
                     record_ids=[f"record_000{index}"], text="body", character_count=4)
            for index in range(3)
        ]
        evidence = [
            GlobalEvidence(
                id="e1", pass_name="facts", pack_id="census_000", chunk_id="c0",
                kind="fact", requirement_key="fact:area", consumer_question_ids=["q1"],
                subject="Alpha", predicate="area", value_raw="1",
                source_quote="body", confidence=1.0,
            )
        ]
        # Before repair both empty records are genuine omissions.
        self.assertEqual(census_gaps(evidence, packs, chunks), (["record_0001", "record_0002"], 3))
        # After one attempt each they are spans with nothing in them, so coverage
        # can close instead of re-scanning them on every run.
        missing, total = census_gaps(
            evidence, packs, chunks, attempted=["record_0001", "record_0002"]
        )
        self.assertEqual(missing, [])
        self.assertEqual(total, 1)

    def test_record_chunks_keeps_every_chunk_of_a_split_record(self) -> None:
        chunks = [
            DocumentChunk(id="c0", order=0, text="head", record_ids=["record_0000"],
                          source_start=0, source_end=4),
            DocumentChunk(id="c1", order=1, text="tail", record_ids=["record_0000"],
                          source_start=4, source_end=8),
            DocumentChunk(id="c2", order=2, text="next", record_ids=["record_0001"],
                          source_start=8, source_end=12),
        ]
        grouped = _record_chunks(chunks)
        self.assertEqual([item.id for item in grouped["record_0000"]], ["c0", "c1"])
        self.assertEqual([item.id for item in grouped["record_0001"]], ["c2"])

    def test_answer_contract_forbids_hedging_under_partial_coverage(self) -> None:
        # Scoring subtracts 0.25 per forbidden claim, and a hedged count reads as
        # one. A partial census must still produce a committed answer.
        contract = " ".join(ANSWER_SYSTEM.split())
        self.assertIn("Commit to one value", contract)
        self.assertIn("Never invent an item to fill a gap", contract)
        # Committing to a figure must not suppress required content: a terse
        # absence answer that drops its positive controls loses half the rubric.
        self.assertIn("not the same as shortening the answer", contract)
        self.assertIn("positive control for each alternative", contract)

    def test_census_record_without_a_usable_name_is_discarded(self) -> None:
        requirement = SharedRequirement(
            key="fact:area", kind="fact", description="area",
            aliases=[], consumer_question_ids=["q1"],
        )
        payload = census_to_scan_payload(
            {"records": [{"record_name": "area_km2", "chunk_id": "c1", "fields": []}]},
            [requirement],
        )
        self.assertEqual(payload["evidence"], [])
        self.assertTrue(payload["uncertain_notes"])

    def test_truncated_census_recovers_closed_records(self) -> None:
        recovered = _recover_complete_evidence_items(
            '{"records": [{"record_name":"Alpha","fields":[]},'
            '{"record_name":"Beta","fields":[]}, {"record_name":"broken"'
        )
        self.assertEqual(len(recovered["records"]), 2)
        self.assertIn("partial recovery", recovered["uncertain_notes"][0])

    def test_census_gap_is_reported_against_the_record_denominator(self) -> None:
        chunks = [
            DocumentChunk(
                id=f"c{index}", order=index, text="body", record_ids=[f"record_000{index}"],
                source_start=0, source_end=4,
            )
            for index in range(3)
        ]
        packs = [
            ScanPack(
                id=f"census_00{index}", order=index, chunk_ids=[f"c{index}"],
                record_ids=[f"record_000{index}"], text="body", character_count=4,
            )
            for index in range(3)
        ]
        evidence = [
            GlobalEvidence(
                id="e1", pass_name="facts", pack_id="census_000", chunk_id="c0",
                kind="fact", requirement_key="fact:area", consumer_question_ids=["q1"],
                subject="Alpha", predicate="area", value_raw="1",
                source_quote="body", confidence=1.0,
            )
        ]
        missing, total = census_gaps(evidence, packs, chunks)
        self.assertEqual(total, 3)
        self.assertEqual(missing, ["record_0001", "record_0002"])
        coverage = coverage_for_question(
            plan(), {"facts": packs},
            {("facts", pack.id): "success" for pack in packs},
            evidence, (total - len(missing), total),
        )
        self.assertFalse(coverage.closure_passed)
        self.assertTrue(any("census incomplete" in item for item in coverage.failures))
        self.assertEqual(coverage.records_with_evidence, 1)
        self.assertEqual(coverage.total_records, 3)

    def test_census_questions_are_never_capped_below_the_record_count(self) -> None:
        configured = settings()
        self.assertEqual(
            evidence_limit_for(plan("aggregation"), configured, 60),
            80,
        )
        self.assertEqual(
            evidence_limit_for(plan("needle"), configured, 60),
            configured.evidence_per_question,
        )
        self.assertEqual(
            evidence_limit_for(plan("aggregation"), configured, 0),
            configured.evidence_per_question,
        )

    def test_census_selection_diversifies_by_record_not_by_chunk(self) -> None:
        evidence = [
            GlobalEvidence(
                id=f"e{index}", pass_name="facts", pack_id="census_000",
                chunk_id=f"c{index}", kind="fact", requirement_key="fact:area",
                consumer_question_ids=["q1"], subject=f"Entry {index}",
                predicate="area", value_raw=str(100 + index), value_normalized=100 + index,
                unit_raw="sq km", source_quote=f"Entry {index} covers {100 + index} sq km",
                original_order=index, confidence=1.0,
            )
            for index in range(4)
        ]
        # Chunks c0/c1 are two views of one record; c2/c3 are separate records.
        records_by_chunk = {
            "c0": ["record_0000"], "c1": ["record_0000"],
            "c2": ["record_0001"], "c3": ["record_0002"],
        }
        selected = select_evidence(
            Question(id="q1", question="How many entries qualify?"),
            plan("aggregation"), evidence, 3, {}, records_by_chunk,
        )
        chosen = {records_by_chunk[item.chunk_id][0] for item in selected}
        self.assertEqual(chosen, {"record_0000", "record_0001", "record_0002"})

    def test_truncated_scan_recovers_only_fully_closed_evidence_items(self) -> None:
        recovered = _recover_complete_evidence_items(
            '{"evidence": [{"chunk_id":"c1","kind":"fact"},'
            '{"chunk_id":"c2","kind":"fact"}, {"chunk_id":"broken"'
        )
        self.assertEqual(len(recovered["evidence"]), 2)
        self.assertIn("partial recovery", recovered["uncertain_notes"][0])

    def test_truncated_scan_is_recovered_without_remote_repair_call(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            async def call(self, **kwargs):
                self.calls += 1
                return ('{"evidence": [{"chunk_id":"c1","kind":"fact"}, {"broken"', {})

        async def exercise() -> None:
            client = FakeClient()
            result = await call_structured(
                client,
                model="ministral-3b-2512",
                fallback_model=None,
                system="scan",
                user="source",
                stage="facts",
                max_tokens=100,
                response_model=ScanDelta,
                normalize=lambda value: {
                    "pass_name": "facts",
                    "pack_id": "p1",
                    "inspected_chunk_ids": ["c1"],
                    "evidence": [],
                    "uncertain_notes": value.get("uncertain_notes", []),
                },
            )
            self.assertEqual(client.calls, 1)
            self.assertIn("partial recovery", result.uncertain_notes[0])

        asyncio.run(exercise())

    def test_non_answer_calls_cannot_consume_final_call_reserve(self) -> None:
        async def exercise() -> None:
            configured = settings().with_overrides(max_total_api_calls=5)
            budget = CallBudget(configured, expected=5, answer_reserve=2)
            for _ in range(3):
                started = await budget.begin("facts")
                await budget.finish(started, True)
            with self.assertRaisesRegex(BudgetExceeded, "final-answer API-call reserve"):
                await budget.begin("audit")
            with self.assertRaisesRegex(BudgetExceeded, "final-answer API-call reserve"):
                await budget.begin("facts_repair")
            started = await budget.begin("answer")
            await budget.finish(started, True)
            started = await budget.begin("answer_repair")
            await budget.finish(started, True)

        asyncio.run(exercise())

    def test_mandatory_scan_bypasses_answer_reserve_but_audit_does_not(self) -> None:
        async def exercise() -> None:
            configured = settings().with_overrides(max_total_api_calls=6)
            budget = CallBudget(configured, expected=6, answer_reserve=2)
            for _ in range(4):
                handle = await budget.begin("facts")
                await budget.finish(handle, True)
            handle = await budget.begin("needle")
            await budget.finish(handle, True)
            with self.assertRaisesRegex(BudgetExceeded, "final-answer API-call reserve"):
                await budget.begin("audit")

        asyncio.run(exercise())

    def test_dry_run_requires_no_key_and_reports_no_answer_shortcuts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "document.txt"
            questions = root / "questions.yaml"
            document.write_text("## Alpha\n\nAlpha Basin has an area of 120 sq km.", encoding="utf-8")
            questions.write_text(
                "- id: q1\n  category: aggregation\n  question: How many basins have area >= 100?\n",
                encoding="utf-8",
            )
            report = dry_run_report(
                settings=settings(api_key=""),
                document_path=document,
                questions_path=questions,
            )
        self.assertEqual(report.estimated_calls["deterministic_answer_shortcuts"], 0)
        self.assertGreater(report.estimated_calls["llm_planning"], 0)
        self.assertGreater(report.estimated_calls["llm_shared_extraction"], 0)
        self.assertGreater(report.estimated_calls["llm_final_answer_calls"], 0)

    def test_actual_run_rejects_missing_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "document.txt"
            questions = root / "questions.yaml"
            document.write_text("source text", encoding="utf-8")
            questions.write_text("- id: q1\n  question: What is stated?\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "COURSE_API_KEY"):
                asyncio.run(
                    run_llm_first_pipeline(
                        settings=settings(api_key=""), document_path=document,
                        questions_path=questions, output_path=root / "out.json",
                        state_db_path=root / "state.db",
                    )
                )

    def test_searchable_pdf_preserves_physical_pages(self) -> None:
        class Page:
            def __init__(self, text: str) -> None:
                self.text = text

            def extract_text(self):
                return self.text

        reader = types.SimpleNamespace(
            is_encrypted=False,
            pages=[Page("first page text " * 10), Page("second page text " * 10)],
        )
        fake_module = types.SimpleNamespace(PdfReader=lambda *args, **kwargs: reader)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pdf"
            path.write_bytes(b"%PDF-fake")
            with patch.dict("sys.modules", {"pypdf": fake_module}):
                document = normalize_document(path)
        self.assertEqual(re.findall(r"page-start-marker-(\d+)", document.text), ["1", "2"])

    def test_cold_run_calls_planning_extraction_and_final_llm(self) -> None:
        stages: list[str] = []

        class FakeClient:
            def __init__(self, configured, store, budget, *, use_cache=True) -> None:
                self.settings = configured
                self.store = store
                self.budget = budget

            async def call(self, *, stage, user, **kwargs):
                stages.append(stage)
                if stage == "health":
                    return "OK", {}
                if stage == "planning":
                    return json.dumps(
                        {
                            "plans": [{
                                "question_id": "q1", "query_type": "aggregation",
                                "target_unit": "basin", "target_entities": ["basin"],
                                "required_fields": ["area"], "candidate_topics": [],
                                "claim_slots": [], "relation_types": [],
                                "aliases": {"area": ["area", "covers"]},
                                "condition": ">= 100", "operations": ["count"],
                                "required_passes": ["facts"],
                                "closure_rules": ["all packs inspected"],
                            }]
                        }
                    ), {}
                if stage == "claims":
                    # An aggregation question enables the claims channel so the
                    # document's own stated collection size can be found; this
                    # tiny fixture states no such total.
                    return json.dumps({"evidence": [], "uncertain_notes": []}), {}
                if stage in {"facts", "audit"}:
                    chunk_id = re.search(r'<<<CHUNK ([^>]+)>>>', user).group(1)
                    requirement_key = re.search(r'"key": "([^"]+)"', user).group(1)
                    return json.dumps(
                        {
                            "evidence": [{
                                "chunk_id": chunk_id, "kind": "fact",
                                "requirement_key": requirement_key,
                                "consumer_question_ids": ["q1"],
                                "subject": "Alpha Basin", "predicate": "area",
                                "value_raw": "120", "unit_raw": "sq km",
                                "source_quote": "Alpha Basin has an area of 120 sq km.",
                                "confidence": 1.0,
                            }],
                            "uncertain_notes": [],
                        }
                    ), {}
                if stage == "answer":
                    self.assert_answer_prompt_is_grounded(user)
                    return json.dumps(
                        {
                            "answers": [{
                                "id": "q1",
                                "answer": "1 basin qualifies: Alpha Basin (120 sq km).",
                                "evidence": ["Alpha / source"],
                            }]
                        }
                    ), {}
                raise AssertionError(stage)

            @staticmethod
            def assert_answer_prompt_is_grounded(user: str) -> None:
                assert "Alpha Basin has an area of 120 sq km." in user
                assert '"qualifying_count": 1' in user

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "document.txt"
            questions = root / "questions.yaml"
            output = root / "submission.json"
            state = root / "state.db"
            document.write_text(
                "## Alpha\n\nAlpha Basin has an area of 120 sq km.", encoding="utf-8"
            )
            questions.write_text(
                "- id: q1\n  category: aggregation\n  question: How many basins have area >= 100?\n",
                encoding="utf-8",
            )
            with patch("pipeline.LLMClient", FakeClient):
                submission, _ = asyncio.run(
                    run_llm_first_pipeline(
                        settings=settings(), document_path=document,
                        questions_path=questions, output_path=output,
                        state_db_path=state,
                    )
                )
            self.assertEqual(
                submission.answers[0].answer,
                "1 basin qualifies: Alpha Basin (120 sq km).",
            )
            self.assertIn("planning", stages)
            self.assertIn("facts", stages)
            self.assertIn("answer", stages)
            with sqlite3.connect(state) as database:
                generator = database.execute("SELECT generator FROM answers").fetchone()[0]
            self.assertEqual(generator, "llm")

    def test_llm_client_rate_limit_retry(self) -> None:
        from unittest.mock import AsyncMock, MagicMock
        from pipeline import LLMClient, SQLiteStore, CallBudget

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            store = SQLiteStore(db_path)
            s = settings()
            budget = CallBudget(s, expected=10)

            client = LLMClient(s, store, budget, use_cache=False)
            mock_create = AsyncMock()
            err = Exception("Rate limit exceeded")
            err.status_code = 429

            success_resp = MagicMock()
            success_resp.choices = [MagicMock(message=MagicMock(content="OK"))]
            success_resp.usage = None

            mock_create.side_effect = [err, err, success_resp]
            client.client.chat.completions.create = mock_create

            with patch("asyncio.sleep", new=AsyncMock()):
                text, _ = asyncio.run(
                    client.call(
                        model="mistral-small-3-2",
                        system="sys",
                        user="user",
                        stage="facts",
                        max_tokens=100,
                    )
                )
            self.assertEqual(text, "OK")
            self.assertEqual(mock_create.call_count, 3)


if __name__ == "__main__":
    unittest.main()
