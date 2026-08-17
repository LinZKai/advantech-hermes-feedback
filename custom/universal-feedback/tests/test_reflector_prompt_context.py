"""Phase 5 Slice 4 tests: the Reflector Prompt Context projection in
tools.reflector_prompt_context.

No DB, no LLM, no network anywhere in this file -- build_reflector_prompt_
context() is a pure function over already-built in-memory objects
(ReflectorInput / ProposalCandidate), so every fixture here is constructed
directly via each dataclass's own constructor, matching test_reflector_
proposals.py's no-DB testing style for the equivalent Slice 2 contracts.

Deliberately a separate file from test_case_reflection_input.py /
test_proposal_matching.py, matching the one-file-per-concern convention
this test suite already uses.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import MappingProxyType

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import CaseAnalysisEvidence  # noqa: E402
from tools.case_reflection_input import CaseIntelligenceRecord, ReflectorInput  # noqa: E402
from tools.proposal_matching import ProposalCandidate  # noqa: E402
from tools.reflector_prompt_context import (  # noqa: E402
    AnalysisWindow,
    CaseIntelligenceProjection,
    ReflectorContextSummary,
    ReflectorPromptContext,
    build_reflector_prompt_context,
    serialize_reflector_prompt_context,
)


def _evidence(**overrides) -> CaseAnalysisEvidence:
    defaults = dict(type="user_text", turn_id="turn-1", fact="User asked how to disable SNMP.")
    defaults.update(overrides)
    return CaseAnalysisEvidence(**defaults)


def _case_record(**overrides) -> CaseIntelligenceRecord:
    defaults = dict(
        case_id="case-1",
        case_title="Cannot disable SNMP",
        issue_summary="User asked how to disable SNMP on ADAM-6266.",
        product_model="ADAM-6266",
        product_source="explicit_user_text",
        product_confidence=0.9,
        issue_type="product_usage_or_application",
        issue_type_confidence=0.8,
        diagnosis="knowledge_gap",
        diagnosis_confidence=0.7,
        evidence=(_evidence(),),
        analysis_version="case-enrichment-v1",
        analyzed_at="2026-01-02T00:00:00+00:00",
        source_evidence_watermark="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return CaseIntelligenceRecord(**defaults)


def _make_reflector_input(
    cases: tuple[CaseIntelligenceRecord, ...] = (),
    *,
    window_start: str | None = None,
    window_end: str | None = None,
    cases_missing_analysis: tuple[str, ...] = (),
    cases_with_unparseable_analysis: tuple[str, ...] = (),
) -> ReflectorInput:
    """Test-only ReflectorInput builder -- constructs a valid instance
    directly (no DB, no FeedbackStoreV2), computing the same aggregates
    tools.case_reflection_input.build_reflector_input would, for whatever
    `cases` this test supplies."""
    analyzed_case_count = len(cases)
    window_case_count = (
        analyzed_case_count + len(cases_missing_analysis) + len(cases_with_unparseable_analysis)
    )
    coverage_ratio = analyzed_case_count / window_case_count if window_case_count > 0 else 0.0

    def _count_by(key_fn):
        counts: dict[str, int] = {}
        for c in cases:
            key = key_fn(c)
            counts[key] = counts.get(key, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

    return ReflectorInput(
        window_start=window_start,
        window_end=window_end,
        cases=cases,
        cases_missing_analysis=cases_missing_analysis,
        cases_with_unparseable_analysis=cases_with_unparseable_analysis,
        window_case_count=window_case_count,
        analyzed_case_count=analyzed_case_count,
        coverage_ratio=coverage_ratio,
        by_product_model=_count_by(lambda c: c.product_model or "__unknown__"),
        by_issue_type=_count_by(lambda c: c.issue_type),
        by_diagnosis=_count_by(lambda c: c.diagnosis),
    )


def _candidate(**overrides) -> ProposalCandidate:
    defaults = dict(
        proposal_id="proposal-1",
        improvement_target="knowledge",
        title="SNMP disable command undocumented",
        review_status="pending",
        latest_pattern_summary="Three Cases ask about SNMP disable.",
        latest_recommended_improvement="Add an SNMP disable procedure to the KB.",
        latest_trend="new",
        latest_supporting_case_count=3,
        latest_confidence=0.7,
    )
    defaults.update(overrides)
    return ProposalCandidate(**defaults)


# ---------------------------------------------------------------------------
# A. Context Projection
# ---------------------------------------------------------------------------


class BuildReflectorPromptContextTests(unittest.TestCase):
    def test_empty_cases_and_proposals(self):
        context = build_reflector_prompt_context(_make_reflector_input(), ())
        self.assertEqual(context.cases, ())
        self.assertEqual(context.existing_proposals, ())
        self.assertEqual(context.summary.analyzed_case_count, 0)

    def test_multiple_cases_correctly_projected(self):
        cases = (_case_record(case_id="case-1"), _case_record(case_id="case-2"))
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        self.assertEqual([c.case_id for c in context.cases], ["case-1", "case-2"])

    def test_multiple_proposal_candidates_correctly_projected(self):
        candidates = (_candidate(proposal_id="p-1"), _candidate(proposal_id="p-2"))
        context = build_reflector_prompt_context(_make_reflector_input(), candidates)
        self.assertEqual([p.proposal_id for p in context.existing_proposals], ["p-1", "p-2"])

    def test_analysis_window_preserved(self):
        reflector_input = _make_reflector_input(
            window_start="2026-01-01T00:00:00+00:00", window_end="2026-02-01T00:00:00+00:00",
        )
        context = build_reflector_prompt_context(reflector_input, ())
        self.assertEqual(context.analysis_window, AnalysisWindow(
            start="2026-01-01T00:00:00+00:00", end="2026-02-01T00:00:00+00:00",
        ))

    def test_analysis_window_none_bounds_preserved(self):
        context = build_reflector_prompt_context(_make_reflector_input(), ())
        self.assertIsNone(context.analysis_window.start)
        self.assertIsNone(context.analysis_window.end)

    def test_analyzed_case_count_correct(self):
        cases = (_case_record(case_id="case-1"), _case_record(case_id="case-2"), _case_record(case_id="case-3"))
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        self.assertEqual(context.summary.analyzed_case_count, 3)

    def test_by_product_model_preserved(self):
        cases = (
            _case_record(case_id="case-1", product_model="ADAM-6266"),
            _case_record(case_id="case-2", product_model="ADAM-6266"),
            _case_record(case_id="case-3", product_model=None, product_source=None, product_confidence=None),
        )
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        self.assertEqual(dict(context.summary.by_product_model), {"ADAM-6266": 2, "__unknown__": 1})

    def test_by_issue_type_preserved(self):
        cases = (
            _case_record(case_id="case-1", issue_type="product_issue"),
            _case_record(case_id="case-2", issue_type="product_issue"),
        )
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        self.assertEqual(dict(context.summary.by_issue_type), {"product_issue": 2})

    def test_by_diagnosis_preserved(self):
        cases = (
            _case_record(case_id="case-1", diagnosis="retrieval_issue"),
            _case_record(case_id="case-2", diagnosis="no_issue_detected"),
        )
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        self.assertEqual(
            dict(context.summary.by_diagnosis), {"retrieval_issue": 1, "no_issue_detected": 1},
        )

    def test_mapping_proxy_type_safely_projected_to_plain_dict(self):
        cases = (_case_record(case_id="case-1"),)
        reflector_input = _make_reflector_input(cases)
        self.assertIsInstance(reflector_input.by_product_model, MappingProxyType)
        context = build_reflector_prompt_context(reflector_input, ())
        self.assertNotIsInstance(context.summary.by_product_model, MappingProxyType)
        self.assertIsInstance(context.summary.by_product_model, dict)

    def test_internal_case_audit_metadata_excluded(self):
        field_names = {f.name for f in CaseIntelligenceProjection.__dataclass_fields__.values()}
        self.assertNotIn("analysis_version", field_names)
        self.assertNotIn("analyzed_at", field_names)
        self.assertNotIn("source_evidence_watermark", field_names)

    def test_missing_and_unparseable_ids_excluded(self):
        field_names = {f.name for f in ReflectorPromptContext.__dataclass_fields__.values()}
        self.assertNotIn("cases_missing_analysis", field_names)
        self.assertNotIn("cases_with_unparseable_analysis", field_names)
        summary_field_names = {f.name for f in ReflectorContextSummary.__dataclass_fields__.values()}
        self.assertNotIn("cases_missing_analysis", summary_field_names)
        self.assertNotIn("cases_with_unparseable_analysis", summary_field_names)
        self.assertNotIn("window_case_count", summary_field_names)

    def test_coverage_ratio_excluded(self):
        summary_field_names = {f.name for f in ReflectorContextSummary.__dataclass_fields__.values()}
        self.assertNotIn("coverage_ratio", summary_field_names)
        context_field_names = {f.name for f in ReflectorPromptContext.__dataclass_fields__.values()}
        self.assertNotIn("coverage_ratio", context_field_names)

    def test_required_case_semantic_fields_retained(self):
        cases = (_case_record(),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        projected = context.cases[0]
        self.assertEqual(projected.case_title, "Cannot disable SNMP")
        self.assertEqual(projected.issue_summary, "User asked how to disable SNMP on ADAM-6266.")
        self.assertEqual(projected.product_model, "ADAM-6266")
        self.assertEqual(projected.product_source, "explicit_user_text")
        self.assertEqual(projected.product_confidence, 0.9)
        self.assertEqual(projected.issue_type, "product_usage_or_application")
        self.assertEqual(projected.issue_type_confidence, 0.8)
        self.assertEqual(projected.diagnosis, "knowledge_gap")
        self.assertEqual(projected.diagnosis_confidence, 0.7)

    def test_evidence_correctly_projected(self):
        cases = (_case_record(evidence=(
            _evidence(type="user_text", turn_id="turn-1", fact="fact one"),
            _evidence(type="retrieval", turn_id="turn-1", fact="fact two"),
        )),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        evidence = context.cases[0].evidence
        self.assertEqual(len(evidence), 2)
        self.assertIsInstance(evidence[0], CaseAnalysisEvidence)
        self.assertEqual(evidence[0].fact, "fact one")
        self.assertEqual(evidence[1].fact, "fact two")

    def test_proposal_matching_semantic_fields_retained(self):
        candidate = _candidate(
            improvement_target="agent_behavior", title="Firmware check missing",
            latest_trend="growing", latest_supporting_case_count=5, latest_confidence=0.42,
        )
        context = build_reflector_prompt_context(_make_reflector_input(), (candidate,))
        projected = context.existing_proposals[0]
        self.assertEqual(projected.improvement_target, "agent_behavior")
        self.assertEqual(projected.title, "Firmware check missing")
        self.assertEqual(projected.latest_trend, "growing")
        self.assertEqual(projected.latest_supporting_case_count, 5)
        self.assertEqual(projected.latest_confidence, 0.42)

    def test_proposal_history_and_created_at_excluded(self):
        field_names = {f.name for f in ProposalCandidate.__dataclass_fields__.values()}
        self.assertNotIn("created_at", field_names)
        self.assertNotIn("reflection_run_id", field_names)
        self.assertNotIn("supporting_case_ids", field_names)
        self.assertNotIn("observations", field_names)

    def test_deterministic_case_ordering(self):
        cases = (_case_record(case_id="case-z"), _case_record(case_id="case-a"), _case_record(case_id="case-m"))
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        # Inherits ReflectorInput.cases's own order unchanged (case-z first
        # here, since the fixture builder does not re-sort) -- confirms
        # this module does not silently re-sort a field its own docstring
        # says it will not touch.
        self.assertEqual([c.case_id for c in context.cases], ["case-z", "case-a", "case-m"])

    def test_deterministic_proposal_ordering_regardless_of_caller_collection_type(self):
        candidates_list = [_candidate(proposal_id="p-z"), _candidate(proposal_id="p-a"), _candidate(proposal_id="p-m")]
        context_from_list = build_reflector_prompt_context(_make_reflector_input(), candidates_list)
        context_from_reversed = build_reflector_prompt_context(_make_reflector_input(), list(reversed(candidates_list)))
        context_from_set = build_reflector_prompt_context(_make_reflector_input(), set(candidates_list))

        expected = ["p-a", "p-m", "p-z"]
        self.assertEqual([p.proposal_id for p in context_from_list.existing_proposals], expected)
        self.assertEqual([p.proposal_id for p in context_from_reversed.existing_proposals], expected)
        self.assertEqual([p.proposal_id for p in context_from_set.existing_proposals], expected)

    def test_deterministic_evidence_ordering(self):
        evidence = (
            _evidence(type="user_text", turn_id="turn-1", fact="first"),
            _evidence(type="assistant_text", turn_id="turn-1", fact="second"),
            _evidence(type="feedback", turn_id="turn-1", fact="third"),
        )
        cases = (_case_record(evidence=evidence),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        self.assertEqual([e.fact for e in context.cases[0].evidence], ["first", "second", "third"])

    def test_duplicate_case_ids_rejected(self):
        # A duplicate can only reach ReflectorPromptContext via a
        # hand-constructed CaseIntelligenceProjection tuple -- confirms
        # the collection-level invariant, not build_reflector_prompt_
        # context() itself (which cannot produce this from a real
        # ReflectorInput, since that type already forbids it).
        projection = CaseIntelligenceProjection(
            case_id="case-1", case_title=None, issue_summary=None,
            product_model=None, product_source=None, product_confidence=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5, evidence=(),
        )
        with self.assertRaises(ValueError):
            ReflectorPromptContext(
                analysis_window=AnalysisWindow(start=None, end=None),
                summary=ReflectorContextSummary(
                    analyzed_case_count=2, by_product_model={"__unknown__": 2},
                    by_issue_type={"product_usage_or_application": 2},
                    by_diagnosis={"knowledge_gap": 2},
                ),
                cases=(projection, projection),
                existing_proposals=(),
            )

    def test_duplicate_proposal_ids_rejected(self):
        candidate = _candidate(proposal_id="p-1")
        with self.assertRaises(ValueError):
            ReflectorPromptContext(
                analysis_window=AnalysisWindow(start=None, end=None),
                summary=ReflectorContextSummary(
                    analyzed_case_count=0, by_product_model={}, by_issue_type={}, by_diagnosis={},
                ),
                cases=(),
                existing_proposals=(candidate, candidate),
            )

    def test_frozen(self):
        context = build_reflector_prompt_context(_make_reflector_input(), ())
        with self.assertRaises(Exception):
            context.cases = ()  # type: ignore[misc]

    def test_analyzed_case_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ReflectorPromptContext(
                analysis_window=AnalysisWindow(start=None, end=None),
                summary=ReflectorContextSummary(
                    analyzed_case_count=5, by_product_model={"__unknown__": 5},
                    by_issue_type={"product_usage_or_application": 5}, by_diagnosis={"knowledge_gap": 5},
                ),
                cases=(),  # zero cases, but summary claims 5
                existing_proposals=(),
            )


# ---------------------------------------------------------------------------
# B. Serialization
# ---------------------------------------------------------------------------


class SerializeReflectorPromptContextTests(unittest.TestCase):
    def test_valid_json(self):
        cases = (_case_record(),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), (_candidate(),))
        parsed = json.loads(serialize_reflector_prompt_context(context))
        self.assertIn("analysis_window", parsed)
        self.assertIn("summary", parsed)
        self.assertIn("cases", parsed)
        self.assertIn("existing_proposals", parsed)

    def test_deterministic_output_same_context_twice(self):
        cases = (_case_record(),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), (_candidate(),))
        first = serialize_reflector_prompt_context(context)
        second = serialize_reflector_prompt_context(context)
        self.assertEqual(first, second)

    def test_semantically_identical_but_differently_constructed_contexts_match(self):
        cases = (_case_record(case_id="case-1"), _case_record(case_id="case-2"))
        candidates_a = [_candidate(proposal_id="p-a"), _candidate(proposal_id="p-b")]
        candidates_b = list(reversed(candidates_a))

        context_a = build_reflector_prompt_context(_make_reflector_input(cases), candidates_a)
        context_b = build_reflector_prompt_context(_make_reflector_input(cases), candidates_b)

        self.assertEqual(
            serialize_reflector_prompt_context(context_a),
            serialize_reflector_prompt_context(context_b),
        )

    def test_unicode_preserved(self):
        cases = (_case_record(case_title="無法停用 SNMP", issue_summary="用戶詢問如何停用 SNMP"),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        serialized = serialize_reflector_prompt_context(context)
        self.assertIn("無法停用 SNMP", serialized)
        self.assertNotIn("\\u", serialized)

    def test_mapping_proxy_type_serializes_through_projection(self):
        cases = (_case_record(product_model="ADAM-6266"),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        parsed = json.loads(serialize_reflector_prompt_context(context))
        self.assertEqual(parsed["summary"]["by_product_model"], {"ADAM-6266": 1})

    def test_tuples_become_json_arrays(self):
        cases = (_case_record(case_id="case-1"), _case_record(case_id="case-2"))
        context = build_reflector_prompt_context(_make_reflector_input(cases), (_candidate(),))
        parsed = json.loads(serialize_reflector_prompt_context(context))
        self.assertIsInstance(parsed["cases"], list)
        self.assertIsInstance(parsed["existing_proposals"], list)
        self.assertIsInstance(parsed["cases"][0]["evidence"], list)

    def test_none_fields_behavior_deterministic(self):
        cases = (_case_record(product_model=None, product_source=None, product_confidence=None),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        parsed = json.loads(serialize_reflector_prompt_context(context))
        self.assertIsNone(parsed["cases"][0]["product_model"])
        self.assertIsNone(parsed["cases"][0]["product_source"])
        self.assertIsNone(parsed["cases"][0]["product_confidence"])

    def test_sort_keys_behavior_stable(self):
        cases = (_case_record(),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), ())
        serialized = serialize_reflector_prompt_context(context)
        # analysis_window sorts before cases, which sorts before
        # existing_proposals, which sorts before summary -- confirms
        # sort_keys=True is actually taking effect at the top level.
        self.assertLess(serialized.index('"analysis_window"'), serialized.index('"cases"'))
        self.assertLess(serialized.index('"cases"'), serialized.index('"existing_proposals"'))
        self.assertLess(serialized.index('"existing_proposals"'), serialized.index('"summary"'))

    def test_no_sqlite_specific_object_leaks_into_json(self):
        cases = (_case_record(),)
        context = build_reflector_prompt_context(_make_reflector_input(cases), (_candidate(),))
        # json.dumps() itself would raise TypeError on any non-JSON-ready
        # object (e.g. a raw sqlite3.Row) -- a successful call already
        # proves nothing sqlite-specific survived projection.
        serialize_reflector_prompt_context(context)


if __name__ == "__main__":
    unittest.main()
