"""Phase 4.5 Stage B tests: the Case enrichment input/output contract in
tools.case_enrichment.

Covers two independent surfaces:
  1. build_case_enrichment_input() -- real execution against a temporary
     FeedbackStoreV2 (matching test_case_analysis.py's fixture style).
  2. parse_case_enrichment_result() / CaseEnrichmentResult /
     CaseAnalysisEvidence -- pure structural validation, no DB involved.

No LLM, no network, no mocking of either, and no call to
FeedbackStoreV2.create_case_analysis() anywhere in this file -- Stage B
never persists an analysis result.
"""
from __future__ import annotations

import copy
import gc
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import (  # noqa: E402
    DIAGNOSIS_VALUES,
    ISSUE_TYPE_VALUES,
    CaseAnalysisEvidence,
    CaseEnrichmentInput,
    CaseEnrichmentResult,
    CaseFeedbackEvidence,
    CaseRetrievalEvidence,
    CaseTurnEvidence,
    build_case_enrichment_input,
    parse_case_enrichment_result,
)
from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Input assembly -- real store execution
# ---------------------------------------------------------------------------


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_session_and_case(self, session_id="sess-1", case_id="case-1"):
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        return session_id, case_id

    def _add_turn(self, case_id, turn_id, *, question="q", answer="a"):
        self.store.create_turn(
            turn_id, case_id,
            platform_user_id="user-1",
            platform_user_message_id=turn_id,
            question_text=question, answer_text=answer,
            feedback_eligible=True,
        )
        return turn_id

    def _add_retrieval_run(self, turn_id, *, result_count=3):
        ok = self.store.add_retrieval_runs(turn_id, [
            RetrievalRunInput(
                execution_status="completed", observation_status="complete",
                result_count=result_count, foundry_iq_ok=True,
            ),
        ])
        assert ok

    def _add_feedback(self, turn_id, *, helpful=True):
        feedback_id = uuid.uuid4().hex
        ok = self.store.submit_feedback(
            feedback_id, turn_id, helpful, feedback_policy_version="v1",
            reason_code=None if helpful else "incomplete",
        )
        assert ok
        return feedback_id


class InputAssemblyTests(_StoreTestCase):
    def test_valid_single_turn_case(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", question="ADAM-6266 怎麼關閉 SNMP？")

        result = build_case_enrichment_input(self.store, "case-1")
        self.assertIsInstance(result, CaseEnrichmentInput)
        self.assertEqual(result.case_id, "case-1")
        self.assertEqual(result.session_id, "sess-1")
        self.assertEqual(len(result.turns), 1)
        self.assertEqual(result.turns[0].turn_id, "turn-1")
        self.assertEqual(result.turns[0].question_text, "ADAM-6266 怎麼關閉 SNMP？")
        self.assertEqual(result.retrievals, ())
        self.assertEqual(result.feedback, ())

    def test_multi_turn_same_case(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_turn("case-1", "turn-2")
        self._add_turn("case-1", "turn-3")

        result = build_case_enrichment_input(self.store, "case-1")
        self.assertEqual([t.turn_id for t in result.turns], ["turn-1", "turn-2", "turn-3"])

    def test_cross_case_isolation(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        self._add_retrieval_run("turn-b")
        self._add_feedback("turn-b", helpful=False)

        result = build_case_enrichment_input(self.store, "case-a")
        self.assertEqual([t.turn_id for t in result.turns], ["turn-a"])
        self.assertEqual(result.retrievals, ())
        self.assertEqual(result.feedback, ())

    def test_retrievals_correctly_attached(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_turn("case-1", "turn-2")
        self._add_retrieval_run("turn-1", result_count=18)
        self._add_retrieval_run("turn-2", result_count=0)

        result = build_case_enrichment_input(self.store, "case-1")
        self.assertEqual(len(result.retrievals), 2)
        self.assertEqual([(r.turn_id, r.result_count) for r in result.retrievals],
                          [("turn-1", 18), ("turn-2", 0)])
        self.assertIsInstance(result.retrievals[0].foundry_iq_ok, bool)

    def test_feedback_correctly_attached(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_feedback("turn-1", helpful=False)

        result = build_case_enrichment_input(self.store, "case-1")
        self.assertEqual(len(result.feedback), 1)
        self.assertEqual(result.feedback[0].turn_id, "turn-1")
        self.assertIs(result.feedback[0].helpful, False)
        self.assertEqual(result.feedback[0].reason_code, "incomplete")

    def test_deterministic_ordering_across_multiple_turns_and_retrievals(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_turn("case-1", "turn-2")
        self._add_retrieval_run("turn-2", result_count=1)
        self._add_retrieval_run("turn-1", result_count=2)
        self._add_retrieval_run("turn-1", result_count=3)

        result = build_case_enrichment_input(self.store, "case-1")
        # turn-1's own two retrievals (in invocation order) come first
        # because turn-1 is iterated before turn-2 (created_at order),
        # regardless of the order add_retrieval_run() was actually called.
        self.assertEqual(
            [(r.turn_id, r.result_count) for r in result.retrievals],
            [("turn-1", 2), ("turn-1", 3), ("turn-2", 1)],
        )

    def test_unknown_case_returns_none(self):
        self.assertIsNone(build_case_enrichment_input(self.store, "does-not-exist"))

    def test_zero_turn_case_returns_none(self):
        self._seed_session_and_case()
        self.assertIsNone(build_case_enrichment_input(self.store, "case-1"))

    def test_evidence_watermark_included_and_matches_store(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        result = build_case_enrichment_input(self.store, "case-1")
        self.assertEqual(result.evidence_watermark, self.store.get_case_evidence_watermark("case-1"))
        self.assertIsNotNone(result.evidence_watermark)

    def test_turn_with_no_retrieval_or_feedback_yields_empty_tuples(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        result = build_case_enrichment_input(self.store, "case-1")
        self.assertEqual(result.retrievals, ())
        self.assertEqual(result.feedback, ())

    def test_case_enrichment_input_structurally_requires_at_least_one_turn(self):
        with self.assertRaises(ValueError):
            CaseEnrichmentInput(
                case_id="c1", session_id="s1",
                turns=(), retrievals=(), feedback=(),
                evidence_watermark="2026-01-01T00:00:00+00:00",
            )


# ---------------------------------------------------------------------------
# 2. Output contract -- pure structural validation, no DB
# ---------------------------------------------------------------------------


def _valid_evidence_dict(**overrides):
    base = {"type": "user_text", "turn_id": "turn-1", "fact": 'User explicitly mentioned "ADAM-6266"'}
    base.update(overrides)
    return base


def _valid_result_dict(**overrides):
    base = {
        "case_title": "ADAM-6266 SNMP configuration issue",
        "issue_summary": "User could not find how to disable SNMP on ADAM-6266.",
        "issue_type": "product_usage_or_application",
        "issue_type_confidence": 0.8,
        "diagnosis": "knowledge_gap",
        "diagnosis_confidence": 0.7,
        "product_model": "ADAM-6266",
        "product_source": "explicit_user_text",
        "product_confidence": 0.95,
        "evidence": [
            _valid_evidence_dict(),
            {"type": "retrieval", "turn_id": "turn-1", "fact": "Foundry IQ retrieval succeeded, result_count=18"},
        ],
    }
    base.update(overrides)
    return base


class OutputValidationValidTests(unittest.TestCase):
    def test_full_valid_result(self):
        result = parse_case_enrichment_result(_valid_result_dict())
        self.assertIsInstance(result, CaseEnrichmentResult)
        self.assertEqual(result.diagnosis, "knowledge_gap")
        self.assertEqual(len(result.evidence), 2)
        self.assertIsInstance(result.evidence[0], CaseAnalysisEvidence)

    def test_product_present_explicit_user_text(self):
        result = parse_case_enrichment_result(_valid_result_dict(product_source="explicit_user_text"))
        self.assertEqual(result.product_source, "explicit_user_text")

    def test_product_present_inference(self):
        result = parse_case_enrichment_result(_valid_result_dict(product_source="inference"))
        self.assertEqual(result.product_source, "inference")

    def test_product_absent_all_product_metadata_none(self):
        result = parse_case_enrichment_result(_valid_result_dict(
            product_model=None, product_source=None, product_confidence=None,
        ))
        self.assertIsNotNone(result)
        self.assertIsNone(result.product_model)
        self.assertIsNone(result.product_source)
        self.assertIsNone(result.product_confidence)

    def test_all_five_diagnosis_values(self):
        # Stage B Contract Refinement: 5 values now (dropped
        # 'workflow_tool_issue', renamed 'unclear_or_other' to
        # 'other_or_unclear') -- see tools.case_enrichment.DIAGNOSIS_VALUES
        # and this project's Stage B Contract Refinement report for why
        # this deliberately diverges from tools.feedback_store_v2.
        # DIAGNOSIS_VALUES (Stage A, still 6 values) rather than importing it.
        self.assertEqual(len(DIAGNOSIS_VALUES), 5)
        self.assertNotIn("workflow_tool_issue", DIAGNOSIS_VALUES)
        self.assertNotIn("unclear_or_other", DIAGNOSIS_VALUES)
        self.assertIn("other_or_unclear", DIAGNOSIS_VALUES)
        for diagnosis in DIAGNOSIS_VALUES:
            with self.subTest(diagnosis=diagnosis):
                result = parse_case_enrichment_result(_valid_result_dict(diagnosis=diagnosis))
                self.assertIsNotNone(result)
                self.assertEqual(result.diagnosis, diagnosis)

    def test_all_four_issue_type_values(self):
        # Stage B Contract Refinement (second pass): collapsed from 5 to 4
        # values -- configuration_or_operation + application_or_integration
        # merged into product_usage_or_application;
        # specification_or_compatibility renamed to
        # product_capability_or_compatibility; product_issue and
        # other_or_unclear unchanged.
        self.assertEqual(len(ISSUE_TYPE_VALUES), 4)
        self.assertIn("product_usage_or_application", ISSUE_TYPE_VALUES)
        self.assertIn("product_capability_or_compatibility", ISSUE_TYPE_VALUES)
        self.assertIn("product_issue", ISSUE_TYPE_VALUES)
        self.assertIn("other_or_unclear", ISSUE_TYPE_VALUES)
        for issue_type in ISSUE_TYPE_VALUES:
            with self.subTest(issue_type=issue_type):
                result = parse_case_enrichment_result(_valid_result_dict(issue_type=issue_type))
                self.assertIsNotNone(result)
                self.assertEqual(result.issue_type, issue_type)

    def test_diagnosis_and_issue_type_are_independent_no_hardcoded_mapping(self):
        # Any issue_type may legitimately pair with any diagnosis -- e.g.
        # product_usage_or_application with each of knowledge_gap,
        # answer_quality_issue, and no_issue_detected, all accepted.
        for diagnosis in ("knowledge_gap", "answer_quality_issue", "no_issue_detected"):
            with self.subTest(diagnosis=diagnosis):
                result = parse_case_enrichment_result(_valid_result_dict(
                    issue_type="product_usage_or_application", diagnosis=diagnosis,
                ))
                self.assertIsNotNone(result)
                self.assertEqual(result.issue_type, "product_usage_or_application")
                self.assertEqual(result.diagnosis, diagnosis)

    def test_issue_type_confidence_boundary_zero(self):
        result = parse_case_enrichment_result(_valid_result_dict(issue_type_confidence=0.0))
        self.assertEqual(result.issue_type_confidence, 0.0)

    def test_issue_type_confidence_boundary_one(self):
        result = parse_case_enrichment_result(_valid_result_dict(issue_type_confidence=1.0))
        self.assertEqual(result.issue_type_confidence, 1.0)

    def test_confidence_boundary_zero(self):
        result = parse_case_enrichment_result(_valid_result_dict(diagnosis_confidence=0.0))
        self.assertEqual(result.diagnosis_confidence, 0.0)

    def test_confidence_boundary_one(self):
        result = parse_case_enrichment_result(_valid_result_dict(diagnosis_confidence=1.0, product_confidence=1.0))
        self.assertEqual(result.diagnosis_confidence, 1.0)
        self.assertEqual(result.product_confidence, 1.0)

    def test_confidence_bare_integer_accepted(self):
        # Matches tools.case_routing's own confidence handling: JSON does
        # not distinguish 1 from 1.0.
        result = parse_case_enrichment_result(_valid_result_dict(
            diagnosis_confidence=1, product_confidence=1, issue_type_confidence=1,
        ))
        self.assertIsNotNone(result)

    def test_multiple_evidence_rows(self):
        result = parse_case_enrichment_result(_valid_result_dict(evidence=[
            _valid_evidence_dict(type="user_text", fact="fact 1"),
            _valid_evidence_dict(type="retrieval", fact="fact 2"),
            _valid_evidence_dict(type="feedback", fact="fact 3"),
        ]))
        self.assertEqual(len(result.evidence), 3)

    def test_evidence_type_assistant_text_accepted(self):
        # Stage B Contract Refinement: added specifically so an
        # answer_quality_issue diagnosis can cite Hermes' own prior answer.
        result = parse_case_enrichment_result(_valid_result_dict(evidence=[
            _valid_evidence_dict(type="assistant_text", fact="Assistant's answer omitted the disable-SNMP command"),
        ]))
        self.assertIsNotNone(result)
        self.assertEqual(result.evidence[0].type, "assistant_text")


class OutputValidationInvalidTests(unittest.TestCase):
    def _assert_rejected(self, payload):
        self.assertIsNone(parse_case_enrichment_result(payload))

    def test_unknown_diagnosis(self):
        self._assert_rejected(_valid_result_dict(diagnosis="not_a_real_diagnosis"))

    def test_diagnosis_dropped_value_workflow_tool_issue_rejected(self):
        # Stage B Contract Refinement dropped this Stage A value entirely.
        self._assert_rejected(_valid_result_dict(diagnosis="workflow_tool_issue"))

    def test_diagnosis_renamed_value_unclear_or_other_rejected(self):
        # Stage B Contract Refinement renamed this Stage A value to
        # 'other_or_unclear' -- the old spelling must no longer validate.
        self._assert_rejected(_valid_result_dict(diagnosis="unclear_or_other"))

    def test_unknown_product_source(self):
        self._assert_rejected(_valid_result_dict(product_source="guessed"))

    def test_confidence_below_zero(self):
        self._assert_rejected(_valid_result_dict(diagnosis_confidence=-0.1))

    def test_confidence_above_one(self):
        self._assert_rejected(_valid_result_dict(diagnosis_confidence=1.1))

    def test_confidence_bool_rejected(self):
        self._assert_rejected(_valid_result_dict(diagnosis_confidence=True))

    def test_confidence_nan_rejected(self):
        self._assert_rejected(_valid_result_dict(diagnosis_confidence=float("nan")))

    def test_unknown_issue_type(self):
        self._assert_rejected(_valid_result_dict(issue_type="not_a_real_issue_type"))

    def test_issue_type_dropped_value_configuration_or_operation_rejected(self):
        # Stage B Contract Refinement (second pass): merged into
        # product_usage_or_application -- old spelling must no longer validate.
        self._assert_rejected(_valid_result_dict(issue_type="configuration_or_operation"))

    def test_issue_type_dropped_value_application_or_integration_rejected(self):
        self._assert_rejected(_valid_result_dict(issue_type="application_or_integration"))

    def test_issue_type_dropped_value_specification_or_compatibility_rejected(self):
        # Renamed to product_capability_or_compatibility.
        self._assert_rejected(_valid_result_dict(issue_type="specification_or_compatibility"))

    def test_issue_type_none_rejected(self):
        self._assert_rejected(_valid_result_dict(issue_type=None))

    def test_issue_type_confidence_below_zero(self):
        self._assert_rejected(_valid_result_dict(issue_type_confidence=-0.1))

    def test_issue_type_confidence_above_one(self):
        self._assert_rejected(_valid_result_dict(issue_type_confidence=1.1))

    def test_issue_type_confidence_bool_rejected(self):
        self._assert_rejected(_valid_result_dict(issue_type_confidence=True))

    def test_issue_type_confidence_nan_rejected(self):
        self._assert_rejected(_valid_result_dict(issue_type_confidence=float("nan")))

    def test_issue_type_confidence_infinity_rejected(self):
        self._assert_rejected(_valid_result_dict(issue_type_confidence=float("inf")))

    def test_missing_issue_type_rejected(self):
        payload = _valid_result_dict()
        del payload["issue_type"]
        self._assert_rejected(payload)

    def test_missing_issue_type_confidence_rejected(self):
        payload = _valid_result_dict()
        del payload["issue_type_confidence"]
        self._assert_rejected(payload)

    def test_product_model_present_missing_product_source(self):
        self._assert_rejected(_valid_result_dict(product_source=None))

    def test_product_model_present_missing_product_confidence(self):
        self._assert_rejected(_valid_result_dict(product_confidence=None))

    def test_no_product_model_but_product_source_present(self):
        self._assert_rejected(_valid_result_dict(product_model=None, product_confidence=None))

    def test_no_product_model_but_product_confidence_present(self):
        self._assert_rejected(_valid_result_dict(product_model=None, product_source=None))

    def test_empty_case_title(self):
        self._assert_rejected(_valid_result_dict(case_title=""))

    def test_whitespace_only_case_title(self):
        self._assert_rejected(_valid_result_dict(case_title="   "))

    def test_empty_issue_summary(self):
        self._assert_rejected(_valid_result_dict(issue_summary=""))

    def test_empty_evidence_fact(self):
        self._assert_rejected(_valid_result_dict(evidence=[_valid_evidence_dict(fact="")]))

    def test_whitespace_only_evidence_fact(self):
        self._assert_rejected(_valid_result_dict(evidence=[_valid_evidence_dict(fact="   ")]))

    def test_unknown_evidence_type(self):
        self._assert_rejected(_valid_result_dict(evidence=[_valid_evidence_dict(type="bogus_type")]))

    def test_evidence_turn_id_none_rejected(self):
        # Stage B Contract Refinement: turn_id is now required -- all four
        # EVIDENCE_TYPES are Turn-level evidence, so None no longer means
        # anything valid.
        self._assert_rejected(_valid_result_dict(evidence=[_valid_evidence_dict(turn_id=None)]))

    def test_evidence_turn_id_empty_rejected(self):
        self._assert_rejected(_valid_result_dict(evidence=[_valid_evidence_dict(turn_id="")]))

    def test_evidence_turn_id_whitespace_rejected(self):
        self._assert_rejected(_valid_result_dict(evidence=[_valid_evidence_dict(turn_id="   ")]))

    def test_zero_evidence_rows_rejected(self):
        self._assert_rejected(_valid_result_dict(evidence=[]))

    def test_missing_required_top_level_field(self):
        payload = _valid_result_dict()
        del payload["diagnosis"]
        self._assert_rejected(payload)

    def test_missing_required_evidence_field(self):
        payload = _valid_result_dict(evidence=[{"type": "user_text", "fact": "x"}])  # missing turn_id key
        self._assert_rejected(payload)

    def test_unknown_top_level_field_rejected(self):
        payload = _valid_result_dict()
        payload["unexpected_secret_field"] = "smuggled"
        self._assert_rejected(payload)

    def test_unknown_evidence_field_rejected(self):
        self._assert_rejected(_valid_result_dict(evidence=[
            {**_valid_evidence_dict(), "extra": "smuggled"},
        ]))

    def test_wrong_root_type_list(self):
        self._assert_rejected([_valid_result_dict()])

    def test_wrong_root_type_string(self):
        self._assert_rejected("not a mapping")

    def test_wrong_root_type_none(self):
        self._assert_rejected(None)

    def test_evidence_not_a_list(self):
        self._assert_rejected(_valid_result_dict(evidence="not a list"))

    def test_evidence_item_not_a_mapping(self):
        self._assert_rejected(_valid_result_dict(evidence=["not a mapping"]))

    def test_diagnosis_wrong_type(self):
        self._assert_rejected(_valid_result_dict(diagnosis=123))

    def test_deep_copy_of_valid_payload_is_still_valid(self):
        # Sanity check that the builder itself produces a genuinely valid
        # payload (guards against every _assert_rejected test above being
        # trivially true because the base payload was already broken).
        self.assertIsNotNone(parse_case_enrichment_result(copy.deepcopy(_valid_result_dict())))


class CaseAnalysisEvidenceDirectConstructionTests(unittest.TestCase):
    """__post_init__ structural checks, exercised directly (not just via
    parse_case_enrichment_result) -- matching CaseRoutingResult's own
    "structural block" test convention."""

    def test_valid_construction(self):
        CaseAnalysisEvidence(type="user_text", turn_id="turn-1", fact="fact")  # must not raise

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            CaseAnalysisEvidence(type="bogus", turn_id="turn-1", fact="fact")

    def test_blank_fact_raises(self):
        with self.assertRaises(ValueError):
            CaseAnalysisEvidence(type="user_text", turn_id="turn-1", fact="")

    def test_whitespace_turn_id_raises(self):
        with self.assertRaises(ValueError):
            CaseAnalysisEvidence(type="user_text", turn_id="  ", fact="fact")

    def test_empty_turn_id_raises(self):
        with self.assertRaises(ValueError):
            CaseAnalysisEvidence(type="user_text", turn_id="", fact="fact")

    def test_none_turn_id_raises(self):
        # Stage B Contract Refinement: turn_id is now required.
        with self.assertRaises(ValueError):
            CaseAnalysisEvidence(type="user_text", turn_id=None, fact="fact")

    def test_assistant_text_type_valid_construction(self):
        CaseAnalysisEvidence(type="assistant_text", turn_id="turn-1", fact="fact")  # must not raise


if __name__ == "__main__":
    unittest.main()
