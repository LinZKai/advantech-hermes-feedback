"""Phase 5 Slice 2 tests: the Cross-case Reflector proposal domain contract
in tools.reflector_proposals -- taxonomy, ImprovementProposal,
ProposalObservation, ReflectionResult, and build_supporting_case_ids().

No DB, no LLM, no network anywhere in this file -- pure dataclass
construction/validation, matching test_case_enrichment.py's
OutputValidationInvalidTests / CaseAnalysisEvidenceDirectConstructionTests
style for the equivalent Phase 4.5 write-side contract.

Deliberately a separate file from test_reflector_proposal_storage.py
(the DB/storage tests for the three new tables this contract mirrors),
matching the one-file-per-concern convention already used across this
suite (test_case_enrichment.py vs. test_case_analysis.py).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import (  # noqa: E402
    IMPROVEMENT_TARGET_VALUES,
    PROPOSAL_TREND_VALUES,
    REVIEW_STATUS_VALUES,
)
from tools.reflector_proposals import (  # noqa: E402
    ImprovementProposal,
    ProposalObservation,
    ReflectionResult,
    build_supporting_case_ids,
)


def _proposal(**overrides) -> ImprovementProposal:
    defaults = dict(
        proposal_id="proposal-1",
        improvement_target="knowledge",
        title="ADAM-6266 SNMP disable command undocumented",
        review_status="pending",
        created_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ImprovementProposal(**defaults)


def _observation(**overrides) -> ProposalObservation:
    defaults = dict(
        observation_id="obs-1",
        proposal_id="proposal-1",
        reflection_run_id="run-1",
        trend="new",
        pattern_summary="Three Cases in the last 30 days ask how to disable SNMP on ADAM-6266.",
        possible_cause="The KB may not clearly document the SNMP disable command for this model.",
        recommended_improvement="Add an explicit SNMP disable procedure to the ADAM-6266 KB article.",
        expected_benefit="Fewer repeat questions on this specific topic.",
        limitations="Based on only three Cases; may not generalize.",
        supporting_case_ids=("case-1", "case-2", "case-3"),
        supporting_case_count=3,
        confidence=0.7,
        observed_at="2026-01-02T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ProposalObservation(**defaults)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TaxonomyTests(unittest.TestCase):
    def test_valid_improvement_target_accepted(self):
        for target in IMPROVEMENT_TARGET_VALUES:
            with self.subTest(improvement_target=target):
                proposal = _proposal(improvement_target=target)
                self.assertEqual(proposal.improvement_target, target)

    def test_invalid_improvement_target_rejected(self):
        with self.assertRaises(ValueError):
            _proposal(improvement_target="not_a_real_target")

    def test_valid_review_status_accepted(self):
        for status in REVIEW_STATUS_VALUES:
            with self.subTest(review_status=status):
                proposal = _proposal(review_status=status)
                self.assertEqual(proposal.review_status, status)

    def test_invalid_review_status_rejected(self):
        with self.assertRaises(ValueError):
            _proposal(review_status="implemented")  # not in this slice's taxonomy

    def test_valid_trend_accepted(self):
        for trend in PROPOSAL_TREND_VALUES:
            with self.subTest(trend=trend):
                kwargs = {"trend": trend}
                if trend == "no_longer_observed":
                    kwargs["supporting_case_ids"] = ()
                    kwargs["supporting_case_count"] = 0
                observation = _observation(**kwargs)
                self.assertEqual(observation.trend, trend)

    def test_invalid_trend_rejected(self):
        with self.assertRaises(ValueError):
            _observation(trend="not_a_real_trend")


# ---------------------------------------------------------------------------
# ImprovementProposal
# ---------------------------------------------------------------------------


class ImprovementProposalTests(unittest.TestCase):
    def test_frozen(self):
        proposal = _proposal()
        with self.assertRaises(Exception):
            proposal.title = "other"  # type: ignore[misc]

    def test_empty_proposal_id_rejected(self):
        with self.assertRaises(ValueError):
            _proposal(proposal_id="")

    def test_blank_title_rejected(self):
        with self.assertRaises(ValueError):
            _proposal(title="   ")

    def test_invalid_review_status_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            _proposal(review_status="archived")  # not in this slice's taxonomy

    def test_first_detected_at_not_a_separate_field(self):
        # Slice 2 design decision -- see ImprovementProposal's docstring:
        # only created_at exists, no separate first_detected_at.
        self.assertNotIn("first_detected_at", ImprovementProposal.__dataclass_fields__)


# ---------------------------------------------------------------------------
# ProposalObservation
# ---------------------------------------------------------------------------


class ProposalObservationTests(unittest.TestCase):
    def test_supporting_case_count_matches_ids(self):
        with self.assertRaises(ValueError):
            _observation(supporting_case_ids=("case-1", "case-2"), supporting_case_count=3)

    def test_duplicate_case_ids_rejected(self):
        with self.assertRaises(ValueError):
            _observation(
                supporting_case_ids=("case-1", "case-1", "case-2"), supporting_case_count=3,
            )

    def test_unsorted_supporting_case_ids_rejected(self):
        with self.assertRaises(ValueError):
            _observation(
                supporting_case_ids=("case-3", "case-1", "case-2"), supporting_case_count=3,
            )

    def test_deterministic_supporting_case_ids_via_helper(self):
        ids = build_supporting_case_ids(["case-3", "case-1", "case-2", "case-1"])
        self.assertEqual(ids, ("case-1", "case-2", "case-3"))
        observation = _observation(supporting_case_ids=ids, supporting_case_count=len(ids))
        self.assertEqual(observation.supporting_case_ids, ("case-1", "case-2", "case-3"))

    def test_confidence_in_range_accepted(self):
        for confidence in (0.0, 0.5, 1.0):
            with self.subTest(confidence=confidence):
                observation = _observation(confidence=confidence)
                self.assertEqual(observation.confidence, confidence)

    def test_invalid_confidence_rejected(self):
        for bad_value in (-0.1, 1.1, "0.5", True):
            with self.subTest(confidence=bad_value):
                with self.assertRaises(ValueError):
                    _observation(confidence=bad_value)

    def test_blank_pattern_summary_rejected(self):
        with self.assertRaises(ValueError):
            _observation(pattern_summary="   ")

    def test_blank_recommended_improvement_rejected(self):
        with self.assertRaises(ValueError):
            _observation(recommended_improvement="")

    def test_possible_cause_nullable(self):
        observation = _observation(possible_cause=None)
        self.assertIsNone(observation.possible_cause)

    def test_expected_benefit_and_limitations_nullable(self):
        observation = _observation(expected_benefit=None, limitations=None)
        self.assertIsNone(observation.expected_benefit)
        self.assertIsNone(observation.limitations)

    def test_optional_field_rejects_blank_string(self):
        # None is allowed; an empty/whitespace string standing in for "no
        # value" is not -- avoids ambiguity between "no cause" and "empty
        # string cause".
        with self.assertRaises(ValueError):
            _observation(possible_cause="   ")

    def test_no_longer_observed_allows_zero_supporting_cases(self):
        observation = _observation(
            trend="no_longer_observed", supporting_case_ids=(), supporting_case_count=0,
        )
        self.assertEqual(observation.supporting_case_ids, ())

    def test_non_no_longer_observed_rejects_zero_supporting_cases(self):
        for trend in ("new", "growing", "stable", "declining"):
            with self.subTest(trend=trend):
                with self.assertRaises(ValueError):
                    _observation(trend=trend, supporting_case_ids=(), supporting_case_count=0)


# ---------------------------------------------------------------------------
# ReflectionResult
# ---------------------------------------------------------------------------


class ReflectionResultTests(unittest.TestCase):
    def test_zero_proposal_result_is_valid(self):
        result = ReflectionResult(
            reflection_run_id="run-1",
            run_summary="本次 analysis horizon 未發現具有實質新意義的 recurring pattern。",
            material_change_detected=False,
            new_proposals=(),
            proposal_observations=(),
        )
        self.assertEqual(result.new_proposals, ())
        self.assertEqual(result.proposal_observations, ())

    def test_material_change_false_with_empty_results_valid(self):
        result = ReflectionResult(
            reflection_run_id="run-1", run_summary="No material change this run.",
            material_change_detected=False, new_proposals=(), proposal_observations=(),
        )
        self.assertFalse(result.material_change_detected)

    def test_material_change_true_with_empty_results_rejected(self):
        with self.assertRaises(ValueError):
            ReflectionResult(
                reflection_run_id="run-1", run_summary="Something changed.",
                material_change_detected=True, new_proposals=(), proposal_observations=(),
            )

    def test_material_change_false_with_nonempty_observations_allowed(self):
        # A routine trend='stable' re-observation, recorded for continuity,
        # does not itself require material_change_detected=True.
        observation = _observation(trend="stable", reflection_run_id="run-1")
        result = ReflectionResult(
            reflection_run_id="run-1", run_summary="Stable, no material change.",
            material_change_detected=False, new_proposals=(),
            proposal_observations=(observation,),
        )
        self.assertFalse(result.material_change_detected)

    def test_new_proposal_requires_matching_observation(self):
        proposal = _proposal(proposal_id="proposal-new")
        with self.assertRaises(ValueError):
            ReflectionResult(
                reflection_run_id="run-1", run_summary="Found a new pattern.",
                material_change_detected=True, new_proposals=(proposal,),
                proposal_observations=(),  # missing the founding Observation
            )

    def test_new_proposal_with_matching_observation_valid(self):
        proposal = _proposal(proposal_id="proposal-new")
        observation = _observation(proposal_id="proposal-new", reflection_run_id="run-1")
        result = ReflectionResult(
            reflection_run_id="run-1", run_summary="Found a new pattern.",
            material_change_detected=True, new_proposals=(proposal,),
            proposal_observations=(observation,),
        )
        self.assertEqual(result.new_proposals[0].proposal_id, "proposal-new")

    def test_duplicate_proposal_id_in_observations_rejected(self):
        obs_a = _observation(observation_id="obs-a", reflection_run_id="run-1")
        obs_b = _observation(observation_id="obs-b", reflection_run_id="run-1")
        with self.assertRaises(ValueError):
            ReflectionResult(
                reflection_run_id="run-1", run_summary="Two observations, same proposal.",
                material_change_detected=True, new_proposals=(),
                proposal_observations=(obs_a, obs_b),
            )

    def test_observation_with_mismatched_reflection_run_id_rejected(self):
        observation = _observation(reflection_run_id="run-DIFFERENT")
        with self.assertRaises(ValueError):
            ReflectionResult(
                reflection_run_id="run-1", run_summary="Mismatched run id.",
                material_change_detected=True, new_proposals=(),
                proposal_observations=(observation,),
            )

    def test_frozen(self):
        result = ReflectionResult(
            reflection_run_id="run-1", run_summary="No change.",
            material_change_detected=False, new_proposals=(), proposal_observations=(),
        )
        with self.assertRaises(Exception):
            result.material_change_detected = True  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
