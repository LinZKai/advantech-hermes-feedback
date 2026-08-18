"""Phase 5 Slice 6A tests: ReflectionResult persistence.

Two layers under test, matching tools.reflector_persistence.py's own
module docstring split:

  * FeedbackStoreV2.persist_reflection_proposals() (tools.feedback_store_v2)
    -- the new atomic batch-insert method, tested directly against a real
    temp-file SQLite DB, matching test_reflector_proposal_storage.py's own
    `_StoreTestCase` style for the equivalent Slice 2 storage API. Some
    scenarios here (two new_proposals sharing a proposal_id within one
    batch) can only be constructed this way -- tools.reflector_proposals.
    ReflectionResult's own __post_init__ already forbids a duplicate
    proposal_id across new_proposals, so that exact shape cannot exist as
    a legally-constructed ReflectionResult; it can still reach the storage
    boundary directly (a public boundary that must defend itself
    regardless of caller, per that method's own docstring), so it is
    tested at that layer instead.

  * persist_reflection_result() (tools.reflector_persistence) -- the
    higher-level lifecycle (create_reflection_run -> atomic Proposal/
    Observation transaction -> complete_reflection_run), tested end to
    end against the same real temp-file DB.

No LLM, no network, no mock/fake DB anywhere in this file -- every test
uses a real FeedbackStoreV2 against a real temporary SQLite file, matching
test_reflector_proposal_storage.py's own precedent (a real DB is the only
way to genuinely exercise transaction atomicity/rollback).
"""
from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.reflector_persistence import (  # noqa: E402
    ReflectionPersistenceError,
    ReflectionPersistenceOutcome,
    persist_reflection_result,
)
from tools.reflector_proposals import ImprovementProposal, ProposalObservation, ReflectionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _proposal(**overrides: Any) -> ImprovementProposal:
    defaults = dict(
        proposal_id="proposal-1",
        improvement_target="knowledge",
        title="ADAM-6266 SNMP disable command undocumented",
        review_status="pending",
        created_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ImprovementProposal(**defaults)


def _observation(**overrides: Any) -> ProposalObservation:
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


def _result(**overrides: Any) -> ReflectionResult:
    defaults: dict[str, Any] = dict(
        reflection_run_id="run-1",
        run_summary="Found one recurring knowledge gap across the analyzed Cases.",
        material_change_detected=True,
        new_proposals=(_proposal(),),
        proposal_observations=(_observation(),),
    )
    defaults.update(overrides)
    return ReflectionResult(**defaults)


class _StoreTestCase(unittest.TestCase):
    """Matches test_reflector_proposal_storage.py's own `_StoreTestCase`
    exactly (real temp-file DB, gc.collect() before cleanup for Windows
    file-lock safety) -- not imported from that file, since this test
    suite's own convention is one self-contained fixture set per file."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()


# ---------------------------------------------------------------------------
# A. FeedbackStoreV2.persist_reflection_proposals() -- direct storage tests
# ---------------------------------------------------------------------------


class PersistReflectionProposalsTests(_StoreTestCase):
    def test_new_proposal_and_observation_committed_together(self):
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=3,
            reflector_version="reflector-v1",
        )
        ok = self.store.persist_reflection_proposals(
            "run-1",
            new_proposals=[{
                "proposal_id": "proposal-1", "improvement_target": "knowledge",
                "title": "ADAM-6266 SNMP disable command undocumented",
                "created_at": "2026-01-01T00:00:00+00:00",
            }],
            observations=[{
                "observation_id": "obs-1", "proposal_id": "proposal-1",
                "trend": "new", "pattern_summary": "Pattern.", "possible_cause": None,
                "recommended_improvement": "Fix it.", "expected_benefit": None, "limitations": None,
                "supporting_case_ids_json": '["case-1", "case-2"]', "supporting_case_count": 2,
                "confidence": 0.7, "observed_at": "2026-01-02T00:00:00+00:00",
            }],
        )
        self.assertTrue(ok)
        self.assertIsNotNone(self.store.get_improvement_proposal("proposal-1"))
        self.assertIsNotNone(self.store.get_latest_proposal_observation("proposal-1"))

    def test_empty_batch_is_a_trivial_success(self):
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=0,
            reflector_version="reflector-v1",
        )
        ok = self.store.persist_reflection_proposals("run-1", new_proposals=[], observations=[])
        self.assertTrue(ok)

    def test_duplicate_proposal_id_within_one_batch_rolls_back_everything(self):
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=3,
            reflector_version="reflector-v1",
        )
        ok = self.store.persist_reflection_proposals(
            "run-1",
            new_proposals=[
                {
                    "proposal_id": "proposal-dup", "improvement_target": "knowledge",
                    "title": "First", "created_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "proposal_id": "proposal-dup", "improvement_target": "agent_behavior",
                    "title": "Second (duplicate id)", "created_at": "2026-01-01T00:00:00+00:00",
                },
            ],
            observations=[],
        )
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_improvement_proposal("proposal-dup"))

    def test_observation_referencing_unknown_proposal_id_rolls_back_everything(self):
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=3,
            reflector_version="reflector-v1",
        )
        ok = self.store.persist_reflection_proposals(
            "run-1",
            new_proposals=[{
                "proposal_id": "proposal-1", "improvement_target": "knowledge",
                "title": "Real proposal", "created_at": "2026-01-01T00:00:00+00:00",
            }],
            observations=[{
                "observation_id": "obs-orphan", "proposal_id": "proposal-does-not-exist",
                "trend": "new", "pattern_summary": "Pattern.", "possible_cause": None,
                "recommended_improvement": "Fix it.", "expected_benefit": None, "limitations": None,
                "supporting_case_ids_json": "[]", "supporting_case_count": 0,
                "confidence": 0.7, "observed_at": "2026-01-02T00:00:00+00:00",
            }],
        )
        self.assertFalse(ok)
        # The Proposal insert executed fine inside the transaction, but the
        # whole transaction rolled back because the Observation after it
        # failed -- the Proposal must not survive either.
        self.assertIsNone(self.store.get_improvement_proposal("proposal-1"))

    def test_invalid_improvement_target_rejected_before_opening_a_transaction(self):
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=1,
            reflector_version="reflector-v1",
        )
        ok = self.store.persist_reflection_proposals(
            "run-1",
            new_proposals=[{
                "proposal_id": "proposal-1", "improvement_target": "not_a_real_target",
                "title": "Title", "created_at": "2026-01-01T00:00:00+00:00",
            }],
            observations=[],
        )
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_improvement_proposal("proposal-1"))

    def test_invalid_confidence_rejected_before_opening_a_transaction(self):
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=1,
            reflector_version="reflector-v1",
        )
        ok = self.store.persist_reflection_proposals(
            "run-1",
            new_proposals=[],
            observations=[{
                "observation_id": "obs-1", "proposal_id": "proposal-x",
                "trend": "new", "pattern_summary": "Pattern.", "possible_cause": None,
                "recommended_improvement": "Fix it.", "expected_benefit": None, "limitations": None,
                "supporting_case_ids_json": "[]", "supporting_case_count": 0,
                "confidence": 1.5, "observed_at": "2026-01-02T00:00:00+00:00",
            }],
        )
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# B/C/D/E. persist_reflection_result() -- end-to-end lifecycle tests
# ---------------------------------------------------------------------------


class PersistReflectionResultSuccessTests(_StoreTestCase):
    def test_create_new_run_succeeds_with_proposal_and_founding_observation(self):
        result = _result()
        outcome = persist_reflection_result(
            self.store, result,
            started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-02T00:05:00+00:00",
            analyzed_case_count=3, reflector_version="reflector-v1",
        )
        self.assertIsInstance(outcome, ReflectionPersistenceOutcome)
        self.assertEqual(outcome.status, "succeeded")

        run_row = self.store.get_reflection_run("run-1")
        self.assertEqual(run_row["status"], "succeeded")
        self.assertEqual(bool(run_row["material_change_detected"]), True)
        self.assertEqual(run_row["run_summary"], result.run_summary)

        proposal_row = self.store.get_improvement_proposal("proposal-1")
        self.assertIsNotNone(proposal_row)
        self.assertEqual(proposal_row["improvement_target"], "knowledge")
        self.assertEqual(proposal_row["review_status"], "pending")

        observation_row = self.store.get_latest_proposal_observation("proposal-1")
        self.assertIsNotNone(observation_row)
        self.assertEqual(observation_row["reflection_run_id"], "run-1")

    def test_match_existing_run_appends_observation_without_new_proposal(self):
        # Simulates an existing Proposal from an earlier Run (pre-created
        # directly, the way build_proposal_candidates()'s underlying data
        # would already exist in the DB before this Reflection Run began).
        self.store.create_improvement_proposal(
            "proposal-existing", improvement_target="knowledge",
            title="ADAM-6266 SNMP Knowledge Improvement", created_at="2025-12-01T00:00:00+00:00",
        )
        result = _result(
            reflection_run_id="run-2",
            new_proposals=(),
            proposal_observations=(_observation(
                observation_id="obs-2", proposal_id="proposal-existing", reflection_run_id="run-2",
                trend="growing",
            ),),
        )
        outcome = persist_reflection_result(
            self.store, result,
            started_at="2026-02-01T00:00:00+00:00", completed_at="2026-02-01T00:05:00+00:00",
            analyzed_case_count=5, reflector_version="reflector-v1",
        )
        self.assertEqual(outcome.status, "succeeded")

        # Still exactly one Proposal -- match_existing must never create a
        # duplicate.
        self.assertEqual(len(self.store.list_improvement_proposals()), 1)
        observation_row = self.store.get_latest_proposal_observation("proposal-existing")
        self.assertEqual(observation_row["reflection_run_id"], "run-2")
        self.assertEqual(observation_row["trend"], "growing")

    def test_no_material_change_run_still_records_summary_and_observation(self):
        # Phase 5 Slice 6A task instruction, section 14: material_change_
        # detected=false must NOT be assumed to mean zero observations.
        self.store.create_improvement_proposal(
            "proposal-existing", improvement_target="knowledge",
            title="ADAM-6266 SNMP Knowledge Improvement", created_at="2025-12-01T00:00:00+00:00",
        )
        result = _result(
            reflection_run_id="run-3",
            run_summary="Routine re-observation only; no material change this run.",
            material_change_detected=False,
            new_proposals=(),
            proposal_observations=(_observation(
                observation_id="obs-3", proposal_id="proposal-existing", reflection_run_id="run-3",
                trend="stable",
            ),),
        )
        outcome = persist_reflection_result(
            self.store, result,
            started_at="2026-03-01T00:00:00+00:00", completed_at="2026-03-01T00:05:00+00:00",
            analyzed_case_count=5, reflector_version="reflector-v1",
        )
        self.assertEqual(outcome.status, "succeeded")

        run_row = self.store.get_reflection_run("run-3")
        self.assertEqual(bool(run_row["material_change_detected"]), False)
        self.assertEqual(run_row["run_summary"], result.run_summary)
        self.assertIsNotNone(self.store.get_latest_proposal_observation("proposal-existing"))

    def test_zero_result_run_succeeds_with_no_proposals_or_observations(self):
        result = _result(
            reflection_run_id="run-4",
            run_summary="No recurring improvement opportunity found in this analysis horizon.",
            material_change_detected=False,
            new_proposals=(),
            proposal_observations=(),
        )
        outcome = persist_reflection_result(
            self.store, result,
            started_at="2026-04-01T00:00:00+00:00", completed_at="2026-04-01T00:05:00+00:00",
            analyzed_case_count=5, reflector_version="reflector-v1",
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(self.store.list_improvement_proposals(), [])


class PersistReflectionResultFailureTests(_StoreTestCase):
    def test_second_observation_failure_rolls_back_the_whole_batch_and_marks_run_failed(self):
        good_proposal = _proposal(proposal_id="proposal-new-1")
        good_observation = _observation(
            observation_id="obs-good", proposal_id="proposal-new-1", reflection_run_id="run-1",
        )
        bad_observation = _observation(
            observation_id="obs-bad", proposal_id="proposal-does-not-exist", reflection_run_id="run-1",
        )
        result = _result(
            new_proposals=(good_proposal,),
            proposal_observations=(good_observation, bad_observation),
        )
        outcome = persist_reflection_result(
            self.store, result,
            started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:05:00+00:00",
            analyzed_case_count=3, reflector_version="reflector-v1",
        )
        self.assertEqual(outcome.status, "failed")

        run_row = self.store.get_reflection_run("run-1")
        self.assertEqual(run_row["status"], "failed")

        # Atomicity: even the Proposal + Observation that WOULD have
        # succeeded on their own must not survive -- the second
        # Observation's failure rolls back the entire batch.
        self.assertIsNone(self.store.get_improvement_proposal("proposal-new-1"))
        self.assertIsNone(self.store.get_latest_proposal_observation("proposal-new-1"))

    def test_match_existing_missing_proposal_fails_run_without_partial_state(self):
        result = _result(
            new_proposals=(),
            proposal_observations=(_observation(proposal_id="proposal-never-created"),),
        )
        outcome = persist_reflection_result(
            self.store, result,
            started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:05:00+00:00",
            analyzed_case_count=3, reflector_version="reflector-v1",
        )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(self.store.get_reflection_run("run-1")["status"], "failed")
        self.assertIsNone(self.store.get_improvement_proposal("proposal-never-created"))
        self.assertIsNone(self.store.get_latest_proposal_observation("proposal-never-created"))

    def test_duplicate_create_new_proposal_id_across_runs_fails_without_clobbering_the_original(self):
        first_result = _result(reflection_run_id="run-1")
        first_outcome = persist_reflection_result(
            self.store, first_result,
            started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:05:00+00:00",
            analyzed_case_count=3, reflector_version="reflector-v1",
        )
        self.assertEqual(first_outcome.status, "succeeded")
        original_title = self.store.get_improvement_proposal("proposal-1")["title"]

        # A second, independent Run's create_new finding happens to collide
        # with the already-persisted proposal_id (e.g. an id-generation bug
        # upstream) -- this must fail closed, not silently overwrite.
        second_result = _result(
            reflection_run_id="run-2",
            new_proposals=(_proposal(proposal_id="proposal-1", title="Different title, same id"),),
            proposal_observations=(_observation(
                observation_id="obs-2", proposal_id="proposal-1", reflection_run_id="run-2",
            ),),
        )
        second_outcome = persist_reflection_result(
            self.store, second_result,
            started_at="2026-01-02T00:00:00+00:00", completed_at="2026-01-02T00:05:00+00:00",
            analyzed_case_count=3, reflector_version="reflector-v1",
        )
        self.assertEqual(second_outcome.status, "failed")
        self.assertEqual(self.store.get_reflection_run("run-2")["status"], "failed")
        # The original Proposal (from run-1) must be completely untouched.
        self.assertEqual(self.store.get_improvement_proposal("proposal-1")["title"], original_title)

    def test_duplicate_reflection_run_id_raises_persistence_error(self):
        result = _result()
        persist_reflection_result(
            self.store, result,
            started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:05:00+00:00",
            analyzed_case_count=3, reflector_version="reflector-v1",
        )
        with self.assertRaises(ReflectionPersistenceError):
            persist_reflection_result(
                self.store, result,
                started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:05:00+00:00",
                analyzed_case_count=3, reflector_version="reflector-v1",
            )


if __name__ == "__main__":
    unittest.main()
