"""Phase 5 Slice 3 tests: the Existing-Proposal candidate + match/new
resolution contract in tools.proposal_matching.

Covers four surfaces:
  1. ProposalCandidate -- direct dataclass construction/validation, no DB.
  2. build_proposal_candidates() -- real SQLite through FeedbackStoreV2,
     matching the _StoreTestCase convention already used across this suite.
  3. ProposalResolution -- direct dataclass construction/validation, no DB.
  4. validate_proposal_resolution() -- deterministic, candidate-bound,
     no DB, no LLM.

Deliberately a separate file from test_reflector_proposals.py /
test_reflector_proposal_storage.py, matching the one-file-per-concern
convention this test suite already uses -- this is a new, independently
reviewable Phase 5 slice.

No LLM, no network, no semantic/fuzzy/embedding matching anywhere in this
file -- Slice 3 has none of those.
"""
from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.proposal_matching import (  # noqa: E402
    ProposalCandidate,
    ProposalCandidateBuildError,
    ProposalResolution,
    build_proposal_candidates,
    validate_proposal_resolution,
)


def _candidate(**overrides) -> ProposalCandidate:
    defaults = dict(
        proposal_id="proposal-1",
        improvement_target="knowledge",
        title="ADAM-6266 SNMP disable command undocumented",
        review_status="pending",
        latest_pattern_summary="Three Cases ask about SNMP disable.",
        latest_recommended_improvement="Add an SNMP disable procedure to the KB.",
        latest_trend="new",
        latest_supporting_case_count=3,
        latest_confidence=0.7,
    )
    defaults.update(overrides)
    return ProposalCandidate(**defaults)


def _resolution(**overrides) -> ProposalResolution:
    defaults = dict(
        action="match_existing",
        proposal_id="proposal-1",
        improvement_target="knowledge",
    )
    defaults.update(overrides)
    return ProposalResolution(**defaults)


# ---------------------------------------------------------------------------
# 1. ProposalCandidate
# ---------------------------------------------------------------------------


class ProposalCandidateTests(unittest.TestCase):
    def test_valid_candidate(self):
        candidate = _candidate()
        self.assertEqual(candidate.proposal_id, "proposal-1")

    def test_frozen(self):
        candidate = _candidate()
        with self.assertRaises(Exception):
            candidate.title = "other"  # type: ignore[misc]

    def test_invalid_id_rejected(self):
        with self.assertRaises(ValueError):
            _candidate(proposal_id="")

    def test_invalid_improvement_target_rejected(self):
        with self.assertRaises(ValueError):
            _candidate(improvement_target="not_a_real_target")

    def test_invalid_review_status_rejected(self):
        with self.assertRaises(ValueError):
            _candidate(review_status="implemented")

    def test_invalid_latest_trend_rejected(self):
        with self.assertRaises(ValueError):
            _candidate(latest_trend="not_a_real_trend")

    def test_invalid_confidence_rejected(self):
        for bad_value in (-0.1, 1.1, "0.5", True):
            with self.subTest(confidence=bad_value):
                with self.assertRaises(ValueError):
                    _candidate(latest_confidence=bad_value)

    def test_invalid_supporting_count_rejected(self):
        for bad_value in (-1, "3", 1.5):
            with self.subTest(count=bad_value):
                with self.assertRaises(ValueError):
                    _candidate(latest_supporting_case_count=bad_value)

    def test_non_no_longer_observed_zero_count_rejected(self):
        with self.assertRaises(ValueError):
            _candidate(latest_trend="growing", latest_supporting_case_count=0)

    def test_no_longer_observed_zero_count_allowed(self):
        candidate = _candidate(latest_trend="no_longer_observed", latest_supporting_case_count=0)
        self.assertEqual(candidate.latest_supporting_case_count, 0)

    def test_review_status_not_restricted_to_pending_at_contract_level(self):
        # The dataclass itself allows any real review_status -- scoping to
        # 'pending' is build_proposal_candidates()'s default, not a
        # ProposalCandidate-level constraint (see this module's docstring).
        candidate = _candidate(review_status="accepted")
        self.assertEqual(candidate.review_status, "accepted")


# ---------------------------------------------------------------------------
# 2. build_proposal_candidates()
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

    def _create_run(self, reflection_run_id: str = "run-1", *, analyzed_case_count: int = 5) -> str:
        ok = self.store.create_reflection_run(
            reflection_run_id, started_at="2026-01-01T00:00:00+00:00",
            analyzed_case_count=analyzed_case_count, reflector_version="reflector-v1",
        )
        assert ok
        return reflection_run_id

    def _create_proposal(
        self, proposal_id: str = "proposal-1", *,
        improvement_target: str = "knowledge",
        title: str = "ADAM-6266 SNMP disable command undocumented",
        created_at: str = "2026-01-01T00:00:00+00:00",
    ) -> str:
        ok = self.store.create_improvement_proposal(
            proposal_id, improvement_target=improvement_target, title=title, created_at=created_at,
        )
        assert ok
        return proposal_id

    def _create_observation(
        self, proposal_id: str, reflection_run_id: str, *,
        observation_id: str | None = None,
        trend: str = "new",
        pattern_summary: str = "Three Cases ask about SNMP disable.",
        recommended_improvement: str = "Add an SNMP disable procedure to the KB.",
        supporting_case_ids: tuple[str, ...] = ("case-1", "case-2"),
        confidence: float = 0.7,
        observed_at: str = "2026-01-02T00:00:00+00:00",
    ) -> str:
        observation_id = observation_id or uuid.uuid4().hex
        ok = self.store.create_proposal_observation(
            observation_id, proposal_id, reflection_run_id,
            trend=trend, pattern_summary=pattern_summary, possible_cause=None,
            recommended_improvement=recommended_improvement, expected_benefit=None, limitations=None,
            supporting_case_ids_json=json.dumps(list(supporting_case_ids)),
            supporting_case_count=len(supporting_case_ids),
            confidence=confidence, observed_at=observed_at,
        )
        assert ok
        return observation_id


class BuildProposalCandidatesTests(_StoreTestCase):
    def test_only_pending_proposals_included(self):
        self._create_run("run-1")
        self._create_proposal("p-pending")
        self._create_observation("p-pending", "run-1")
        candidates = build_proposal_candidates(self.store)
        self.assertEqual([c.proposal_id for c in candidates], ["p-pending"])

    def test_accepted_excluded(self):
        self._create_run("run-1")
        self._create_proposal("p-accepted")
        self._create_observation("p-accepted", "run-1")
        self.store.update_proposal_review_status("p-accepted", "accepted")
        candidates = build_proposal_candidates(self.store)
        self.assertEqual(candidates, ())

    def test_rejected_excluded(self):
        self._create_run("run-1")
        self._create_proposal("p-rejected")
        self._create_observation("p-rejected", "run-1")
        self.store.update_proposal_review_status("p-rejected", "rejected")
        candidates = build_proposal_candidates(self.store)
        self.assertEqual(candidates, ())

    def test_latest_observation_used(self):
        self._create_proposal("p1")
        self._create_run("run-1")
        self._create_run("run-2")
        self._create_observation(
            "p1", "run-1", observation_id="o1", trend="new", observed_at="2026-01-01T00:00:01+00:00",
        )
        self._create_observation(
            "p1", "run-2", observation_id="o2", trend="growing", observed_at="2026-01-08T00:00:01+00:00",
        )
        candidates = build_proposal_candidates(self.store)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].latest_trend, "growing")

    def test_historical_observation_not_duplicated(self):
        self._create_proposal("p1")
        self._create_run("run-1")
        self._create_run("run-2")
        self._create_observation("p1", "run-1", observation_id="o1", observed_at="2026-01-01T00:00:01+00:00")
        self._create_observation("p1", "run-2", observation_id="o2", observed_at="2026-01-08T00:00:01+00:00")
        candidates = build_proposal_candidates(self.store)
        # Exactly one Candidate for p1, never one per historical Observation.
        self.assertEqual(len(candidates), 1)

    def test_deterministic_ordering(self):
        self._create_run("run-1")
        for proposal_id in ("p-z", "p-a", "p-m"):
            self._create_proposal(proposal_id)
            self._create_observation(proposal_id, "run-1", observation_id=f"o-{proposal_id}")
        candidates = build_proposal_candidates(self.store)
        self.assertEqual([c.proposal_id for c in candidates], ["p-a", "p-m", "p-z"])

    def test_pending_proposal_without_observation_raises(self):
        self._create_proposal("p-orphan")  # no observation ever created
        with self.assertRaises(ProposalCandidateBuildError):
            build_proposal_candidates(self.store)

    def test_empty_proposal_set_returns_empty_tuple(self):
        candidates = build_proposal_candidates(self.store)
        self.assertEqual(candidates, ())

    def test_custom_review_status_scope(self):
        self._create_run("run-1")
        self._create_proposal("p-accepted")
        self._create_observation("p-accepted", "run-1")
        self.store.update_proposal_review_status("p-accepted", "accepted")
        candidates = build_proposal_candidates(self.store, review_status="accepted")
        self.assertEqual([c.proposal_id for c in candidates], ["p-accepted"])

    def test_candidate_fields_reflect_source_rows(self):
        self._create_run("run-1")
        self._create_proposal("p1", improvement_target="agent_behavior", title="Firmware check missing")
        self._create_observation(
            "p1", "run-1", trend="growing", pattern_summary="s", recommended_improvement="r",
            supporting_case_ids=("c1", "c2", "c3"), confidence=0.42,
        )
        candidate = build_proposal_candidates(self.store)[0]
        self.assertEqual(candidate.improvement_target, "agent_behavior")
        self.assertEqual(candidate.title, "Firmware check missing")
        self.assertEqual(candidate.latest_trend, "growing")
        self.assertEqual(candidate.latest_pattern_summary, "s")
        self.assertEqual(candidate.latest_recommended_improvement, "r")
        self.assertEqual(candidate.latest_supporting_case_count, 3)
        self.assertEqual(candidate.latest_confidence, 0.42)


# ---------------------------------------------------------------------------
# 3. ProposalResolution
# ---------------------------------------------------------------------------


class ProposalResolutionTests(unittest.TestCase):
    def test_valid_match_existing(self):
        resolution = _resolution(action="match_existing", proposal_id="proposal-1")
        self.assertEqual(resolution.proposal_id, "proposal-1")

    def test_valid_create_new(self):
        resolution = _resolution(action="create_new", proposal_id=None)
        self.assertIsNone(resolution.proposal_id)

    def test_match_existing_without_proposal_id_rejected(self):
        with self.assertRaises(ValueError):
            _resolution(action="match_existing", proposal_id=None)

    def test_create_new_with_proposal_id_rejected(self):
        with self.assertRaises(ValueError):
            _resolution(action="create_new", proposal_id="proposal-1")

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            _resolution(action="not_a_real_action")

    def test_invalid_improvement_target_rejected(self):
        with self.assertRaises(ValueError):
            _resolution(improvement_target="not_a_real_target")

    def test_frozen(self):
        resolution = _resolution()
        with self.assertRaises(Exception):
            resolution.action = "create_new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. validate_proposal_resolution()
# ---------------------------------------------------------------------------


class ValidateProposalResolutionTests(unittest.TestCase):
    def test_match_id_in_candidates_accepted(self):
        candidate = _candidate(proposal_id="proposal-1", improvement_target="knowledge")
        resolution = _resolution(
            action="match_existing", proposal_id="proposal-1", improvement_target="knowledge",
        )
        self.assertTrue(validate_proposal_resolution(resolution, candidates=(candidate,)))

    def test_hallucinated_id_rejected(self):
        candidate = _candidate(proposal_id="proposal-1")
        resolution = _resolution(action="match_existing", proposal_id="proposal-does-not-exist")
        self.assertFalse(validate_proposal_resolution(resolution, candidates=(candidate,)))

    def test_create_new_with_candidate_list_valid(self):
        candidate = _candidate(proposal_id="proposal-1")
        resolution = _resolution(action="create_new", proposal_id=None)
        self.assertTrue(validate_proposal_resolution(resolution, candidates=(candidate,)))

    def test_duplicate_candidate_ids_rejected(self):
        candidate_a = _candidate(proposal_id="proposal-1")
        candidate_b = _candidate(proposal_id="proposal-1")  # same id, malformed caller input
        resolution = _resolution(action="match_existing", proposal_id="proposal-1")
        with self.assertRaises(ValueError):
            validate_proposal_resolution(resolution, candidates=(candidate_a, candidate_b))

    def test_improvement_target_mismatch_rejected(self):
        candidate = _candidate(proposal_id="proposal-1", improvement_target="knowledge")
        resolution = _resolution(
            action="match_existing", proposal_id="proposal-1", improvement_target="agent_behavior",
        )
        self.assertFalse(validate_proposal_resolution(resolution, candidates=(candidate,)))

    def test_improvement_target_match_accepted(self):
        candidate = _candidate(proposal_id="proposal-1", improvement_target="agent_behavior")
        resolution = _resolution(
            action="match_existing", proposal_id="proposal-1", improvement_target="agent_behavior",
        )
        self.assertTrue(validate_proposal_resolution(resolution, candidates=(candidate,)))

    def test_empty_candidate_set_create_new_valid(self):
        resolution = _resolution(action="create_new", proposal_id=None)
        self.assertTrue(validate_proposal_resolution(resolution, candidates=()))

    def test_empty_candidate_set_match_existing_invalid(self):
        resolution = _resolution(action="match_existing", proposal_id="proposal-1")
        self.assertFalse(validate_proposal_resolution(resolution, candidates=()))


if __name__ == "__main__":
    unittest.main()
