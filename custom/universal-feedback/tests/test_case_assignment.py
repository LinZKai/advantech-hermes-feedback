"""Phase 4A Stage B tests: real Case assignment policy + persistence in
tools.retrieval_runtime.

Covers the routing decision policy (_resolve_case_assignment), deterministic
new-Case ids (real_case_id), idempotent Case/Turn creation
(_get_or_create_real_case / _get_or_create_turn), and the
persist_turn_and_retrieval / observe_and_persist_turn / build_turn_
observation_context / persist_turn_observation_context signature
extensions -- all via a temporary on-disk SQLite DB through
FeedbackStoreV2, matching the fixture convention in
test_retrieval_runtime.py.

Deliberately a SEPARATE file from test_retrieval_runtime.py rather than an
extension of it: test_retrieval_runtime.py stays untouched as the frozen
Phase 3A/Stage-A-integration regression baseline (all of it still passes
unmodified against the Stage B code -- see the Stage B report), and this
file owns the substantial new Stage B-only test surface, matching the
one-file-per-concern pattern already used elsewhere in this test suite.
"""
from __future__ import annotations

import gc
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_routing import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    ROUTING_VERSION,
    CaseRoutingResult,
)
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.retrieval_observer import (  # noqa: E402
    FoundryRetrievalObservation,
    TurnRetrievalObservation,
)
from tools.retrieval_runtime import (  # noqa: E402
    CASE_ASSIGNMENT_METHOD_EXISTING,
    CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID,
    CASE_ASSIGNMENT_METHOD_FALLBACK_LOW_CONFIDENCE,
    CASE_ASSIGNMENT_METHOD_FALLBACK_MISSING,
    CASE_ASSIGNMENT_METHOD_FALLBACK_UNCERTAIN,
    CASE_ASSIGNMENT_METHOD_FIRST_TURN,
    CASE_ASSIGNMENT_METHOD_NEW,
    DEFAULT_CASE_ASSIGNMENT_METHOD,
    _get_or_create_real_case,
    _get_or_create_turn,
    build_turn_observation_context,
    default_case_id,
    observe_and_persist_turn,
    persist_turn_and_retrieval,
    persist_turn_observation_context,
    real_case_id,
)


def _empty_observation() -> TurnRetrievalObservation:
    return TurnRetrievalObservation(
        boundary_trusted=True,
        turn_observation_status="complete",
        turn_observation_reason=None,
        retrievals=(),
    )


def _completed_observation(n=1) -> TurnRetrievalObservation:
    retrievals = tuple(
        FoundryRetrievalObservation(
            tool_call_id=f"call-{i}",
            execution_status="completed",
            foundry_iq_ok=True,
            observation_status="complete",
            observation_reason=None,
            error_code=None,
            http_status=None,
            result_count=3,
            reference_count=1,
            foundry_schema_version="foundry-iq-result-v2",
            request_attempted=True,
        )
        for i in range(n)
    )
    return TurnRetrievalObservation(
        boundary_trusted=True,
        turn_observation_status="complete",
        turn_observation_reason=None,
        retrievals=retrievals,
    )


def _existing(case_id, confidence):
    return CaseRoutingResult(
        status="valid", case_action="existing", case_id=case_id,
        confidence=confidence, routing_version=ROUTING_VERSION,
    )


def _new(confidence=0.9):
    return CaseRoutingResult(
        status="valid", case_action="new", case_id=None,
        confidence=confidence, routing_version=ROUTING_VERSION,
    )


def _uncertain(confidence=0.3):
    return CaseRoutingResult(
        status="valid", case_action="uncertain", case_id=None,
        confidence=confidence, routing_version=ROUTING_VERSION,
    )


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_case(self, case_id, session_id):
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        return case_id

    def _persist(
        self,
        *,
        session_id="sess-1",
        turn_id="turn-1",
        observation=None,
        case_routing=None,
        candidate_case_ids=None,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        **overrides,
    ):
        kwargs = dict(
            session_id=session_id,
            platform="telegram",
            platform_chat_id="chat-1",
            turn_id=turn_id,
            platform_user_id="user-1",
            platform_user_message_id=f"{turn_id}-msg",
            platform_assistant_message_id=None,
            question_text="What is the reset procedure?",
            answer_text="Hold the reset button for 5 seconds.",
            feedback_eligible=True,
            model="claude-sonnet-5",
            provider=None,
            hermes_version=None,
            support_config_commit=None,
            feedback_code_commit=None,
            observation=observation or _empty_observation(),
            case_routing=case_routing,
            candidate_case_ids=candidate_case_ids,
            confidence_threshold=confidence_threshold,
        )
        kwargs.update(overrides)
        return persist_turn_and_retrieval(self.store, **kwargs)


class DecisionPolicyTests(_StoreTestCase):
    def test_first_turn_no_candidates_is_deterministic_new(self):
        ok = self._persist(turn_id="turn-1", candidate_case_ids=frozenset())
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FIRST_TURN)
        self.assertIsNone(turn["case_assignment_confidence"])
        self.assertEqual(turn["case_assignment_classifier_version"], ROUTING_VERSION)
        self.assertEqual(turn["case_id"], real_case_id("sess-1", "turn-1"))

    def test_first_turn_overrides_any_case_routing_content(self):
        # Not a fallback, not a model decision -- first turn wins
        # regardless of what case_routing says, even a high-confidence
        # "existing".
        routing = _existing("case-a", 0.99)
        self._persist(turn_id="turn-1", case_routing=routing, candidate_case_ids=frozenset())
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FIRST_TURN)
        self.assertIsNone(turn["case_assignment_confidence"])
        self.assertNotEqual(turn["case_id"], "case-a")

    def test_valid_existing_high_confidence_assigned(self):
        self._seed_case("case-a", "sess-1")
        self._persist(
            turn_id="turn-1", case_routing=_existing("case-a", 0.92),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_id"], "case-a")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_EXISTING)
        self.assertEqual(turn["case_assignment_confidence"], "0.92")
        self.assertEqual(turn["case_assignment_classifier_version"], ROUTING_VERSION)

    def test_valid_existing_exactly_threshold_assigned(self):
        self._seed_case("case-a", "sess-1")
        self._persist(
            turn_id="turn-1",
            case_routing=_existing("case-a", DEFAULT_CONFIDENCE_THRESHOLD),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_EXISTING)
        self.assertEqual(turn["case_id"], "case-a")

    def test_valid_existing_below_threshold_falls_back_new(self):
        self._seed_case("case-a", "sess-1")
        self._persist(
            turn_id="turn-1", case_routing=_existing("case-a", 0.40),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_LOW_CONFIDENCE)
        # Original model confidence preserved verbatim -- never rewritten
        # to the threshold, never zeroed.
        self.assertEqual(turn["case_assignment_confidence"], "0.4")
        self.assertNotEqual(turn["case_id"], "case-a")

    def test_valid_new_creates_new_case(self):
        self._persist(
            turn_id="turn-1", case_routing=_new(0.81),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_NEW)
        self.assertEqual(turn["case_assignment_confidence"], "0.81")
        self.assertEqual(turn["case_id"], real_case_id("sess-1", "turn-1"))

    def test_valid_uncertain_falls_back_new(self):
        self._persist(
            turn_id="turn-1", case_routing=_uncertain(0.3),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_UNCERTAIN)
        self.assertEqual(turn["case_assignment_confidence"], "0.3")

    def test_missing_routing_none_falls_back_new(self):
        self._persist(turn_id="turn-1", case_routing=None, candidate_case_ids=frozenset({"case-a"}))
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_MISSING)
        self.assertIsNone(turn["case_assignment_confidence"])

    def test_absent_status_routing_falls_back_new(self):
        self._persist(
            turn_id="turn-1", case_routing=CaseRoutingResult(status="absent"),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_MISSING)

    def test_invalid_routing_falls_back_new(self):
        self._persist(
            turn_id="turn-1",
            case_routing=CaseRoutingResult(status="invalid", invalid_reason="json_invalid"),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID)
        # Stage A never populates confidence on an invalid CaseRoutingResult.
        self.assertIsNone(turn["case_assignment_confidence"])


class ExistingCaseAssignmentTests(_StoreTestCase):
    def test_existing_oldest_case(self):
        self._seed_case("case-old", "sess-1")
        self._seed_case("case-new", "sess-1")
        self._persist(
            turn_id="turn-1", case_routing=_existing("case-old", 0.9),
            candidate_case_ids=frozenset({"case-old", "case-new"}),
        )
        self.assertEqual(self.store.get_turn("turn-1")["case_id"], "case-old")

    def test_existing_newest_case(self):
        self._seed_case("case-old", "sess-1")
        self._seed_case("case-new", "sess-1")
        self._persist(
            turn_id="turn-1", case_routing=_existing("case-new", 0.9),
            candidate_case_ids=frozenset({"case-old", "case-new"}),
        )
        self.assertEqual(self.store.get_turn("turn-1")["case_id"], "case-new")

    def test_unknown_case_id_cannot_be_accepted(self):
        self._seed_case("case-a", "sess-1")
        self._persist(
            turn_id="turn-1", case_routing=_existing("case-z", 0.99),
            candidate_case_ids=frozenset({"case-a"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertNotEqual(turn["case_id"], "case-z")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID)

    def test_cross_session_case_cannot_be_written(self):
        # case-shared genuinely exists, but under a DIFFERENT session.
        # Even if a buggy/malicious caller's candidate_case_ids includes
        # it, the persistence layer must reject it -- never write
        # turns.case_id=case-shared under session sess-1.
        self._seed_case("case-shared", "sess-other")
        self._persist(
            session_id="sess-1", turn_id="turn-1",
            case_routing=_existing("case-shared", 0.95),
            candidate_case_ids=frozenset({"case-shared"}),
        )
        turn = self.store.get_turn("turn-1")
        self.assertNotEqual(turn["case_id"], "case-shared")
        self.assertEqual(turn["session_id"], "sess-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID)


class DeterministicCaseIdTests(unittest.TestCase):
    def test_same_turn_retry_same_case_id(self):
        self.assertEqual(real_case_id("sess-1", "turn-1"), real_case_id("sess-1", "turn-1"))

    def test_different_turn_different_case_id(self):
        self.assertNotEqual(real_case_id("sess-1", "turn-1"), real_case_id("sess-1", "turn-2"))

    def test_different_session_same_turn_id_different_case_id(self):
        self.assertNotEqual(real_case_id("sess-1", "turn-x"), real_case_id("sess-2", "turn-x"))

    def test_real_case_id_distinct_from_default_case_id(self):
        self.assertNotEqual(real_case_id("sess-1", "turn-1"), default_case_id("sess-1"))


class NewCaseIdempotencyTests(_StoreTestCase):
    def test_retry_persist_turn_and_retrieval_creates_only_one_case(self):
        routing = _new(0.9)
        first = self._persist(turn_id="turn-1", case_routing=routing, candidate_case_ids=frozenset())
        second = self._persist(turn_id="turn-1", case_routing=routing, candidate_case_ids=frozenset())
        self.assertTrue(first)
        self.assertTrue(second)
        cases = self.store.list_cases_for_session("sess-1")
        self.assertEqual(len(cases), 1)
        self.assertEqual(self.store.get_turn("turn-1")["case_id"], cases[0]["case_id"])

    def test_get_or_create_real_case_direct_idempotent_retry(self):
        self.store.create_or_update_session("sess-1", "telegram")
        case_id_1 = _get_or_create_real_case(self.store, session_id="sess-1", turn_id="turn-1")
        case_id_2 = _get_or_create_real_case(self.store, session_id="sess-1", turn_id="turn-1")
        self.assertEqual(case_id_1, case_id_2)
        self.assertEqual(len(self.store.list_cases_for_session("sess-1")), 1)

    def test_get_or_create_real_case_rejects_session_mismatch(self):
        # Force the exact real_case_id collision scenario: a row with the
        # SAME deterministic id already exists, but for a DIFFERENT
        # session. Must raise, never silently treated as an idempotent
        # retry -- this is the "IntegrityError but expected case cannot be
        # found" guard, not a theoretical uuid5-collision worry.
        forged_case_id = real_case_id("sess-1", "turn-1")
        self.store.create_or_update_session("sess-other", "telegram")
        self.store.create_case(forged_case_id, "sess-other")
        with self.assertRaises(RuntimeError):
            _get_or_create_real_case(self.store, session_id="sess-1", turn_id="turn-1")


class TurnRetryTests(_StoreTestCase):
    def test_turn_retry_still_runs_retrieval_persistence(self):
        # Simulates a first attempt that created the turn row but died
        # before add_retrieval_runs ever ran. Retry must still attempt
        # retrieval persistence -- a duplicate turn_id never implies
        # retrieval already succeeded.
        self._seed_case("case-a", "sess-1")
        self.store.create_turn(
            "turn-1", "case-a",
            platform_user_id="user-1", platform_user_message_id="turn-1-msg",
            question_text="q", answer_text="a", feedback_eligible=True,
        )
        self.assertEqual(self.store.list_retrieval_runs_for_turn("turn-1"), [])

        ok = self._persist(
            turn_id="turn-1", observation=_completed_observation(1),
            candidate_case_ids=frozenset({"case-a"}),
            case_routing=_existing("case-a", 0.9),
        )
        self.assertTrue(ok)
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(len(runs), 1)
        # The turn row itself is unchanged/still attached to case-a.
        self.assertEqual(self.store.get_turn("turn-1")["case_id"], "case-a")

    def test_get_or_create_turn_verifies_same_case_on_retry(self):
        self._seed_case("case-a", "sess-1")
        _get_or_create_turn(
            self.store, "turn-1", "case-a",
            platform_user_id="u", platform_user_message_id="turn-1-msg",
            question_text="q", answer_text="a", feedback_eligible=True,
        )
        # Second call for the SAME (turn_id, case_id) pair must not raise.
        _get_or_create_turn(
            self.store, "turn-1", "case-a",
            platform_user_id="u", platform_user_message_id="turn-1-msg",
            question_text="q", answer_text="a", feedback_eligible=True,
        )
        self.assertIsNotNone(self.store.get_turn("turn-1"))

    def test_get_or_create_turn_rejects_case_mismatch(self):
        self.store.create_or_update_session("sess-1", "telegram")
        self.store.create_case("case-a", "sess-1")
        self.store.create_case("case-b", "sess-1")
        self.store.create_turn(
            "turn-1", "case-a",
            platform_user_id="u", platform_user_message_id="turn-1-msg",
            question_text="q", answer_text="a", feedback_eligible=True,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            _get_or_create_turn(
                self.store, "turn-1", "case-b",
                platform_user_id="u", platform_user_message_id="turn-1-msg",
                question_text="q", answer_text="a", feedback_eligible=True,
            )


class RetrievalPersistenceRegressionTests(_StoreTestCase):
    def test_normal_retrieval_runs_preserved_with_real_case(self):
        self._persist(
            turn_id="turn-1", observation=_completed_observation(2),
            case_routing=_new(0.9), candidate_case_ids=frozenset(),
        )
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(len(runs), 2)

    def test_retrieval_failure_status_behavior_unchanged_with_real_case(self):
        bad_obs = TurnRetrievalObservation(
            boundary_trusted=True,
            turn_observation_status="complete",
            turn_observation_reason=None,
            retrievals=(
                FoundryRetrievalObservation(
                    tool_call_id="call-1",
                    execution_status="http_error",
                    foundry_iq_ok=False,
                    observation_status="complete",
                    observation_reason=None,
                    error_code="http_error",
                    http_status=999,  # violates the DB's own CHECK (100-599)
                    result_count=None,
                    reference_count=None,
                    foundry_schema_version=None,
                    request_attempted=True,
                ),
            ),
        )
        ok = self._persist(
            turn_id="turn-1", observation=bad_obs,
            case_routing=_new(0.9), candidate_case_ids=frozenset(),
        )
        self.assertTrue(ok)  # Turn creation itself still succeeded.
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "retrieval_insert_failed")
        self.assertEqual(self.store.list_retrieval_runs_for_turn("turn-1"), [])


class LegacyCallerBackwardCompatibilityTests(_StoreTestCase):
    """candidate_case_ids is the explicit Phase 4 opt-in signal (see
    persist_turn_and_retrieval docstring). A caller that omits it entirely
    must see byte-for-byte the same Phase 3A behavior as before Stage B."""

    def test_omitting_candidate_case_ids_keeps_phase3_default_behavior(self):
        ok = self._persist(turn_id="turn-1")
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], DEFAULT_CASE_ASSIGNMENT_METHOD)
        self.assertIsNone(turn["case_assignment_confidence"])
        self.assertIsNone(turn["case_assignment_classifier_version"])
        self.assertEqual(turn["case_id"], default_case_id("sess-1"))

    def test_omitting_candidate_case_ids_multiple_turns_share_default_case(self):
        self._persist(session_id="sess-1", turn_id="turn-1")
        self._persist(session_id="sess-1", turn_id="turn-2")
        t1 = self.store.get_turn("turn-1")
        t2 = self.store.get_turn("turn-2")
        self.assertEqual(t1["case_id"], t2["case_id"])

    def test_observe_and_persist_turn_without_phase4_kwargs_uses_default_case(self):
        response = {
            "final_response": "answer", "model": None,
            "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True,
        }
        ok = observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        self.assertTrue(ok)
        self.assertEqual(self.store.get_turn("turn-1")["case_assignment_method"], DEFAULT_CASE_ASSIGNMENT_METHOD)


class ContextExtensionTests(_StoreTestCase):
    def test_build_context_stashes_candidate_case_ids(self):
        agent_result = {
            "final_response": "answer", "model": None, "messages": [], "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
            candidate_case_ids=frozenset({"case-a", "case-b"}),
        )
        self.assertIsNotNone(ctx)
        json.dumps(ctx)  # still a plain, JSON-safe dict
        self.assertEqual(sorted(ctx["candidate_case_ids"]), ["case-a", "case-b"])

    def test_build_context_omitted_candidate_case_ids_is_none(self):
        agent_result = {
            "final_response": "answer", "model": None, "messages": [], "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
        )
        self.assertIsNone(ctx["candidate_case_ids"])

    def test_persist_context_with_case_routing_creates_real_case(self):
        agent_result = {
            "final_response": "answer", "model": None, "messages": [], "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
            candidate_case_ids=frozenset(),
        )
        ok = persist_turn_observation_context(
            ctx,
            platform_assistant_message_id="asst-1",
            question_text="q", answer_text="a", feedback_eligible=True,
            store=self.store, case_routing=_new(0.88),
        )
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        # candidate_case_ids stashed as an explicitly-empty set -> first
        # turn short-circuit, not the "new" method case_routing requested.
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FIRST_TURN)

    def test_persist_context_without_candidate_case_ids_uses_default_case(self):
        agent_result = {
            "final_response": "answer", "model": None, "messages": [], "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
        )
        ok = persist_turn_observation_context(
            ctx,
            platform_assistant_message_id="asst-1",
            question_text="q", answer_text="a", feedback_eligible=True,
            store=self.store,
        )
        self.assertTrue(ok)
        self.assertEqual(self.store.get_turn("turn-1")["case_assignment_method"], DEFAULT_CASE_ASSIGNMENT_METHOD)


if __name__ == "__main__":
    unittest.main()
