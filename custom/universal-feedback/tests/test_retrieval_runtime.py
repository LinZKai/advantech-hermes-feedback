"""Phase 3A storage-integration and wiring-glue tests: tools.retrieval_runtime.

Uses a temporary on-disk SQLite DB via FeedbackStoreV2, matching the
Windows-safe cleanup convention in test_feedback_store_v2.py (sqlite3
connections only fully close on GC, so `del store; gc.collect()` must run
before the temp dir is removed).
"""
from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.retrieval_observer import (  # noqa: E402
    FoundryRetrievalObservation,
    TurnRetrievalObservation,
)
from tools.retrieval_runtime import (  # noqa: E402
    build_turn_observation_context,
    default_case_id,
    get_or_create_default_case,
    observe_and_persist_turn,
    persist_turn_and_retrieval,
    persist_turn_observation_context,
    resolve_boundary_trust,
)
from tools.universal_feedback import FOUNDARY_SCRIPT  # noqa: E402

CANARY_SECRET = "sk-CANARY-SECRET-DO-NOT-LEAK-1234567890"


def _foundry_messages(question="question"):
    command = f"python3 {FOUNDARY_SCRIPT} '{question}'"
    inner = json.dumps({
        "schema_version": "foundry-iq-result-v2",
        "ok": True,
        "request_attempted": True,
        "error_code": None,
        "question": question,
        "http_status": None,
        "documents": [{"ref_id": 0, "content": "doc"}],
        "references": [],
        "activity": [],
    })
    outer = json.dumps({"output": inner, "exit_code": 0, "error": None})
    return [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "function": {"name": "terminal", "arguments": json.dumps({"command": command})}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": outer},
    ]


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


def _empty_observation() -> TurnRetrievalObservation:
    return TurnRetrievalObservation(
        boundary_trusted=True,
        turn_observation_status="complete",
        turn_observation_reason=None,
        retrievals=(),
    )


def _unavailable_observation() -> TurnRetrievalObservation:
    return TurnRetrievalObservation(
        boundary_trusted=False,
        turn_observation_status="unavailable",
        turn_observation_reason="untrusted_turn_boundary",
        retrievals=(),
    )


class _ExplodingStore:
    """A store double whose every method raises -- used to prove Phase 3A
    persistence failures never propagate."""

    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated failure in {name}")
        return _boom


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _persist(self, *, session_id="sess-1", turn_id="turn-1", observation=None, **overrides):
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
        )
        kwargs.update(overrides)
        return persist_turn_and_retrieval(self.store, **kwargs)


class DefaultCaseTests(_StoreTestCase):
    def test_default_case_id_is_deterministic(self):
        self.assertEqual(default_case_id("sess-abc"), default_case_id("sess-abc"))

    def test_default_case_id_differs_by_session(self):
        self.assertNotEqual(default_case_id("sess-abc"), default_case_id("sess-xyz"))

    def test_default_case_id_reasonable_length_no_raw_session_id(self):
        case_id = default_case_id("sess-super-secret-session-identifier")
        self.assertLess(len(case_id), 80)
        self.assertNotIn("super-secret", case_id)

    def test_get_or_create_default_case_idempotent(self):
        self.store.create_or_update_session("sess-1", "telegram")
        first = get_or_create_default_case(self.store, "sess-1")
        second = get_or_create_default_case(self.store, "sess-1")
        self.assertEqual(first, second)
        cases = self.store.list_cases_for_session("sess-1")
        self.assertEqual(len(cases), 1)


class StorageIntegrationTests(_StoreTestCase):
    def test_1_creates_session_case_turn_and_retrieval(self):
        ok = self._persist(session_id="sess-1", turn_id="turn-1", observation=_completed_observation(1))
        self.assertTrue(ok)
        self.assertIsNotNone(self.store.get_session("sess-1"))
        turn = self.store.get_turn("turn-1")
        self.assertIsNotNone(turn)
        self.assertEqual(turn["retrieval_observation_status"], "complete")
        self.assertEqual(turn["case_assignment_method"], "phase3_default")
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["execution_status"], "completed")

    def test_2_same_session_multiple_turns_share_default_case(self):
        self._persist(session_id="sess-1", turn_id="turn-1")
        self._persist(session_id="sess-1", turn_id="turn-2")
        t1 = self.store.get_turn("turn-1")
        t2 = self.store.get_turn("turn-2")
        self.assertEqual(t1["case_id"], t2["case_id"])
        self.assertEqual(len(self.store.list_cases_for_session("sess-1")), 1)

    def test_3_new_session_gets_different_case(self):
        self._persist(session_id="sess-1", turn_id="turn-1")
        self._persist(session_id="sess-2", turn_id="turn-2")
        t1 = self.store.get_turn("turn-1")
        t2 = self.store.get_turn("turn-2")
        self.assertNotEqual(t1["case_id"], t2["case_id"])

    def test_4_zero_retrievals_creates_no_row(self):
        self._persist(turn_id="turn-1", observation=_empty_observation())
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(runs, [])
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["retrieval_observation_status"], "complete")
        self.assertIsNone(turn["retrieval_observation_reason"])

    def test_5_multiple_retrievals_one_row_each_in_order(self):
        self._persist(turn_id="turn-1", observation=_completed_observation(3))
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(len(runs), 3)
        self.assertEqual([r["tool_call_id"] for r in runs], ["call-0", "call-1", "call-2"])
        self.assertEqual([r["invocation_order"] for r in runs], [1, 2, 3])

    def test_6_parser_unavailable_written_to_turn(self):
        self._persist(turn_id="turn-1", observation=_unavailable_observation())
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "untrusted_turn_boundary")
        self.assertEqual(self.store.list_retrieval_runs_for_turn("turn-1"), [])

    def test_6b_parser_partial_row_written(self):
        obs = TurnRetrievalObservation(
            boundary_trusted=True,
            turn_observation_status="complete",
            turn_observation_reason=None,
            retrievals=(
                FoundryRetrievalObservation(
                    tool_call_id="call-1",
                    execution_status="unparseable",
                    foundry_iq_ok=None,
                    observation_status="partial",
                    observation_reason="outer_result_unparseable",
                    error_code=None,
                    http_status=None,
                    result_count=None,
                    reference_count=None,
                    foundry_schema_version=None,
                    request_attempted=None,
                ),
            ),
        )
        self._persist(turn_id="turn-1", observation=obs)
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["observation_status"], "partial")
        self.assertEqual(runs[0]["observation_reason"], "outer_result_unparseable")
        # Turn-level status is unaffected by a single partial row -- the
        # boundary itself was reliably observed.
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["retrieval_observation_status"], "complete")

    def test_7_retrieval_insert_failure_preserves_turn(self):
        # http_status=999 passes retrieval_observer/RetrievalRunInput
        # (neither validates the range) but violates the DB's own CHECK
        # constraint (100-599), forcing add_retrieval_runs's real recovery
        # path: rollback the batch, downgrade the turn, never raise.
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
                    http_status=999,
                    result_count=None,
                    reference_count=None,
                    foundry_schema_version=None,
                    request_attempted=True,
                ),
            ),
        )
        ok = self._persist(turn_id="turn-1", observation=bad_obs)
        self.assertTrue(ok)  # Turn creation itself succeeded.
        turn = self.store.get_turn("turn-1")
        self.assertIsNotNone(turn)
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "retrieval_insert_failed")
        self.assertEqual(self.store.list_retrieval_runs_for_turn("turn-1"), [])

    def test_8_v2_failure_does_not_affect_simulated_legacy_hook(self):
        legacy_calls = []

        def _legacy_feedback_hook():
            legacy_calls.append(1)

        exploding_store = _ExplodingStore()
        ok = persist_turn_and_retrieval(
            exploding_store,
            session_id="sess-1",
            platform="telegram",
            platform_chat_id="chat-1",
            turn_id="turn-1",
            platform_user_id="user-1",
            platform_user_message_id="msg-1",
            platform_assistant_message_id=None,
            question_text="q",
            answer_text="a",
            feedback_eligible=True,
            model=None,
            provider=None,
            hermes_version=None,
            support_config_commit=None,
            feedback_code_commit=None,
            observation=_completed_observation(1),
        )
        # Simulates run.py/base.py's own call order: Phase 3A first, then
        # the legacy feedback hook unconditionally afterward.
        _legacy_feedback_hook()

        self.assertFalse(ok)
        self.assertEqual(legacy_calls, [1])

    def test_question_and_answer_text_redacted(self):
        # safe_feedback_text redacts known credential *patterns*
        # (Authorization: Bearer, api_key=, token=, ...), not arbitrary
        # secret-looking substrings -- use a pattern it actually matches to
        # prove persist_turn_and_retrieval applies it before storing.
        self._persist(
            turn_id="turn-1",
            question_text="my token=" + CANARY_SECRET,
            answer_text="Authorization: Bearer " + CANARY_SECRET,
        )
        turn = self.store.get_turn("turn-1")
        self.assertNotIn(CANARY_SECRET, turn["question_text"])
        self.assertNotIn(CANARY_SECRET, turn["answer_text"])
        self.assertIn("[REDACTED]", turn["question_text"])
        self.assertIn("[REDACTED]", turn["answer_text"])


class ResolveBoundaryTrustTests(unittest.TestCase):
    """resolve_boundary_trust reads a single explicit field
    (phase3a_boundary_trusted) that run.py's _run_agent_inner sets
    directly at its three response-dict construction sites (see
    run.py:~17831, ~17939, ~18408 -- the two normal-completion return
    points and the inactivity-timeout synthetic dict). It must never
    reconstruct trust by comparing possibly-missing fields."""

    def test_explicit_true_is_trusted(self):
        self.assertTrue(resolve_boundary_trust({"phase3a_boundary_trusted": True}))

    def test_explicit_false_is_untrusted(self):
        self.assertFalse(resolve_boundary_trust({"phase3a_boundary_trusted": False}))

    def test_missing_field_is_untrusted_not_ambiguous(self):
        # A response dict that simply never sets this key (e.g. a code
        # path not yet audited, or the proxy-mode delegation path) must
        # fail closed -- never crash, never be treated as trusted.
        self.assertFalse(resolve_boundary_trust({"final_response": "hi"}))
        self.assertFalse(resolve_boundary_trust({}))

    def test_non_mapping_input_is_untrusted(self):
        self.assertFalse(resolve_boundary_trust(None))
        self.assertFalse(resolve_boundary_trust("not a dict"))
        self.assertFalse(resolve_boundary_trust(["not", "a", "dict"]))

    def test_field_name_matches_real_runtime_producer(self):
        # Pin the exact key name this function reads against the literal
        # string used at all three run.py producer sites, so a rename on
        # either side is caught immediately instead of silently going
        # untrusted everywhere.
        import re

        run_py = (_OVERLAY_ROOT / "gateway" / "run.py").read_text(encoding="utf-8")
        occurrences = re.findall(r'"phase3a_boundary_trusted":\s*[^\n]+', run_py)
        self.assertGreaterEqual(len(occurrences), 3, "expected 3 producer sites in run.py")
        self.assertTrue(resolve_boundary_trust({"phase3a_boundary_trusted": True}))


class ObserveAndPersistTurnTests(_StoreTestCase):
    def test_1_genuine_first_turn_offset_zero_trusted(self):
        response = {
            "final_response": "answer",
            "model": "claude-sonnet-5",
            "messages": _foundry_messages(),
            "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        ok = observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
            platform_assistant_message_id="asst-1", question_text="q", feedback_eligible=True,
            store=self.store,
        )
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["retrieval_observation_status"], "complete")
        self.assertEqual(len(self.store.list_retrieval_runs_for_turn("turn-1")), 1)

    def test_2_subsequent_turn_offset_positive_scans_only_current(self):
        old_call = {
            "role": "assistant",
            "tool_calls": [{"id": "old", "function": {"name": "terminal", "arguments": json.dumps({"command": f"python3 {FOUNDARY_SCRIPT} 'old'"})}}],
        }
        old_result = {"role": "tool", "tool_call_id": "old", "content": json.dumps({"output": None, "exit_code": 0, "error": None})}
        messages = [old_call, old_result, {"role": "user", "content": "new question"}, {"role": "assistant", "content": "no tools this time"}]
        response = {
            "final_response": "answer", "model": None, "messages": messages,
            "history_offset": 2, "phase3a_boundary_trusted": True,
        }
        ok = observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-2", platform_user_id="user-1", platform_user_message_id="msg-2",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-2")
        self.assertEqual(turn["retrieval_observation_status"], "complete")
        self.assertEqual(self.store.list_retrieval_runs_for_turn("turn-2"), [])

    def test_3_old_history_foundry_current_none_not_misattributed(self):
        # Same fixture as #2 -- the key assertion is that the OLD call's
        # tool_call_id ("old") never appears in this turn's retrieval_runs.
        old_call = {
            "role": "assistant",
            "tool_calls": [{"id": "old", "function": {"name": "terminal", "arguments": json.dumps({"command": f"python3 {FOUNDARY_SCRIPT} 'old'"})}}],
        }
        old_result = {"role": "tool", "tool_call_id": "old", "content": json.dumps({"output": json.dumps({"schema_version": "foundry-iq-result-v2", "ok": True, "request_attempted": True, "error_code": None, "question": "old", "http_status": None, "documents": [], "references": [], "activity": []}), "exit_code": 0, "error": None})}
        messages = [old_call, old_result, {"role": "user", "content": "q2"}, {"role": "assistant", "content": "no tools"}]
        response = {
            "final_response": "answer", "model": None, "messages": messages,
            "history_offset": 2, "phase3a_boundary_trusted": True,
        }
        observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-3", platform_user_id="user-1", platform_user_message_id="msg-3",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        runs = self.store.list_retrieval_runs_for_turn("turn-3")
        self.assertEqual(runs, [])
        self.assertNotIn("old", [r["tool_call_id"] for r in runs])

    def test_4_compacted_in_place_untrusted(self):
        response = {
            "final_response": "answer", "model": None,
            "messages": _foundry_messages(), "history_offset": 0,
            "compacted_in_place": True,
            "phase3a_boundary_trusted": False,  # set by run.py precisely because compacted_in_place was True
        }
        observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-4", platform_user_id="user-1", platform_user_message_id="msg-4",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        turn = self.store.get_turn("turn-4")
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "untrusted_turn_boundary")
        self.assertEqual(self.store.list_retrieval_runs_for_turn("turn-4"), [])

    def test_5_session_split_untrusted(self):
        response = {
            "final_response": "answer", "model": None,
            "messages": _foundry_messages(), "history_offset": 0,
            "session_id": "sess-1-rotated",
            "phase3a_boundary_trusted": False,  # set by run.py precisely because _session_was_split was True
        }
        observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-5", platform_user_id="user-1", platform_user_message_id="msg-5",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        turn = self.store.get_turn("turn-5")
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "untrusted_turn_boundary")

    def test_6_missing_session_id_field_not_ambiguous(self):
        # No "session_id" key at all on the response (e.g. the
        # inactivity-timeout synthetic dict, or the proxy-mode path) --
        # must not silently be treated as trusted just because there is no
        # session_id to "differ" from anything; the explicit
        # phase3a_boundary_trusted field is the only thing that matters,
        # and this dict does not set it (defaults False).
        response = {
            "final_response": "answer", "model": None,
            "messages": _foundry_messages(), "history_offset": 0,
        }
        self.assertNotIn("session_id", response)
        self.assertNotIn("phase3a_boundary_trusted", response)
        observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-6", platform_user_id="user-1", platform_user_message_id="msg-6",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        turn = self.store.get_turn("turn-6")
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "untrusted_turn_boundary")

    def test_returns_true_on_success(self):
        response = {
            "final_response": "answer", "model": None,
            "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True,
        }
        ok = observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-7", platform_user_id="user-1", platform_user_message_id="msg-7",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=self.store,
        )
        self.assertIs(ok, True)

    def test_returns_false_never_none_when_store_unavailable(self):
        # store=None falls through to the process-wide get_store()
        # singleton, which may resolve successfully on a machine where the
        # default DB path happens to be writable -- force the "unavailable"
        # branch deterministically instead of relying on the real default
        # path failing, so this test can't spuriously write to it.
        from unittest import mock

        import tools.retrieval_runtime as rr

        response = {"final_response": "answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        with mock.patch.object(rr, "get_store", return_value=None):
            result = observe_and_persist_turn(
                response,
                session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
                turn_id="turn-8", platform_user_id="user-1", platform_user_message_id="msg-8",
                platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
                store=None,
            )
        self.assertIs(result, False)

    def test_returns_false_never_none_on_exploding_store(self):
        response = {"final_response": "answer", "messages": _foundry_messages(), "history_offset": 0, "phase3a_boundary_trusted": True}
        result = observe_and_persist_turn(
            response,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-9", platform_user_id="user-1", platform_user_message_id="msg-9",
            platform_assistant_message_id=None, question_text="q", feedback_eligible=True,
            store=_ExplodingStore(),
        )
        self.assertIs(result, False)

    def test_never_raises_on_malformed_response(self):
        for bad_response in (
            {"messages": "not a list", "history_offset": "not an int", "phase3a_boundary_trusted": True},
            {},
            {"phase3a_boundary_trusted": True},
        ):
            with self.subTest(bad_response=bad_response):
                try:
                    result = observe_and_persist_turn(
                        bad_response,
                        session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
                        turn_id=f"turn-bad-{id(bad_response)}", platform_user_id="user-1",
                        platform_user_message_id="msg-bad", platform_assistant_message_id=None,
                        question_text="q", feedback_eligible=True, store=self.store,
                    )
                except Exception as exc:  # pragma: no cover - failure path
                    self.fail(f"observe_and_persist_turn raised: {exc}")
                self.assertIsInstance(result, bool)


class NonStreamingContextSplitTests(_StoreTestCase):
    """Covers the run.py-parse / base.py-persist split used for the
    non-streaming delivery path."""

    def test_context_built_then_persisted_matches_combined_path(self):
        agent_result = {
            "final_response": "answer", "model": "claude-sonnet-5",
            "messages": _foundry_messages(), "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
        )
        self.assertIsNotNone(ctx)
        # ctx must be JSON-safe (plain dict/list/primitives) -- this is
        # what actually gets parked on MessageEvent.metadata.
        json.dumps(ctx)
        self.assertEqual(ctx["model"], "claude-sonnet-5")

        ok = persist_turn_observation_context(
            ctx,
            platform_assistant_message_id="asst-1",
            question_text="question text",
            answer_text="answer text",
            feedback_eligible=True,
            store=self.store,
        )
        self.assertIs(ok, True)
        turn = self.store.get_turn("turn-1")
        self.assertIsNotNone(turn)
        self.assertEqual(turn["answer_text"], "answer text")
        self.assertEqual(turn["platform_assistant_message_id"], "asst-1")
        runs = self.store.list_retrieval_runs_for_turn("turn-1")
        self.assertEqual(len(runs), 1)

    def test_persist_never_raises_on_bad_context_and_returns_false(self):
        try:
            result = persist_turn_observation_context(
                {"garbage": True},
                platform_assistant_message_id=None,
                question_text="q",
                answer_text="a",
                feedback_eligible=True,
                store=self.store,
            )
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"persist_turn_observation_context raised: {exc}")
        self.assertIs(result, False)
        # No turn should have been created from a garbage context.
        self.assertEqual(self.store.list_turns_for_session("sess-1"), [])

    def test_build_context_never_raises_on_malformed_agent_result(self):
        result = build_turn_observation_context(
            {"messages": "not a list", "history_offset": "not an int"},
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
        )
        # Malformed messages/offset still yield a valid (untrusted) context,
        # not an exception -- build_turn_observation_context only returns
        # None on an *internal* failure, not on parser-level untrusted input.
        self.assertIsNotNone(result)
        self.assertEqual(result["observation"]["turn_observation_status"], "unavailable")

    def test_build_context_returns_none_on_internal_failure(self):
        result = build_turn_observation_context(
            None,  # not a Mapping at all -- triggers the internal except
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
        )
        self.assertIsNone(result)


class GetStoreRetryTests(unittest.TestCase):
    """get_store()'s process-wide singleton must retry after a failed
    construction attempt, never latch permanently -- a transient failure
    (e.g. the data directory not yet mounted at process start) must not
    wedge every turn for the rest of the process's lifetime."""

    def setUp(self):
        import tools.retrieval_runtime as rr
        self.rr = rr
        self._saved_singleton = rr._store_singleton
        rr._store_singleton = None

    def tearDown(self):
        self.rr._store_singleton = self._saved_singleton

    def test_failed_construction_returns_none_without_raising(self):
        from unittest import mock

        with mock.patch.object(
            self.rr, "FeedbackStoreV2", side_effect=RuntimeError("simulated")
        ):
            self.assertIsNone(self.rr.get_store())

    def test_next_call_after_failure_retries_and_can_succeed(self):
        from unittest import mock

        calls = {"n": 0}

        def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated transient failure")
            return mock.sentinel.store

        with mock.patch.object(self.rr, "FeedbackStoreV2", side_effect=_flaky):
            first = self.rr.get_store()
            self.assertIsNone(first)
            second = self.rr.get_store()
            self.assertIs(second, mock.sentinel.store)

    def test_successful_construction_is_cached_not_reconstructed(self):
        from unittest import mock

        with mock.patch.object(
            self.rr, "FeedbackStoreV2", return_value=mock.sentinel.store
        ) as constructor:
            first = self.rr.get_store()
            second = self.rr.get_store()
        self.assertIs(first, mock.sentinel.store)
        self.assertIs(second, mock.sentinel.store)
        constructor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
