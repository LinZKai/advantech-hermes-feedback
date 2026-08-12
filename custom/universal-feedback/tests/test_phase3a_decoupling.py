"""Phase 3A / Blocker 2 behavioral decoupling tests.

test_phase3a_wiring.py proves the *source text* of run.py/base.py has the
right shape (right flag names, no dependency on universal_feedback_handled,
etc.) since those files can't be executed directly in this repo (no
proprietary Hermes agent runtime available -- see AGENTS.md). This file
proves the *behavior* is actually correct, by re-running the same call
sequence those source blocks perform -- independent try/except around
tools.retrieval_runtime, then independent try/except around a stand-in
legacy feedback hook -- against the real tools.retrieval_runtime functions
and a real temporary SQLite-backed FeedbackStoreV2, with a stand-in legacy
hook that can be configured to succeed, raise, or simply not apply. The
harness functions below are a deliberately small, literal mirror of the
actual source (cross-checked structurally by test_phase3a_wiring.py),
including the attempted/persisted flag split introduced in this revision,
not a reimplementation of unrelated logic.
"""
from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.retrieval_runtime import (  # noqa: E402
    build_turn_observation_context,
    observe_and_persist_turn,
    persist_turn_observation_context,
)


class _ExplodingStore:
    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated v2 failure in {name}")
        return _boom


class _RecordingHook:
    """Stand-in for the legacy universal-feedback send call
    (adapter.send_universal_feedback / self.send_universal_feedback)."""

    def __init__(self, *, raises: bool = False):
        self.raises = raises
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.raises:
            raise RuntimeError("simulated legacy feedback hook failure")


def _simulate_streaming_turn(*, response: dict, legacy_hook, store) -> str:
    """Literal mirror of run.py's _run_agent_inner block ordering (see
    test_phase3a_wiring.py's IndependenceFromLegacyFeedbackTests for the
    structural proof this matches the real source): Phase 3A first, in its
    own try/except, gated on already_sent + not already
    phase3a_turn_persistence_attempted; then the legacy hook, in its own
    try/except, entirely independent of whether Phase 3A ran or succeeded.
    "attempted" is set before the call (blocks a retry regardless of
    outcome); "persisted" is set after, from the helper's real boolean
    return value. Returns the final answer text, exactly like the real
    function always still delivers it regardless of either side's outcome.
    """
    try:
        if (
            isinstance(response, dict)
            and bool(response.get("already_sent"))
            and not response.get("phase3a_turn_persistence_attempted")
        ):
            response["phase3a_turn_persistence_attempted"] = True
            response["phase3a_turn_persisted"] = observe_and_persist_turn(
                response,
                session_id="sess-1",
                platform="telegram",
                platform_chat_id="chat-1",
                turn_id=f"telegram:chat-1:{response.get('_msg_id', 'm1')}",
                platform_user_id="user-1",
                platform_user_message_id=str(response.get("_msg_id", "m1")),
                platform_assistant_message_id="asst-1",
                question_text="question",
                feedback_eligible=bool(response.get("_feedback_eligible", True)),
                store=store,
            )
    except Exception:
        pass  # mirrors: logger.debug("Phase 3A retrieval observation failed", exc_info=True)

    try:
        if legacy_hook is not None:
            legacy_hook()
    except Exception:
        pass  # mirrors: logger.debug("Universal feedback hook failed", exc_info=True)

    return response.get("final_response")


def _simulate_non_streaming_turn(
    *, agent_result: dict, event_metadata: dict, delivery_success: bool,
    legacy_hook_applicable: bool, legacy_hook, store, turn_id: str,
) -> str:
    """Literal mirror of the run.py parse step + base.py persist step +
    base.py legacy hook, in that order, each independently guarded. The
    parse step's own gate checks "attempted" (never "persisted"), matching
    run.py: a streaming attempt that failed still blocks a non-streaming
    retry -- this design deliberately does not retry across paths."""
    # run.py: _handle_message_with_agent's parse-only step.
    try:
        if isinstance(agent_result, dict) and not agent_result.get("phase3a_turn_persistence_attempted"):
            ctx = build_turn_observation_context(
                agent_result,
                session_id="sess-1",
                platform="telegram",
                platform_chat_id="chat-1",
                turn_id=turn_id,
                platform_user_id="user-1",
                platform_user_message_id=turn_id.rsplit(":", 1)[-1],
            )
            if ctx is not None:
                event_metadata["phase3a_turn_context"] = ctx
    except Exception:
        pass

    text_content = agent_result.get("final_response") or ""

    # base.py: persist step, gated on delivery confirmation.
    try:
        ctx = event_metadata.get("phase3a_turn_context")
        if ctx is not None and delivery_success:
            persist_turn_observation_context(
                ctx,
                platform_assistant_message_id="asst-1",
                question_text="question",
                answer_text=text_content,
                feedback_eligible=True,
                store=store,
            )
    except Exception:
        pass

    # base.py: legacy feedback hook, independent of the block above.
    try:
        if legacy_hook_applicable and legacy_hook is not None:
            legacy_hook()
    except Exception:
        pass

    return text_content


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()


def _trusted_response(**overrides) -> dict:
    base = {
        "final_response": "the answer",
        "already_sent": True,
        "messages": [],
        "history_offset": 0,
        "phase3a_boundary_trusted": True,
    }
    base.update(overrides)
    return base


class StreamingDecouplingTests(_StoreTestCase):
    def test_legacy_hook_succeeds_v2_written_once(self):
        hook = _RecordingHook(raises=False)
        response = _trusted_response(_msg_id="m1")
        result = _simulate_streaming_turn(response=response, legacy_hook=hook, store=self.store)
        self.assertEqual(result, "the answer")
        self.assertEqual(hook.calls, 1)
        self.assertTrue(response["phase3a_turn_persistence_attempted"])
        self.assertTrue(response["phase3a_turn_persisted"])
        turns = self.store.list_turns_for_session("sess-1")
        self.assertEqual(len(turns), 1)

    def test_legacy_hook_raises_v2_still_written_and_final_answer_present(self):
        hook = _RecordingHook(raises=True)
        response = _trusted_response(_msg_id="m2")
        result = _simulate_streaming_turn(response=response, legacy_hook=hook, store=self.store)
        self.assertEqual(result, "the answer")  # final answer still delivered
        self.assertEqual(hook.calls, 1)  # hook was still attempted
        self.assertTrue(response["phase3a_turn_persisted"])
        turns = self.store.list_turns_for_session("sess-1")
        self.assertEqual(len(turns), 1)  # v2 still got its row despite the hook raising

    def test_v2_store_failure_legacy_hook_still_called_final_answer_present(self):
        hook = _RecordingHook(raises=False)
        response = _trusted_response(_msg_id="m3")
        result = _simulate_streaming_turn(response=response, legacy_hook=hook, store=_ExplodingStore())
        self.assertEqual(result, "the answer")
        self.assertEqual(hook.calls, 1)  # legacy hook unaffected by v2's failure
        # attempted=True (the call was made), persisted must NOT be True.
        self.assertTrue(response["phase3a_turn_persistence_attempted"])
        self.assertFalse(response["phase3a_turn_persisted"])

    def test_legacy_hook_not_applicable_v2_still_written(self):
        # No hook at all (e.g. platform not eligible for feedback UI, or
        # send_universal_feedback unavailable) -- a delivered final answer
        # must still reach v2.
        response = _trusted_response(_msg_id="m4")
        result = _simulate_streaming_turn(response=response, legacy_hook=None, store=self.store)
        self.assertEqual(result, "the answer")
        turns = self.store.list_turns_for_session("sess-1")
        self.assertEqual(len(turns), 1)

    def test_universal_feedback_handled_absent_does_not_block_v2(self):
        # No universal_feedback_handled key at all on the response dict --
        # v2 must not require it to be present, let alone True.
        response = _trusted_response(_msg_id="m5")
        self.assertNotIn("universal_feedback_handled", response)
        _simulate_streaming_turn(response=response, legacy_hook=_RecordingHook(), store=self.store)
        turns = self.store.list_turns_for_session("sess-1")
        self.assertEqual(len(turns), 1)

    def test_not_delivered_yet_does_not_write_v2(self):
        # already_sent False (e.g. streaming did not actually deliver) --
        # the streaming side must not attempt v2 at all; this is the
        # "delivery confirmed" boundary, not a feedback-eligibility one.
        response = _trusted_response(already_sent=False, _msg_id="m6")
        _simulate_streaming_turn(response=response, legacy_hook=_RecordingHook(), store=self.store)
        turns = self.store.list_turns_for_session("sess-1")
        self.assertEqual(len(turns), 0)
        self.assertNotIn("phase3a_turn_persistence_attempted", response)

    def test_untrusted_boundary_still_attempts_but_writes_unavailable_turn(self):
        # already_sent True but the gateway did not assert boundary trust
        # (e.g. compaction/session-split happened) -- persistence is still
        # ATTEMPTED (delivery was confirmed) and a Turn row is written, but
        # with retrieval_observation_status="unavailable" and no Retrieval
        # rows, never a fabricated one.
        response = _trusted_response(_msg_id="m7", phase3a_boundary_trusted=False)
        _simulate_streaming_turn(response=response, legacy_hook=_RecordingHook(), store=self.store)
        self.assertTrue(response["phase3a_turn_persistence_attempted"])
        self.assertTrue(response["phase3a_turn_persisted"])
        turn = self.store.get_turn("telegram:chat-1:m7")
        self.assertIsNotNone(turn)
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "untrusted_turn_boundary")
        self.assertEqual(self.store.list_retrieval_runs_for_turn("telegram:chat-1:m7"), [])


class NonStreamingDecouplingTests(_StoreTestCase):
    def test_legacy_hook_succeeds_v2_written_once(self):
        hook = _RecordingHook(raises=False)
        agent_result = {"final_response": "the answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        event_metadata = {}
        result = _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata=event_metadata,
            delivery_success=True, legacy_hook_applicable=True, legacy_hook=hook,
            store=self.store, turn_id="telegram:chat-1:n1",
        )
        self.assertEqual(result, "the answer")
        self.assertEqual(hook.calls, 1)
        self.assertEqual(len(self.store.list_turns_for_session("sess-1")), 1)

    def test_legacy_hook_raises_v2_still_written(self):
        hook = _RecordingHook(raises=True)
        agent_result = {"final_response": "the answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        result = _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata={},
            delivery_success=True, legacy_hook_applicable=True, legacy_hook=hook,
            store=self.store, turn_id="telegram:chat-1:n2",
        )
        self.assertEqual(result, "the answer")
        self.assertEqual(hook.calls, 1)
        self.assertEqual(len(self.store.list_turns_for_session("sess-1")), 1)

    def test_v2_failure_legacy_hook_still_called(self):
        hook = _RecordingHook(raises=False)
        agent_result = {"final_response": "the answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        result = _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata={},
            delivery_success=True, legacy_hook_applicable=True, legacy_hook=hook,
            store=_ExplodingStore(), turn_id="telegram:chat-1:n3",
        )
        self.assertEqual(result, "the answer")
        self.assertEqual(hook.calls, 1)

    def test_legacy_hook_not_applicable_v2_still_written(self):
        agent_result = {"final_response": "the answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        result = _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata={},
            delivery_success=True, legacy_hook_applicable=False, legacy_hook=_RecordingHook(),
            store=self.store, turn_id="telegram:chat-1:n4",
        )
        self.assertEqual(result, "the answer")
        self.assertEqual(len(self.store.list_turns_for_session("sess-1")), 1)

    def test_delivery_not_confirmed_does_not_write_v2(self):
        agent_result = {"final_response": "the answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata={},
            delivery_success=False, legacy_hook_applicable=True, legacy_hook=_RecordingHook(),
            store=self.store, turn_id="telegram:chat-1:n5",
        )
        self.assertEqual(len(self.store.list_turns_for_session("sess-1")), 0)


class NoDoubleWriteTests(_StoreTestCase):
    def test_streaming_claims_turn_non_streaming_does_not_double_write(self):
        # Streaming path runs first (as it does in the real call graph --
        # _run_agent_inner's block executes before _run_agent_inner
        # returns up through _handle_message_with_agent) and claims the
        # turn via phase3a_turn_persistence_attempted on the shared
        # response dict.
        turn_id = "telegram:chat-1:shared1"
        response = _trusted_response(_msg_id="shared1")
        _simulate_streaming_turn(response=response, legacy_hook=_RecordingHook(), store=self.store)
        self.assertTrue(response.get("phase3a_turn_persistence_attempted"))
        self.assertTrue(response.get("phase3a_turn_persisted"))

        # agent_result IS the same dict as `response` in the real call
        # graph (_handle_message_with_agent's agent_result is exactly what
        # _run_agent_inner returned) -- simulate that identity directly.
        agent_result = response
        event_metadata = {}
        _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata=event_metadata,
            delivery_success=True, legacy_hook_applicable=True, legacy_hook=_RecordingHook(),
            store=self.store, turn_id=turn_id,
        )
        # The non-streaming parse step must have skipped entirely (no
        # context stashed), because attempted was already True.
        self.assertNotIn("phase3a_turn_context", event_metadata)

        turns = self.store.list_turns_for_session("sess-1")
        self.assertEqual(len(turns), 1)

    def test_streaming_attempt_failed_non_streaming_still_does_not_retry(self):
        # Streaming attempted but the store failed (persisted=False) --
        # the non-streaming path must NOT retry (attempted is still True),
        # per the design's explicit no-cross-path-retry choice.
        turn_id = "telegram:chat-1:sharedfail"
        response = _trusted_response(_msg_id="sharedfail")
        _simulate_streaming_turn(response=response, legacy_hook=_RecordingHook(), store=_ExplodingStore())
        self.assertTrue(response.get("phase3a_turn_persistence_attempted"))
        self.assertFalse(response.get("phase3a_turn_persisted"))

        event_metadata = {}
        _simulate_non_streaming_turn(
            agent_result=response, event_metadata=event_metadata,
            delivery_success=True, legacy_hook_applicable=True, legacy_hook=_RecordingHook(),
            store=self.store, turn_id=turn_id,
        )
        self.assertNotIn("phase3a_turn_context", event_metadata)
        # Not retried, and the earlier failed attempt wrote nothing either.
        self.assertEqual(self.store.list_turns_for_session("sess-1"), [])

    def test_non_streaming_only_path_writes_exactly_once(self):
        # The streaming side never claimed this turn (already_sent stayed
        # False, e.g. streaming wasn't used) -- only the non-streaming
        # path should write, exactly once.
        agent_result = {"final_response": "the answer", "messages": [], "history_offset": 0, "phase3a_boundary_trusted": True}
        event_metadata = {}
        _simulate_non_streaming_turn(
            agent_result=agent_result, event_metadata=event_metadata,
            delivery_success=True, legacy_hook_applicable=True, legacy_hook=_RecordingHook(),
            store=self.store, turn_id="telegram:chat-1:onlynonstream",
        )
        self.assertIn("phase3a_turn_context", event_metadata)
        self.assertEqual(len(self.store.list_turns_for_session("sess-1")), 1)


if __name__ == "__main__":
    unittest.main()
