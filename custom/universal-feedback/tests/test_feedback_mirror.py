"""Phase 3B tests: tools.feedback_mirror.

Real temp-file SQLite DB via FeedbackStoreV2, matching the Windows-safe
cleanup convention used throughout this test suite (sqlite3 connections
only fully close on GC).
"""
from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_mirror import (  # noqa: E402
    mirror_negative_feedback,
    mirror_positive_feedback,
    mirror_suggestion,
)
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.universal_feedback import POLICY_VERSION  # noqa: E402

CANARY_SECRET = "sk-CANARY-SECRET-DO-NOT-LEAK-1234567890"


class _ExplodingStore:
    """A store double whose every method raises -- proves mirror failures
    never propagate."""

    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated v2 failure in {name}")
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

    def _new_turn(self, turn_id, *, session_id="sess-1", case_id="case-1", **overrides):
        self.store.create_or_update_session(session_id, "telegram")
        if self.store.get_case(case_id) is None:
            self.store.create_case(case_id, session_id)
        kwargs = dict(
            platform_user_id="user-1",
            platform_user_message_id=f"{turn_id}-msg",
            question_text="What is the reset procedure?",
            answer_text="Hold the reset button for 5 seconds.",
            feedback_eligible=True,
        )
        kwargs.update(overrides)
        self.store.create_turn(turn_id, case_id, **kwargs)
        return turn_id


class PositiveMirrorTests(_StoreTestCase):
    def test_creates_row_with_correct_final_state(self):
        turn_id = self._new_turn("turn-1")
        ok = mirror_positive_feedback(turn_id, store=self.store)
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["helpful"], 1)
        self.assertIsNone(row["reason_code"])
        self.assertIsNone(row["suggestion_text"])
        self.assertIsNotNone(row["submitted_at"])
        self.assertEqual(row["feedback_policy_version"], POLICY_VERSION)

    def test_duplicate_positive_does_not_create_second_row(self):
        turn_id = self._new_turn("turn-1")
        first = mirror_positive_feedback(turn_id, store=self.store)
        second = mirror_positive_feedback(turn_id, store=self.store)
        self.assertTrue(first)
        self.assertFalse(second)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["helpful"], 1)  # still the original, legal state

    def test_missing_turn_id_is_noop(self):
        for bad in (None, "", 0):
            with self.subTest(bad=bad):
                ok = mirror_positive_feedback(bad, store=self.store)
                self.assertFalse(ok)

    def test_unknown_turn_id_fk_failure_returns_false(self):
        # No turns row for this id at all -- FK constraint rejects the
        # INSERT; must return False, never raise.
        ok = mirror_positive_feedback("turn-does-not-exist", store=self.store)
        self.assertFalse(ok)

    def test_store_failure_returns_false_never_raises(self):
        try:
            ok = mirror_positive_feedback("turn-1", store=_ExplodingStore())
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"mirror_positive_feedback raised: {exc}")
        self.assertFalse(ok)

    def test_question_answer_text_never_touched(self):
        # Sanity: mirror_positive_feedback takes no question/answer
        # parameter at all -- it cannot leak or mis-associate text.
        turn_id = self._new_turn("turn-1", question_text="secret q " + CANARY_SECRET, answer_text="secret a")
        mirror_positive_feedback(turn_id, store=self.store)
        row = self.store.get_feedback(turn_id)
        # feedback row itself has no question/answer columns at all.
        self.assertNotIn("question_text", row.keys())


class NegativeMirrorTests(_StoreTestCase):
    def test_creates_row_with_correct_final_state(self):
        turn_id = self._new_turn("turn-1")
        ok = mirror_negative_feedback(turn_id, "incomplete", store=self.store)
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["helpful"], 0)
        self.assertEqual(row["reason_code"], "incomplete")
        self.assertIsNone(row["suggestion_text"])
        self.assertIsNotNone(row["submitted_at"])
        self.assertEqual(row["feedback_policy_version"], POLICY_VERSION)

    def test_invalid_reason_code_rejected(self):
        turn_id = self._new_turn("turn-1")
        ok = mirror_negative_feedback(turn_id, "not-a-real-reason", store=self.store)
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_reason_taxonomy_matches_existing_reason_codes(self):
        from tools.feedback_callbacks import REASON_CODES

        for i, code in enumerate(REASON_CODES):
            turn_id = self._new_turn(f"turn-{code}")
            with self.subTest(code=code):
                ok = mirror_negative_feedback(turn_id, code, store=self.store)
                self.assertTrue(ok)

    def test_duplicate_negative_callback_does_not_duplicate(self):
        turn_id = self._new_turn("turn-1")
        first = mirror_negative_feedback(turn_id, "incomplete", store=self.store)
        second = mirror_negative_feedback(turn_id, "unclear", store=self.store)
        self.assertTrue(first)
        self.assertFalse(second)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["reason_code"], "incomplete")  # first write kept

    def test_missing_turn_id_is_noop(self):
        ok = mirror_negative_feedback(None, "incomplete", store=self.store)
        self.assertFalse(ok)

    def test_store_failure_returns_false_never_raises(self):
        try:
            ok = mirror_negative_feedback("turn-1", "incomplete", store=_ExplodingStore())
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"mirror_negative_feedback raised: {exc}")
        self.assertFalse(ok)


class SuggestionMirrorTests(_StoreTestCase):
    def test_first_suggestion_updates_the_negative_row(self):
        turn_id = self._new_turn("turn-1")
        mirror_negative_feedback(turn_id, "incomplete", store=self.store)
        ok = mirror_suggestion(turn_id, "Please add more detail.", store=self.store)
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["suggestion_text"], "Please add more detail.")

    def test_second_suggestion_does_not_overwrite_first(self):
        turn_id = self._new_turn("turn-1")
        mirror_negative_feedback(turn_id, "incomplete", store=self.store)
        first = mirror_suggestion(turn_id, "first suggestion", store=self.store)
        second = mirror_suggestion(turn_id, "second suggestion", store=self.store)
        self.assertTrue(first)
        self.assertFalse(second)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["suggestion_text"], "first suggestion")

    def test_positive_turn_cannot_receive_suggestion(self):
        turn_id = self._new_turn("turn-1")
        mirror_positive_feedback(turn_id, store=self.store)
        ok = mirror_suggestion(turn_id, "should not apply", store=self.store)
        self.assertFalse(ok)
        row = self.store.get_feedback(turn_id)
        self.assertIsNone(row["suggestion_text"])

    def test_orphan_suggestion_no_feedback_row_creates_nothing(self):
        turn_id = self._new_turn("turn-1")
        # No positive/negative feedback mirrored yet -- no feedback row at all.
        ok = mirror_suggestion(turn_id, "orphan", store=self.store)
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_missing_turn_id_is_noop(self):
        ok = mirror_suggestion("", "text", store=self.store)
        self.assertFalse(ok)

    def test_store_failure_returns_false_never_raises(self):
        try:
            ok = mirror_suggestion("turn-1", "text", store=_ExplodingStore())
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"mirror_suggestion raised: {exc}")
        self.assertFalse(ok)


class TurnLinkageTests(_StoreTestCase):
    """No heuristic linkage: only the exact turn_id supplied by the caller
    is ever written to -- never "latest Turn in session", never inferred
    from anything else."""

    def test_binds_to_the_exact_turn_supplied_not_the_newest(self):
        older = self._new_turn("turn-older", platform_user_message_id="m1")
        newer = self._new_turn("turn-newer", platform_user_message_id="m2")

        # Feedback pressed on the OLDER turn, even though a newer turn now
        # exists in the same session.
        ok = mirror_positive_feedback(older, store=self.store)
        self.assertTrue(ok)

        self.assertIsNotNone(self.store.get_feedback(older))
        self.assertIsNone(self.store.get_feedback(newer))

    def test_next_turn_started_does_not_reroute_pending_feedback(self):
        first = self._new_turn("turn-1", platform_user_message_id="m1")
        # A second Turn begins in the same session/chat before the user
        # answers feedback on the first -- the caller (adapter.py) always
        # resolves turn_id from the specific feedback_runs row it looked
        # up by run_id/feedback_message_id, so this must still bind first.
        self._new_turn("turn-2", platform_user_message_id="m2")

        ok = mirror_negative_feedback(first, "incomplete", store=self.store)
        self.assertTrue(ok)
        self.assertEqual(self.store.get_feedback(first)["reason_code"], "incomplete")
        self.assertIsNone(self.store.get_feedback("turn-2"))

    def test_different_sessions_never_cross_bind(self):
        turn_a = self._new_turn("turn-a", session_id="sess-a", case_id="case-a")
        turn_b = self._new_turn("turn-b", session_id="sess-b", case_id="case-b")

        mirror_positive_feedback(turn_a, store=self.store)
        mirror_negative_feedback(turn_b, "unclear", store=self.store)

        self.assertEqual(self.store.get_feedback(turn_a)["helpful"], 1)
        self.assertEqual(self.store.get_feedback(turn_b)["helpful"], 0)

    def test_unresolvable_turn_id_fails_closed_no_fake_row(self):
        # Simulates the caller finding no turn_key at all on the
        # feedback_runs row (e.g. an older/legacy row predating Phase 3A).
        ok = mirror_positive_feedback(None, store=self.store)
        self.assertFalse(ok)
        # No turns exist yet in this fresh store, and no feedback row was
        # fabricated for any placeholder id either.
        self.assertEqual(self.store.list_turns_for_session("sess-1"), [])


class StoreUnavailableTests(unittest.TestCase):
    def test_get_store_unavailable_returns_false_never_raises(self):
        import tools.feedback_mirror as fm

        with mock.patch.object(fm, "get_store", return_value=None):
            for fn, args in (
                (mirror_positive_feedback, ("turn-1",)),
                (mirror_negative_feedback, ("turn-1", "incomplete")),
                (mirror_suggestion, ("turn-1", "text")),
            ):
                with self.subTest(fn=fn.__name__):
                    try:
                        result = fn(*args)
                    except Exception as exc:  # pragma: no cover - failure path
                        self.fail(f"{fn.__name__} raised: {exc}")
                    self.assertIs(result, False)


if __name__ == "__main__":
    unittest.main()
