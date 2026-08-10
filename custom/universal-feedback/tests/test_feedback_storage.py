"""Tests for tools.feedback_storage.FeedbackStore.submit_negative() and the
surrounding first-write-wins / three-column invariant it must preserve.

Run locally with:
    python -m unittest discover -s custom/universal-feedback/tests -v
No production dependency is added; this uses only the standard library
(sqlite3 against a temp file per test).
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

from tools.feedback_callbacks import REASON_CODES  # noqa: E402
from tools.feedback_storage import FeedbackStore  # noqa: E402


class FeedbackStoreSubmitNegativeTests(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: FeedbackStore._connect() uses sqlite3's
        # `with conn:` only to commit/rollback (stdlib sqlite3 never closes
        # the connection on __exit__), so short-lived connections are only
        # closed when garbage-collected. On Windows an open sqlite3 file
        # handle blocks deleting the temp dir; gc.collect() in tearDown
        # closes them deterministically, and ignore_cleanup_errors is a
        # belt-and-suspenders fallback. This is test-harness hygiene only —
        # production _connect() is intentionally left untouched.
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStore(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _new_run(self, run_id: str) -> None:
        created = self.store.create_run(run_id, "chat-1")
        self.assertTrue(created)

    def test_each_allowlisted_reason_can_be_submitted(self):
        for idx, code in enumerate(REASON_CODES):
            run_id = f"run-{idx}"
            with self.subTest(code=code):
                self._new_run(run_id)
                accepted = self.store.submit_negative(run_id, code)
                self.assertTrue(accepted)

    def test_successful_submit_sets_all_three_columns(self):
        run_id = "run-success"
        self._new_run(run_id)
        accepted = self.store.submit_negative(run_id, "incomplete")
        self.assertTrue(accepted)

        row = self.store.get(run_id)
        self.assertEqual(row["helpful"], 0)
        self.assertEqual(row["reason_code"], "incomplete")
        self.assertIsNotNone(row["submitted_at"])

    def test_illegal_reason_is_rejected_and_not_written(self):
        run_id = "run-illegal"
        self._new_run(run_id)
        accepted = self.store.submit_negative(run_id, "bogus_reason")
        self.assertFalse(accepted)

        row = self.store.get(run_id)
        self.assertIsNone(row["helpful"])
        self.assertIsNone(row["reason_code"])
        self.assertIsNone(row["submitted_at"])

    def test_second_submission_on_same_row_is_rejected_and_does_not_overwrite(self):
        run_id = "run-double-submit"
        self._new_run(run_id)
        first = self.store.submit_negative(run_id, "incorrect")
        self.assertTrue(first)
        row_after_first = self.store.get(run_id)

        second = self.store.submit_negative(run_id, "other")
        self.assertFalse(second)

        row_after_second = self.store.get(run_id)
        self.assertEqual(row_after_second["reason_code"], "incorrect")
        self.assertEqual(row_after_second["submitted_at"], row_after_first["submitted_at"])

    def test_row_with_existing_positive_feedback_cannot_become_negative(self):
        run_id = "run-already-positive"
        self._new_run(run_id)
        positive = self.store.submit_helpful(run_id, True)
        self.assertTrue(positive)

        accepted = self.store.submit_negative(run_id, "unclear")
        self.assertFalse(accepted)

        row = self.store.get(run_id)
        self.assertEqual(row["helpful"], 1)
        self.assertIsNone(row["reason_code"])

    def test_row_with_existing_submitted_at_cannot_be_overwritten(self):
        # Legacy /feedback_test resolved-flow submission also stamps
        # submitted_at; submit_negative must respect that too, even though
        # it never touched helpful/reason_code.
        run_id = "run-legacy-resolved"
        self._new_run(run_id)
        legacy_accepted = self.store.submit(run_id, True)
        self.assertTrue(legacy_accepted)

        accepted = self.store.submit_negative(run_id, "not_relevant")
        self.assertFalse(accepted)

        row = self.store.get(run_id)
        self.assertIsNone(row["helpful"])
        self.assertIsNone(row["reason_code"])
        self.assertEqual(row["resolved"], 1)

    def test_submit_helpful_after_submit_negative_is_also_rejected(self):
        # Cross-check the other direction of the same invariant: once
        # submit_negative() has claimed submitted_at, the pre-existing
        # submit_helpful() (unmodified by this change) must not be able to
        # flip the row back to positive.
        run_id = "run-negative-then-positive"
        self._new_run(run_id)
        negative = self.store.submit_negative(run_id, "incorrect")
        self.assertTrue(negative)

        flipped = self.store.submit_helpful(run_id, True)
        self.assertFalse(flipped)

        row = self.store.get(run_id)
        self.assertEqual(row["helpful"], 0)
        self.assertEqual(row["reason_code"], "incorrect")

    def test_incomplete_row_keeps_three_column_invariant_until_submission(self):
        run_id = "run-incomplete"
        self._new_run(run_id)
        row = self.store.get(run_id)
        self.assertIsNone(row["helpful"])
        self.assertIsNone(row["reason_code"])
        self.assertIsNone(row["submitted_at"])

    def test_storage_revalidates_reason_allowlist_independent_of_caller(self):
        # submit_negative must not trust a pre-validated value; calling it
        # directly with an out-of-allowlist code (bypassing the parser
        # entirely) must still be rejected.
        run_id = "run-bypass-parser"
        self._new_run(run_id)
        self.assertFalse(self.store.submit_negative(run_id, "not-a-real-code"))
        self.assertFalse(self.store.submit_negative(run_id, ""))

    def test_unhashable_or_non_string_reason_is_rejected_without_raising(self):
        # submit_negative() must fail closed (return False) rather than
        # raise, even for unhashable types, and must not disturb the row.
        run_id = "run-unhashable-reason"
        self._new_run(run_id)
        for bad_reason in (None, 123, ["incorrect"], {"incorrect"}, {"code": "incorrect"}):
            with self.subTest(bad_reason=bad_reason):
                accepted = self.store.submit_negative(run_id, bad_reason)
                self.assertFalse(accepted)

        row = self.store.get(run_id)
        self.assertIsNone(row["helpful"])
        self.assertIsNone(row["reason_code"])
        self.assertIsNone(row["submitted_at"])

    def test_unknown_run_id_is_rejected(self):
        self.assertFalse(self.store.submit_negative("does-not-exist", "incorrect"))

    def test_schema_init_and_migration_are_repeatable(self):
        # Re-opening the same DB file (re-running _migrate_schema) must be
        # idempotent and must not disturb existing rows.
        run_id = "run-repeat-init"
        self._new_run(run_id)
        self.store.submit_negative(run_id, "incorrect")

        reopened = FeedbackStore(self.db_path)
        reopened_again = FeedbackStore(self.db_path)
        row = reopened_again.get(run_id)
        self.assertEqual(row["reason_code"], "incorrect")
        self.assertEqual(row["helpful"], 0)


class FeedbackStoreSuggestionTests(unittest.TestCase):
    """Tests for get_by_feedback_message_id() and submit_suggestion()."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStore(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _new_finalized_negative_run(
        self,
        run_id: str,
        *,
        chat_id: str = "chat-1",
        telegram_user_id: str = "user-1",
        feedback_message_id: str = "msg-1",
        reason_code: str | None = "incorrect",
        submitted_at: str | None = "2026-01-01T00:00:00+00:00",
        helpful: int | None = 0,
    ) -> None:
        """Create a row already in the post-submit_negative() finalized shape.

        Uses create_run()'s **fields passthrough directly instead of the
        submit_negative() flow so tests can construct edge-case rows (e.g.
        a missing reason_code) that submit_negative() itself would never
        produce, in order to prove submit_suggestion()'s own predicate
        guards against them independently.
        """
        created = self.store.create_run(
            run_id,
            chat_id,
            telegram_user_id=telegram_user_id,
            feedback_message_id=feedback_message_id,
            helpful=helpful,
            reason_code=reason_code,
            submitted_at=submitted_at,
        )
        self.assertTrue(created)

    # -- get_by_feedback_message_id ---------------------------------------

    def test_get_by_feedback_message_id_returns_correct_row(self):
        self._new_finalized_negative_run("run-a", chat_id="chat-1", feedback_message_id="msg-1")
        row = self.store.get_by_feedback_message_id("chat-1", "msg-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["run_id"], "run-a")

    def test_get_by_feedback_message_id_scoped_per_chat(self):
        # Telegram message_id is only unique within a chat — the same
        # numeric id in two different chats must resolve to two distinct
        # rows, never cross-contaminate.
        self._new_finalized_negative_run("run-chat-1", chat_id="chat-1", feedback_message_id="msg-100")
        self._new_finalized_negative_run("run-chat-2", chat_id="chat-2", feedback_message_id="msg-100")

        row1 = self.store.get_by_feedback_message_id("chat-1", "msg-100")
        row2 = self.store.get_by_feedback_message_id("chat-2", "msg-100")
        self.assertEqual(row1["run_id"], "run-chat-1")
        self.assertEqual(row2["run_id"], "run-chat-2")

    def test_get_by_feedback_message_id_fail_closed_on_duplicate_rows(self):
        # No DB constraint currently enforces uniqueness on
        # (chat_id, feedback_message_id); if it is ever violated, the
        # lookup must fail closed (None) rather than arbitrarily pick one.
        self._new_finalized_negative_run("run-dup-1", chat_id="chat-1", feedback_message_id="msg-dup")
        self._new_finalized_negative_run("run-dup-2", chat_id="chat-1", feedback_message_id="msg-dup")

        row = self.store.get_by_feedback_message_id("chat-1", "msg-dup")
        self.assertIsNone(row)

    def test_get_by_feedback_message_id_unknown_returns_none(self):
        self.assertIsNone(self.store.get_by_feedback_message_id("chat-1", "does-not-exist"))

    # -- submit_suggestion: happy path -------------------------------------

    def test_submit_suggestion_succeeds_for_finalized_negative_row(self):
        self._new_finalized_negative_run("run-ok")
        accepted = self.store.submit_suggestion(
            "run-ok", "chat-1", "user-1", "msg-1", "請補充更多範例。"
        )
        self.assertTrue(accepted)
        row = self.store.get("run-ok")
        self.assertEqual(row["suggestion_text"], "請補充更多範例。")

    def test_submit_suggestion_does_not_change_other_columns(self):
        self._new_finalized_negative_run(
            "run-columns", reason_code="unclear", submitted_at="2026-02-02T00:00:00+00:00"
        )
        before = self.store.get("run-columns")
        accepted = self.store.submit_suggestion(
            "run-columns", "chat-1", "user-1", "msg-1", "建議內容"
        )
        self.assertTrue(accepted)
        after = self.store.get("run-columns")
        self.assertEqual(after["helpful"], before["helpful"])
        self.assertEqual(after["reason_code"], before["reason_code"])
        self.assertEqual(after["submitted_at"], before["submitted_at"])

    # -- submit_suggestion: eligibility gates ------------------------------

    def test_submit_suggestion_rejected_for_positive_row(self):
        self._new_finalized_negative_run(
            "run-positive", helpful=1, reason_code=None
        )
        accepted = self.store.submit_suggestion(
            "run-positive", "chat-1", "user-1", "msg-1", "建議內容"
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-positive")["suggestion_text"])

    def test_submit_suggestion_rejected_for_unfinalized_row(self):
        self._new_finalized_negative_run(
            "run-unfinalized", helpful=None, reason_code=None, submitted_at=None
        )
        accepted = self.store.submit_suggestion(
            "run-unfinalized", "chat-1", "user-1", "msg-1", "建議內容"
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-unfinalized")["suggestion_text"])

    def test_submit_suggestion_rejected_when_reason_code_missing(self):
        # Constructs a state submit_negative() itself could never produce
        # (helpful=0/submitted_at set but reason_code NULL), to prove
        # submit_suggestion() independently re-validates the reason
        # allowlist rather than trusting helpful/submitted_at alone.
        self._new_finalized_negative_run("run-no-reason", reason_code=None)
        accepted = self.store.submit_suggestion(
            "run-no-reason", "chat-1", "user-1", "msg-1", "建議內容"
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-no-reason")["suggestion_text"])

    # -- submit_suggestion: binding mismatches -----------------------------

    def test_submit_suggestion_rejected_on_chat_mismatch(self):
        self._new_finalized_negative_run("run-chat-mismatch", chat_id="chat-1")
        accepted = self.store.submit_suggestion(
            "run-chat-mismatch", "chat-WRONG", "user-1", "msg-1", "建議內容"
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-chat-mismatch")["suggestion_text"])

    def test_submit_suggestion_rejected_on_owner_mismatch(self):
        self._new_finalized_negative_run("run-owner-mismatch", telegram_user_id="user-1")
        accepted = self.store.submit_suggestion(
            "run-owner-mismatch", "chat-1", "user-WRONG", "msg-1", "建議內容"
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-owner-mismatch")["suggestion_text"])

    def test_submit_suggestion_rejected_on_message_id_mismatch(self):
        self._new_finalized_negative_run("run-msgid-mismatch", feedback_message_id="msg-1")
        accepted = self.store.submit_suggestion(
            "run-msgid-mismatch", "chat-1", "user-1", "msg-WRONG", "建議內容"
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-msgid-mismatch")["suggestion_text"])

    # -- submit_suggestion: first-write-wins -------------------------------

    def test_submit_suggestion_second_attempt_does_not_overwrite(self):
        self._new_finalized_negative_run("run-replay")
        first = self.store.submit_suggestion("run-replay", "chat-1", "user-1", "msg-1", "第一次建議")
        self.assertTrue(first)
        second = self.store.submit_suggestion("run-replay", "chat-1", "user-1", "msg-1", "第二次建議")
        self.assertFalse(second)
        self.assertEqual(self.store.get("run-replay")["suggestion_text"], "第一次建議")

    # -- submit_suggestion: text validation ---------------------------------

    def test_submit_suggestion_rejects_empty_string(self):
        self._new_finalized_negative_run("run-empty")
        accepted = self.store.submit_suggestion("run-empty", "chat-1", "user-1", "msg-1", "")
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-empty")["suggestion_text"])

    def test_submit_suggestion_rejects_whitespace_only(self):
        self._new_finalized_negative_run("run-whitespace")
        accepted = self.store.submit_suggestion(
            "run-whitespace", "chat-1", "user-1", "msg-1", "   \n\t  "
        )
        self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-whitespace")["suggestion_text"])

    def test_submit_suggestion_rejects_non_string(self):
        self._new_finalized_negative_run("run-non-string")
        for bad_value in (None, 123, ["a suggestion"], {"text": "a suggestion"}):
            with self.subTest(bad_value=bad_value):
                accepted = self.store.submit_suggestion(
                    "run-non-string", "chat-1", "user-1", "msg-1", bad_value
                )
                self.assertFalse(accepted)
        self.assertIsNone(self.store.get("run-non-string")["suggestion_text"])

    def test_submit_suggestion_accepts_1000_chars_rejects_1001(self):
        self._new_finalized_negative_run("run-at-limit")
        self._new_finalized_negative_run(
            "run-over-limit", feedback_message_id="msg-2"
        )

        exactly_1000 = "a" * 1000
        accepted = self.store.submit_suggestion(
            "run-at-limit", "chat-1", "user-1", "msg-1", exactly_1000
        )
        self.assertTrue(accepted)
        self.assertEqual(self.store.get("run-at-limit")["suggestion_text"], exactly_1000)

        over_1000 = "a" * 1001
        rejected = self.store.submit_suggestion(
            "run-over-limit", "chat-1", "user-1", "msg-2", over_1000
        )
        self.assertFalse(rejected)
        self.assertIsNone(self.store.get("run-over-limit")["suggestion_text"])


if __name__ == "__main__":
    unittest.main()
