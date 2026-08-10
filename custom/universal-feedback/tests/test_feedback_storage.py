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


if __name__ == "__main__":
    unittest.main()
