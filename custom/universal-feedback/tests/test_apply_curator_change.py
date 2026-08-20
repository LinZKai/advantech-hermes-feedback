"""Curator Slice 2 tests: the deterministic apply pipeline in tools.
apply_curator_change.

No LLM, no network -- every test uses a real temp-file SQLite DB and a
real temp AGENTS.md file (matching test_run_curator.py's own
`_StoreTestCase` style), because genuinely exercising the guard sequence
and the write/DB-status ordering invariant needs real I/O, not a mock.
"""
from __future__ import annotations

import contextlib
import gc
import io
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.apply_curator_change import ApplyOutcome, apply_curator_change, main  # noqa: E402
from tools.curator_domain import CURATOR_TARGET_FILE  # noqa: E402
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402

_BEFORE_CONTENT = "# Advantech Technical Support Instructions\n\nPut the direct answer first.\n"
_PROPOSED_CONTENT = "# Advantech Technical Support Instructions\n\nBe direct. New rule added.\n"


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)
        self.agents_file = Path(self._tmpdir.name) / "AGENTS.md"
        self.agents_file.write_text(_BEFORE_CONTENT, encoding="utf-8")

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_change(
        self, *,
        status: str = "approved",
        target_file: str = CURATOR_TARGET_FILE,
        before_content: str = _BEFORE_CONTENT,
        proposed_content: str | None = _PROPOSED_CONTENT,
        change_type: str = "modify_rule",
    ) -> str:
        proposal_id = "proposal-1"
        self.store.create_improvement_proposal(
            proposal_id, improvement_target="agent_behavior", title="title",
            created_at="2026-01-01T00:00:00+00:00",
        )
        change_id = uuid.uuid4().hex
        ok = self.store.create_curator_change(
            change_id, proposal_id, target_file,
            change_type=change_type, rationale="r", before_content=before_content,
            proposed_content=proposed_content, expected_effect=None, confidence=0.8,
            created_at="2026-01-02T00:00:00+00:00",
        )
        assert ok
        if status != "proposed":
            self.store.review_curator_change(change_id, "approved" if status != "rejected" else "rejected", reviewed_at="2026-01-03T00:00:00+00:00")
            if status == "applied":
                self.store.mark_curator_change_applied(change_id, applied_at="2026-01-04T00:00:00+00:00")
            elif status == "failed":
                self.store.mark_curator_change_failed(change_id)
        return change_id


# ---------------------------------------------------------------------------
# A. Status guard
# ---------------------------------------------------------------------------


class StatusGuardTests(_StoreTestCase):
    def test_proposed_cannot_apply(self):
        change_id = self._seed_change(status="proposed")
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "not_approved")
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _BEFORE_CONTENT)

    def test_rejected_cannot_apply(self):
        change_id = self._seed_change(status="rejected")
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "not_approved")
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _BEFORE_CONTENT)

    def test_already_applied_cannot_apply_again(self):
        change_id = self._seed_change(status="applied")
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "not_approved")

    def test_failed_cannot_apply(self):
        change_id = self._seed_change(status="failed")
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "not_approved")

    def test_unknown_change_id(self):
        outcome = apply_curator_change(self.store, change_id="does-not-exist", agents_file=self.agents_file)
        self.assertEqual(outcome.status, "not_found")


# ---------------------------------------------------------------------------
# B. Target / content guards
# ---------------------------------------------------------------------------


class TargetAndContentGuardTests(_StoreTestCase):
    def test_wrong_target_file_fails_closed(self):
        # CuratorChange normally refuses to construct with a wrong target_file,
        # but create_curator_change() is a lower-level storage boundary --
        # simulate a row that somehow has one anyway (e.g. hand-inserted) to
        # prove apply_curator_change() re-checks independently.
        change_id = self._seed_change(status="approved")
        with self.store._connect() as db:
            db.execute("UPDATE curator_changes SET target_file=? WHERE change_id=?", ("/sandbox/SOUL.md", change_id))
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "wrong_target")
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _BEFORE_CONTENT)

    def test_empty_proposed_content_fails_closed(self):
        change_id = self._seed_change(status="approved")
        with self.store._connect() as db:
            db.execute("UPDATE curator_changes SET proposed_content=? WHERE change_id=?", ("   ", change_id))
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "empty_content")
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _BEFORE_CONTENT)

    def test_null_proposed_content_fails_closed(self):
        change_id = self._seed_change(status="approved")
        with self.store._connect() as db:
            db.execute("UPDATE curator_changes SET proposed_content=NULL WHERE change_id=?", (change_id,))
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "empty_content")

    def test_missing_agents_file_fails_closed(self):
        change_id = self._seed_change(status="approved")
        missing = Path(self._tmpdir.name) / "does-not-exist" / "AGENTS.md"
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=missing)
        self.assertEqual(outcome.status, "agents_file_unreadable")

    def test_source_changed_since_proposal_fails_closed_without_overwrite(self):
        change_id = self._seed_change(status="approved", before_content=_BEFORE_CONTENT)
        # Someone else edited AGENTS.md after Curator produced this change.
        self.agents_file.write_text("SOMEONE ELSE'S EDIT\n", encoding="utf-8")

        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)

        self.assertEqual(outcome.status, "source_changed")
        # File must be left exactly as the other edit left it -- never overwritten.
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), "SOMEONE ELSE'S EDIT\n")
        # Status must NOT silently flip to failed -- the change itself isn't
        # broken, its precondition is stale; it stays 'approved'.
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "approved")


# ---------------------------------------------------------------------------
# C. Happy path
# ---------------------------------------------------------------------------


class HappyPathTests(_StoreTestCase):
    def test_successful_apply_writes_file_and_marks_applied(self):
        change_id = self._seed_change(status="approved")

        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)

        self.assertEqual(outcome.status, "applied")
        self.assertIsNotNone(outcome.applied_at)
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _PROPOSED_CONTENT)

        row = self.store.get_curator_change(change_id)
        self.assertEqual(row["status"], "applied")
        self.assertEqual(row["applied_at"], outcome.applied_at)

    def test_no_change_recommended_change_type_still_needs_content_and_will_be_empty(self):
        # A no_change_recommended change legitimately has proposed_content=None
        # -- confirms it is correctly refused as "nothing to apply", not
        # silently treated as a successful no-op apply.
        change_id = self._seed_change(status="approved", change_type="no_change_recommended", proposed_content=None)
        outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        self.assertEqual(outcome.status, "empty_content")


# ---------------------------------------------------------------------------
# D. Write failure -- never marked applied
# ---------------------------------------------------------------------------


class WriteFailureTests(_StoreTestCase):
    def test_write_failure_marks_failed_not_applied(self):
        change_id = self._seed_change(status="approved")

        with mock.patch("tools.apply_curator_change.os.replace", side_effect=OSError("disk full")):
            outcome = apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)

        self.assertEqual(outcome.status, "write_failed")
        row = self.store.get_curator_change(change_id)
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["applied_at"])
        # Original file content must be untouched -- the atomic write never
        # got far enough to replace it.
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _BEFORE_CONTENT)

    def test_write_failure_never_leaves_a_stray_temp_file(self):
        change_id = self._seed_change(status="approved")
        with mock.patch("tools.apply_curator_change.os.replace", side_effect=OSError("disk full")):
            apply_curator_change(self.store, change_id=change_id, agents_file=self.agents_file)
        leftovers = [p for p in self.agents_file.parent.iterdir() if p.name.startswith(f".{self.agents_file.name}.")]
        self.assertEqual(leftovers, [])


# ---------------------------------------------------------------------------
# E. main() -- the thin CLI boundary
# ---------------------------------------------------------------------------


class MainCliTests(_StoreTestCase):
    def _run_main(self, *extra_argv: str) -> tuple[int, str, str]:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        argv = ["--db", str(self.db_path), "--agents-file", str(self.agents_file), *extra_argv]
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(argv)
        return exit_code, buf_out.getvalue(), buf_err.getvalue()

    def test_apply_via_cli_succeeds(self):
        change_id = self._seed_change(status="approved")
        exit_code, out, _ = self._run_main("--change-id", change_id)
        self.assertEqual(exit_code, 0)
        self.assertIn("status=applied", out)
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _PROPOSED_CONTENT)

    def test_apply_via_cli_on_proposed_change_fails_closed(self):
        change_id = self._seed_change(status="proposed")
        exit_code, out, _ = self._run_main("--change-id", change_id)
        self.assertEqual(exit_code, 1)
        self.assertIn("status=not_approved", out)
        self.assertEqual(self.agents_file.read_text(encoding="utf-8"), _BEFORE_CONTENT)


if __name__ == "__main__":
    unittest.main()
