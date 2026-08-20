"""Curator Slice 2 tests: the human-review helper CLI in tools.
review_curator_change.

No LLM, no network -- this module only wraps tools.feedback_store_v2.
FeedbackStoreV2.review_curator_change(), matching test_review_proposal.py's
own real-temp-file-DB `_StoreTestCase` style.
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

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.curator_domain import CURATOR_TARGET_FILE  # noqa: E402
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.review_curator_change import main, review_curator_change  # noqa: E402


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_change(self, *, status: str = "proposed") -> str:
        proposal_id = "proposal-1"
        self.store.create_improvement_proposal(
            proposal_id, improvement_target="agent_behavior", title="title",
            created_at="2026-01-01T00:00:00+00:00",
        )
        change_id = uuid.uuid4().hex
        self.store.create_curator_change(
            change_id, proposal_id, CURATOR_TARGET_FILE,
            change_type="modify_rule", rationale="r", before_content="before",
            proposed_content="after", expected_effect=None, confidence=0.8,
            created_at="2026-01-02T00:00:00+00:00",
        )
        if status != "proposed":
            if status in ("approved", "rejected"):
                self.store.review_curator_change(change_id, status, reviewed_at="2026-01-03T00:00:00+00:00")
            elif status == "applied":
                self.store.review_curator_change(change_id, "approved", reviewed_at="2026-01-03T00:00:00+00:00")
                self.store.mark_curator_change_applied(change_id, applied_at="2026-01-04T00:00:00+00:00")
            elif status == "failed":
                self.store.review_curator_change(change_id, "approved", reviewed_at="2026-01-03T00:00:00+00:00")
                self.store.mark_curator_change_failed(change_id)
        return change_id


class ReviewCuratorChangeFunctionTests(_StoreTestCase):
    def test_proposed_to_approved(self):
        change_id = self._seed_change()
        result = review_curator_change(self.store, change_id=change_id, status="approved")
        self.assertEqual(result, "reviewed")
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "approved")

    def test_proposed_to_rejected(self):
        change_id = self._seed_change()
        result = review_curator_change(self.store, change_id=change_id, status="rejected")
        self.assertEqual(result, "reviewed")
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "rejected")

    def test_unknown_change_id(self):
        result = review_curator_change(self.store, change_id="does-not-exist", status="approved")
        self.assertEqual(result, "not_found")

    def test_rejected_to_approved_refused(self):
        change_id = self._seed_change(status="rejected")
        result = review_curator_change(self.store, change_id=change_id, status="approved")
        self.assertEqual(result, "not_proposed")
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "rejected")

    def test_approved_to_rejected_refused(self):
        change_id = self._seed_change(status="approved")
        result = review_curator_change(self.store, change_id=change_id, status="rejected")
        self.assertEqual(result, "not_proposed")
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "approved")

    def test_applied_to_approved_refused(self):
        change_id = self._seed_change(status="applied")
        result = review_curator_change(self.store, change_id=change_id, status="approved")
        self.assertEqual(result, "not_proposed")
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "applied")

    def test_failed_to_approved_refused(self):
        change_id = self._seed_change(status="failed")
        result = review_curator_change(self.store, change_id=change_id, status="approved")
        self.assertEqual(result, "not_proposed")
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "failed")


class MainCliTests(_StoreTestCase):
    def _run_main(self, *extra_argv: str) -> tuple[int, str, str]:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        argv = ["--db", str(self.db_path), *extra_argv]
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(argv)
        return exit_code, buf_out.getvalue(), buf_err.getvalue()

    def test_approve_via_cli(self):
        change_id = self._seed_change()
        exit_code, out, _ = self._run_main("--change-id", change_id, "--status", "approved")
        self.assertEqual(exit_code, 0)
        self.assertIn("result=reviewed", out)
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "approved")

    def test_illegal_transition_nonzero_exit(self):
        change_id = self._seed_change(status="applied")
        exit_code, out, _ = self._run_main("--change-id", change_id, "--status", "approved")
        self.assertEqual(exit_code, 1)
        self.assertIn("result=not_proposed", out)

    def test_unknown_status_rejected_by_argparse(self):
        change_id = self._seed_change()
        with self.assertRaises(SystemExit):
            self._run_main("--change-id", change_id, "--status", "proposed")


if __name__ == "__main__":
    unittest.main()
