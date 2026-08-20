"""Curator Slice 1 tests: the human-review helper CLI in tools.
review_proposal.

No LLM, no network -- this module only wraps tools.feedback_store_v2.
FeedbackStoreV2.update_proposal_review_status(), matching test_run_
reflector.py's/test_run_curator.py's own real-temp-file-DB `_StoreTestCase`
style.
"""
from __future__ import annotations

import contextlib
import gc
import io
import sys
import tempfile
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.review_proposal import main, review_proposal  # noqa: E402


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_pending_proposal(self, proposal_id: str = "proposal-1") -> str:
        ok = self.store.create_improvement_proposal(
            proposal_id, improvement_target="agent_behavior", title="title",
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert ok
        return proposal_id


class ReviewProposalFunctionTests(_StoreTestCase):
    def test_pending_to_accepted(self):
        proposal_id = self._seed_pending_proposal()
        result = review_proposal(self.store, proposal_id=proposal_id, review_status="accepted")
        self.assertEqual(result, "reviewed")
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")

    def test_pending_to_rejected(self):
        proposal_id = self._seed_pending_proposal()
        result = review_proposal(self.store, proposal_id=proposal_id, review_status="rejected")
        self.assertEqual(result, "reviewed")
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "rejected")

    def test_unknown_proposal_id(self):
        result = review_proposal(self.store, proposal_id="does-not-exist", review_status="accepted")
        self.assertEqual(result, "not_found")

    def test_already_accepted_refuses_to_overwrite(self):
        proposal_id = self._seed_pending_proposal()
        review_proposal(self.store, proposal_id=proposal_id, review_status="accepted")
        result = review_proposal(self.store, proposal_id=proposal_id, review_status="rejected")
        self.assertEqual(result, "not_pending")
        # Original decision must survive untouched.
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")

    def test_already_rejected_refuses_to_overwrite(self):
        proposal_id = self._seed_pending_proposal()
        review_proposal(self.store, proposal_id=proposal_id, review_status="rejected")
        result = review_proposal(self.store, proposal_id=proposal_id, review_status="accepted")
        self.assertEqual(result, "not_pending")
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "rejected")


class MainCliTests(_StoreTestCase):
    def _run_main(self, *extra_argv: str) -> tuple[int, str, str]:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        argv = ["--db", str(self.db_path), *extra_argv]
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(argv)
        return exit_code, buf_out.getvalue(), buf_err.getvalue()

    def test_accept_via_cli(self):
        proposal_id = self._seed_pending_proposal()
        exit_code, out, _ = self._run_main("--proposal-id", proposal_id, "--status", "accepted")
        self.assertEqual(exit_code, 0)
        self.assertIn("result=reviewed", out)
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")

    def test_unknown_status_rejected_by_argparse(self):
        proposal_id = self._seed_pending_proposal()
        with self.assertRaises(SystemExit):
            self._run_main("--proposal-id", proposal_id, "--status", "pending")

    def test_not_found_exit_code(self):
        exit_code, out, _ = self._run_main("--proposal-id", "does-not-exist", "--status", "accepted")
        self.assertEqual(exit_code, 1)
        self.assertIn("result=not_found", out)


if __name__ == "__main__":
    unittest.main()
