"""Curator Slice 1 tests: the curator_changes schema and storage API in
tools.feedback_store_v2.

Deliberately a separate file from test_reflector_proposal_storage.py,
matching the one-file-per-concern convention this test suite already uses.

No LLM, no network, no runner anywhere in this file.
"""
from __future__ import annotations

import gc
import sqlite3
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


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _raw_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_proposal(
        self, proposal_id: str = "proposal-1", *,
        improvement_target: str = "agent_behavior",
        title: str = "技術支援回答應優先呈現直接結論、必要條件與可執行步驟",
        created_at: str = "2026-01-01T00:00:00+00:00",
    ) -> str:
        ok = self.store.create_improvement_proposal(
            proposal_id, improvement_target=improvement_target, title=title, created_at=created_at,
        )
        assert ok
        return proposal_id

    def _create_change(
        self, proposal_id: str, *,
        change_id: str | None = None,
        target_file: str = CURATOR_TARGET_FILE,
        change_type: str = "modify_rule",
        rationale: str = "多筆案例顯示回答過於冗長。",
        before_content: str = "# Advantech Technical Support Instructions\n",
        proposed_content: str | None = "# Advantech Technical Support Instructions\n\nBe direct.\n",
        expected_effect: str | None = "回答更精簡。",
        confidence: float = 0.8,
        created_at: str = "2026-01-02T00:00:00+00:00",
    ) -> str:
        change_id = change_id or uuid.uuid4().hex
        ok = self.store.create_curator_change(
            change_id, proposal_id, target_file,
            change_type=change_type, rationale=rationale, before_content=before_content,
            proposed_content=proposed_content, expected_effect=expected_effect,
            confidence=confidence, created_at=created_at,
        )
        assert ok
        return change_id


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


class SchemaTests(_StoreTestCase):
    def test_table_created(self):
        with self._raw_connect() as conn:
            names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertIn("curator_changes", names)

    def test_index_created(self):
        with self._raw_connect() as conn:
            names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
        self.assertIn("idx_curator_changes_proposal", names)


# ---------------------------------------------------------------------------
# 2. create_curator_change -- happy path + status/timestamps
# ---------------------------------------------------------------------------


class CreateCuratorChangeTests(_StoreTestCase):
    def test_valid_change_persists_with_status_proposed(self):
        proposal_id = self._create_proposal()
        change_id = self._create_change(proposal_id)

        row = self.store.get_curator_change(change_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "proposed")
        self.assertEqual(row["proposal_id"], proposal_id)
        self.assertEqual(row["target_file"], CURATOR_TARGET_FILE)
        self.assertIsNone(row["reviewed_at"])
        self.assertIsNone(row["applied_at"])

    def test_no_change_recommended_persists_with_null_proposed_content(self):
        proposal_id = self._create_proposal()
        change_id = self._create_change(
            proposal_id, change_type="no_change_recommended", proposed_content=None,
        )
        row = self.store.get_curator_change(change_id)
        self.assertEqual(row["change_type"], "no_change_recommended")
        self.assertIsNone(row["proposed_content"])
        self.assertEqual(row["status"], "proposed")

    def test_unknown_proposal_id_fails_fk_violation(self):
        ok = self.store.create_curator_change(
            uuid.uuid4().hex, "proposal-does-not-exist", CURATOR_TARGET_FILE,
            change_type="modify_rule", rationale="r", before_content="b",
            proposed_content="p", expected_effect=None, confidence=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_duplicate_change_id_fails(self):
        proposal_id = self._create_proposal()
        change_id = self._create_change(proposal_id)
        ok = self.store.create_curator_change(
            change_id, proposal_id, CURATOR_TARGET_FILE,
            change_type="modify_rule", rationale="r", before_content="b",
            proposed_content="p", expected_effect=None, confidence=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 3. create_curator_change -- fail-closed Python-side pre-validation
# ---------------------------------------------------------------------------


class CreateCuratorChangeValidationTests(_StoreTestCase):
    def test_invalid_change_type_rejected(self):
        proposal_id = self._create_proposal()
        ok = self.store.create_curator_change(
            uuid.uuid4().hex, proposal_id, CURATOR_TARGET_FILE,
            change_type="rewrite_everything", rationale="r", before_content="b",
            proposed_content="p", expected_effect=None, confidence=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_blank_rationale_rejected(self):
        proposal_id = self._create_proposal()
        ok = self.store.create_curator_change(
            uuid.uuid4().hex, proposal_id, CURATOR_TARGET_FILE,
            change_type="modify_rule", rationale="   ", before_content="b",
            proposed_content="p", expected_effect=None, confidence=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_missing_proposed_content_for_real_change_rejected(self):
        proposal_id = self._create_proposal()
        ok = self.store.create_curator_change(
            uuid.uuid4().hex, proposal_id, CURATOR_TARGET_FILE,
            change_type="add_rule", rationale="r", before_content="b",
            proposed_content=None, expected_effect=None, confidence=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_non_null_proposed_content_for_no_change_recommended_rejected(self):
        proposal_id = self._create_proposal()
        ok = self.store.create_curator_change(
            uuid.uuid4().hex, proposal_id, CURATOR_TARGET_FILE,
            change_type="no_change_recommended", rationale="r", before_content="b",
            proposed_content="should be null", expected_effect=None, confidence=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_invalid_confidence_rejected(self):
        proposal_id = self._create_proposal()
        ok = self.store.create_curator_change(
            uuid.uuid4().hex, proposal_id, CURATOR_TARGET_FILE,
            change_type="modify_rule", rationale="r", before_content="b",
            proposed_content="p", expected_effect=None, confidence=1.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
