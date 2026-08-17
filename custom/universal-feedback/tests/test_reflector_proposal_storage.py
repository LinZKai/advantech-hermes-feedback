"""Phase 5 Slice 2 tests: the reflection_runs / improvement_proposals /
proposal_observations schema and storage API in tools.feedback_store_v2.

Deliberately a separate file from test_case_analysis.py / test_case_
reflection_input.py, matching the one-file-per-concern convention this
test suite already uses -- this is a new, independently reviewable Phase 5
slice, not an extension of the Phase 4.5 / Slice 1 test surface.

No LLM, no network, no matching algorithm, no runner, no scheduler
anywhere in this file -- Slice 2 has none of those.
"""
from __future__ import annotations

import gc
import json
import sqlite3
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

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

    def _create_run(
        self, reflection_run_id: str = "run-1", *,
        started_at: str = "2026-01-01T00:00:00+00:00",
        analyzed_case_count: int = 5,
        reflector_version: str = "reflector-v1",
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> str:
        ok = self.store.create_reflection_run(
            reflection_run_id, started_at=started_at, analyzed_case_count=analyzed_case_count,
            reflector_version=reflector_version, window_start=window_start, window_end=window_end,
        )
        assert ok
        return reflection_run_id

    def _create_proposal(
        self, proposal_id: str = "proposal-1", *,
        improvement_target: str = "knowledge",
        title: str = "ADAM-6266 SNMP disable command undocumented",
        created_at: str = "2026-01-01T00:00:00+00:00",
    ) -> str:
        ok = self.store.create_improvement_proposal(
            proposal_id, improvement_target=improvement_target, title=title, created_at=created_at,
        )
        assert ok
        return proposal_id

    def _create_observation(
        self, proposal_id: str, reflection_run_id: str, *,
        observation_id: str | None = None,
        trend: str = "new",
        pattern_summary: str = "Three Cases ask about SNMP disable.",
        possible_cause: str | None = "KB may not document this clearly.",
        recommended_improvement: str = "Add an SNMP disable procedure to the KB.",
        expected_benefit: str | None = "Fewer repeat questions.",
        limitations: str | None = "Small sample.",
        supporting_case_ids: tuple[str, ...] = ("case-1", "case-2"),
        confidence: float = 0.7,
        observed_at: str = "2026-01-02T00:00:00+00:00",
    ) -> str:
        observation_id = observation_id or uuid.uuid4().hex
        ok = self.store.create_proposal_observation(
            observation_id, proposal_id, reflection_run_id,
            trend=trend, pattern_summary=pattern_summary, possible_cause=possible_cause,
            recommended_improvement=recommended_improvement, expected_benefit=expected_benefit,
            limitations=limitations,
            supporting_case_ids_json=json.dumps(list(supporting_case_ids)),
            supporting_case_count=len(supporting_case_ids),
            confidence=confidence, observed_at=observed_at,
        )
        assert ok
        return observation_id


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


class SchemaTests(_StoreTestCase):
    def test_tables_created(self):
        with self._raw_connect() as conn:
            names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertIn("reflection_runs", names)
        self.assertIn("improvement_proposals", names)
        self.assertIn("proposal_observations", names)

    def test_indexes_created(self):
        with self._raw_connect() as conn:
            names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='proposal_observations'"
                ).fetchall()
            }
        self.assertIn("idx_proposal_observations_proposal", names)
        self.assertIn("idx_proposal_observations_run", names)

    def test_migration_idempotent_when_reapplied(self):
        # Re-constructing FeedbackStoreV2 against the same DB file must not
        # raise -- CREATE TABLE IF NOT EXISTS is a no-op the second time.
        FeedbackStoreV2(self.db_path)

    def test_reflection_run_status_check_rejects_unknown_value(self):
        self._create_run("run-1")
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE reflection_runs SET status=? WHERE reflection_run_id=?",
                    ("not_a_real_status", "run-1"),
                )

    def test_improvement_target_check_rejects_unknown_value(self):
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO improvement_proposals "
                    "(proposal_id, improvement_target, title, review_status, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("p1", "not_a_real_target", "t", "pending", "2026-01-01T00:00:00+00:00"),
                )

    def test_review_status_check_rejects_unknown_value(self):
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO improvement_proposals "
                    "(proposal_id, improvement_target, title, review_status, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("p1", "knowledge", "t", "implemented", "2026-01-01T00:00:00+00:00"),
                )

    def test_trend_check_rejects_unknown_value(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO proposal_observations ("
                    "observation_id, proposal_id, reflection_run_id, trend, "
                    "pattern_summary, recommended_improvement, "
                    "supporting_case_ids_json, supporting_case_count, confidence, observed_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "o1", "p1", "run-1", "not_a_real_trend",
                        "summary", "improvement", "[]", 0, 0.5, "2026-01-01T00:00:00+00:00",
                    ),
                )

    def test_confidence_range_check(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            for bad_value in (-0.1, 1.1):
                with self.subTest(confidence=bad_value):
                    with self.assertRaises(sqlite3.IntegrityError):
                        conn.execute(
                            "INSERT INTO proposal_observations ("
                            "observation_id, proposal_id, reflection_run_id, trend, "
                            "pattern_summary, recommended_improvement, "
                            "supporting_case_ids_json, supporting_case_count, confidence, observed_at"
                            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                f"o-{bad_value}", "p1", "run-1", "new",
                                "summary", "improvement", "[]", 0, bad_value,
                                "2026-01-01T00:00:00+00:00",
                            ),
                        )

    def test_proposal_observation_fk_to_unknown_proposal_rejected(self):
        self._create_run("run-1")
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO proposal_observations ("
                    "observation_id, proposal_id, reflection_run_id, trend, "
                    "pattern_summary, recommended_improvement, "
                    "supporting_case_ids_json, supporting_case_count, confidence, observed_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "o1", "does-not-exist", "run-1", "new",
                        "summary", "improvement", "[]", 0, 0.5, "2026-01-01T00:00:00+00:00",
                    ),
                )

    def test_proposal_observation_fk_to_unknown_run_rejected(self):
        self._create_proposal("p1")
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO proposal_observations ("
                    "observation_id, proposal_id, reflection_run_id, trend, "
                    "pattern_summary, recommended_improvement, "
                    "supporting_case_ids_json, supporting_case_count, confidence, observed_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "o1", "p1", "does-not-exist", "new",
                        "summary", "improvement", "[]", 0, 0.5, "2026-01-01T00:00:00+00:00",
                    ),
                )

    def test_unique_proposal_id_reflection_run_id(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        self._create_observation("p1", "run-1", observation_id="o1", observed_at="2026-01-02T00:00:00+00:00")
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO proposal_observations ("
                    "observation_id, proposal_id, reflection_run_id, trend, "
                    "pattern_summary, recommended_improvement, "
                    "supporting_case_ids_json, supporting_case_count, confidence, observed_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "o2", "p1", "run-1", "growing",
                        "summary2", "improvement2", "[]", 0, 0.5, "2026-01-03T00:00:00+00:00",
                    ),
                )


# ---------------------------------------------------------------------------
# 2. reflection_runs lifecycle
# ---------------------------------------------------------------------------


class ReflectionRunLifecycleTests(_StoreTestCase):
    def test_create_then_get(self):
        self._create_run("run-1", analyzed_case_count=7, reflector_version="reflector-v1")
        row = self.store.get_reflection_run("run-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["analyzed_case_count"], 7)
        self.assertIsNone(row["completed_at"])

    def test_get_unknown_run_returns_none(self):
        self.assertIsNone(self.store.get_reflection_run("does-not-exist"))

    def test_duplicate_reflection_run_id_rejected(self):
        self._create_run("run-1")
        ok = self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=1,
            reflector_version="reflector-v1",
        )
        self.assertFalse(ok)

    def test_negative_analyzed_case_count_rejected(self):
        ok = self.store.create_reflection_run(
            "run-1", started_at="2026-01-01T00:00:00+00:00", analyzed_case_count=-1,
            reflector_version="reflector-v1",
        )
        self.assertFalse(ok)

    def test_complete_run_succeeded(self):
        self._create_run("run-1")
        ok = self.store.complete_reflection_run(
            "run-1", status="succeeded", completed_at="2026-01-01T01:00:00+00:00",
            material_change_detected=True, run_summary="Found a new pattern.",
        )
        self.assertTrue(ok)
        row = self.store.get_reflection_run("run-1")
        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["completed_at"], "2026-01-01T01:00:00+00:00")
        self.assertEqual(row["material_change_detected"], 1)
        self.assertEqual(row["run_summary"], "Found a new pattern.")

    def test_complete_run_failed(self):
        self._create_run("run-1")
        ok = self.store.complete_reflection_run(
            "run-1", status="failed", completed_at="2026-01-01T01:00:00+00:00",
        )
        self.assertTrue(ok)
        row = self.store.get_reflection_run("run-1")
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["material_change_detected"])

    def test_complete_run_invalid_status_rejected(self):
        self._create_run("run-1")
        ok = self.store.complete_reflection_run(
            "run-1", status="running", completed_at="2026-01-01T01:00:00+00:00",
        )
        self.assertFalse(ok)
        self.assertEqual(self.store.get_reflection_run("run-1")["status"], "running")

    def test_cannot_complete_already_completed_run(self):
        self._create_run("run-1")
        self.store.complete_reflection_run(
            "run-1", status="succeeded", completed_at="2026-01-01T01:00:00+00:00",
        )
        ok = self.store.complete_reflection_run(
            "run-1", status="failed", completed_at="2026-01-01T02:00:00+00:00",
        )
        self.assertFalse(ok)
        # First completion is never overwritten by the rejected second call.
        self.assertEqual(self.store.get_reflection_run("run-1")["status"], "succeeded")

    def test_complete_unknown_run_returns_false(self):
        ok = self.store.complete_reflection_run(
            "does-not-exist", status="succeeded", completed_at="2026-01-01T01:00:00+00:00",
        )
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 3. improvement_proposals lifecycle
# ---------------------------------------------------------------------------


class ImprovementProposalLifecycleTests(_StoreTestCase):
    def test_create_then_get(self):
        self._create_proposal("p1", improvement_target="agent_behavior", title="t")
        row = self.store.get_improvement_proposal("p1")
        self.assertIsNotNone(row)
        self.assertEqual(row["improvement_target"], "agent_behavior")
        self.assertEqual(row["review_status"], "pending")

    def test_get_unknown_proposal_returns_none(self):
        self.assertIsNone(self.store.get_improvement_proposal("does-not-exist"))

    def test_duplicate_proposal_id_rejected(self):
        self._create_proposal("p1")
        ok = self.store.create_improvement_proposal(
            "p1", improvement_target="knowledge", title="t2", created_at="2026-01-02T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_blank_title_rejected(self):
        ok = self.store.create_improvement_proposal(
            "p1", improvement_target="knowledge", title="   ", created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_invalid_improvement_target_rejected(self):
        ok = self.store.create_improvement_proposal(
            "p1", improvement_target="not_a_real_target", title="t",
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_update_review_status(self):
        self._create_proposal("p1")
        ok = self.store.update_proposal_review_status("p1", "accepted")
        self.assertTrue(ok)
        self.assertEqual(self.store.get_improvement_proposal("p1")["review_status"], "accepted")

    def test_update_review_status_invalid_value_rejected(self):
        self._create_proposal("p1")
        ok = self.store.update_proposal_review_status("p1", "implemented")
        self.assertFalse(ok)
        self.assertEqual(self.store.get_improvement_proposal("p1")["review_status"], "pending")

    def test_update_review_status_unknown_proposal_returns_false(self):
        ok = self.store.update_proposal_review_status("does-not-exist", "accepted")
        self.assertFalse(ok)

    def test_list_all_proposals(self):
        self._create_proposal("p1", created_at="2026-01-01T00:00:00+00:00")
        self._create_proposal("p2", created_at="2026-01-02T00:00:00+00:00")
        rows = self.store.list_improvement_proposals()
        self.assertEqual([r["proposal_id"] for r in rows], ["p1", "p2"])

    def test_list_filtered_by_review_status(self):
        self._create_proposal("p1")
        self._create_proposal("p2")
        self.store.update_proposal_review_status("p2", "accepted")
        rows = self.store.list_improvement_proposals(review_status="accepted")
        self.assertEqual([r["proposal_id"] for r in rows], ["p2"])


# ---------------------------------------------------------------------------
# 4. proposal_observations append-only lifecycle
# ---------------------------------------------------------------------------


class ProposalObservationLifecycleTests(_StoreTestCase):
    def test_create_then_get_latest(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        observation_id = self._create_observation("p1", "run-1")
        row = self.store.get_latest_proposal_observation("p1")
        self.assertIsNotNone(row)
        self.assertEqual(row["observation_id"], observation_id)

    def test_get_latest_with_no_observation_returns_none(self):
        self._create_proposal("p1")
        self.assertIsNone(self.store.get_latest_proposal_observation("p1"))

    def test_second_observation_becomes_latest_first_is_not_overwritten(self):
        self._create_proposal("p1")
        self._create_run("run-1", started_at="2026-01-01T00:00:00+00:00")
        self._create_run("run-2", started_at="2026-01-08T00:00:00+00:00")
        first_id = self._create_observation(
            "p1", "run-1", observation_id="o1", trend="new", observed_at="2026-01-01T00:00:01+00:00",
        )
        second_id = self._create_observation(
            "p1", "run-2", observation_id="o2", trend="growing", observed_at="2026-01-08T00:00:01+00:00",
        )
        latest = self.store.get_latest_proposal_observation("p1")
        self.assertEqual(latest["observation_id"], second_id)

        with self._raw_connect() as conn:
            ids = {
                row["observation_id"] for row in
                conn.execute("SELECT observation_id FROM proposal_observations WHERE proposal_id=?", ("p1",))
            }
        self.assertEqual(ids, {first_id, second_id})

    def test_duplicate_proposal_run_pair_rejected(self):
        self._create_proposal("p1")
        self._create_run("run-1")
        self._create_observation("p1", "run-1", observation_id="o1", observed_at="2026-01-02T00:00:00+00:00")
        ok = self.store.create_proposal_observation(
            "o2", "p1", "run-1",
            trend="growing", pattern_summary="s", possible_cause=None,
            recommended_improvement="r", expected_benefit=None, limitations=None,
            supporting_case_ids_json="[]", supporting_case_count=0,
            confidence=0.5, observed_at="2026-01-03T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_invalid_trend_rejected(self):
        self._create_proposal("p1")
        self._create_run("run-1")
        ok = self.store.create_proposal_observation(
            "o1", "p1", "run-1",
            trend="not_a_real_trend", pattern_summary="s", possible_cause=None,
            recommended_improvement="r", expected_benefit=None, limitations=None,
            supporting_case_ids_json="[]", supporting_case_count=0,
            confidence=0.5, observed_at="2026-01-02T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_unknown_proposal_id_rejected(self):
        self._create_run("run-1")
        ok = self.store.create_proposal_observation(
            "o1", "does-not-exist", "run-1",
            trend="new", pattern_summary="s", possible_cause=None,
            recommended_improvement="r", expected_benefit=None, limitations=None,
            supporting_case_ids_json="[]", supporting_case_count=0,
            confidence=0.5, observed_at="2026-01-02T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_list_latest_proposal_observations_multiple_proposals(self):
        self._create_run("run-1")
        self._create_proposal("p-a")
        self._create_proposal("p-b")
        obs_a = self._create_observation("p-a", "run-1", observation_id="oa", observed_at="2026-01-02T00:00:00+00:00")
        obs_b = self._create_observation("p-b", "run-1", observation_id="ob", observed_at="2026-01-02T00:00:00+00:01")
        rows = self.store.list_latest_proposal_observations()
        by_proposal = {r["proposal_id"]: r["observation_id"] for r in rows}
        self.assertEqual(by_proposal, {"p-a": obs_a, "p-b": obs_b})

    def test_list_latest_proposal_observations_proposal_ids_subset(self):
        self._create_run("run-1")
        self._create_proposal("p-a")
        self._create_proposal("p-b")
        self._create_observation("p-a", "run-1", observation_id="oa", observed_at="2026-01-02T00:00:00+00:00")
        self._create_observation("p-b", "run-1", observation_id="ob", observed_at="2026-01-02T00:00:00+01:00")
        rows = self.store.list_latest_proposal_observations(proposal_ids=["p-a"])
        self.assertEqual([r["proposal_id"] for r in rows], ["p-a"])

    def test_list_latest_proposal_observations_empty_list_returns_empty(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        self._create_observation("p1", "run-1")
        rows = self.store.list_latest_proposal_observations(proposal_ids=[])
        self.assertEqual(rows, [])

    def test_list_latest_proposal_observations_none_returns_all_observed(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        self._create_proposal("p2")  # never observed
        self._create_observation("p1", "run-1")
        rows = self.store.list_latest_proposal_observations(proposal_ids=None)
        self.assertEqual([r["proposal_id"] for r in rows], ["p1"])

    def test_supporting_case_ids_json_round_trips(self):
        self._create_run("run-1")
        self._create_proposal("p1")
        self._create_observation("p1", "run-1", supporting_case_ids=("case-a", "case-b"))
        row = self.store.get_latest_proposal_observation("p1")
        self.assertEqual(json.loads(row["supporting_case_ids_json"]), ["case-a", "case-b"])
        self.assertEqual(row["supporting_case_count"], 2)


if __name__ == "__main__":
    unittest.main()
