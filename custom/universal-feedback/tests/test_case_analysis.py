"""Phase 4.5 Stage A tests: the case_analysis table and its storage API in
tools.feedback_store_v2 -- schema, the evidence-watermark computation, and
the needs-analysis lifecycle contract.

Deliberately a separate file from test_feedback_store_v2.py, matching the
one-file-per-concern convention already used for test_case_routing.py /
test_case_assignment.py: this is a substantial, independently reviewable
slice, not an extension of the existing v2-schema test surface.

Stage A scope only: schema, storage queries, and their tests. No LLM call,
no enrichment service, no runner, no scheduler, and no Human Override logic
exist yet -- see tools/feedback_store_v2.py's module docstring and this
file's NeedsAnalysisLifecycleTests for what Stage A actually guarantees.
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

_TESTS_DIR = Path(__file__).resolve().parent
_OVERLAY_ROOT = _TESTS_DIR.parents[0] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

_002_SQL_PATH = _OVERLAY_ROOT / "migrations" / "002_feedback_schema_v2.sql"

from tools.feedback_store_v2 import (  # noqa: E402
    DIAGNOSIS_VALUES,
    ISSUE_TYPE_VALUES,
    PRODUCT_SOURCE_VALUES,
    FeedbackStoreV2,
    RetrievalRunInput,
)


# ---------------------------------------------------------------------------
# Fixture base: real on-disk SQLite through the actual store, matching the
# _StoreTestCase convention already used across this test suite.
# ---------------------------------------------------------------------------


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

    def _seed_session_and_case(self, session_id="sess-1", case_id="case-1"):
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        return session_id, case_id

    def _add_turn(self, case_id, turn_id, *, created_at=None):
        self.store.create_turn(
            turn_id, case_id,
            platform_user_id="user-1",
            platform_user_message_id=turn_id,
            question_text="q", answer_text="a",
            feedback_eligible=True,
        )
        if created_at is not None:
            with self._raw_connect() as conn:
                conn.execute("UPDATE turns SET created_at=? WHERE turn_id=?", (created_at, turn_id))
        return turn_id

    def _add_retrieval_run(self, turn_id, *, created_at=None):
        ok = self.store.add_retrieval_runs(turn_id, [
            RetrievalRunInput(execution_status="completed", observation_status="complete", result_count=3),
        ])
        assert ok
        if created_at is not None:
            with self._raw_connect() as conn:
                conn.execute(
                    "UPDATE retrieval_runs SET created_at=? WHERE turn_id=? "
                    "AND retrieval_id = (SELECT retrieval_id FROM retrieval_runs WHERE turn_id=? ORDER BY invocation_order DESC LIMIT 1)",
                    (created_at, turn_id, turn_id),
                )

    def _add_feedback(self, turn_id, *, helpful=True, submitted_at=None):
        feedback_id = uuid.uuid4().hex
        ok = self.store.submit_feedback(
            feedback_id, turn_id, helpful, feedback_policy_version="v1",
            reason_code=None if helpful else "incomplete",
        )
        assert ok
        if submitted_at is not None:
            with self._raw_connect() as conn:
                conn.execute("UPDATE feedback SET submitted_at=? WHERE feedback_id=?", (submitted_at, feedback_id))
        return feedback_id

    def _create_analysis(
        self, case_id, *, analyzed_at, source_evidence_watermark, analysis_id=None,
        diagnosis="knowledge_gap", diagnosis_confidence=0.5,
        issue_type="product_usage_or_application", issue_type_confidence=0.5,
    ):
        analysis_id = analysis_id or uuid.uuid4().hex
        ok = self.store.create_case_analysis(
            analysis_id, case_id,
            case_title="title", issue_summary="summary",
            issue_type=issue_type, issue_type_confidence=issue_type_confidence,
            diagnosis=diagnosis, diagnosis_confidence=diagnosis_confidence,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None,
            analysis_version="v1",
            analyzed_at=analyzed_at,
            source_evidence_watermark=source_evidence_watermark,
        )
        assert ok
        return analysis_id


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


class SchemaTests(_StoreTestCase):
    def test_case_analysis_table_and_columns_exist(self):
        with self._raw_connect() as conn:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(case_analysis)").fetchall()}
        expected = {
            "analysis_id", "case_id", "case_title", "issue_summary",
            "issue_type", "issue_type_confidence",
            "diagnosis", "diagnosis_confidence",
            "product_model", "product_source", "product_confidence",
            "evidence_json", "analysis_version", "analyzed_at",
            "source_evidence_watermark",
        }
        self.assertEqual(cols, expected)

    _BASE_INSERT_COLS = (
        "analysis_id, case_id, issue_type, issue_type_confidence, "
        "diagnosis, diagnosis_confidence, analysis_version, analyzed_at, source_evidence_watermark"
    )

    def test_fk_to_unknown_case_id_rejected(self):
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "does-not-exist", "product_usage_or_application", 0.5, "knowledge_gap", 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_diagnosis_check_constraint_rejects_unknown_value(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "case-1", "product_usage_or_application", 0.5, "not_a_real_diagnosis", 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_diagnosis_check_constraint_rejects_dropped_legacy_values(self):
        # workflow_tool_issue / unclear_or_other are no longer valid at the
        # SQL level either -- confirms the CHECK itself (not just Python
        # pre-validation) was actually rebuilt to the new 5-value taxonomy.
        self._seed_session_and_case()
        for legacy_value in ("workflow_tool_issue", "unclear_or_other"):
            with self.subTest(diagnosis=legacy_value):
                with self._raw_connect() as conn:
                    conn.execute("PRAGMA foreign_keys=ON")
                    with self.assertRaises(sqlite3.IntegrityError):
                        conn.execute(
                            f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (f"a-{legacy_value}", "case-1", "product_usage_or_application", 0.5, legacy_value, 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                        )

    def test_issue_type_check_constraint_rejects_unknown_value(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "case-1", "not_a_real_issue_type", 0.5, "knowledge_gap", 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_issue_type_not_null(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO case_analysis (analysis_id, case_id, issue_type_confidence, diagnosis, diagnosis_confidence, analysis_version, analyzed_at, source_evidence_watermark) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "case-1", 0.5, "knowledge_gap", 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_diagnosis_confidence_not_null(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO case_analysis (analysis_id, case_id, issue_type, issue_type_confidence, diagnosis, analysis_version, analyzed_at, source_evidence_watermark) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "case-1", "product_usage_or_application", 0.5, "knowledge_gap", "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_issue_type_confidence_range_check(self):
        self._seed_session_and_case()
        for bad_value in (-0.1, 1.1):
            with self.subTest(issue_type_confidence=bad_value):
                with self._raw_connect() as conn:
                    conn.execute("PRAGMA foreign_keys=ON")
                    with self.assertRaises(sqlite3.IntegrityError):
                        conn.execute(
                            f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ("a1", "case-1", "product_usage_or_application", bad_value, "knowledge_gap", 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                        )

    def test_diagnosis_confidence_range_check(self):
        self._seed_session_and_case()
        for bad_value in (-0.1, 1.1):
            with self.subTest(diagnosis_confidence=bad_value):
                with self._raw_connect() as conn:
                    conn.execute("PRAGMA foreign_keys=ON")
                    with self.assertRaises(sqlite3.IntegrityError):
                        conn.execute(
                            f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ("a1", "case-1", "product_usage_or_application", 0.5, "knowledge_gap", bad_value, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                        )

    def test_product_confidence_range_check_when_not_null(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO case_analysis "
                    "(analysis_id, case_id, issue_type, issue_type_confidence, diagnosis, diagnosis_confidence, product_confidence, analysis_version, analyzed_at, source_evidence_watermark) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "case-1", "product_usage_or_application", 0.5, "knowledge_gap", 0.5, 1.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_product_confidence_null_is_allowed(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                f"INSERT INTO case_analysis ({self._BASE_INSERT_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("a1", "case-1", "product_usage_or_application", 0.5, "knowledge_gap", 0.5, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            )
            conn.commit()  # must not raise -- product_confidence defaults NULL

    def test_product_source_check_constraint_rejects_unknown_value(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO case_analysis (analysis_id, case_id, issue_type, issue_type_confidence, diagnosis, diagnosis_confidence, product_source, analysis_version, analyzed_at, source_evidence_watermark) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("a1", "case-1", "product_usage_or_application", 0.5, "knowledge_gap", 0.5, "guessed", "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

    def test_product_source_null_is_allowed(self):
        self._seed_session_and_case()
        with self._raw_connect() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "INSERT INTO case_analysis (analysis_id, case_id, issue_type, issue_type_confidence, diagnosis, diagnosis_confidence, product_source, analysis_version, analyzed_at, source_evidence_watermark) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("a1", "case-1", "product_usage_or_application", 0.5, "knowledge_gap", 0.5, None, "v1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            )
            conn.commit()  # must not raise

    def test_index_exists(self):
        with self._raw_connect() as conn:
            names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='case_analysis'"
                ).fetchall()
            }
        self.assertIn("idx_case_analysis_case", names)

    def test_migration_idempotent_when_reapplied(self):
        # Calling the schema migration again must not raise and must not
        # duplicate/alter the table.
        self.store._migrate_schema()
        self.store._migrate_schema()
        with self._raw_connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='case_analysis'"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_opening_a_second_store_instance_against_the_same_file_is_safe(self):
        # A fresh FeedbackStoreV2(...) construction (migrate=True default)
        # against a database that already has case_analysis must not raise.
        second = FeedbackStoreV2(self.db_path)
        self.assertIsNone(second.get_case_evidence_watermark("does-not-exist"))

    def test_existing_pre_004_database_gets_case_analysis_added(self):
        """An existing v2 database created before case_analysis existed
        (built directly from 002_feedback_schema_v2.sql's own frozen file
        content) must gain case_analysis (and its index) the next time
        FeedbackStoreV2 opens it -- without losing any of its existing
        tables/data. Purely additive (CREATE TABLE IF NOT EXISTS); no
        rebuild involved."""
        old_db_path = Path(self._tmpdir.name) / "pre_004.db"
        sql_text = _002_SQL_PATH.read_text(encoding="utf-8")
        conn = sqlite3.connect(old_db_path)
        try:
            conn.executescript(sql_text)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "INSERT INTO sessions (session_id, platform, platform_chat_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("sess-old", "telegram", "chat-old", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            )
            conn.execute(
                "INSERT INTO cases (case_id, session_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("case-old", "sess-old", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            )
            conn.commit()
        finally:
            conn.close()

        upgraded = FeedbackStoreV2(old_db_path)
        with sqlite3.connect(old_db_path) as raw:
            raw.row_factory = sqlite3.Row
            table_names = {row["name"] for row in raw.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            index_names = {row["name"] for row in raw.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        self.assertIn("case_analysis", table_names)
        self.assertIn("idx_case_analysis_case", index_names)
        # Pre-existing data survives untouched.
        self.assertIsNotNone(upgraded.get_case("case-old"))
        del upgraded
        gc.collect()



# ---------------------------------------------------------------------------
# 2. list_feedback_for_case
# ---------------------------------------------------------------------------


class ListFeedbackForCaseTests(_StoreTestCase):
    def test_returns_only_this_cases_feedback(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        turn_a = self._add_turn("case-a", "turn-a")
        turn_b = self._add_turn("case-b", "turn-b")
        self._add_feedback(turn_a, helpful=True)
        self._add_feedback(turn_b, helpful=False)

        rows = self.store.list_feedback_for_case("case-a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["turn_id"], turn_a)

    def test_deterministic_ordering_by_submitted_at(self):
        self._seed_session_and_case()
        t1 = self._add_turn("case-1", "turn-1")
        t2 = self._add_turn("case-1", "turn-2")
        # Submitted out of chronological order -- list_feedback_for_case
        # must still return them ordered by submitted_at.
        self._add_feedback(t2, helpful=False, submitted_at="2026-01-01T00:00:02+00:00")
        self._add_feedback(t1, helpful=True, submitted_at="2026-01-01T00:00:01+00:00")

        rows = self.store.list_feedback_for_case("case-1")
        self.assertEqual([r["turn_id"] for r in rows], [t1, t2])

    def test_empty_case_returns_empty_list(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self.assertEqual(self.store.list_feedback_for_case("case-1"), [])

    def test_unknown_case_returns_empty_list(self):
        self.assertEqual(self.store.list_feedback_for_case("does-not-exist"), [])


# ---------------------------------------------------------------------------
# 3. get_case_evidence_watermark
# ---------------------------------------------------------------------------


class EvidenceWatermarkTests(_StoreTestCase):
    def test_case_does_not_exist_returns_none(self):
        self.assertIsNone(self.store.get_case_evidence_watermark("does-not-exist"))

    def test_case_exists_zero_turns_returns_none(self):
        self._seed_session_and_case()
        self.assertIsNone(self.store.get_case_evidence_watermark("case-1"))

    def test_turn_only(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T10:00:00+00:00")

    def test_turn_and_retrieval_retrieval_is_latest(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self._add_retrieval_run("turn-1", created_at="2026-01-01T10:00:07+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T10:00:07+00:00")

    def test_turn_and_feedback_feedback_is_latest(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self._add_feedback("turn-1", submitted_at="2026-01-01T10:00:10+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T10:00:10+00:00")

    def test_turn_is_latest_when_newer_than_retrieval_and_feedback(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self._add_retrieval_run("turn-1", created_at="2026-01-01T10:00:01+00:00")
        self._add_feedback("turn-1", submitted_at="2026-01-01T10:00:02+00:00")
        self._add_turn("case-1", "turn-2", created_at="2026-01-01T11:00:00+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T11:00:00+00:00")

    def test_multiple_turns_latest_wins(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self._add_turn("case-1", "turn-2", created_at="2026-01-01T09:00:00+00:00")
        self._add_turn("case-1", "turn-3", created_at="2026-01-01T12:00:00+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T12:00:00+00:00")

    def test_multiple_retrieval_runs_on_different_turns(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self._add_turn("case-1", "turn-2", created_at="2026-01-01T10:05:00+00:00")
        self._add_retrieval_run("turn-1", created_at="2026-01-01T10:00:07+00:00")
        self._add_retrieval_run("turn-2", created_at="2026-01-01T10:05:07+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T10:05:07+00:00")

    def test_multiple_feedback_rows_across_turns(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self._add_turn("case-1", "turn-2", created_at="2026-01-01T10:05:00+00:00")
        self._add_feedback("turn-1", submitted_at="2026-01-01T10:00:20+00:00")
        self._add_feedback("turn-2", submitted_at="2026-01-01T10:05:20+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T10:05:20+00:00")

    def test_cross_case_isolation(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a", created_at="2026-01-01T10:00:00+00:00")
        self._add_turn("case-b", "turn-b", created_at="2026-01-01T20:00:00+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-a"), "2026-01-01T10:00:00+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-b"), "2026-01-01T20:00:00+00:00")

    def test_zero_microsecond_and_nonzero_microsecond_timestamps_still_order_correctly(self):
        # datetime.isoformat() omits the microsecond component entirely
        # when it is exactly 0 -- a real _now() value can look shorter than
        # a neighboring one. Confirms lexical TEXT comparison in SQL still
        # orders these correctly (the shorter, no-microsecond string is a
        # valid string PREFIX of any later-in-that-second timestamp, so it
        # always sorts first, matching real chronological order).
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")  # no microseconds
        self._add_retrieval_run("turn-1", created_at="2026-01-01T10:00:00.500000+00:00")
        self.assertEqual(self.store.get_case_evidence_watermark("case-1"), "2026-01-01T10:00:00.500000+00:00")


# ---------------------------------------------------------------------------
# 4. Analysis persistence (create_case_analysis / get_latest_case_analysis)
# ---------------------------------------------------------------------------


class AnalysisPersistenceTests(_StoreTestCase):
    def test_create_then_get_latest(self):
        self._seed_session_and_case()
        analysis_id = self._create_analysis(
            "case-1", analyzed_at="2026-01-01T00:00:00+00:00", source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        row = self.store.get_latest_case_analysis("case-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["analysis_id"], analysis_id)

    def test_get_latest_with_no_analysis_returns_none(self):
        self._seed_session_and_case()
        self.assertIsNone(self.store.get_latest_case_analysis("case-1"))

    def test_second_analysis_becomes_latest_first_is_not_overwritten(self):
        self._seed_session_and_case()
        first_id = self._create_analysis(
            "case-1", analyzed_at="2026-01-01T00:00:00+00:00", source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        second_id = self._create_analysis(
            "case-1", analyzed_at="2026-01-02T00:00:00+00:00", source_evidence_watermark="2026-01-02T00:00:00+00:00",
        )
        latest = self.store.get_latest_case_analysis("case-1")
        self.assertEqual(latest["analysis_id"], second_id)

        with self._raw_connect() as conn:
            ids = {row["analysis_id"] for row in conn.execute(
                "SELECT analysis_id FROM case_analysis WHERE case_id=?", ("case-1",)
            ).fetchall()}
        self.assertEqual(ids, {first_id, second_id})

    def test_invalid_diagnosis_rejected(self):
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="not_a_real_diagnosis", diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_latest_case_analysis("case-1"))

    def test_diagnosis_dropped_legacy_values_rejected(self):
        self._seed_session_and_case()
        for legacy_value in ("workflow_tool_issue", "unclear_or_other"):
            with self.subTest(diagnosis=legacy_value):
                ok = self.store.create_case_analysis(
                    f"a-{legacy_value}", "case-1",
                    case_title=None, issue_summary=None,
                    issue_type="product_usage_or_application", issue_type_confidence=0.5,
                    diagnosis=legacy_value, diagnosis_confidence=0.5,
                    product_model=None, product_source=None, product_confidence=None,
                    evidence_json=None, analysis_version="v1",
                    analyzed_at="2026-01-01T00:00:00+00:00",
                    source_evidence_watermark="2026-01-01T00:00:00+00:00",
                )
                self.assertFalse(ok)

    def test_invalid_product_source_rejected(self):
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model="ADAM-6266", product_source="guessed", product_confidence=0.5,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_none_product_source_accepted(self):
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="no_issue_detected", diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertTrue(ok)

    def test_all_diagnosis_values_accepted(self):
        self._seed_session_and_case()
        self.assertEqual(len(DIAGNOSIS_VALUES), 5)
        for i, diagnosis in enumerate(DIAGNOSIS_VALUES):
            with self.subTest(diagnosis=diagnosis):
                ok = self.store.create_case_analysis(
                    f"a{i}", "case-1",
                    case_title=None, issue_summary=None,
                    issue_type="product_usage_or_application", issue_type_confidence=0.5,
                    diagnosis=diagnosis, diagnosis_confidence=0.5,
                    product_model=None, product_source=None, product_confidence=None,
                    evidence_json=None, analysis_version="v1",
                    analyzed_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                    source_evidence_watermark=f"2026-01-0{i + 1}T00:00:00+00:00",
                )
                self.assertTrue(ok)

    def test_all_issue_type_values_accepted(self):
        self._seed_session_and_case()
        self.assertEqual(len(ISSUE_TYPE_VALUES), 4)
        for i, issue_type in enumerate(ISSUE_TYPE_VALUES):
            with self.subTest(issue_type=issue_type):
                ok = self.store.create_case_analysis(
                    f"a{i}", "case-1",
                    case_title=None, issue_summary=None,
                    issue_type=issue_type, issue_type_confidence=0.5,
                    diagnosis="knowledge_gap", diagnosis_confidence=0.5,
                    product_model=None, product_source=None, product_confidence=None,
                    evidence_json=None, analysis_version="v1",
                    analyzed_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                    source_evidence_watermark=f"2026-01-0{i + 1}T00:00:00+00:00",
                )
                self.assertTrue(ok)

    def test_invalid_issue_type_rejected(self):
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="not_a_real_issue_type", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_all_product_source_values_accepted(self):
        self._seed_session_and_case()
        for i, source in enumerate(PRODUCT_SOURCE_VALUES):
            with self.subTest(product_source=source):
                ok = self.store.create_case_analysis(
                    f"a{i}", "case-1",
                    case_title=None, issue_summary=None,
                    issue_type="product_usage_or_application", issue_type_confidence=0.5,
                    diagnosis="knowledge_gap", diagnosis_confidence=0.5,
                    product_model="ADAM-6266", product_source=source, product_confidence=0.9,
                    evidence_json=None, analysis_version="v1",
                    analyzed_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                    source_evidence_watermark=f"2026-01-0{i + 1}T00:00:00+00:00",
                )
                self.assertTrue(ok)

    def test_product_model_present_missing_product_source_rejected(self):
        # Stage B.5 Schema Alignment: the storage layer now enforces the
        # same product_model/product_source/product_confidence consistency
        # rule CaseEnrichmentResult.__post_init__ already enforces -- this
        # was NOT previously checked at the storage layer (a pre-alignment
        # audit found the gap).
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model="ADAM-6266", product_source=None, product_confidence=0.9,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_product_model_present_missing_product_confidence_rejected(self):
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_no_product_model_but_product_confidence_present_rejected(self):
        self._seed_session_and_case()
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=0.5,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_confidence_boundary_and_invalid_values(self):
        self._seed_session_and_case()
        base_kwargs = dict(
            case_title=None, issue_summary=None,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        cases = [
            ("boundary_zero", {"issue_type_confidence": 0.0, "diagnosis_confidence": 0.5}, True),
            ("boundary_one", {"issue_type_confidence": 1.0, "diagnosis_confidence": 0.5}, True),
            ("diag_boundary_zero", {"issue_type_confidence": 0.5, "diagnosis_confidence": 0.0}, True),
            ("diag_boundary_one", {"issue_type_confidence": 0.5, "diagnosis_confidence": 1.0}, True),
            ("issue_below_zero", {"issue_type_confidence": -0.1, "diagnosis_confidence": 0.5}, False),
            ("issue_above_one", {"issue_type_confidence": 1.1, "diagnosis_confidence": 0.5}, False),
            ("issue_bool", {"issue_type_confidence": True, "diagnosis_confidence": 0.5}, False),
            ("issue_nan", {"issue_type_confidence": float("nan"), "diagnosis_confidence": 0.5}, False),
            ("issue_inf", {"issue_type_confidence": float("inf"), "diagnosis_confidence": 0.5}, False),
            ("diag_below_zero", {"issue_type_confidence": 0.5, "diagnosis_confidence": -0.1}, False),
            ("diag_above_one", {"issue_type_confidence": 0.5, "diagnosis_confidence": 1.1}, False),
            ("diag_bool", {"issue_type_confidence": 0.5, "diagnosis_confidence": True}, False),
            ("diag_nan", {"issue_type_confidence": 0.5, "diagnosis_confidence": float("nan")}, False),
            ("diag_inf", {"issue_type_confidence": 0.5, "diagnosis_confidence": float("inf")}, False),
        ]
        for i, (name, overrides, expected) in enumerate(cases):
            with self.subTest(case=name):
                ok = self.store.create_case_analysis(
                    f"a{i}", "case-1",
                    issue_type="product_usage_or_application",
                    diagnosis="knowledge_gap",
                    analyzed_at=f"2026-02-{i + 1:02d}T00:00:00+00:00",
                    **base_kwargs, **overrides,
                )
                self.assertEqual(ok, expected, name)

    def test_product_confidence_invalid_values_rejected(self):
        self._seed_session_and_case()
        for bad_value in (-0.1, 1.1, True, float("nan"), float("inf")):
            with self.subTest(product_confidence=bad_value):
                ok = self.store.create_case_analysis(
                    "a1", "case-1",
                    case_title=None, issue_summary=None,
                    issue_type="product_usage_or_application", issue_type_confidence=0.5,
                    diagnosis="knowledge_gap", diagnosis_confidence=0.5,
                    product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=bad_value,
                    evidence_json=None, analysis_version="v1",
                    analyzed_at="2026-01-01T00:00:00+00:00",
                    source_evidence_watermark="2026-01-01T00:00:00+00:00",
                )
                self.assertFalse(ok)

    def test_unknown_case_id_rejected_not_raised(self):
        ok = self.store.create_case_analysis(
            "a1", "does-not-exist",
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_duplicate_case_id_and_analyzed_at_rejected_not_raised(self):
        self._seed_session_and_case()
        self._create_analysis("case-1", analyzed_at="2026-01-01T00:00:00+00:00", source_evidence_watermark="2026-01-01T00:00:00+00:00", analysis_id="a1")
        ok = self.store.create_case_analysis(
            "a2", "case-1",  # different analysis_id, SAME (case_id, analyzed_at)
            case_title=None, issue_summary=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(ok)

    def test_evidence_json_and_case_title_round_trip(self):
        self._seed_session_and_case()
        evidence = json.dumps([{"type": "user_text", "turn_id": "turn-1", "fact": "User wrote ADAM-6266"}])
        ok = self.store.create_case_analysis(
            "a1", "case-1",
            case_title="ADAM-6266 SNMP issue", issue_summary="summary text",
            issue_type="product_capability_or_compatibility", issue_type_confidence=0.85,
            diagnosis="knowledge_gap", diagnosis_confidence=0.7,
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.95,
            evidence_json=evidence, analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        self.assertTrue(ok)
        row = self.store.get_latest_case_analysis("case-1")
        self.assertEqual(row["case_title"], "ADAM-6266 SNMP issue")
        self.assertEqual(row["issue_type"], "product_capability_or_compatibility")
        self.assertEqual(row["issue_type_confidence"], 0.85)
        self.assertEqual(row["diagnosis"], "knowledge_gap")
        self.assertEqual(row["diagnosis_confidence"], 0.7)
        self.assertEqual(row["product_model"], "ADAM-6266")
        self.assertEqual(row["product_source"], "explicit_user_text")
        self.assertEqual(row["product_confidence"], 0.95)
        self.assertEqual(json.loads(row["evidence_json"]), json.loads(evidence))


# ---------------------------------------------------------------------------
# 5. Needs-analysis lifecycle -- the most important Stage A regression
#    contract, per the task: this must hold WITHOUT ever writing
#    cases.updated_at.
# ---------------------------------------------------------------------------


class NeedsAnalysisLifecycleTests(_StoreTestCase):
    def _needs_analysis(self, case_id: str) -> bool:
        return case_id in {row["case_id"] for row in self.store.list_cases_needing_analysis()}

    def test_full_lifecycle_cycle(self):
        self._seed_session_and_case()

        # 1. Case exists, no turns, no analysis -- zero evidence, must NOT
        #    be reported (nothing to analyze yet).
        self.assertFalse(self._needs_analysis("case-1"))

        # Turn 1 arrives.
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        self.assertTrue(self._needs_analysis("case-1"))  # 1. needs analysis

        # 2. Analyze at the current watermark -- no longer needs analysis.
        watermark_1 = self.store.get_case_evidence_watermark("case-1")
        self.assertEqual(watermark_1, "2026-01-01T10:00:00+00:00")
        self._create_analysis("case-1", analyzed_at="2026-01-01T10:00:01+00:00", source_evidence_watermark=watermark_1)
        self.assertFalse(self._needs_analysis("case-1"))

        # 3. Add a new Turn -- needs analysis again.
        self._add_turn("case-1", "turn-2", created_at="2026-01-01T11:00:00+00:00")
        self.assertTrue(self._needs_analysis("case-1"))

        # 4. Create new analysis -- no longer needs analysis.
        watermark_2 = self.store.get_case_evidence_watermark("case-1")
        self.assertEqual(watermark_2, "2026-01-01T11:00:00+00:00")
        self._create_analysis("case-1", analyzed_at="2026-01-01T11:00:01+00:00", source_evidence_watermark=watermark_2)
        self.assertFalse(self._needs_analysis("case-1"))

        # 5. Add Feedback -- needs analysis again.
        self._add_feedback("turn-2", submitted_at="2026-01-01T12:00:00+00:00")
        self.assertTrue(self._needs_analysis("case-1"))

        # 6. Create new analysis -- no longer needs analysis.
        watermark_3 = self.store.get_case_evidence_watermark("case-1")
        self.assertEqual(watermark_3, "2026-01-01T12:00:00+00:00")
        self._create_analysis("case-1", analyzed_at="2026-01-01T12:00:01+00:00", source_evidence_watermark=watermark_3)
        self.assertFalse(self._needs_analysis("case-1"))

        # 7. Add Retrieval evidence -- needs analysis again.
        self._add_retrieval_run("turn-2", created_at="2026-01-01T13:00:00+00:00")
        self.assertTrue(self._needs_analysis("case-1"))

        # This entire cycle held without ever writing cases.updated_at.
        case = self.store.get_case("case-1")
        self.assertEqual(case["updated_at"], case["created_at"])

    def test_case_with_zero_turns_never_appears(self):
        self._seed_session_and_case()
        self.assertFalse(self._needs_analysis("case-1"))

    def test_cross_case_isolation(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a", created_at="2026-01-01T10:00:00+00:00")
        self._add_turn("case-b", "turn-b", created_at="2026-01-01T10:00:00+00:00")
        # Fully analyze case-a only.
        wm = self.store.get_case_evidence_watermark("case-a")
        self._create_analysis("case-a", analyzed_at="2026-01-01T10:00:01+00:00", source_evidence_watermark=wm)

        result = {row["case_id"] for row in self.store.list_cases_needing_analysis()}
        self.assertNotIn("case-a", result)
        self.assertIn("case-b", result)

    def test_session_id_filter(self):
        self.store.create_or_update_session("sess-a", "telegram")
        self.store.create_or_update_session("sess-b", "telegram")
        self.store.create_case("case-a", "sess-a")
        self.store.create_case("case-b", "sess-b")
        self._add_turn("case-a", "turn-a", created_at="2026-01-01T10:00:00+00:00")
        self._add_turn("case-b", "turn-b", created_at="2026-01-01T10:00:00+00:00")

        result = {row["case_id"] for row in self.store.list_cases_needing_analysis(session_id="sess-a")}
        self.assertEqual(result, {"case-a"})

    def test_limit_applied_after_ordering_oldest_evidence_first(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a", created_at="2026-01-01T12:00:00+00:00")
        self._add_turn("case-b", "turn-b", created_at="2026-01-01T09:00:00+00:00")

        result = self.store.list_cases_needing_analysis(limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["case_id"], "case-b")  # older evidence favored first

    def test_evidence_watermark_column_present_on_returned_rows(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        rows = self.store.list_cases_needing_analysis()
        self.assertEqual(rows[0]["case_id"], "case-1")
        self.assertEqual(rows[0]["evidence_watermark"], "2026-01-01T10:00:00+00:00")

    def test_never_needs_analysis_after_watermark_tied_analysis(self):
        # A boundary check on the strict "greater than" comparison: an
        # analysis whose source_evidence_watermark exactly equals the
        # current evidence watermark must count as fully up to date (not
        # stale), never re-flagged just because it isn't strictly newer.
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", created_at="2026-01-01T10:00:00+00:00")
        wm = self.store.get_case_evidence_watermark("case-1")
        self._create_analysis("case-1", analyzed_at="2026-01-01T10:00:01+00:00", source_evidence_watermark=wm)
        self.assertFalse(self._needs_analysis("case-1"))


if __name__ == "__main__":
    unittest.main()
