"""Migration 003 tests: widening retrieval_runs.execution_status to allow
'blocked'/'unparseable' on a database that already has the original
(Phase 2, commit 099abb5) 002 schema applied.

These tests deliberately build the "old" database from
002_feedback_schema_v2.sql's own file content (not from
tools.feedback_store_v2's current _SCHEMA_STATEMENTS), so a future
accidental edit to 002_feedback_schema_v2.sql would be caught by
test_002_sql_doc_matches_099abb5_exactly below, and any regression in the
upgrade path would be caught by everything else here -- both against the
actual historical schema, not a paraphrase of it.
"""
from __future__ import annotations

import gc
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
_OVERLAY_ROOT = _TESTS_DIR.parents[0] / "overlay"
_REPO_ROOT = _TESTS_DIR.parents[2]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

_MIGRATIONS_DIR = _OVERLAY_ROOT / "migrations"
_002_SQL_PATH = _MIGRATIONS_DIR / "002_feedback_schema_v2.sql"
_003_SQL_PATH = _MIGRATIONS_DIR / "003_retrieval_execution_statuses.sql"
_PHASE2_COMMIT = "099abb5"

_NOW = "2026-01-01T00:00:00+00:00"


def _seed_old_v2_schema(db_path: Path) -> None:
    """Build a database using 002_feedback_schema_v2.sql's own file
    content -- the original (pre-003) 8-value execution_status CHECK --
    then insert one legitimate old-shape retrieval row."""
    sql_text = _002_SQL_PATH.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql_text)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO sessions (session_id, platform, platform_chat_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("sess-old", "telegram", "chat-old", _NOW, _NOW),
        )
        conn.execute(
            "INSERT INTO cases (case_id, session_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("case-old", "sess-old", _NOW, _NOW),
        )
        conn.execute(
            """
            INSERT INTO turns (
                turn_id, case_id, session_id, platform_user_id, platform_user_message_id,
                question_text, answer_text, feedback_eligible, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("turn-old", "case-old", "sess-old", "user-old", "msg-old", "old question", "old answer", 1, _NOW, _NOW),
        )
        conn.execute(
            """
            INSERT INTO retrieval_runs (
                retrieval_id, turn_id, invocation_order, tool_call_id,
                execution_status, foundry_iq_ok, observation_status,
                result_count, reference_count, foundry_schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("retrieval-old", "turn-old", 1, "call-old", "completed", 1, "complete", 2, 1, "foundry-iq-result-v2", _NOW),
        )
        conn.commit()
    finally:
        conn.close()


class _MigrationTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"

    def tearDown(self):
        gc.collect()
        self._tmpdir.cleanup()

    def _raw_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


class OldSchemaFixtureTests(_MigrationTestCase):
    """Sanity: the fixture itself really does build the pre-003 shape,
    with no 'blocked'/'unparseable' allowed -- otherwise every test below
    would be exercising nothing."""

    def test_fixture_old_schema_rejects_blocked_before_upgrade(self):
        _seed_old_v2_schema(self.db_path)
        conn = self._raw_connect()
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO retrieval_runs (retrieval_id, turn_id, invocation_order, execution_status, observation_status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("retrieval-blocked", "turn-old", 2, "blocked", "complete", _NOW),
                )
        finally:
            conn.close()


class UpgradePathTests(_MigrationTestCase):
    def test_old_row_fully_preserved_after_upgrade(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        store = FeedbackStoreV2(self.db_path)
        rows = store.list_retrieval_runs_for_turn("turn-old")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["retrieval_id"], "retrieval-old")
        self.assertEqual(row["tool_call_id"], "call-old")
        self.assertEqual(row["execution_status"], "completed")
        self.assertEqual(row["foundry_iq_ok"], 1)
        self.assertEqual(row["observation_status"], "complete")
        self.assertEqual(row["result_count"], 2)
        self.assertEqual(row["reference_count"], 1)
        self.assertEqual(row["foundry_schema_version"], "foundry-iq-result-v2")
        self.assertEqual(row["created_at"], _NOW)
        del store
        gc.collect()

    def test_blocked_insertable_after_upgrade(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput

        store = FeedbackStoreV2(self.db_path)
        ok = store.add_retrieval_runs(
            "turn-old", [RetrievalRunInput(execution_status="blocked", observation_status="complete")]
        )
        self.assertTrue(ok)
        statuses = {r["execution_status"] for r in store.list_retrieval_runs_for_turn("turn-old")}
        self.assertIn("blocked", statuses)
        del store
        gc.collect()

    def test_unparseable_insertable_after_upgrade(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput

        store = FeedbackStoreV2(self.db_path)
        ok = store.add_retrieval_runs(
            "turn-old",
            [RetrievalRunInput(execution_status="unparseable", observation_status="partial", observation_reason="missing_inner_result")],
        )
        self.assertTrue(ok)
        statuses = {r["execution_status"] for r in store.list_retrieval_runs_for_turn("turn-old")}
        self.assertIn("unparseable", statuses)
        del store
        gc.collect()

    def test_fk_unique_and_indexes_preserved(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        store = FeedbackStoreV2(self.db_path)
        conn = self._raw_connect()
        try:
            fks = conn.execute("PRAGMA foreign_key_list(retrieval_runs)").fetchall()
            self.assertEqual(len(fks), 1)
            self.assertEqual(fks[0]["table"], "turns")
            self.assertEqual(fks[0]["from"], "turn_id")
            self.assertEqual(fks[0]["to"], "turn_id")

            index_names = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='retrieval_runs'"
                ).fetchall()
            }
            self.assertIn("idx_retrieval_runs_turn", index_names)

            create_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertIn("UNIQUE (turn_id, invocation_order)", create_sql)
            for other_check in (
                "request_attempted IN (0, 1)",
                "foundry_iq_ok IN (0, 1)",
                "observation_status IN ('complete', 'partial', 'unavailable')",
                "http_status IS NULL OR (http_status BETWEEN 100 AND 599)",
            ):
                self.assertIn(other_check, create_sql)

            # UNIQUE(turn_id, invocation_order) still actually enforced.
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO retrieval_runs (retrieval_id, turn_id, invocation_order, execution_status, observation_status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("dup", "turn-old", 1, "completed", "complete", _NOW),
                )
            conn.rollback()

            # FK to turns(turn_id) still actually enforced.
            conn.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO retrieval_runs (retrieval_id, turn_id, invocation_order, execution_status, observation_status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("orphan", "no-such-turn", 99, "completed", "complete", _NOW),
                )
        finally:
            conn.close()
        del store
        gc.collect()

    def test_upgrade_marker_detection_false_after_upgrade(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        store = FeedbackStoreV2(self.db_path)
        conn = self._raw_connect()
        try:
            self.assertFalse(store._retrieval_runs_needs_execution_status_upgrade(conn))
        finally:
            conn.close()
        del store
        gc.collect()

    def test_repeated_init_is_idempotent(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        store1 = FeedbackStoreV2(self.db_path)
        del store1
        gc.collect()

        # Second init against an already-upgraded database must not fail
        # and must not duplicate or lose the row.
        store2 = FeedbackStoreV2(self.db_path)
        rows = store2.list_retrieval_runs_for_turn("turn-old")
        self.assertEqual(len(rows), 1)
        del store2
        gc.collect()

        store3 = FeedbackStoreV2(self.db_path)
        rows3 = store3.list_retrieval_runs_for_turn("turn-old")
        self.assertEqual(len(rows3), 1)
        del store3
        gc.collect()

    def test_fresh_database_never_needs_upgrade(self):
        # A brand-new database (no old schema seeded) is created directly
        # in the post-003 shape by _SCHEMA_STATEMENTS -- the upgrade path
        # must recognize this as already-current, not attempt a rebuild
        # against a table that was never in the old shape.
        from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput

        store = FeedbackStoreV2(self.db_path)
        conn = self._raw_connect()
        try:
            self.assertFalse(store._retrieval_runs_needs_execution_status_upgrade(conn))
        finally:
            conn.close()
        store.create_or_update_session("s1", "telegram")
        store.create_case("c1", "s1")
        store.create_turn(
            "t1", "c1", platform_user_id="u1", platform_user_message_id="m1",
            question_text="q", answer_text="a", feedback_eligible=True,
        )
        ok = store.add_retrieval_runs("t1", [RetrievalRunInput(execution_status="blocked", observation_status="complete")])
        self.assertTrue(ok)
        del store
        gc.collect()


class CompletionMarkerTests(_MigrationTestCase):
    """Item 2: _retrieval_runs_needs_execution_status_upgrade must require
    BOTH 'blocked' and 'unparseable' in the live CHECK constraint text --
    checking only one would treat an abnormal partial schema as already
    complete and silently skip the rebuild."""

    def test_old_002_neither_marker_present_needs_upgrade(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        conn = self._raw_connect()
        try:
            sql_text = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertNotIn("'blocked'", sql_text)
            self.assertNotIn("'unparseable'", sql_text)
        finally:
            conn.close()

        store = FeedbackStoreV2.__new__(FeedbackStoreV2)
        store.path = self.db_path
        conn2 = self._raw_connect()
        try:
            self.assertTrue(store._retrieval_runs_needs_execution_status_upgrade(conn2))
        finally:
            conn2.close()

    def test_complete_003_both_markers_present_is_noop(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        FeedbackStoreV2(self.db_path)  # runs the real upgrade once
        gc.collect()

        store = FeedbackStoreV2.__new__(FeedbackStoreV2)
        store.path = self.db_path
        conn = self._raw_connect()
        try:
            sql_text = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertIn("'blocked'", sql_text)
            self.assertIn("'unparseable'", sql_text)
            self.assertFalse(store._retrieval_runs_needs_execution_status_upgrade(conn))
        finally:
            conn.close()

    def test_abnormal_partial_schema_blocked_only_still_needs_upgrade(self):
        # Simulates a hand-edited / partially-migrated database: the CHECK
        # constraint has 'blocked' but not 'unparseable'. Checking only
        # the 'blocked' marker would wrongly call this "already upgraded".
        _seed_old_v2_schema(self.db_path)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                PRAGMA foreign_keys=OFF;
                BEGIN;
                ALTER TABLE retrieval_runs RENAME TO retrieval_runs_old;
                CREATE TABLE retrieval_runs (
                    retrieval_id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                    invocation_order INTEGER NOT NULL,
                    tool_call_id TEXT,
                    request_attempted INTEGER CHECK (request_attempted IN (0, 1)),
                    execution_status TEXT NOT NULL CHECK (execution_status IN (
                        'completed', 'failed', 'timed_out', 'http_error',
                        'network_error', 'invalid_response', 'no_documents', 'unknown',
                        'blocked'
                    )),
                    foundry_iq_ok INTEGER CHECK (foundry_iq_ok IN (0, 1)),
                    observation_status TEXT NOT NULL
                        CHECK (observation_status IN ('complete', 'partial', 'unavailable')),
                    observation_reason TEXT,
                    error_code TEXT,
                    http_status INTEGER CHECK (http_status IS NULL OR (http_status BETWEEN 100 AND 599)),
                    result_count INTEGER,
                    reference_count INTEGER,
                    foundry_schema_version TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (turn_id, invocation_order)
                );
                INSERT INTO retrieval_runs SELECT * FROM retrieval_runs_old;
                DROP TABLE retrieval_runs_old;
                CREATE INDEX IF NOT EXISTS idx_retrieval_runs_turn ON retrieval_runs(turn_id);
                COMMIT;
                PRAGMA foreign_keys=ON;
                """
            )
        finally:
            conn.close()

        from tools.feedback_store_v2 import FeedbackStoreV2

        store = FeedbackStoreV2.__new__(FeedbackStoreV2)
        store.path = self.db_path
        conn2 = self._raw_connect()
        try:
            sql_text = conn2.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertIn("'blocked'", sql_text)
            self.assertNotIn("'unparseable'", sql_text)
            self.assertTrue(
                store._retrieval_runs_needs_execution_status_upgrade(conn2),
                "must still need upgrade when only one of the two markers is present",
            )
        finally:
            conn2.close()

        # And a real init must actually fix it, not skip.
        real_store = FeedbackStoreV2(self.db_path)
        conn3 = self._raw_connect()
        try:
            sql_text2 = conn3.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertIn("'unparseable'", sql_text2)
        finally:
            conn3.close()
        self.assertEqual(len(real_store.list_retrieval_runs_for_turn("turn-old")), 1)
        del real_store
        gc.collect()

    def test_repeated_init_safe_across_marker_check(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_store_v2 import FeedbackStoreV2

        for _ in range(3):
            store = FeedbackStoreV2(self.db_path)
            self.assertEqual(len(store.list_retrieval_runs_for_turn("turn-old")), 1)
            del store
            gc.collect()


class ForeignKeyCheckReallyEnforcedTests(_MigrationTestCase):
    """Item 2: PRAGMA foreign_key_check(retrieval_runs)'s result must be
    fetched and acted on, not just executed and ignored -- prove this with
    a genuine orphaned row, not just an injected-SQL-error failure."""

    def test_orphaned_row_triggers_rollback_not_silent_success(self):
        _seed_old_v2_schema(self.db_path)
        # Insert a retrieval_runs row whose turn_id does not exist in
        # turns -- only possible with foreign_keys off, exactly like a
        # real database that had FK enforcement disabled at some point.
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(
                "INSERT INTO retrieval_runs (retrieval_id, turn_id, invocation_order, execution_status, observation_status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("orphan-retrieval", "turn-does-not-exist", 1, "completed", "complete", _NOW),
            )
            conn.commit()
        finally:
            conn.close()

        from tools.feedback_store_v2 import FeedbackStoreV2

        with self.assertRaises(sqlite3.IntegrityError):
            FeedbackStoreV2(self.db_path)

        # Rolled back: retrieval_runs must still be in the OLD shape, with
        # BOTH the legitimate row and the orphan still present unchanged
        # (a real rollback undoes the whole rebuild, it doesn't try to
        # selectively fix the offending row).
        conn2 = self._raw_connect()
        try:
            sql_text = conn2.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertNotIn("'blocked'", sql_text)
            rows = conn2.execute("SELECT retrieval_id FROM retrieval_runs ORDER BY retrieval_id").fetchall()
            self.assertEqual({r[0] for r in rows}, {"retrieval-old", "orphan-retrieval"})
        finally:
            conn2.close()


class MigrationFailureRollbackTests(_MigrationTestCase):
    def test_induced_failure_leaves_original_table_and_data_intact(self):
        _seed_old_v2_schema(self.db_path)
        from tools import feedback_store_v2 as fsv2

        broken_statements = fsv2._RETRIEVAL_RUNS_REBUILD_STATEMENTS + (
            "SELECT * FROM this_table_does_not_exist_at_all",
        )
        with mock.patch.object(fsv2, "_RETRIEVAL_RUNS_REBUILD_STATEMENTS", broken_statements):
            with self.assertRaises(sqlite3.OperationalError):
                fsv2.FeedbackStoreV2(self.db_path)

        # The failed attempt must not have left a stray shadow table, must
        # not have dropped/renamed the original table, and the original
        # row must be untouched -- verified with a fresh raw connection
        # (not the broken store instance).
        conn = self._raw_connect()
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertIn("retrieval_runs", tables)
            self.assertNotIn("retrieval_runs_new", tables)

            create_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
            ).fetchone()[0]
            self.assertNotIn("'blocked'", create_sql)  # still the old, pre-003 constraint

            rows = conn.execute("SELECT * FROM retrieval_runs").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["retrieval_id"], "retrieval-old")
            self.assertEqual(rows[0]["execution_status"], "completed")
        finally:
            conn.close()

    def test_can_retry_successfully_after_a_failed_attempt(self):
        _seed_old_v2_schema(self.db_path)
        from tools import feedback_store_v2 as fsv2

        broken_statements = fsv2._RETRIEVAL_RUNS_REBUILD_STATEMENTS + (
            "SELECT * FROM this_table_does_not_exist_at_all",
        )
        with mock.patch.object(fsv2, "_RETRIEVAL_RUNS_REBUILD_STATEMENTS", broken_statements):
            with self.assertRaises(sqlite3.OperationalError):
                fsv2.FeedbackStoreV2(self.db_path)

        # Retrying with the real (unpatched) migration must succeed and
        # preserve the row, proving the failed attempt left no residue
        # that would block a subsequent real run.
        store = fsv2.FeedbackStoreV2(self.db_path)
        rows = store.list_retrieval_runs_for_turn("turn-old")
        self.assertEqual(len(rows), 1)
        ok = store.add_retrieval_runs(
            "turn-old", [fsv2.RetrievalRunInput(execution_status="blocked", observation_status="complete")]
        )
        self.assertTrue(ok)
        del store
        gc.collect()


class LegacyFeedbackRunsUntouchedTests(_MigrationTestCase):
    def test_legacy_feedback_runs_schema_and_rows_unchanged(self):
        _seed_old_v2_schema(self.db_path)
        from tools.feedback_storage import FeedbackStore

        legacy_store = FeedbackStore(self.db_path)
        legacy_store.create_run(
            "run-1", "chat-1", turn_key="telegram:chat-1:msg-1",
            question_text="legacy q", answer_text="legacy a",
        )
        del legacy_store
        gc.collect()

        conn_before = self._raw_connect()
        try:
            schema_before = conn_before.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback_runs'"
            ).fetchone()[0]
            rows_before = [dict(r) for r in conn_before.execute("SELECT * FROM feedback_runs").fetchall()]
        finally:
            conn_before.close()

        from tools.feedback_store_v2 import FeedbackStoreV2

        store = FeedbackStoreV2(self.db_path)
        del store
        gc.collect()

        conn_after = self._raw_connect()
        try:
            schema_after = conn_after.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback_runs'"
            ).fetchone()[0]
            rows_after = [dict(r) for r in conn_after.execute("SELECT * FROM feedback_runs").fetchall()]
        finally:
            conn_after.close()

        self.assertEqual(schema_before, schema_after)
        self.assertEqual(rows_before, rows_after)
        self.assertEqual(len(rows_after), 1)
        self.assertEqual(rows_after[0]["run_id"], "run-1")


class MigrationDocumentationTests(unittest.TestCase):
    def test_002_sql_doc_matches_099abb5_exactly(self):
        try:
            result = subprocess.run(
                ["git", "show", f"{_PHASE2_COMMIT}:custom/universal-feedback/overlay/migrations/002_feedback_schema_v2.sql"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            self.skipTest(f"git show {_PHASE2_COMMIT}:... unavailable in this environment: {exc}")
            return
        original_bytes = result.stdout
        current_bytes = _002_SQL_PATH.read_bytes()
        self.assertEqual(
            current_bytes, original_bytes,
            "002_feedback_schema_v2.sql must stay byte-for-byte identical to commit 099abb5",
        )

    def test_003_sql_file_exists_and_is_additive_only(self):
        self.assertTrue(_003_SQL_PATH.exists())
        text = _003_SQL_PATH.read_text(encoding="utf-8")
        self.assertNotIn("DROP TABLE feedback_runs", text)
        self.assertNotIn("DROP TABLE sessions", text)
        self.assertNotIn("DROP TABLE cases", text)
        self.assertNotIn("DROP TABLE turns", text)
        self.assertNotIn("DROP TABLE feedback", text)
        self.assertIn("retrieval_runs", text)


if __name__ == "__main__":
    unittest.main()
