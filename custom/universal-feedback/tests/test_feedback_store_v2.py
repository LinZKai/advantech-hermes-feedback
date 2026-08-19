"""Tests for tools.feedback_store_v2 (Session -> Case -> Turn ->
{Retrieval Run, Feedback} schema v2).

Run locally with:
    python -m unittest discover -s custom/universal-feedback/tests -v

No production dependency is added; this uses only the standard library
(sqlite3 against a temp file per test). Never touches
/sandbox/.hermes/data/support_feedback.db.
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

from tools.feedback_storage import FeedbackStore, MAX_SUGGESTION_TEXT_LENGTH  # noqa: E402
from tools.feedback_store_v2 import (  # noqa: E402
    FeedbackStoreV2,
    RetrievalRunInput,
    _SCHEMA_STATEMENTS,
)
from tools.universal_feedback import POLICY_VERSION  # noqa: E402

_MIGRATION_002_PATH = (
    Path(__file__).resolve().parents[1]
    / "overlay" / "migrations" / "002_feedback_schema_v2.sql"
)


class _V2StoreTestCase(unittest.TestCase):
    """Shared temp-DB setup/teardown, matching test_feedback_storage.py's
    Windows-safe cleanup pattern (sqlite3 connections are only closed on
    garbage collection, and an open handle blocks temp-dir deletion)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    # -- fixtures ---------------------------------------------------------

    def _new_session(self, session_id: str = "sess-1", **kwargs) -> str:
        return self.store.create_or_update_session(session_id, "telegram", **kwargs)

    def _new_case(self, case_id: str = "case-1", session_id: str = "sess-1", **kwargs) -> str:
        return self.store.create_case(case_id, session_id, **kwargs)

    def _new_turn(
        self,
        turn_id: str = "turn-1",
        case_id: str = "case-1",
        *,
        platform_user_message_id: str | None = None,
        **kwargs,
    ) -> str:
        fields = dict(
            platform_user_id="user-1",
            platform_user_message_id=platform_user_message_id or f"msg-{turn_id}",
            question_text="ADAM-6266 如何開啟 SNMP？",
            answer_text="請至設定頁面啟用 SNMP。",
            feedback_eligible=True,
        )
        fields.update(kwargs)
        return self.store.create_turn(turn_id, case_id, **fields)

    def _new_session_case_turn(self, suffix: str = "1") -> tuple[str, str, str]:
        session_id = self._new_session(f"sess-{suffix}")
        case_id = self._new_case(f"case-{suffix}", session_id)
        turn_id = self._new_turn(f"turn-{suffix}", case_id, platform_user_message_id=f"msg-{suffix}")
        return session_id, case_id, turn_id


class MigrationTests(_V2StoreTestCase):
    def test_migration_applies_to_fresh_db(self):
        with self.store._connect() as db:
            tables = {
                row[0]
                for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        for table in ("sessions", "cases", "turns", "retrieval_runs", "feedback"):
            self.assertIn(table, tables)

    def test_migration_applies_on_top_of_legacy_001_schema_and_preserves_data(self):
        legacy_store = FeedbackStore(self.db_path)
        created = legacy_store.create_run("legacy-run-1", "chat-1")
        self.assertTrue(created)
        del legacy_store
        gc.collect()

        v2_store = FeedbackStoreV2(self.db_path)  # must not raise

        with v2_store._connect() as db:
            tables = {
                row[0]
                for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self.assertIn("feedback_runs", tables)
            row = db.execute(
                "SELECT * FROM feedback_runs WHERE run_id=?", ("legacy-run-1",)
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["chat_id"], "chat-1")

    def test_migration_is_repeatable(self):
        # Re-instantiating (re-running _migrate_schema) must not raise or
        # disturb existing rows.
        self._new_session_case_turn()
        FeedbackStoreV2(self.db_path)
        FeedbackStoreV2(self.db_path)
        row = self.store.get_turn("turn-1")
        self.assertIsNotNone(row)

    def test_migration_sql_file_has_no_drop_table(self):
        content = _MIGRATION_002_PATH.read_text(encoding="utf-8")
        self.assertNotIn("DROP TABLE", content.upper())

    def test_migration_sql_file_does_not_alter_legacy_table(self):
        content = _MIGRATION_002_PATH.read_text(encoding="utf-8")
        self.assertNotIn("ALTER TABLE feedback_runs", content.upper())

    def test_schema_statements_constant_has_no_drop(self):
        for statement in _SCHEMA_STATEMENTS:
            self.assertNotIn("DROP", statement.upper())


class ConnectionPragmaTests(_V2StoreTestCase):
    def test_foreign_keys_enabled(self):
        with self.store._connect() as db:
            value = db.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(value, 1)

    def test_wal_journal_mode(self):
        with self.store._connect() as db:
            mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")

    def test_busy_timeout_set(self):
        with self.store._connect() as db:
            timeout_ms = db.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(timeout_ms, 5000)


class SessionCaseTurnTests(_V2StoreTestCase):
    def test_session_case_turn_write_and_read(self):
        session_id, case_id, turn_id = self._new_session_case_turn()

        session = self.store.get_session(session_id)
        self.assertEqual(session["platform"], "telegram")

        case = self.store.get_case(case_id)
        self.assertEqual(case["session_id"], session_id)

        turn = self.store.get_turn(turn_id)
        self.assertEqual(turn["case_id"], case_id)
        self.assertEqual(turn["session_id"], session_id)
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")

    def test_create_or_update_session_is_idempotent_upsert(self):
        self.store.create_or_update_session("sess-x", "telegram", platform_chat_id="chat-1")
        self.store.create_or_update_session("sess-x", "telegram", platform_chat_id="chat-2")
        row = self.store.get_session("sess-x")
        self.assertEqual(row["platform_chat_id"], "chat-2")

    def test_list_cases_for_session(self):
        session_id = self._new_session()
        self._new_case("case-a", session_id)
        self._new_case("case-b", session_id)
        cases = self.store.list_cases_for_session(session_id)
        self.assertEqual({c["case_id"] for c in cases}, {"case-a", "case-b"})

    def test_list_turns_for_case_and_session(self):
        session_id = self._new_session()
        case_id = self._new_case(session_id=session_id)
        self._new_turn("turn-a", case_id, platform_user_message_id="m-a")
        self._new_turn("turn-b", case_id, platform_user_message_id="m-b")

        by_case = self.store.list_turns_for_case(case_id)
        self.assertEqual({t["turn_id"] for t in by_case}, {"turn-a", "turn-b"})

        by_session = self.store.list_turns_for_session(session_id)
        self.assertEqual({t["turn_id"] for t in by_session}, {"turn-a", "turn-b"})

    def test_list_turn_questions_for_cases_orders_by_case_then_created_at(self):
        session_id = self._new_session()
        case_a = self._new_case("case-a", session_id)
        case_b = self._new_case("case-b", session_id)
        self._new_turn("turn-a1", case_a, platform_user_message_id="m-a1", question_text="a first")
        self._new_turn("turn-a2", case_a, platform_user_message_id="m-a2", question_text="a second")
        self._new_turn("turn-b1", case_b, platform_user_message_id="m-b1", question_text="b first")

        rows = self.store.list_turn_questions_for_cases([case_a, case_b])
        by_case: dict[str, list[str]] = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row["question_text"])

        self.assertEqual(by_case[case_a], ["a first", "a second"])
        self.assertEqual(by_case[case_b], ["b first"])

    def test_list_turn_questions_for_cases_empty_input_returns_empty_without_query(self):
        self.assertEqual(self.store.list_turn_questions_for_cases([]), [])

    def test_turn_session_id_is_derived_from_case_not_caller(self):
        session_id = self._new_session()
        case_id = self._new_case(session_id=session_id)
        turn_id = self._new_turn(case_id=case_id)
        turn = self.store.get_turn(turn_id)
        self.assertEqual(turn["session_id"], session_id)

    def test_create_turn_unknown_case_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            self._new_turn(case_id="does-not-exist")

    def test_unicode_question_answer_round_trip(self):
        session_id = self._new_session()
        case_id = self._new_case(session_id=session_id)
        turn_id = self._new_turn(
            case_id=case_id,
            question_text="ADAM-6266 支援 SNMP v2c 嗎？🤔",
            answer_text="支援。請參考文件「ADAM-6266 SNMP.pdf」🎉 — includes emoji and CJK.",
        )
        row = self.store.get_turn(turn_id)
        self.assertEqual(row["question_text"], "ADAM-6266 支援 SNMP v2c 嗎？🤔")
        self.assertEqual(
            row["answer_text"],
            "支援。請參考文件「ADAM-6266 SNMP.pdf」🎉 — includes emoji and CJK.",
        )


class RetrievalRunTests(_V2StoreTestCase):
    def test_turn_with_zero_retrievals(self):
        _s, _c, turn_id = self._new_session_case_turn()
        rows = self.store.list_retrieval_runs_for_turn(turn_id)
        self.assertEqual(rows, [])

    def test_turn_with_one_retrieval(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.add_retrieval_runs(
            turn_id,
            [
                RetrievalRunInput(
                    execution_status="completed",
                    observation_status="complete",
                    request_attempted=True,
                    foundry_iq_ok=True,
                    result_count=3,
                    reference_count=2,
                    foundry_schema_version="foundry-iq-result-v2",
                )
            ],
        )
        self.assertTrue(ok)
        rows = self.store.list_retrieval_runs_for_turn(turn_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["invocation_order"], 1)
        self.assertEqual(rows[0]["execution_status"], "completed")
        self.assertEqual(rows[0]["foundry_iq_ok"], 1)

    def test_turn_with_multiple_retrievals_preserves_order(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.add_retrieval_runs(
            turn_id,
            [
                RetrievalRunInput(execution_status="failed", observation_status="complete", foundry_iq_ok=False),
                RetrievalRunInput(execution_status="completed", observation_status="complete", foundry_iq_ok=True),
            ],
        )
        self.assertTrue(ok)
        rows = self.store.list_retrieval_runs_for_turn(turn_id)
        self.assertEqual([r["invocation_order"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["execution_status"], "failed")
        self.assertEqual(rows[1]["execution_status"], "completed")

    def test_second_add_retrieval_runs_call_continues_invocation_order(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self.store.add_retrieval_runs(
            turn_id, [RetrievalRunInput(execution_status="failed", observation_status="complete")]
        )
        self.store.add_retrieval_runs(
            turn_id, [RetrievalRunInput(execution_status="completed", observation_status="complete")]
        )
        rows = self.store.list_retrieval_runs_for_turn(turn_id)
        self.assertEqual([r["invocation_order"] for r in rows], [1, 2])

    def test_retrieval_insert_failure_preserves_turn(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.add_retrieval_runs(
            turn_id,
            [RetrievalRunInput(execution_status="not-a-real-status", observation_status="complete")],
        )
        self.assertFalse(ok)

        # The turn itself must still exist and be intact.
        turn = self.store.get_turn(turn_id)
        self.assertIsNotNone(turn)
        self.assertEqual(turn["question_text"], "ADAM-6266 如何開啟 SNMP？")

    def test_retrieval_insert_failure_rolls_back_entire_batch_not_partially(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.add_retrieval_runs(
            turn_id,
            [
                RetrievalRunInput(execution_status="completed", observation_status="complete"),
                RetrievalRunInput(execution_status="bogus-status", observation_status="complete"),
                RetrievalRunInput(execution_status="completed", observation_status="complete"),
            ],
        )
        self.assertFalse(ok)
        # Zero rows -- not "the first one made it through".
        rows = self.store.list_retrieval_runs_for_turn(turn_id)
        self.assertEqual(rows, [])

    def test_retrieval_insert_failure_sets_observation_unavailable_with_safe_reason(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self.store.add_retrieval_runs(
            turn_id,
            [RetrievalRunInput(execution_status="bogus", observation_status="complete")],
        )
        turn = self.store.get_turn(turn_id)
        self.assertEqual(turn["retrieval_observation_status"], "unavailable")
        self.assertEqual(turn["retrieval_observation_reason"], "retrieval_insert_failed")
        # The reason must never be (or contain) the raw exception text.
        self.assertNotIn("bogus", turn["retrieval_observation_reason"])

    def test_retrieval_insert_failure_for_unknown_turn_returns_false_without_raising(self):
        ok = self.store.add_retrieval_runs(
            "does-not-exist",
            [RetrievalRunInput(execution_status="completed", observation_status="complete")],
        )
        self.assertFalse(ok)

    def test_update_turn_retrieval_observation_directly(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.update_turn_retrieval_observation(turn_id, "complete")
        self.assertTrue(ok)
        turn = self.store.get_turn(turn_id)
        self.assertEqual(turn["retrieval_observation_status"], "complete")

    def test_retrieval_run_input_rejects_forbidden_fields(self):
        forbidden_kwargs = (
            "documents",
            "content",
            "raw_output",
            "terminal_output",
            "response_body",
            "headers",
            "query_key",
            "authorization",
        )
        for kwarg in forbidden_kwargs:
            with self.subTest(kwarg=kwarg):
                with self.assertRaises(TypeError):
                    RetrievalRunInput(
                        execution_status="completed",
                        observation_status="complete",
                        **{kwarg: "must never be accepted"},
                    )


class FeedbackTests(_V2StoreTestCase):
    def _submit(self, turn_id: str, helpful: bool, **kwargs) -> bool:
        kwargs.setdefault("feedback_policy_version", POLICY_VERSION)
        return self.store.submit_feedback(uuid.uuid4().hex, turn_id, helpful, **kwargs)

    def test_feedback_at_most_one_per_turn(self):
        _s, _c, turn_id = self._new_session_case_turn()
        first = self._submit(turn_id, True)
        self.assertTrue(first)
        second = self._submit(turn_id, True)
        self.assertFalse(second)
        self.assertEqual(len(self._all_feedback_rows(turn_id)), 1)

    def test_positive_feedback_cannot_carry_reason(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self._submit(turn_id, True, reason_code="incorrect")
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_positive_feedback_cannot_carry_suggestion(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self._submit(turn_id, True, suggestion_text="不需要，但還是給個建議")
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_negative_feedback_missing_reason_fails(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self._submit(turn_id, False)
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_negative_feedback_illegal_reason_code_fails(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self._submit(turn_id, False, reason_code="not-a-real-code")
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_negative_feedback_with_valid_reason_succeeds(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self._submit(turn_id, False, reason_code="incomplete")
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["helpful"], 0)
        self.assertEqual(row["reason_code"], "incomplete")

    def test_negative_feedback_with_suggestion_succeeds(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self._submit(
            turn_id, False, reason_code="unclear", suggestion_text="請補充更多範例 🎉"
        )
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["suggestion_text"], "請補充更多範例 🎉")

    def test_every_allowlisted_reason_code_accepted(self):
        from tools.feedback_callbacks import REASON_CODES

        session_id = self._new_session()
        case_id = self._new_case(session_id=session_id)
        for idx, code in enumerate(REASON_CODES):
            with self.subTest(code=code):
                turn_id = f"turn-reason-{idx}"
                self._new_turn(turn_id, case_id, platform_user_message_id=f"m-reason-{idx}")
                ok = self._submit(turn_id, False, reason_code=code)
                self.assertTrue(ok)

    def test_retrieval_failure_does_not_block_feedback_submission(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self.store.add_retrieval_runs(
            turn_id,
            [RetrievalRunInput(execution_status="bogus", observation_status="complete")],
        )
        ok = self._submit(turn_id, False, reason_code="incorrect")
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["reason_code"], "incorrect")

    def test_db_level_check_constraint_backstops_python_validation(self):
        # Bypass submit_feedback()'s own Python guard entirely and hit the
        # table directly, to prove the CHECK constraint is real and not
        # just enforced in application code. A valid feedback_policy_version
        # is included so the NOT NULL/blank check on that column doesn't
        # mask the helpful/reason_code CHECK this test actually targets.
        _s, _c, turn_id = self._new_session_case_turn()
        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO feedback (feedback_id, turn_id, helpful, reason_code, suggestion_text, feedback_policy_version, submitted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (uuid.uuid4().hex, turn_id, 1, "incorrect", None, POLICY_VERSION, "now"),
                )

    def _all_feedback_rows(self, turn_id: str) -> list[sqlite3.Row]:
        with self.store._connect() as db:
            return db.execute(
                "SELECT * FROM feedback WHERE turn_id=?", (turn_id,)
            ).fetchall()


class ForeignKeyEnforcementTests(_V2StoreTestCase):
    def test_case_with_unknown_session_rejected_at_db_level(self):
        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO cases (case_id, session_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    ("case-x", "does-not-exist", "now", "now"),
                )

    def test_turn_with_unknown_case_rejected_at_db_level(self):
        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    """
                    INSERT INTO turns (
                        turn_id, case_id, session_id, platform_user_id, platform_user_message_id,
                        question_text, answer_text, feedback_eligible, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("turn-x", "does-not-exist", "does-not-exist", "u1", "m1", "q", "a", 0, "now", "now"),
                )

    def test_retrieval_run_with_unknown_turn_rejected_at_db_level(self):
        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    """
                    INSERT INTO retrieval_runs (
                        retrieval_id, turn_id, invocation_order, execution_status, observation_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (uuid.uuid4().hex, "does-not-exist", 1, "completed", "complete", "now"),
                )

    def test_feedback_with_unknown_turn_rejected_at_db_level(self):
        # A valid feedback_policy_version is included so the NOT NULL/blank
        # check on that column doesn't mask the FK violation this test
        # actually targets.
        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO feedback (feedback_id, turn_id, helpful, feedback_policy_version, submitted_at) VALUES (?, ?, ?, ?, ?)",
                    (uuid.uuid4().hex, "does-not-exist", 1, POLICY_VERSION, "now"),
                )


class CompositeForeignKeyTests(_V2StoreTestCase):
    """FOREIGN KEY (case_id, session_id) REFERENCES cases(case_id, session_id)
    -- proves a mismatched pair is rejected at the DB level, not just by
    create_turn()'s own Python-side derivation of session_id from case_id."""

    def test_mismatched_case_session_pair_rejected_on_fresh_db(self):
        session_a = self._new_session("sess-A")
        case_a = self._new_case("case-A", session_a)
        session_b = self._new_session("sess-B")

        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    """
                    INSERT INTO turns (
                        turn_id, case_id, session_id, platform_user_id, platform_user_message_id,
                        question_text, answer_text, feedback_eligible, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("turn-mismatch", case_a, session_b, "u1", "m1", "q", "a", 0, "now", "now"),
                )

    def test_matching_case_session_pair_still_succeeds(self):
        session_a = self._new_session("sess-A")
        case_a = self._new_case("case-A", session_a)

        with self.store._connect() as db:
            db.execute(
                """
                INSERT INTO turns (
                    turn_id, case_id, session_id, platform_user_id, platform_user_message_id,
                    question_text, answer_text, feedback_eligible, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("turn-ok", case_a, session_a, "u1", "m1", "q", "a", 0, "now", "now"),
            )
        self.assertIsNotNone(self.store.get_turn("turn-ok"))

    def test_composite_fk_active_after_migration_on_legacy_db(self):
        # Same mismatch check as above, but starting from a DB that already
        # has legacy 001 data -- proves the composite FK is created
        # regardless of whether migration 002 runs against an empty DB or
        # one with the legacy table already populated.
        legacy_store = FeedbackStore(self.db_path)
        legacy_store.create_run("legacy-run-x", "chat-1")
        del legacy_store
        gc.collect()

        v2_store = FeedbackStoreV2(self.db_path)
        session_a = v2_store.create_or_update_session("sess-A2", "telegram")
        case_a = v2_store.create_case("case-A2", session_a)
        session_b = v2_store.create_or_update_session("sess-B2", "telegram")

        with v2_store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    """
                    INSERT INTO turns (
                        turn_id, case_id, session_id, platform_user_id, platform_user_message_id,
                        question_text, answer_text, feedback_eligible, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("turn-mismatch-2", case_a, session_b, "u1", "m1", "q", "a", 0, "now", "now"),
                )

    def test_create_turn_still_derives_session_id_and_never_mismatches(self):
        # The Python API path (create_turn) never lets a mismatch happen in
        # the first place -- the composite FK is a backstop for direct SQL,
        # not create_turn()'s own behavior, which is unchanged.
        session_id = self._new_session()
        case_id = self._new_case(session_id=session_id)
        turn_id = self._new_turn(case_id=case_id)
        turn = self.store.get_turn(turn_id)
        self.assertEqual(turn["session_id"], session_id)


class AddSuggestionTests(_V2StoreTestCase):
    def _submit_negative(self, turn_id: str, *, reason_code: str = "incorrect") -> None:
        ok = self.store.submit_feedback(
            uuid.uuid4().hex, turn_id, False,
            feedback_policy_version=POLICY_VERSION,
            reason_code=reason_code,
        )
        self.assertTrue(ok)

    def test_add_suggestion_after_negative_reason_succeeds(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self._submit_negative(turn_id)
        ok = self.store.add_suggestion(turn_id, "請補充範例 🎉")
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["suggestion_text"], "請補充範例 🎉")

    def test_add_suggestion_rejected_for_positive_feedback(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.submit_feedback(
            uuid.uuid4().hex, turn_id, True, feedback_policy_version=POLICY_VERSION,
        )
        self.assertTrue(ok)
        rejected = self.store.add_suggestion(turn_id, "不該被接受")
        self.assertFalse(rejected)
        self.assertIsNone(self.store.get_feedback(turn_id)["suggestion_text"])

    def test_add_suggestion_rejected_for_unknown_turn(self):
        ok = self.store.add_suggestion("does-not-exist", "建議內容")
        self.assertFalse(ok)

    def test_add_suggestion_rejected_when_no_feedback_row_yet(self):
        _s, _c, turn_id = self._new_session_case_turn()
        # Turn exists, but no feedback has been submitted for it at all --
        # add_suggestion() must not create a half-formed row.
        ok = self.store.add_suggestion(turn_id, "建議內容")
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_add_suggestion_rejects_blank(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self._submit_negative(turn_id)
        for bad in ("", "   ", "\n\t"):
            with self.subTest(bad=repr(bad)):
                ok = self.store.add_suggestion(turn_id, bad)
                self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id)["suggestion_text"])

    def test_add_suggestion_rejects_non_string(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self._submit_negative(turn_id)
        for bad in (None, 123, ["a suggestion"], {"text": "a suggestion"}):
            with self.subTest(bad=bad):
                ok = self.store.add_suggestion(turn_id, bad)
                self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id)["suggestion_text"])

    def test_add_suggestion_accepts_at_limit_rejects_over_limit(self):
        # Same MAX_SUGGESTION_TEXT_LENGTH bound as legacy submit_suggestion().
        _s, _c, turn_id = self._new_session_case_turn(suffix="1")
        self._submit_negative(turn_id)
        exactly_limit = "a" * MAX_SUGGESTION_TEXT_LENGTH
        self.assertTrue(self.store.add_suggestion(turn_id, exactly_limit))

        _s2, _c2, turn_id_2 = self._new_session_case_turn(suffix="2")
        self._submit_negative(turn_id_2)
        over_limit = "a" * (MAX_SUGGESTION_TEXT_LENGTH + 1)
        self.assertFalse(self.store.add_suggestion(turn_id_2, over_limit))

    def test_add_suggestion_duplicate_submission_is_first_write_wins(self):
        # Matches legacy submit_suggestion()'s behavior exactly: a second
        # attempt is rejected and does not overwrite the first.
        _s, _c, turn_id = self._new_session_case_turn()
        self._submit_negative(turn_id)
        first = self.store.add_suggestion(turn_id, "第一次建議")
        self.assertTrue(first)
        second = self.store.add_suggestion(turn_id, "第二次建議")
        self.assertFalse(second)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["suggestion_text"], "第一次建議")

    def test_add_suggestion_does_not_disturb_other_columns(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self._submit_negative(turn_id, reason_code="unclear")
        before = self.store.get_feedback(turn_id)
        ok = self.store.add_suggestion(turn_id, "建議內容")
        self.assertTrue(ok)
        after = self.store.get_feedback(turn_id)
        self.assertEqual(after["helpful"], before["helpful"])
        self.assertEqual(after["reason_code"], before["reason_code"])
        self.assertEqual(after["submitted_at"], before["submitted_at"])
        self.assertEqual(after["feedback_policy_version"], before["feedback_policy_version"])


class FeedbackPolicyVersionTests(_V2StoreTestCase):
    def test_policy_version_saved_and_readable(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.submit_feedback(
            uuid.uuid4().hex, turn_id, True, feedback_policy_version=POLICY_VERSION,
        )
        self.assertTrue(ok)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["feedback_policy_version"], POLICY_VERSION)

    def test_different_policy_versions_saved_correctly(self):
        versions = (POLICY_VERSION, "universal-message-v2-test")
        for idx, version in enumerate(versions):
            with self.subTest(version=version):
                _s, _c, turn_id = self._new_session_case_turn(suffix=str(idx))
                ok = self.store.submit_feedback(
                    uuid.uuid4().hex, turn_id, True, feedback_policy_version=version,
                )
                self.assertTrue(ok)
                row = self.store.get_feedback(turn_id)
                self.assertEqual(row["feedback_policy_version"], version)

    def test_missing_policy_version_argument_raises_type_error(self):
        # feedback_policy_version has no default -- omitting it entirely is
        # a caller programming error, not a silent fallback.
        _s, _c, turn_id = self._new_session_case_turn()
        with self.assertRaises(TypeError):
            self.store.submit_feedback(uuid.uuid4().hex, turn_id, True)

    def test_none_policy_version_rejected(self):
        _s, _c, turn_id = self._new_session_case_turn()
        ok = self.store.submit_feedback(
            uuid.uuid4().hex, turn_id, True, feedback_policy_version=None,
        )
        self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_blank_policy_version_rejected(self):
        _s, _c, turn_id = self._new_session_case_turn()
        for bad in ("", "   ", "\n\t"):
            with self.subTest(bad=repr(bad)):
                ok = self.store.submit_feedback(
                    uuid.uuid4().hex, turn_id, True, feedback_policy_version=bad,
                )
                self.assertFalse(ok)
        self.assertIsNone(self.store.get_feedback(turn_id))

    def test_db_check_constraint_rejects_blank_policy_version_directly(self):
        # Bypass submit_feedback()'s own Python guard, hit the table
        # directly, to prove the CHECK is real, not just app-enforced.
        _s, _c, turn_id = self._new_session_case_turn()
        with self.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, turn_id, helpful, reason_code, suggestion_text,
                        feedback_policy_version, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (uuid.uuid4().hex, turn_id, 1, None, None, "   ", "now"),
                )

    def test_add_suggestion_does_not_change_policy_version(self):
        _s, _c, turn_id = self._new_session_case_turn()
        self.store.submit_feedback(
            uuid.uuid4().hex, turn_id, False,
            feedback_policy_version="universal-message-v1",
            reason_code="incorrect",
        )
        self.store.add_suggestion(turn_id, "建議內容")
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["feedback_policy_version"], "universal-message-v1")


if __name__ == "__main__":
    unittest.main()
