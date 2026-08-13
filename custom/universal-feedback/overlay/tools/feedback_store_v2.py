"""SQLite storage for the Session -> Case -> Turn -> {Retrieval Run, Feedback}
schema (v2).

This is deliberately a separate module from tools.feedback_storage, which
keeps serving the legacy `feedback_runs` table unchanged. Migration 002
(see ../migrations/002_feedback_schema_v2.sql) is purely additive: it adds
five new tables to the same database file feedback_storage.DEFAULT_PATH
already points at, alongside the untouched legacy table.

Phase 2 scope only: this module provides the schema and storage API. It is
not wired into the gateway, the Telegram adapter, or any messages parser --
callers (a later phase) are responsible for deciding what to write and when.

Migration 003 (see ../migrations/003_retrieval_execution_statuses.sql,
Phase 3A): widens retrieval_runs.execution_status to also allow 'blocked'
and 'unparseable'. 002_feedback_schema_v2.sql itself is never edited to
reflect this -- it stays byte-for-byte what Phase 2 commit 099abb5
originally created, so migration history stays traceable. _SCHEMA_STATEMENTS
below creates a brand-new database directly in the post-003 shape (a fresh
CREATE TABLE IF NOT EXISTS is a no-op against an existing table regardless
of its CHECK constraint, so this alone would silently fail to widen an
*existing* v2 database); _migrate_schema() separately runs
_upgrade_retrieval_runs_execution_statuses() to detect and rebuild an
existing pre-003 retrieval_runs table in place, using SQLite's own
documented 12-step ALTER TABLE recipe (SQLite has no ALTER TABLE ... DROP/
ADD CONSTRAINT).
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.feedback_callbacks import REASON_CODES, is_valid_reason_code
from tools.feedback_storage import DEFAULT_PATH, MAX_SUGGESTION_TEXT_LENGTH
from tools.universal_feedback import safe_feedback_text

# feedback.feedback_policy_version has no default in this module: callers
# (a later, gateway-wiring phase) are expected to pass
# tools.universal_feedback.POLICY_VERSION explicitly (the same constant the
# legacy universal-feedback send path already stamps on every row, see
# feedback_policy_version usage in the overlay's feedback-hook call sites),
# not a value invented here. Not re-exported from this module -- import it
# from tools.universal_feedback directly, its actual source, to avoid two
# import paths for one constant.
_REASON_CODE_PLACEHOLDERS = ",".join("?" for _ in REASON_CODES)

# Every connection this module opens sets these explicitly -- see _connect().
# 5s is a conservative default for a low-concurrency single-file SQLite POC:
# long enough to ride out a competing writer's transaction, short enough
# that a genuinely stuck lock still surfaces as an error instead of hanging
# indefinitely.
_BUSY_TIMEOUT_MS = 5000

EXECUTION_STATUSES: tuple[str, ...] = (
    "completed",
    "failed",
    "timed_out",
    "http_error",
    "network_error",
    "invalid_response",
    "no_documents",
    "unknown",
    # Added by migration 003 (tools/retrieval_observer.py, Phase 3A):
    # "blocked" is a fully-readable outer-terminal signal (policy blocked
    # the command before Foundry IQ's own script ever ran), paired with
    # observation_status="complete". "unparseable" is a truncated/
    # placeholder/malformed outer or inner result, paired with
    # observation_status="partial".
    "blocked",
    "unparseable",
)
_EXECUTION_STATUS_SET = frozenset(EXECUTION_STATUSES)

OBSERVATION_STATUSES: tuple[str, ...] = ("complete", "partial", "unavailable")
_OBSERVATION_STATUS_SET = frozenset(OBSERVATION_STATUSES)

# Fixed, safe reason stored on turns.retrieval_observation_reason when a
# retrieval-insert batch fails and is rolled back -- never the underlying
# exception's own text, which is not a value this module controls.
_RETRIEVAL_INSERT_FAILED_REASON = "retrieval_insert_failed"

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id       TEXT PRIMARY KEY,
        platform         TEXT NOT NULL,
        platform_chat_id TEXT,
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cases (
        case_id       TEXT PRIMARY KEY,
        session_id    TEXT NOT NULL REFERENCES sessions(session_id),
        title         TEXT,
        product_model TEXT,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL,
        -- Referenced by turns' composite FK below. case_id alone is
        -- already unique (it's the PK), but SQLite requires the exact
        -- (case_id, session_id) column pair to have its own named
        -- UNIQUE/PK constraint before a composite FK can reference it --
        -- individual-column uniqueness is not enough.
        UNIQUE (case_id, session_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cases_session ON cases(session_id)",
    """
    CREATE TABLE IF NOT EXISTS turns (
        turn_id                        TEXT PRIMARY KEY,
        case_id                        TEXT NOT NULL,
        session_id                     TEXT NOT NULL,

        platform_user_id               TEXT NOT NULL,
        platform_user_message_id       TEXT NOT NULL,
        platform_assistant_message_id  TEXT,

        question_text                  TEXT NOT NULL,
        answer_text                    TEXT NOT NULL,

        feedback_eligible              INTEGER NOT NULL CHECK (feedback_eligible IN (0, 1)),

        retrieval_observation_status   TEXT NOT NULL DEFAULT 'unavailable'
            CHECK (retrieval_observation_status IN ('complete', 'partial', 'unavailable')),
        retrieval_observation_reason   TEXT,

        support_config_commit          TEXT,
        feedback_code_commit           TEXT,
        hermes_version                 TEXT,
        model                          TEXT,
        provider                       TEXT,

        case_assignment_method              TEXT,
        case_assignment_confidence          TEXT,
        case_assignment_classifier_version  TEXT,
        case_assignment_overridden_by       TEXT,
        case_assignment_overridden_at       TEXT,

        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,

        UNIQUE (session_id, platform_user_message_id),
        -- Composite FK: guarantees case_id and session_id name a
        -- *consistent* pair (this case really does belong to this
        -- session), not just that each individually exists somewhere.
        -- Supersedes separate single-column FKs to cases(case_id) and
        -- sessions(session_id): both are implied by this one (the latter
        -- transitively, via cases.session_id's own FK to sessions), so
        -- keeping either alongside this would be a redundant constraint
        -- that adds no additional guarantee.
        FOREIGN KEY (case_id, session_id) REFERENCES cases(case_id, session_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turns_case ON turns(case_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_created_at ON turns(created_at)",
    """
    CREATE TABLE IF NOT EXISTS retrieval_runs (
        retrieval_id           TEXT PRIMARY KEY,
        turn_id                TEXT NOT NULL REFERENCES turns(turn_id),
        invocation_order       INTEGER NOT NULL,
        tool_call_id           TEXT,

        request_attempted      INTEGER CHECK (request_attempted IN (0, 1)),
        -- Post-003 shape (see migration 003 / module docstring above): a
        -- brand-new database is created directly with 'blocked' and
        -- 'unparseable' already allowed. An *existing* pre-003 database
        -- does not get this from CREATE TABLE IF NOT EXISTS alone (a
        -- no-op against an existing table) -- see
        -- _upgrade_retrieval_runs_execution_statuses().
        execution_status        TEXT NOT NULL CHECK (execution_status IN (
            'completed', 'failed', 'timed_out', 'http_error',
            'network_error', 'invalid_response', 'no_documents', 'unknown',
            'blocked', 'unparseable'
        )),
        foundry_iq_ok            INTEGER CHECK (foundry_iq_ok IN (0, 1)),
        observation_status       TEXT NOT NULL
            CHECK (observation_status IN ('complete', 'partial', 'unavailable')),
        observation_reason       TEXT,

        error_code                TEXT,
        http_status                INTEGER CHECK (http_status IS NULL OR (http_status BETWEEN 100 AND 599)),
        result_count                INTEGER,
        reference_count               INTEGER,
        foundry_schema_version          TEXT,

        created_at TEXT NOT NULL,

        UNIQUE (turn_id, invocation_order)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_retrieval_runs_turn ON retrieval_runs(turn_id)",
    """
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id     TEXT PRIMARY KEY,
        turn_id         TEXT NOT NULL UNIQUE REFERENCES turns(turn_id),

        helpful         INTEGER NOT NULL CHECK (helpful IN (0, 1)),
        reason_code     TEXT,
        suggestion_text TEXT,

        -- Which feedback-collection policy (button text, reason-code set,
        -- collection rules) was active when this row was submitted, so a
        -- future dashboard can tell whether two periods' feedback are even
        -- comparable. NOT NULL and non-blank at the DB level, independent
        -- of submit_feedback()'s own Python-side validation.
        feedback_policy_version TEXT NOT NULL CHECK (length(trim(feedback_policy_version)) > 0),

        submitted_at TEXT NOT NULL,

        CHECK (
            (helpful = 1 AND reason_code IS NULL AND suggestion_text IS NULL)
            OR
            (helpful = 0 AND reason_code IN ('incorrect', 'incomplete', 'not_relevant', 'unclear', 'other'))
        )
    )
    """,
)

# Migration 003 (see ../migrations/003_retrieval_execution_statuses.sql):
# rebuild retrieval_runs in place to widen its execution_status CHECK
# constraint to also allow 'blocked'/'unparseable', for a database whose
# retrieval_runs table was already created under the original (Phase 2,
# commit 099abb5) 002 migration. SQLite has no `ALTER TABLE ... DROP/ADD
# CONSTRAINT`, so this is the standard 12-step ALTER TABLE recipe: create a
# shadow table with the new constraint, copy every row unchanged, drop the
# old table, rename the shadow table into place, recreate its indexes.
# Column list is explicit (never `SELECT *`) so a future column addition to
# one side can't silently misalign with the other. Never touches
# feedback_runs, sessions, cases, turns, or feedback.
_RETRIEVAL_RUNS_REBUILD_COLUMNS = (
    "retrieval_id, turn_id, invocation_order, tool_call_id, "
    "request_attempted, execution_status, foundry_iq_ok, "
    "observation_status, observation_reason, "
    "error_code, http_status, result_count, reference_count, "
    "foundry_schema_version, created_at"
)

_RETRIEVAL_RUNS_REBUILD_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE retrieval_runs_new (
        retrieval_id           TEXT PRIMARY KEY,
        turn_id                TEXT NOT NULL REFERENCES turns(turn_id),
        invocation_order       INTEGER NOT NULL,
        tool_call_id           TEXT,

        request_attempted      INTEGER CHECK (request_attempted IN (0, 1)),
        execution_status        TEXT NOT NULL CHECK (execution_status IN (
            'completed', 'failed', 'timed_out', 'http_error',
            'network_error', 'invalid_response', 'no_documents', 'unknown',
            'blocked', 'unparseable'
        )),
        foundry_iq_ok            INTEGER CHECK (foundry_iq_ok IN (0, 1)),
        observation_status       TEXT NOT NULL
            CHECK (observation_status IN ('complete', 'partial', 'unavailable')),
        observation_reason       TEXT,

        error_code                TEXT,
        http_status                INTEGER CHECK (http_status IS NULL OR (http_status BETWEEN 100 AND 599)),
        result_count                INTEGER,
        reference_count               INTEGER,
        foundry_schema_version          TEXT,

        created_at TEXT NOT NULL,

        UNIQUE (turn_id, invocation_order)
    )
    """,
    f"INSERT INTO retrieval_runs_new ({_RETRIEVAL_RUNS_REBUILD_COLUMNS}) "
    f"SELECT {_RETRIEVAL_RUNS_REBUILD_COLUMNS} FROM retrieval_runs",
    "DROP TABLE retrieval_runs",
    "ALTER TABLE retrieval_runs_new RENAME TO retrieval_runs",
    "CREATE INDEX IF NOT EXISTS idx_retrieval_runs_turn ON retrieval_runs(turn_id)",
)

# A pre-003 retrieval_runs table's own CREATE TABLE text (recorded verbatim
# in sqlite_master.sql) cannot contain either of these literals -- the
# original 002 migration's CHECK constraint enumerates exactly the 8
# pre-003 values. BOTH markers must be present for the table to count as
# fully upgraded: checking only one (e.g. only 'blocked') would treat an
# abnormal partial/hand-edited schema that has 'blocked' but not
# 'unparseable' as already-complete and skip the rebuild, silently leaving
# 'unparseable' inserts to fail their CHECK constraint forever. Detection
# is on the live table's actual recorded DDL, never on a separately-
# tracked "migrations applied" version number, so it stays correct even if
# this module's own history of migrations is ever replayed out of order or
# against a hand-edited database.
_EXECUTION_STATUS_UPGRADE_MARKERS = ("'blocked'", "'unparseable'")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RetrievalRunInput:
    """One Foundry IQ invocation's safe, structured telemetry.

    Deliberately has no field for documents/content, full document text,
    full terminal output, the full command string, a Query Key, an
    Authorization header, a SAS token, a response body, or headers.
    Constructing this with any such keyword argument raises TypeError
    before the value ever reaches SQL, because a dataclass rejects unknown
    keyword arguments -- this is the structural block, not a runtime filter
    applied after the fact.
    """

    execution_status: str
    observation_status: str
    tool_call_id: str | None = None
    request_attempted: bool | None = None
    foundry_iq_ok: bool | None = None
    observation_reason: str | None = None
    error_code: str | None = None
    http_status: int | None = None
    result_count: int | None = None
    reference_count: int | None = None
    foundry_schema_version: str | None = None


class FeedbackStoreV2:
    """Storage API for sessions/cases/turns/retrieval_runs/feedback.

    Not wired into the gateway or any adapter in this phase -- callers are
    expected to supply every value explicitly.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH, *, migrate: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if migrate:
            self._migrate_schema()

    # -- connection / schema ------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        # Set explicitly on every connection -- foreign_keys and
        # busy_timeout are per-connection settings in SQLite (not persisted
        # in the database file), so they must be reissued each time rather
        # than assumed from a previous connection.
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return db

    def _migrate_schema(self) -> None:
        """Idempotently create the v2 tables/indexes, then apply migration
        003 (widen retrieval_runs.execution_status) if needed.

        Every statement in _SCHEMA_STATEMENTS is CREATE ... IF NOT EXISTS,
        so re-running this (e.g. on every process start, matching
        FeedbackStore's own _migrate_schema convention) against a database
        that already has these tables, or against one that only has the
        legacy 001 schema applied, is always safe. Never touches
        feedback_runs.

        On a brand-new database, _SCHEMA_STATEMENTS already creates
        retrieval_runs directly in the post-003 shape, so the upgrade step
        below is a no-op there; it only does real work against a database
        whose retrieval_runs table was already created under the original
        (pre-003) 002 migration.
        """
        with self._connect() as db:
            for statement in _SCHEMA_STATEMENTS:
                db.execute(statement)
        self._upgrade_retrieval_runs_execution_statuses()

    def _retrieval_runs_needs_execution_status_upgrade(self, db: sqlite3.Connection) -> bool:
        """Detect an existing pre-003 retrieval_runs table by inspecting
        its own recorded CREATE TABLE text in sqlite_master -- never a
        separately-tracked "migrations applied" version number, so this
        stays correct even against a database whose migration history
        this module did not itself apply. Returns False (nothing to do)
        both when the table already has the widened constraint (BOTH
        'blocked' and 'unparseable' present -- checking only one would
        wrongly treat an abnormal partial schema as fully upgraded) AND
        when the table does not exist yet at all (a fresh database,
        already created in the post-003 shape by _SCHEMA_STATEMENTS
        above)."""
        row = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='retrieval_runs'"
        ).fetchone()
        if row is None:
            return False
        sql_text = row[0] or ""
        return not all(marker in sql_text for marker in _EXECUTION_STATUS_UPGRADE_MARKERS)

    def _upgrade_retrieval_runs_execution_statuses(self) -> None:
        """Migration 003 (see ../migrations/003_retrieval_execution_statuses.sql):
        rebuild retrieval_runs in place so its execution_status CHECK
        constraint also allows 'blocked'/'unparseable'.

        Uses a dedicated connection with explicit BEGIN IMMEDIATE / COMMIT /
        ROLLBACK (rather than self._connect()'s implicit-transaction
        `with db:` convention used elsewhere in this class), because
        `PRAGMA foreign_keys` can only be changed outside an open
        transaction -- it must be turned off *before* BEGIN and back on
        *after* COMMIT/ROLLBACK, which the implicit-transaction pattern
        cannot express. Every retrieval_runs row is copied across
        unchanged via an explicit column list (never `SELECT *`); indexes
        are recreated by name after the rename since dropping a table also
        drops its indexes. `PRAGMA foreign_key_check` is verified before
        COMMIT; any failure at any step (including a RETRIEVAL_RUNS_new
        table left over from a previous failed attempt -- IF NOT EXISTS on
        its own CREATE TABLE statement -- see below) rolls back the whole
        transaction, leaving the original retrieval_runs table and its
        data completely untouched. Never touches feedback_runs, sessions,
        cases, turns, or feedback.
        """
        db = sqlite3.connect(self.path)
        try:
            db.row_factory = sqlite3.Row
            db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if not self._retrieval_runs_needs_execution_status_upgrade(db):
                return
            db.execute("PRAGMA foreign_keys=OFF")
            try:
                db.execute("BEGIN IMMEDIATE")
                try:
                    # DROP TABLE IF EXISTS guards a shadow table left over
                    # from an earlier attempt that failed after creating it
                    # but before this same transaction could complete (the
                    # failed attempt's own ROLLBACK already undid the DROP/
                    # RENAME/INSERT below, but a table created *outside* any
                    # transaction would not roll back -- CREATE TABLE here
                    # runs inside BEGIN IMMEDIATE, so in practice this is
                    # defensive belt-and-suspenders, not a known gap).
                    db.execute("DROP TABLE IF EXISTS retrieval_runs_new")
                    for statement in _RETRIEVAL_RUNS_REBUILD_STATEMENTS:
                        db.execute(statement)
                    violations = db.execute(
                        "PRAGMA foreign_key_check(retrieval_runs)"
                    ).fetchall()
                    if violations:
                        raise sqlite3.IntegrityError(
                            f"retrieval_runs rebuild left {len(violations)} "
                            "foreign_key_check violation(s)"
                        )
                except Exception:
                    db.execute("ROLLBACK")
                    raise
                else:
                    db.execute("COMMIT")
            finally:
                db.execute("PRAGMA foreign_keys=ON")
        finally:
            db.close()

    # -- sessions -------------------------------------------------------

    def create_or_update_session(
        self,
        session_id: str,
        platform: str,
        *,
        platform_chat_id: str | None = None,
    ) -> str:
        """Idempotent upsert: safe to call on every turn of an ongoing
        session, not just the first one."""
        now = _now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO sessions (session_id, platform, platform_chat_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    platform=excluded.platform,
                    platform_chat_id=excluded.platform_chat_id,
                    updated_at=excluded.updated_at
                """,
                (str(session_id), str(platform), platform_chat_id, now, now),
            )
        return str(session_id)

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()

    # -- cases ------------------------------------------------------------

    def create_case(
        self,
        case_id: str,
        session_id: str,
        *,
        title: str | None = None,
        product_model: str | None = None,
    ) -> str:
        now = _now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO cases (case_id, session_id, title, product_model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(case_id), str(session_id), title, product_model, now, now),
            )
        return str(case_id)

    def get_case(self, case_id: str) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()

    def list_cases_for_session(self, session_id: str) -> list[sqlite3.Row]:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM cases WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()

    # -- turns --------------------------------------------------------------

    def create_turn(
        self,
        turn_id: str,
        case_id: str,
        *,
        platform_user_id: str,
        platform_user_message_id: str,
        question_text: str,
        answer_text: str,
        feedback_eligible: bool,
        platform_assistant_message_id: str | None = None,
        retrieval_observation_status: str = "unavailable",
        retrieval_observation_reason: str | None = None,
        support_config_commit: str | None = None,
        feedback_code_commit: str | None = None,
        hermes_version: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        case_assignment_method: str | None = None,
        case_assignment_confidence: str | None = None,
        case_assignment_classifier_version: str | None = None,
    ) -> str:
        """Create one turn under an existing case.

        session_id is derived from the case's own session_id (not accepted
        as a caller-supplied parameter), so the denormalized
        turns.session_id column can never disagree with turns.case_id's
        actual parent session. Raises LookupError if case_id does not
        exist -- the FK constraint would also catch this, but this gives a
        clearer error before any SQL runs.
        """
        if retrieval_observation_status not in _OBSERVATION_STATUS_SET:
            raise ValueError(f"invalid retrieval_observation_status: {retrieval_observation_status!r}")

        now = _now()
        with self._connect() as db:
            case_row = db.execute(
                "SELECT session_id FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
            if case_row is None:
                raise LookupError(f"unknown case_id: {case_id!r}")
            session_id = case_row["session_id"]

            db.execute(
                """
                INSERT INTO turns (
                    turn_id, case_id, session_id,
                    platform_user_id, platform_user_message_id, platform_assistant_message_id,
                    question_text, answer_text, feedback_eligible,
                    retrieval_observation_status, retrieval_observation_reason,
                    support_config_commit, feedback_code_commit, hermes_version, model, provider,
                    case_assignment_method, case_assignment_confidence, case_assignment_classifier_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(turn_id), str(case_id), session_id,
                    str(platform_user_id), str(platform_user_message_id), platform_assistant_message_id,
                    str(question_text), str(answer_text), int(bool(feedback_eligible)),
                    retrieval_observation_status, retrieval_observation_reason,
                    support_config_commit, feedback_code_commit, hermes_version, model, provider,
                    case_assignment_method, case_assignment_confidence, case_assignment_classifier_version,
                    now, now,
                ),
            )
        return str(turn_id)

    def get_turn(self, turn_id: str) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM turns WHERE turn_id=?", (turn_id,)
            ).fetchone()

    def list_turns_for_case(self, case_id: str) -> list[sqlite3.Row]:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM turns WHERE case_id=? ORDER BY created_at",
                (case_id,),
            ).fetchall()

    def list_turns_for_session(self, session_id: str) -> list[sqlite3.Row]:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM turns WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()

    def update_turn_retrieval_observation(
        self, turn_id: str, status: str, reason: str | None = None
    ) -> bool:
        if status not in _OBSERVATION_STATUS_SET:
            raise ValueError(f"invalid retrieval_observation_status: {status!r}")
        with self._connect() as db:
            cur = db.execute(
                "UPDATE turns SET retrieval_observation_status=?, retrieval_observation_reason=?, updated_at=? WHERE turn_id=?",
                (status, reason, _now(), turn_id),
            )
            return cur.rowcount == 1

    # -- retrieval runs -------------------------------------------------

    def add_retrieval_runs(
        self, turn_id: str, retrieval_runs: list[RetrievalRunInput]
    ) -> bool:
        """Insert 0..n retrieval runs for an existing turn as one atomic
        batch, protected by an explicit SAVEPOINT.

        If any insert in the batch fails, the whole batch is rolled back to
        the savepoint (never a half-inserted set of attempts) and
        turns.retrieval_observation_status is safely downgraded to
        'unavailable' with a fixed, non-sensitive reason -- all within the
        same connection/transaction, so the turn row itself is never
        touched by a ROLLBACK: only the savepoint's own inserts are undone.
        A plain `except` around the whole call could not do this: an
        uncaught exception leaving the `with self._connect()` block rolls
        back the *entire* transaction, and there is no later opportunity to
        still run the recovery UPDATE inside the same commit. The savepoint
        is what makes "undo just this batch, then keep going" possible.

        Returns True if every row in the batch was inserted, False if the
        batch was rolled back. An empty list is a no-op that returns True
        and leaves turns.retrieval_observation_status untouched -- callers
        that already know "no retrieval happened, reliably observed" should
        call update_turn_retrieval_observation(turn_id, "complete") instead
        of calling this with an empty list.
        """
        if not retrieval_runs:
            return True

        with self._connect() as db:
            existing = db.execute(
                "SELECT COALESCE(MAX(invocation_order), 0) FROM retrieval_runs WHERE turn_id=?",
                (turn_id,),
            ).fetchone()[0]

            db.execute("SAVEPOINT retrieval_insert")
            try:
                for offset, run in enumerate(retrieval_runs, start=1):
                    if run.execution_status not in _EXECUTION_STATUS_SET:
                        raise ValueError(f"invalid execution_status: {run.execution_status!r}")
                    if run.observation_status not in _OBSERVATION_STATUS_SET:
                        raise ValueError(f"invalid observation_status: {run.observation_status!r}")
                    db.execute(
                        """
                        INSERT INTO retrieval_runs (
                            retrieval_id, turn_id, invocation_order, tool_call_id,
                            request_attempted, execution_status, foundry_iq_ok,
                            observation_status, observation_reason,
                            error_code, http_status, result_count, reference_count,
                            foundry_schema_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            str(turn_id),
                            existing + offset,
                            run.tool_call_id,
                            None if run.request_attempted is None else int(bool(run.request_attempted)),
                            run.execution_status,
                            None if run.foundry_iq_ok is None else int(bool(run.foundry_iq_ok)),
                            run.observation_status,
                            run.observation_reason,
                            run.error_code,
                            run.http_status,
                            run.result_count,
                            run.reference_count,
                            run.foundry_schema_version,
                            _now(),
                        ),
                    )
            except Exception:
                db.execute("ROLLBACK TO SAVEPOINT retrieval_insert")
                db.execute("RELEASE SAVEPOINT retrieval_insert")
                db.execute(
                    "UPDATE turns SET retrieval_observation_status=?, retrieval_observation_reason=?, updated_at=? WHERE turn_id=?",
                    ("unavailable", _RETRIEVAL_INSERT_FAILED_REASON, _now(), str(turn_id)),
                )
                return False
            else:
                db.execute("RELEASE SAVEPOINT retrieval_insert")
                return True

    def list_retrieval_runs_for_turn(self, turn_id: str) -> list[sqlite3.Row]:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM retrieval_runs WHERE turn_id=? ORDER BY invocation_order",
                (turn_id,),
            ).fetchall()

    # -- feedback -------------------------------------------------------

    def submit_feedback(
        self,
        feedback_id: str,
        turn_id: str,
        helpful: bool,
        *,
        feedback_policy_version: str,
        reason_code: str | None = None,
        suggestion_text: str | None = None,
    ) -> bool:
        """Submit the one and only feedback row for a turn.

        Re-validates every invariant in Python before attempting the
        INSERT (fail closed, no partial write attempted for an invalid
        combination), and the table's own CHECK constraints are the
        authoritative backstop if this method is ever bypassed. Returns
        False, never raises, for: a positive rating carrying a reason or
        suggestion, a negative rating with a missing/illegal reason code, a
        missing/blank feedback_policy_version, or a second submission for a
        turn that already has one (UNIQUE turn_id).

        feedback_policy_version has no default -- callers pass
        tools.universal_feedback.POLICY_VERSION (or whatever policy was
        actually active) explicitly, so a caller can never submit a row
        without recording which policy produced it.
        """
        helpful_bool = bool(helpful)

        if not isinstance(feedback_policy_version, str) or not feedback_policy_version.strip():
            return False

        if helpful_bool:
            if reason_code is not None or suggestion_text is not None:
                return False
        else:
            if not is_valid_reason_code(reason_code):
                return False

        with self._connect() as db:
            try:
                db.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, turn_id, helpful, reason_code, suggestion_text,
                        feedback_policy_version, submitted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(feedback_id),
                        str(turn_id),
                        int(helpful_bool),
                        reason_code,
                        suggestion_text,
                        feedback_policy_version,
                        _now(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def add_suggestion(self, turn_id: str, suggestion_text: str) -> bool:
        """Attach an optional suggestion to an already-submitted negative
        feedback row -- the second step of the reason-then-suggestion
        two-phase UI flow (a guarded UPDATE, never a new row).

        Validation/normalization intentionally mirrors
        tools.feedback_storage.FeedbackStore.submit_suggestion() exactly:
        same MAX_SUGGESTION_TEXT_LENGTH bound, same non-string/blank
        rejection, same safe_feedback_text() redaction, same
        first-write-wins guard (suggestion_text IS NULL) -- so v1 and v2
        enforce one consistent suggestion policy instead of two that could
        silently drift apart. Unlike v1's submit_suggestion(), this has no
        chat/user/feedback_message_id binding parameters: that
        authorization concern belongs to whatever gateway/adapter code
        calls this (out of scope for this phase, which is not wired to any
        adapter), not to the storage layer's own data invariants.

        Returns False, never raises, for: non-string/blank/over-length
        suggestion_text, an unknown turn_id, a turn with no feedback row
        yet, a positive (helpful=1) feedback row, a negative row with no
        legal reason_code, or a row that already has a suggestion. Never
        touches feedback_policy_version, reason_code, helpful, or
        submitted_at.
        """
        if not isinstance(suggestion_text, str):
            return False
        text = suggestion_text.strip()
        if not text or len(text) > MAX_SUGGESTION_TEXT_LENGTH:
            return False
        text = safe_feedback_text(text)

        with self._connect() as db:
            cur = db.execute(
                f"""
                UPDATE feedback SET suggestion_text=?
                WHERE turn_id=? AND helpful=0
                  AND reason_code IN ({_REASON_CODE_PLACEHOLDERS})
                  AND suggestion_text IS NULL
                """,
                (text, str(turn_id), *REASON_CODES),
            )
            return cur.rowcount == 1

    def get_feedback(self, turn_id: str) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM feedback WHERE turn_id=?", (turn_id,)
            ).fetchone()


__all__ = [
    "FeedbackStoreV2",
    "RetrievalRunInput",
    "EXECUTION_STATUSES",
    "OBSERVATION_STATUSES",
    "REASON_CODES",
]
