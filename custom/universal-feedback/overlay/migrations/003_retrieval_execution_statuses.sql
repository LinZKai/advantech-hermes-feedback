-- Migration 003: widen retrieval_runs.execution_status to also allow
-- 'blocked' and 'unparseable' (Phase 3A / tools/retrieval_observer.py):
--   * 'blocked'     -- a fully-readable outer-terminal signal that a
--                      command was stopped by policy before Foundry IQ's
--                      own script ever ran (observation_status='complete').
--   * 'unparseable' -- a truncated, placeholder, or otherwise malformed
--                      outer/inner result (observation_status='partial').
--
-- 002_feedback_schema_v2.sql (see that file) is left byte-for-byte
-- unchanged: it is the committed historical record of exactly what Phase 2
-- commit 099abb5 created, and must stay that way so migration history
-- stays traceable. This file documents the INCREMENTAL change on top of
-- it, the same way 002 itself was additive on top of 001.
--
-- SQLite has no `ALTER TABLE ... DROP CONSTRAINT` / `ADD CONSTRAINT`, so a
-- CHECK constraint cannot be widened in place. The only supported way is
-- SQLite's own documented 12-step "ALTER TABLE" procedure
-- (https://www.sqlite.org/lang_altertable.html#otheralter): build a shadow
-- table with the new constraint, copy every row across unchanged, drop the
-- old table, rename the shadow table into place, recreate its indexes,
-- and verify referential integrity with `PRAGMA foreign_key_check` before
-- committing.
--
-- This script is idempotent in effect (re-running it against a database
-- that has already been upgraded just rebuilds an identical-shape table
-- again -- wasteful but harmless), but the runtime migration actually
-- executed by tools.feedback_store_v2.FeedbackStoreV2 additionally checks
-- sqlite_master.sql first and skips entirely once 'blocked' is already
-- present in the live CHECK constraint text, so it only ever runs the
-- rebuild once per database. See
-- FeedbackStoreV2._upgrade_retrieval_runs_execution_statuses() for the
-- actual, real code path this file documents -- that Python method (not
-- this .sql file) is what a running process executes.
--
-- Never touches: `feedback_runs` (legacy table, see
-- 001_universal_feedback.sql), `sessions`, `cases`, `turns`, `feedback`.
-- Every retrieval_runs row's data is preserved unchanged; only the table's
-- own execution_status CHECK constraint (and, incidentally, its identity
-- as a SQLite object -- indexes are recreated by name) changes.

PRAGMA foreign_keys=OFF;

BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS retrieval_runs_new (
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
);

INSERT INTO retrieval_runs_new (
    retrieval_id, turn_id, invocation_order, tool_call_id,
    request_attempted, execution_status, foundry_iq_ok,
    observation_status, observation_reason,
    error_code, http_status, result_count, reference_count,
    foundry_schema_version, created_at
)
SELECT
    retrieval_id, turn_id, invocation_order, tool_call_id,
    request_attempted, execution_status, foundry_iq_ok,
    observation_status, observation_reason,
    error_code, http_status, result_count, reference_count,
    foundry_schema_version, created_at
FROM retrieval_runs;

DROP TABLE retrieval_runs;

ALTER TABLE retrieval_runs_new RENAME TO retrieval_runs;

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_turn ON retrieval_runs(turn_id);

-- Caller must verify `PRAGMA foreign_key_check(retrieval_runs);` returns no
-- rows before COMMIT; if it does, ROLLBACK instead. The Python migration
-- performs this check itself (see FeedbackStoreV2 above).

COMMIT;

PRAGMA foreign_keys=ON;
