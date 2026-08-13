-- Staged migration documentation. Runtime migration is applied idempotently
-- by tools.feedback_store_v2.FeedbackStoreV2 using
-- "CREATE TABLE IF NOT EXISTS" / "CREATE INDEX IF NOT EXISTS" -- the same
-- pattern 001_universal_feedback.sql documents for feedback_runs. This file
-- is the committed record of what that idempotent DDL creates; the DDL
-- constants in feedback_store_v2.py are the source of truth actually
-- executed and must be kept in sync with this file.
--
-- This migration is purely additive:
--   * The legacy `feedback_runs` table (see 001_universal_feedback.sql) is
--     left completely untouched -- no DROP, no ALTER, no data migration.
--   * Every statement below is safe to re-run against a database that
--     already has these tables (IF NOT EXISTS everywhere), and safe to run
--     against a brand-new database or one that only has 001 applied.
--
-- Hierarchy: sessions -> cases -> turns -> {retrieval_runs (0..n), feedback (0..1)}.
--
-- Reflector-related tables are intentionally NOT created by this migration.
-- They are deferred to a later migration once the Phase 5 proposal/evidence
-- workflow is settled, so this schema does not lock in shape for tables
-- nothing yet writes to.

CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    platform         TEXT NOT NULL,
    platform_chat_id TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    case_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    title         TEXT,
    product_model TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    -- Referenced by turns' composite FK below. case_id alone is already
    -- unique (it's the PK), but SQLite requires the exact (case_id,
    -- session_id) column pair to have its own named UNIQUE/PK constraint
    -- before a composite FK can reference it -- individual-column
    -- uniqueness is not enough.
    UNIQUE (case_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_cases_session ON cases(session_id);

CREATE TABLE IF NOT EXISTS turns (
    turn_id                        TEXT PRIMARY KEY,
    -- case_id/session_id have no single-column FK of their own -- the
    -- composite FK at the bottom of this table covers both, and also
    -- guarantees the pair is internally consistent (this case really does
    -- belong to this session), which two separate single-column FKs
    -- cannot express. session_id is denormalized from cases.session_id so
    -- "all turns in this session" does not require a join through cases;
    -- the storage API derives it from case_id at insert time rather than
    -- accepting it from the caller (see feedback_store_v2.py).
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

    -- Provenance, kept as separate columns rather than one JSON blob so
    -- each is independently queryable/indexable if ever needed.
    support_config_commit          TEXT,
    feedback_code_commit           TEXT,
    hermes_version                 TEXT,
    model                          TEXT,
    provider                       TEXT,

    -- Case-assignment provenance lives on the turn (the thing that was
    -- actually classified/assigned), not inferred at the case level.
    -- No classifier exists yet (Phase 2 scope), so these are nullable and
    -- populated by whatever assigns case_id at turn-creation time; a
    -- manual override records who/when without losing the original
    -- automated assignment's method/confidence/classifier_version.
    case_assignment_method              TEXT,
    case_assignment_confidence          TEXT,
    case_assignment_classifier_version  TEXT,
    case_assignment_overridden_by       TEXT,
    case_assignment_overridden_at       TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    UNIQUE (session_id, platform_user_message_id),
    -- Composite FK: case_id and session_id must name a *consistent* pair
    -- (this case really does belong to this session), not just each exist
    -- somewhere independently. Supersedes separate single-column FKs to
    -- cases(case_id) and sessions(session_id): both are implied by this
    -- one (the latter transitively, via cases.session_id's own FK to
    -- sessions), so keeping either alongside would be a redundant
    -- constraint adding no further guarantee.
    FOREIGN KEY (case_id, session_id) REFERENCES cases(case_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_turns_case ON turns(case_id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_created_at ON turns(created_at);

-- One row per Foundry IQ invocation attempt. Zero rows for a turn that made
-- no attempt -- turns.retrieval_observation_status is what distinguishes a
-- reliably-observed "no attempt" (complete) from "we don't know" (partial /
-- unavailable); a fabricated placeholder row is never created to represent
-- "no call happened".
--
-- Never stores: documents[].content, full document text, full terminal
-- output, the full command string, Query Key, Authorization headers, SAS
-- tokens, response bodies, or request/response headers.
CREATE TABLE IF NOT EXISTS retrieval_runs (
    retrieval_id           TEXT PRIMARY KEY,
    turn_id                TEXT NOT NULL REFERENCES turns(turn_id),
    invocation_order       INTEGER NOT NULL,
    tool_call_id           TEXT,

    request_attempted      INTEGER CHECK (request_attempted IN (0, 1)),
    execution_status        TEXT NOT NULL CHECK (execution_status IN (
        'completed', 'failed', 'timed_out', 'http_error',
        'network_error', 'invalid_response', 'no_documents', 'unknown'
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

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_turn ON retrieval_runs(turn_id);

-- At most one row per turn (0..1). Created by FeedbackStoreV2.submit_feedback()
-- at actual submission time (helpful + any reason known together); there is
-- no separate "prompt was sent" pending state in this schema -- that is a
-- Telegram/gateway delivery concern, out of scope for this phase.
--
-- suggestion_text may start NULL on a negative row and be attached later by
-- FeedbackStoreV2.add_suggestion() (a guarded UPDATE, never a new row) --
-- this mirrors the legacy feedback_runs reason-then-suggestion two-phase UI
-- flow. add_suggestion() never touches feedback_policy_version.
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     TEXT PRIMARY KEY,
    turn_id         TEXT NOT NULL UNIQUE REFERENCES turns(turn_id),

    helpful         INTEGER NOT NULL CHECK (helpful IN (0, 1)),
    reason_code     TEXT,
    suggestion_text TEXT,

    -- Which feedback-collection policy (button text, reason-code set,
    -- collection rules) was active when this row was submitted, so a
    -- future dashboard can tell whether two periods' feedback are even
    -- comparable. NOT NULL and non-blank at the DB level, independent of
    -- submit_feedback()'s own Python-side validation.
    feedback_policy_version TEXT NOT NULL CHECK (length(trim(feedback_policy_version)) > 0),

    submitted_at TEXT NOT NULL,

    CHECK (
        (helpful = 1 AND reason_code IS NULL AND suggestion_text IS NULL)
        OR
        (helpful = 0 AND reason_code IN ('incorrect', 'incomplete', 'not_relevant', 'unclear', 'other'))
    )
);
