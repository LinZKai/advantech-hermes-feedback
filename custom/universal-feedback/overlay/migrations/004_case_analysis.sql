-- Staged migration documentation. Runtime migration is applied idempotently
-- by tools.feedback_store_v2.FeedbackStoreV2 using
-- "CREATE TABLE IF NOT EXISTS" / "CREATE INDEX IF NOT EXISTS" -- the same
-- pattern 001_universal_feedback.sql / 002_feedback_schema_v2.sql document.
-- This file is the committed record of what that idempotent DDL creates;
-- the DDL constants in feedback_store_v2.py are the source of truth
-- actually executed and must be kept in sync with this file.
--
-- Phase 4.5 Stage A: adds one new, purely-additive table, `case_analysis`.
-- Unlike migration 003 (which had to rebuild retrieval_runs in place to
-- widen an existing CHECK constraint), this needs no rebuild step: the
-- table never existed before, so CREATE TABLE IF NOT EXISTS alone is
-- sufficient and idempotent for both a brand-new database and one already
-- at the 002/003 shape.
--
-- case_analysis is append-only: one row per analysis run for a Case, never
-- updated in place. Re-analysis is always a new INSERT (UNIQUE(case_id,
-- analyzed_at), not UNIQUE(case_id) alone), so a Case's analysis history is
-- never lost or silently overwritten. `cases.title` / `cases.product_model`
-- are not written by anything in this migration or by
-- create_case_analysis() -- deciding whether/how an analysis result
-- updates a Case's own identity fields (and any Human Override policy) is
-- deferred to a later stage.
--
-- source_evidence_watermark is deliberately NOT cases.updated_at: that
-- column is set once, at Case creation, and is never updated by any
-- existing write path (create_turn, submit_feedback, add_suggestion,
-- add_retrieval_runs all leave `cases` untouched -- verified by inspection,
-- there is no `UPDATE cases` anywhere in this codebase). Instead,
-- source_evidence_watermark records MAX(turns.created_at,
-- feedback.submitted_at, retrieval_runs.created_at) for the Case's Turns
-- at the time of that analysis run -- see
-- FeedbackStoreV2.get_case_evidence_watermark().

CREATE TABLE IF NOT EXISTS case_analysis (
    analysis_id TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL REFERENCES cases(case_id),

    case_title      TEXT,
    issue_summary   TEXT,

    diagnosis            TEXT NOT NULL CHECK (diagnosis IN (
        'knowledge_gap', 'retrieval_issue', 'answer_quality_issue',
        'workflow_tool_issue', 'unclear_or_other', 'no_issue_detected'
    )),
    diagnosis_confidence REAL,

    product_model      TEXT,
    product_source      TEXT CHECK (
        product_source IN ('explicit_user_text', 'inference')
        OR product_source IS NULL
    ),
    product_confidence  REAL,

    evidence_json TEXT,

    analysis_version TEXT NOT NULL,
    analyzed_at      TEXT NOT NULL,

    source_evidence_watermark TEXT NOT NULL,

    UNIQUE (case_id, analyzed_at)
);

CREATE INDEX IF NOT EXISTS idx_case_analysis_case ON case_analysis(case_id);
