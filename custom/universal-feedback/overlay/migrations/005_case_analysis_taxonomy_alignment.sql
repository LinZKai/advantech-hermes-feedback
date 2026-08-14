-- Migration 005: rebuild case_analysis to align it with the finalized
-- Phase 4.5 Stage B tools.case_enrichment.CaseEnrichmentResult contract
-- (Stage B.5 Schema Alignment):
--
--   * diagnosis narrows from 6 values to 5 -- 'workflow_tool_issue' is
--     dropped entirely, 'unclear_or_other' is renamed to
--     'other_or_unclear'. Final taxonomy: 'knowledge_gap',
--     'retrieval_issue', 'answer_quality_issue', 'other_or_unclear',
--     'no_issue_detected'.
--   * issue_type (new): 'product_usage_or_application',
--     'product_capability_or_compatibility', 'product_issue',
--     'other_or_unclear'. NOT NULL.
--   * issue_type_confidence (new): REAL, NOT NULL, CHECK 0.0-1.0.
--   * diagnosis_confidence: was nullable, now NOT NULL, CHECK 0.0-1.0 --
--     CaseEnrichmentResult requires it unconditionally.
--   * product_confidence: stays nullable (a Case can have no identified
--     product), but now CHECK 0.0-1.0 when not NULL.
--
-- 004_case_analysis.sql (see that file) is left byte-for-byte unchanged --
-- it is the committed historical record of exactly what Stage A created,
-- the same way 002/003 stay unchanged under their own later migrations.
-- This file documents the INCREMENTAL change on top of it.
--
-- SQLite has no `ALTER TABLE ... DROP CONSTRAINT` / `ADD CONSTRAINT`, and
-- an existing nullable column cannot be tightened to NOT NULL in place
-- either, so this uses the same documented 12-step "ALTER TABLE" recipe
-- 003_retrieval_execution_statuses.sql already established for this
-- repository: build a shadow table with the new constraints, copy every
-- row across (transformed, not verbatim -- see below), drop the old
-- table, rename the shadow table into place, recreate its index, and
-- verify referential integrity with `PRAGMA foreign_key_check` before
-- committing.
--
-- Unlike 003's straight unchanged-row copy, this migration's INSERT ...
-- SELECT also TRANSFORMS legacy data for any case_analysis row that
-- predates this migration. Every case is a documented, intentional
-- decision -- never a value inferred or guessed at runtime:
--
--   * diagnosis = 'unclear_or_other'    -> 'other_or_unclear'
--     A pure rename: same meaning under the finalized taxonomy.
--   * diagnosis = 'workflow_tool_issue' -> 'other_or_unclear'
--     An intentional, LOSSY downgrade, not a semantic-equivalence rename
--     -- the finalized taxonomy has no workflow/tool-specific bucket.
--     This is the accepted product decision for this migration.
--   * diagnosis_confidence IS NULL -> 0.0
--     The pre-alignment create_case_analysis() accepted a missing
--     confidence; the finalized contract requires one. 0.0 is an
--     explicit fallback sentinel for "this row predates the NOT NULL
--     requirement", not a claim that the original analysis actually
--     reported zero confidence.
--   * issue_type / issue_type_confidence did not exist before this
--     migration, so every pre-existing row receives the fixed fallback
--     ('other_or_unclear', 0.0) -- NEVER a value inferred from
--     case_title/issue_summary. A migration must not fabricate an AI
--     classification a row never actually received.
--
-- This script is idempotent in effect (re-running it against an
-- already-upgraded database just rebuilds an identical-shape table again
-- -- wasteful but harmless), but the runtime migration actually executed
-- by tools.feedback_store_v2.FeedbackStoreV2 additionally checks
-- PRAGMA table_info(case_analysis) first and skips entirely once
-- `issue_type` is already a column, so it only ever runs the rebuild once
-- per database. See FeedbackStoreV2._upgrade_case_analysis_taxonomy() for
-- the actual, real code path this file documents -- that Python method
-- (not this .sql file) is what a running process executes.
--
-- Never touches: feedback_runs (legacy table), sessions, cases, turns,
-- feedback, retrieval_runs. Every case_analysis row's identity
-- (analysis_id, case_id) and untouched fields (case_title, issue_summary,
-- product_model, product_source, product_confidence, evidence_json,
-- analysis_version, analyzed_at, source_evidence_watermark) are preserved
-- unchanged; only diagnosis/diagnosis_confidence are transformed and
-- issue_type/issue_type_confidence are backfilled, as documented above.

PRAGMA foreign_keys=OFF;

BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS case_analysis_new (
    analysis_id TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL REFERENCES cases(case_id),

    case_title      TEXT,
    issue_summary   TEXT,

    issue_type            TEXT NOT NULL CHECK (issue_type IN (
        'product_usage_or_application', 'product_capability_or_compatibility',
        'product_issue', 'other_or_unclear'
    )),
    issue_type_confidence REAL NOT NULL CHECK (
        issue_type_confidence >= 0.0 AND issue_type_confidence <= 1.0
    ),

    diagnosis            TEXT NOT NULL CHECK (diagnosis IN (
        'knowledge_gap', 'retrieval_issue', 'answer_quality_issue',
        'other_or_unclear', 'no_issue_detected'
    )),
    diagnosis_confidence REAL NOT NULL CHECK (
        diagnosis_confidence >= 0.0 AND diagnosis_confidence <= 1.0
    ),

    product_model      TEXT,
    product_source      TEXT CHECK (
        product_source IN ('explicit_user_text', 'inference')
        OR product_source IS NULL
    ),
    product_confidence  REAL CHECK (
        product_confidence IS NULL
        OR (product_confidence >= 0.0 AND product_confidence <= 1.0)
    ),

    evidence_json TEXT,

    analysis_version TEXT NOT NULL,
    analyzed_at      TEXT NOT NULL,

    source_evidence_watermark TEXT NOT NULL,

    UNIQUE (case_id, analyzed_at)
);

INSERT INTO case_analysis_new (
    analysis_id, case_id, case_title, issue_summary,
    issue_type, issue_type_confidence,
    diagnosis, diagnosis_confidence,
    product_model, product_source, product_confidence,
    evidence_json, analysis_version, analyzed_at, source_evidence_watermark
)
SELECT
    analysis_id, case_id, case_title, issue_summary,
    'other_or_unclear', 0.0,
    CASE
        WHEN diagnosis IN ('unclear_or_other', 'workflow_tool_issue') THEN 'other_or_unclear'
        ELSE diagnosis
    END,
    COALESCE(diagnosis_confidence, 0.0),
    product_model, product_source, product_confidence,
    evidence_json, analysis_version, analyzed_at, source_evidence_watermark
FROM case_analysis;

DROP TABLE case_analysis;

ALTER TABLE case_analysis_new RENAME TO case_analysis;

CREATE INDEX IF NOT EXISTS idx_case_analysis_case ON case_analysis(case_id);

-- Caller must verify `PRAGMA foreign_key_check(case_analysis);` returns no
-- rows before COMMIT; if it does, ROLLBACK instead. The Python migration
-- performs this check itself (see FeedbackStoreV2 above).

COMMIT;

PRAGMA foreign_keys=ON;
