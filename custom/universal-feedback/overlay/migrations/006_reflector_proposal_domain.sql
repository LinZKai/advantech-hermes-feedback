-- Staged migration documentation. Runtime migration is applied idempotently
-- by tools.feedback_store_v2.FeedbackStoreV2 using
-- "CREATE TABLE IF NOT EXISTS" / "CREATE INDEX IF NOT EXISTS" -- the same
-- pattern 001_universal_feedback.sql / 002_feedback_schema_v2.sql /
-- 004_case_analysis.sql document. This file is the committed record of what
-- that idempotent DDL creates; the DDL constants in feedback_store_v2.py are
-- the source of truth actually executed and must be kept in sync with this
-- file.
--
-- Phase 5 Slice 2 (Cross-case Reflector proposal domain): adds three new,
-- purely-additive tables -- reflection_runs, improvement_proposals,
-- proposal_observations. Like migration 004, this needs no rebuild step:
-- every table is brand new, so CREATE TABLE IF NOT EXISTS alone is
-- sufficient and idempotent for both a brand-new database and one already
-- at the 002/003/004/005 shape.
--
-- Three different row-mutation lifecycles, deliberately not uniform:
--
--   * reflection_runs is a MUTABLE batch-run record, not append-only: one
--     row per Reflection Run, created with status='running'
--     (create_reflection_run()), then updated exactly once to a terminal
--     status ('succeeded' or 'failed') via complete_reflection_run() --
--     mirrors tools.feedback_store_v2.FeedbackStoreV2.
--     update_turn_retrieval_observation()'s existing "validated UPDATE,
--     rowcount==1" convention, the only other row-mutation precedent
--     already in that module.
--
--   * improvement_proposals is "identity + HUMAN lifecycle": created once
--     per newly detected recurring pattern (review_status='pending', the
--     only legal initial value), and its review_status is later moved by
--     update_proposal_review_status() -- a HUMAN reviewer action only,
--     never an AI/Reflector code path. This module cannot enforce that
--     boundary at the type level; it is a process boundary, documented
--     here and in tools.reflector_proposals's module docstring. Does NOT
--     carry per-observation fields (pattern_summary, recommended_
--     improvement, confidence, trend, supporting Cases, ...) -- those
--     change on every re-observation and live on proposal_observations
--     instead; a Proposal row is identity + human lifecycle only.
--
--   * proposal_observations is APPEND-ONLY, exactly like case_analysis:
--     one row per (proposal_id, reflection_run_id) pair, never updated in
--     place -- a Proposal re-observed in a later Reflection Run is always
--     a new row, so its observation history is never lost or silently
--     overwritten.
--
-- Taxonomy authority: tools.feedback_store_v2 now also owns
-- IMPROVEMENT_TARGET_VALUES / REVIEW_STATUS_VALUES / PROPOSAL_TREND_VALUES /
-- REFLECTION_RUN_STATUS_VALUES -- tools.reflector_proposals imports all four
-- rather than maintaining its own copies, matching tools.case_enrichment's
-- existing relationship to this module's case_analysis taxonomy constants.
--
-- Supporting Cases (Phase 5 Slice 2 task instruction, section 15):
-- proposal_observations.supporting_case_ids_json is a JSON array of
-- case_id strings -- deliberately NOT a normalized
-- improvement_proposal_cases join table in this slice. The current need is
-- report/drilldown output, not relational query (Case -> all Proposals,
-- proposal overlap, cross-proposal query); normalize only if/when a future
-- Dashboard actually needs that access pattern.
--
-- Proposal review history (section 16): review_status lives directly on
-- improvement_proposals, not in a separate proposal_reviews audit table --
-- deferred until a real human-review audit/history requirement exists.
--
-- Self-improvement boundary (section 17): improvement_target='agent_behavior'
-- on a Proposal is a CLASSIFICATION of what the proposal is about, never
-- permission for any code path to edit Hermes' production behavior/KB/
-- Skill/SOUL directly. No such write path exists anywhere in this schema or
-- in tools.reflector_proposals -- see that module's own docstring for the
-- explicit boundary statement.

CREATE TABLE IF NOT EXISTS reflection_runs (
    reflection_run_id TEXT PRIMARY KEY,

    window_start TEXT,
    window_end   TEXT,

    started_at   TEXT NOT NULL,
    completed_at TEXT,

    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),

    analyzed_case_count INTEGER NOT NULL CHECK (analyzed_case_count >= 0),

    material_change_detected INTEGER CHECK (material_change_detected IN (0, 1)),
    run_summary TEXT,

    reflector_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS improvement_proposals (
    proposal_id TEXT PRIMARY KEY,

    improvement_target TEXT NOT NULL CHECK (improvement_target IN (
        'knowledge', 'agent_behavior', 'retrieval', 'workflow', 'other'
    )),
    title TEXT NOT NULL,

    review_status TEXT NOT NULL CHECK (review_status IN (
        'pending', 'accepted', 'rejected'
    )),

    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal_observations (
    observation_id TEXT PRIMARY KEY,

    proposal_id        TEXT NOT NULL REFERENCES improvement_proposals(proposal_id),
    reflection_run_id  TEXT NOT NULL REFERENCES reflection_runs(reflection_run_id),

    trend TEXT NOT NULL CHECK (trend IN (
        'new', 'growing', 'stable', 'declining', 'no_longer_observed'
    )),

    pattern_summary          TEXT NOT NULL,
    possible_cause            TEXT,
    recommended_improvement  TEXT NOT NULL,
    expected_benefit          TEXT,
    limitations                TEXT,

    supporting_case_ids_json TEXT NOT NULL,
    supporting_case_count    INTEGER NOT NULL CHECK (supporting_case_count >= 0),

    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),

    observed_at TEXT NOT NULL,

    UNIQUE (proposal_id, reflection_run_id),
    UNIQUE (proposal_id, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_proposal_observations_proposal ON proposal_observations(proposal_id);
CREATE INDEX IF NOT EXISTS idx_proposal_observations_run ON proposal_observations(reflection_run_id);
