-- Staged migration documentation. Runtime migration is applied idempotently
-- by tools.feedback_store_v2.FeedbackStoreV2 using
-- "CREATE TABLE IF NOT EXISTS" / "CREATE INDEX IF NOT EXISTS" -- the same
-- pattern every prior migration file documents. This file is the committed
-- record of what that idempotent DDL creates; the DDL constants in
-- feedback_store_v2.py are the source of truth actually executed and must
-- be kept in sync with this file.
--
-- Curator Slice 1 (Accepted Improvement Proposal -> Curator ->
-- /sandbox/AGENTS.md proposed change): adds one new, purely-additive
-- table -- curator_changes. Like migration 006, this needs no rebuild
-- step: the table is brand new, so CREATE TABLE IF NOT EXISTS alone is
-- sufficient and idempotent for both a brand-new database and one already
-- at the 002-006 shape.
--
-- Row-mutation lifecycle:
--
--   * curator_changes is NOT append-only in the case_analysis/proposal_
--     observations sense (a re-run against the same proposal_id simply
--     inserts ANOTHER row with a new change_id -- nothing ever updates a
--     prior row's proposed_content/rationale/confidence in place), but
--     status IS mutable across a row's own lifetime: 'proposed' (the only
--     legal initial value create_curator_change() ever inserts) ->
--     'approved'/'rejected' (a future human review step) ->
--     'applied'/'failed' (a future apply step). Neither the review nor
--     the apply transition is implemented anywhere in this codebase yet
--     -- see tools.curator_domain's module docstring for the explicit
--     self-improvement boundary this slice does not cross.
--
-- target_file is a plain TEXT column, not a CHECK-constrained enum: Slice
-- 1 always writes /sandbox/AGENTS.md (see tools.curator_domain.
-- CURATOR_TARGET_FILE), enforced in Python (CuratorChange.__post_init__
-- and create_curator_change()'s own pre-validation) and in
-- tools.run_curator's deterministic guards, not at the SQL layer -- a
-- future slice that lets Curator target more than one file would only
-- need to widen the Python-side allowlist, not this column's shape.
--
-- proposed_content is nullable: NULL exactly when
-- change_type='no_change_recommended' (a fully legitimate, expected
-- outcome -- Curator judged /sandbox/AGENTS.md not to be an appropriate
-- lever for this Proposal). Every other change_type
-- (add_rule/modify_rule/remove_rule) requires a non-blank proposed_content
-- holding the COMPLETE replacement content of target_file -- not a diff,
-- not a patch. There is no diff/patch engine anywhere in this slice; a
-- future "show a diff" step, if ever built, derives it deterministically
-- from before_content/proposed_content in Python, never stored separately.
--
-- Taxonomy authority: tools.feedback_store_v2 now also owns
-- CHANGE_TYPE_VALUES / CURATOR_CHANGE_STATUS_VALUES -- tools.curator_domain
-- imports both rather than maintaining its own copies, matching how
-- tools.reflector_proposals already imports IMPROVEMENT_TARGET_VALUES/
-- REVIEW_STATUS_VALUES/PROPOSAL_TREND_VALUES from this same module.
--
-- Self-improvement boundary: an accepted, agent_behavior
-- improvement_proposals row is NEVER permission for any code path to
-- write to /sandbox/AGENTS.md (or any other file) directly. No function
-- anywhere in tools.curator_domain/tools.curator_analyzer/
-- tools.curator_prompt_context/tools.run_curator writes to that file --
-- this slice only produces one reviewable curator_changes row with
-- status='proposed'. Applying a change is explicitly out of scope and not
-- implemented.

CREATE TABLE IF NOT EXISTS curator_changes (
    change_id TEXT PRIMARY KEY,

    proposal_id TEXT NOT NULL REFERENCES improvement_proposals(proposal_id),
    target_file TEXT NOT NULL,

    change_type TEXT NOT NULL CHECK (change_type IN (
        'add_rule', 'modify_rule', 'remove_rule', 'no_change_recommended'
    )),

    rationale        TEXT NOT NULL,
    before_content   TEXT NOT NULL,
    proposed_content TEXT,
    expected_effect  TEXT,

    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),

    status TEXT NOT NULL CHECK (status IN (
        'proposed', 'approved', 'rejected', 'applied', 'failed'
    )),

    created_at  TEXT NOT NULL,
    reviewed_at TEXT,
    applied_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_curator_changes_proposal ON curator_changes(proposal_id);
