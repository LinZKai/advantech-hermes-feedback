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

Migration 004 (see ../migrations/004_case_analysis.sql, Phase 4.5 Stage A):
adds a new, purely-additive `case_analysis` table -- one append-only row per
analysis run for a Case (never updated in place; re-analysis is always a new
row, so history is never lost). Unlike migration 003, this needs no rebuild
step: CREATE TABLE IF NOT EXISTS is sufficient and idempotent for a table
that never existed before, on both a brand-new database and an existing one
already at the 002/003 shape. Stage A only adds the schema and query
surface below -- no LLM call, no enrichment service, and no runner exist
yet; those are later stages.

Migration 005 (see ../migrations/005_case_analysis_taxonomy_alignment.sql,
Phase 4.5 Stage B.5 Schema Alignment): rebuilds `case_analysis` so it
matches the finalized Stage B tools.case_enrichment.CaseEnrichmentResult
contract -- narrows `diagnosis` from 6 values to 5 (drops
'workflow_tool_issue', renames 'unclear_or_other' to 'other_or_unclear'),
adds NOT NULL `issue_type`/`issue_type_confidence`, and adds range CHECK
constraints (0.0-1.0) to all three confidence columns. This module is now
the taxonomy authority for DIAGNOSIS_VALUES/ISSUE_TYPE_VALUES/
PRODUCT_SOURCE_VALUES -- tools.case_enrichment imports all three from here
rather than maintaining its own copies. Like migration 003 (and unlike
004), this needs the full rebuild recipe, because it changes an existing
CHECK constraint and tightens existing columns to NOT NULL, not just adds
new tables. See _upgrade_case_analysis_taxonomy() below for the legacy-data
handling this rebuild applies (never a silent guess at Issue Type for
pre-existing rows -- see that method's docstring).

Phase 5 Slice 1 (Cross-case Reflector input foundation): adds two new
read-only query methods, list_case_ids_created_in_window() and
list_latest_case_analysis() -- no schema change, no migration; both tables
they read (cases, case_analysis) are unchanged. Each is a general-purpose,
single-responsibility primitive (Case-creation-time filtering; latest-
analysis-per-Case batch lookup) -- neither method knows about the other,
about Reflector, or about any notion of an "analysis window"; that
composition lives in tools.case_reflection_input, which calls both. See
each method's own docstring for its exact semantics.

Migration 006 (see ../migrations/006_reflector_proposal_domain.sql, Phase 5
Slice 2): adds three new, purely-additive tables -- reflection_runs,
improvement_proposals, proposal_observations. Like migration 004, this
needs no rebuild step: every table is brand new, so CREATE TABLE IF NOT
EXISTS is sufficient and idempotent for both a fresh database and an
existing one already at the 002/003/004/005 shape.

  * reflection_runs is a mutable batch-run record (create with
    status='running', then a single terminal transition to 'succeeded' or
    'failed' via complete_reflection_run()) -- NOT append-only, unlike
    case_analysis/proposal_observations, because a run is one ongoing
    activity with exactly one lifecycle, not a history of independent
    snapshots. Mirrors update_turn_retrieval_observation()'s existing
    "validated UPDATE, rowcount==1" convention -- the only other row-
    mutation precedent already in this module.

  * improvement_proposals is "identity + human lifecycle": created once,
    per detected recurring pattern, and its review_status
    (pending/accepted/rejected) is later moved by a HUMAN reviewer via
    update_proposal_review_status() -- never by any AI/Reflector code
    path. Nothing in this module enforces that boundary at the type
    level (a Python caller could call it from anywhere); it is a process
    boundary, documented here and in tools.reflector_proposals.

  * proposal_observations is append-only, exactly like case_analysis:
    one row per (proposal_id, reflection_run_id) pair (UNIQUE constraint,
    same purpose as case_analysis's own append-only design -- history is
    never lost, re-observation is always a new row), plus a second
    UNIQUE(proposal_id, observed_at) that makes get_latest_proposal_
    observation()/list_latest_proposal_observations()'s "ORDER BY
    observed_at DESC" unambiguous for any one proposal_id, mirroring
    case_analysis's own UNIQUE(case_id, analyzed_at) tie-elimination
    exactly.

This module is now also the taxonomy authority for IMPROVEMENT_TARGET_
VALUES/REVIEW_STATUS_VALUES/PROPOSAL_TREND_VALUES/REFLECTION_RUN_STATUS_
VALUES -- tools.reflector_proposals imports all four from here rather
than maintaining its own copies, matching tools.case_enrichment's existing
relationship to this module's case_analysis taxonomy constants.

Slice 2 scope only: schema and storage query surface. No LLM, no matching
algorithm, no runner, no scheduler exist yet -- see
tools.reflector_proposals for the typed domain contracts these tables
persist.
"""
from __future__ import annotations

import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping

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

# Phase 4.5 Stage B.5 Schema Alignment: case_analysis.diagnosis /
# case_analysis.issue_type / case_analysis.product_source fixed
# vocabularies -- kept in sync with the CHECK constraints in
# _SCHEMA_STATEMENTS below, so create_case_analysis() can fail closed in
# Python (matching submit_feedback()'s pre-validation convention) before
# ever attempting the INSERT, rather than relying solely on the DB-level
# CHECK to reject bad input.
#
# This module is the taxonomy authority for all three constants below --
# tools.case_enrichment.CaseEnrichmentResult imports DIAGNOSIS_VALUES/
# ISSUE_TYPE_VALUES/PRODUCT_SOURCE_VALUES from here rather than maintaining
# its own copies (an earlier Stage B draft did keep a local, deliberately
# diverging DIAGNOSIS_VALUES while the two contracts were still being
# finalized; that divergence is resolved now that both are aligned to the
# same 5-value diagnosis / 4-value issue_type taxonomy below).
DIAGNOSIS_VALUES: tuple[str, ...] = (
    "knowledge_gap",
    "retrieval_issue",
    "answer_quality_issue",
    "other_or_unclear",
    "no_issue_detected",
)
_DIAGNOSIS_VALUE_SET = frozenset(DIAGNOSIS_VALUES)

ISSUE_TYPE_VALUES: tuple[str, ...] = (
    "product_usage_or_application",
    "product_capability_or_compatibility",
    "product_issue",
    "other_or_unclear",
)
_ISSUE_TYPE_VALUE_SET = frozenset(ISSUE_TYPE_VALUES)

PRODUCT_SOURCE_VALUES: tuple[str, ...] = ("explicit_user_text", "inference")
_PRODUCT_SOURCE_VALUE_SET = frozenset(PRODUCT_SOURCE_VALUES)

# Phase 5 Slice 2 (Cross-case Reflector proposal domain) taxonomies -- kept
# in sync with the CHECK constraints in _SCHEMA_STATEMENTS below, same
# fail-closed-in-Python-before-the-CHECK-fires convention as the case_
# analysis taxonomies above. tools.reflector_proposals imports all four
# from here rather than maintaining its own copies (see this module's
# docstring).
#
# improvement_target: WHAT KIND of change a proposal is about -- never a
# claim that Hermes/KB/Skill/SOUL may be edited automatically because of
# it (see tools.reflector_proposals's module docstring for the explicit
# self-improvement boundary this taxonomy value does NOT cross).
#
#   knowledge        - KB/FAQ/documentation may have a gap or be
#                       insufficient.
#   agent_behavior    - Hermes' own response strategy/instructions/
#                       reasoning pattern may be the improvement target --
#                       NOT permission to self-edit production behavior.
#   retrieval          - knowledge may exist, but retrieval/query/indexing/
#                       search behavior may be the improvement target.
#   workflow          - tool usage, support workflow, or Case handling
#                       process may be the improvement target.
#   other              - does not safely fit the four categories above.
IMPROVEMENT_TARGET_VALUES: tuple[str, ...] = (
    "knowledge",
    "agent_behavior",
    "retrieval",
    "workflow",
    "other",
)
_IMPROVEMENT_TARGET_VALUE_SET = frozenset(IMPROVEMENT_TARGET_VALUES)

# review_status: the HUMAN lifecycle of an ImprovementProposal -- moved
# only by a human reviewer, never by any AI/Reflector code path (a process
# boundary this module cannot enforce at the type level; see
# update_proposal_review_status()'s own docstring). Deliberately only
# three values in this slice -- 'implemented'/'resolved'/'archived' are
# NOT added yet because the implementation workflow they would represent
# does not exist yet (see the Phase 5 Slice 2 task instruction).
REVIEW_STATUS_VALUES: tuple[str, ...] = (
    "pending",
    "accepted",
    "rejected",
)
_REVIEW_STATUS_VALUE_SET = frozenset(REVIEW_STATUS_VALUES)

# trend: the AI's OBSERVATION of how a pattern's supporting evidence has
# changed since this Proposal's previous Observation -- never a claim
# about review_status (a pattern can be trend='growing' while
# review_status stays 'pending' forever; 'no_longer_observed' is not
# 'rejected' -- it is an observation that current evidence no longer
# supports the pattern as active, nothing more).
PROPOSAL_TREND_VALUES: tuple[str, ...] = (
    "new",
    "growing",
    "stable",
    "declining",
    "no_longer_observed",
)
_PROPOSAL_TREND_VALUE_SET = frozenset(PROPOSAL_TREND_VALUES)

# status: a reflection_runs row's own batch lifecycle -- 'running' is the
# only legal value create_reflection_run() ever inserts;
# complete_reflection_run() then makes the single legal terminal
# transition to 'succeeded' or 'failed' (never back to 'running').
REFLECTION_RUN_STATUS_VALUES: tuple[str, ...] = (
    "running",
    "succeeded",
    "failed",
)
_REFLECTION_RUN_STATUS_VALUE_SET = frozenset(REFLECTION_RUN_STATUS_VALUES)
_REFLECTION_RUN_TERMINAL_STATUS_VALUE_SET = frozenset({"succeeded", "failed"})


def _is_valid_confidence(value: Any) -> bool:
    """True only for a real, finite number in [0.0, 1.0].

    Mirrors the same defensive pattern independently maintained in
    tools.case_routing._is_valid_confidence and (until this alignment)
    tools.case_enrichment._is_valid_confidence -- bool is rejected before
    the int check (bool is an int subclass in Python, so `True` would
    otherwise silently pass as confidence=1), and non-standard JSON tokens
    NaN/Infinity/-Infinity (which Python's json module accepts by default)
    are rejected explicitly. This module keeps its own copy rather than
    importing either of those modules' leading-underscore helpers --
    neither exports it, and this storage-layer boundary should not depend
    on either higher-level module to validate a primitive value.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return 0.0 <= value <= 1.0

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
    # Phase 4.5 Stage B.5 Schema Alignment: append-only analysis history
    # for a Case, aligned to the finalized tools.case_enrichment.
    # CaseEnrichmentResult contract. Never updated in place --
    # create_case_analysis() only ever INSERTs, and UNIQUE(case_id,
    # analyzed_at) is the only uniqueness constraint (not UNIQUE(case_id)
    # alone), so re-analysis is always a new row, never an overwrite of a
    # prior one. source_evidence_watermark is NOT cases.updated_at (that
    # column is only ever set once, at Case creation -- see
    # get_case_evidence_watermark()'s docstring) -- it is the
    # MAX(turns.created_at, feedback.submitted_at, retrieval_runs.
    # created_at) this analysis run actually saw, recorded so a later run
    # can tell whether new evidence has arrived since.
    #
    # issue_type/issue_type_confidence and diagnosis/diagnosis_confidence
    # are both NOT NULL: CaseEnrichmentResult requires all four fields
    # unconditionally (never None), so the storage layer now enforces the
    # same invariant rather than only relying on the dataclass layer.
    # product_source/product_confidence stay nullable together (a Case can
    # legitimately have no identified product) but are range/membership
    # CHECKed when present -- see create_case_analysis()'s docstring for
    # the exact product_model/product_source/product_confidence
    # consistency rule this mirrors.
    """
    CREATE TABLE IF NOT EXISTS case_analysis (
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

        -- Not a foreign key or CHECK -- just a recorded TEXT timestamp,
        -- same ISO-8601 (datetime.now(timezone.utc).isoformat()) shape as
        -- every other *_at/*_watermark column in this module, so lexical
        -- TEXT comparison ("<", ">", MAX()) matches chronological order.
        source_evidence_watermark TEXT NOT NULL,

        -- Prevents two rows for the same case at the exact same
        -- analyzed_at (practically only reachable by a caller-supplied
        -- duplicate, since this module's own _now() has microsecond
        -- resolution); also what makes get_latest_case_analysis()'s
        -- "ORDER BY analyzed_at DESC LIMIT 1" unambiguous for any one
        -- case_id -- ties are structurally impossible per case.
        UNIQUE (case_id, analyzed_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_case_analysis_case ON case_analysis(case_id)",
    # Phase 5 Slice 2 (see ../migrations/006_reflector_proposal_domain.sql):
    # a mutable batch-run record, NOT append-only -- create_reflection_run()
    # inserts exactly one row with status='running'; complete_reflection_run()
    # later makes the single legal terminal UPDATE to 'succeeded'/'failed'.
    # window_start/window_end/material_change_detected/run_summary are all
    # nullable: the first two may genuinely be unbounded (an "all Cases"
    # run), the last two are only known once the run actually completes.
    # analyzed_case_count is NOT NULL because by the time a caller creates
    # this row, tools.case_reflection_input.build_reflector_input() has
    # already run and that count is already known -- there is no
    # "in-progress, not yet counted" state for it to represent.
    """
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
    )
    """,
    # improvement_proposals: identity + HUMAN lifecycle only -- never
    # updated by this module except review_status (via
    # update_proposal_review_status()), and nothing in
    # tools.reflector_proposals.ImprovementProposal carries the
    # per-observation fields (pattern_summary, recommended_improvement,
    # confidence, trend, ...) that change on every re-observation; those
    # live on proposal_observations instead. created_at is set once, at
    # INSERT, and never rewritten -- same "set once, never revisited"
    # convention as cases.created_at (see get_case_evidence_watermark's
    # docstring for that precedent).
    """
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
    )
    """,
    # proposal_observations: append-only, exactly like case_analysis --
    # one row per (proposal_id, reflection_run_id) pair, never updated in
    # place; a Proposal re-observed in a later run is always a new row.
    # possible_cause/expected_benefit/limitations are nullable: the AI may
    # legitimately have no supportable hypothesis/benefit/caveat to state
    # for a given observation (see tools.reflector_proposals.
    # ProposalObservation's docstring) -- pattern_summary and
    # recommended_improvement are NOT NULL because an Observation with
    # neither would carry no actionable content at all.
    # supporting_case_ids_json is the Phase 5 Slice 2 POC shape (a JSON
    # array of case_id strings) rather than a normalized join table -- see
    # this migration's own doc file for why that is deliberately deferred.
    """
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

        -- One Observation per Proposal per Run (the actual business
        -- invariant -- see this migration's doc file).
        UNIQUE (proposal_id, reflection_run_id),
        -- Tie-elimination for get_latest_proposal_observation()'s "ORDER
        -- BY observed_at DESC LIMIT 1", mirroring case_analysis's own
        -- UNIQUE(case_id, analyzed_at) exactly.
        UNIQUE (proposal_id, observed_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_proposal_observations_proposal ON proposal_observations(proposal_id)",
    "CREATE INDEX IF NOT EXISTS idx_proposal_observations_run ON proposal_observations(reflection_run_id)",
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

# Migration 005 (see ../migrations/005_case_analysis_taxonomy_alignment.sql):
# rebuild case_analysis in place to align it with the finalized Stage B
# CaseEnrichmentResult contract -- narrows diagnosis to 5 values, adds NOT
# NULL issue_type/issue_type_confidence, and adds range CHECK constraints to
# all three confidence columns. Same recipe as migration 003 (SQLite has no
# `ALTER TABLE ... DROP/ADD CONSTRAINT`, so a CHECK/NOT NULL change requires
# the shadow-table rebuild), but unlike 003's straight unchanged-row copy,
# this INSERT...SELECT also TRANSFORMS legacy data -- see
# FeedbackStoreV2._upgrade_case_analysis_taxonomy()'s docstring for the
# exact, documented (never silently guessed) legacy-value handling:
#   * diagnosis 'unclear_or_other'   -> 'other_or_unclear' (pure rename)
#   * diagnosis 'workflow_tool_issue' -> 'other_or_unclear' (lossy downgrade;
#     the new taxonomy has no workflow/tool-specific bucket)
#   * diagnosis_confidence NULL      -> 0.0 (explicit fallback sentinel;
#     the old API accepted a missing confidence, the new contract does not)
#   * issue_type / issue_type_confidence did not exist before this
#     migration -> every pre-existing row gets the fixed fallback
#     ('other_or_unclear', 0.0), never a value inferred from case_title/
#     issue_summary -- a migration must never fabricate an AI classification.
# Never touches feedback_runs, sessions, cases, turns, feedback, or
# retrieval_runs.
_CASE_ANALYSIS_REBUILD_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE case_analysis_new (
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
    )
    """,
    """
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
    FROM case_analysis
    """,
    "DROP TABLE case_analysis",
    "ALTER TABLE case_analysis_new RENAME TO case_analysis",
    "CREATE INDEX IF NOT EXISTS idx_case_analysis_case ON case_analysis(case_id)",
)


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
        self._upgrade_case_analysis_taxonomy()

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

    def _case_analysis_needs_taxonomy_upgrade(self, db: sqlite3.Connection) -> bool:
        """Detect a case_analysis table still at the pre-Stage-B.5
        (migration 004) shape.

        Unlike _retrieval_runs_needs_execution_status_upgrade (which reads
        the live CHECK constraint text out of sqlite_master.sql, because
        migration 003 changes a CHECK without changing the column set),
        this checks column presence via PRAGMA table_info -- sufficient
        and simpler here because migration 005 also adds real columns
        (issue_type/issue_type_confidence), so there is no scenario where
        the columns already exist but the CHECK is still stale on its own.
        Returns False (nothing to do) both when already upgraded AND when
        the table does not exist yet at all (a fresh database, already
        created in the latest shape by _SCHEMA_STATEMENTS above).
        """
        cols = {row["name"] for row in db.execute("PRAGMA table_info(case_analysis)").fetchall()}
        if not cols:
            return False
        return "issue_type" not in cols

    def _upgrade_case_analysis_taxonomy(self) -> None:
        """Migration 005 (see
        ../migrations/005_case_analysis_taxonomy_alignment.sql): rebuild
        case_analysis in place to match the finalized Stage B
        CaseEnrichmentResult contract -- narrows diagnosis to 5 values,
        adds NOT NULL issue_type/issue_type_confidence, and adds range
        CHECK constraints to all three confidence columns.

        Same rebuild choreography as
        _upgrade_retrieval_runs_execution_statuses (migration 003): a
        dedicated connection, PRAGMA foreign_keys turned OFF before BEGIN
        and back ON after COMMIT/ROLLBACK (in a finally), BEGIN IMMEDIATE,
        a shadow table, an explicit-column INSERT...SELECT (never
        `SELECT *`), DROP + RENAME, the index recreated by name, and
        PRAGMA foreign_key_check verified before COMMIT -- any failure at
        any step rolls back the whole transaction, leaving the original
        case_analysis table and its data completely untouched.

        Legacy data handling in the INSERT...SELECT
        (_CASE_ANALYSIS_REBUILD_STATEMENTS) -- every case documented, none
        silently guessed:
          * diagnosis 'unclear_or_other' -> 'other_or_unclear': a pure
            rename, same meaning, per the finalized Stage B taxonomy.
          * diagnosis 'workflow_tool_issue' -> 'other_or_unclear': a real,
            intentional LOSSY downgrade, not a semantic-equivalence rename
            -- the new taxonomy has no workflow/tool-specific bucket at
            all. This is the documented product decision for this
            migration (see migrations/005_case_analysis_taxonomy_alignment.sql),
            not something inferred here.
          * diagnosis_confidence NULL -> 0.0: the pre-alignment
            create_case_analysis() accepted a missing confidence
            (diagnosis_confidence was optional); the new contract requires
            it. 0.0 is an explicit fallback sentinel for "this row predates
            the NOT NULL requirement", never a claim that the original
            analysis actually reported zero confidence.
          * issue_type/issue_type_confidence did not exist before this
            migration, so every pre-existing row gets the fixed fallback
            ('other_or_unclear', 0.0) -- NEVER a value inferred from
            case_title/issue_summary. A migration must not fabricate an AI
            classification a row never actually received; the fixed
            fallback honestly represents "this row was never classified
            for Issue Type."

        Never touches feedback_runs, sessions, cases, turns, feedback, or
        retrieval_runs.
        """
        db = sqlite3.connect(self.path)
        try:
            db.row_factory = sqlite3.Row
            db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if not self._case_analysis_needs_taxonomy_upgrade(db):
                return
            db.execute("PRAGMA foreign_keys=OFF")
            try:
                db.execute("BEGIN IMMEDIATE")
                try:
                    # DROP TABLE IF EXISTS guards a shadow table left over
                    # from an earlier attempt that failed after creating
                    # it but before this same transaction could complete
                    # -- same defensive belt-and-suspenders as the
                    # retrieval_runs rebuild above.
                    db.execute("DROP TABLE IF EXISTS case_analysis_new")
                    for statement in _CASE_ANALYSIS_REBUILD_STATEMENTS:
                        db.execute(statement)
                    violations = db.execute(
                        "PRAGMA foreign_key_check(case_analysis)"
                    ).fetchall()
                    if violations:
                        raise sqlite3.IntegrityError(
                            f"case_analysis rebuild left {len(violations)} "
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

    def list_feedback_for_case(self, case_id: str) -> list[sqlite3.Row]:
        """Every Feedback row for every Turn in this Case, oldest first.

        A Case-level counterpart to get_feedback() (single-turn) -- joins
        through turns rather than requiring the caller to loop
        list_turns_for_case() + get_feedback() per turn. feedback_id is a
        secondary sort key purely to make row order fully deterministic
        even on the (currently unreachable, since submitted_at has
        microsecond resolution) chance of two rows sharing the same
        submitted_at -- it carries no meaning of its own.
        """
        with self._connect() as db:
            return db.execute(
                """
                SELECT feedback.*
                FROM feedback
                JOIN turns ON feedback.turn_id = turns.turn_id
                WHERE turns.case_id = ?
                ORDER BY feedback.submitted_at, feedback.feedback_id
                """,
                (case_id,),
            ).fetchall()

    # -- case analysis (Phase 4.5 Stage A) -------------------------------

    def get_case_evidence_watermark(self, case_id: str) -> str | None:
        """The latest timestamp across every piece of evidence this Case
        currently has: MAX(turns.created_at, feedback.submitted_at,
        retrieval_runs.created_at) for Turns belonging to this case_id.

        Deliberately NOT cases.updated_at -- that column is set once, at
        Case creation (FeedbackStoreV2.create_case()), and is never touched
        by create_turn(), submit_feedback(), add_suggestion(), or
        add_retrieval_runs() (verified: no code path anywhere issues
        ``UPDATE cases``). Using it as a "Case evidence changed" watermark
        would make every Case with more than one Turn look permanently
        not-stale after its first Turn.

        Implementation note: combines all three sources via UNION ALL
        into one column, then applies the AGGREGATE MAX() over that column
        -- never SQLite's scalar ``max(a, b, c)`` form, which returns NULL
        if ANY argument is NULL (a Case with turns but no feedback yet, or
        no retrieval evidence yet, would silently lose its real watermark).
        Aggregate MAX() over a UNION ALL correctly ignores the branches
        that contribute zero rows and returns NULL only when the whole
        union is empty.

        Returns None if the case has zero turns (create_case() does not
        require an immediate turn -- a Case can theoretically exist with no
        Turn yet if a caller creates one directly, or if turn persistence
        failed after case persistence already committed) or does not
        exist. Never raises on a missing/unknown case_id -- an empty result
        set is a normal SQL outcome, not an error.
        """
        with self._connect() as db:
            row = db.execute(
                """
                SELECT MAX(ts) AS watermark FROM (
                    SELECT created_at AS ts FROM turns WHERE case_id = ?
                    UNION ALL
                    SELECT feedback.submitted_at AS ts
                    FROM feedback JOIN turns ON feedback.turn_id = turns.turn_id
                    WHERE turns.case_id = ?
                    UNION ALL
                    SELECT retrieval_runs.created_at AS ts
                    FROM retrieval_runs JOIN turns ON retrieval_runs.turn_id = turns.turn_id
                    WHERE turns.case_id = ?
                )
                """,
                (case_id, case_id, case_id),
            ).fetchone()
        return row["watermark"] if row is not None else None

    def get_latest_case_analysis(self, case_id: str) -> sqlite3.Row | None:
        """The most recent case_analysis row for this case_id (by
        analyzed_at), or None if this Case has never been analyzed.

        Unambiguous by construction: UNIQUE(case_id, analyzed_at) means two
        rows for the SAME case_id can never share the same analyzed_at, so
        "ORDER BY analyzed_at DESC LIMIT 1" can never have a tie to break
        for any one case_id (ties are only structurally possible ACROSS
        different case_ids, which this query never compares against each
        other).
        """
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM case_analysis WHERE case_id=? ORDER BY analyzed_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()

    def create_case_analysis(
        self,
        analysis_id: str,
        case_id: str,
        *,
        case_title: str | None,
        issue_summary: str | None,
        issue_type: str,
        issue_type_confidence: float,
        diagnosis: str,
        diagnosis_confidence: float,
        product_model: str | None,
        product_source: str | None,
        product_confidence: float | None,
        evidence_json: str | None,
        analysis_version: str,
        analyzed_at: str,
        source_evidence_watermark: str,
    ) -> bool:
        """Insert one analysis row for a Case. Append-only: this never
        updates a prior row for the same case_id -- re-analysis is always a
        new row (analyzed_at must differ from every prior row for this
        case_id, enforced by UNIQUE(case_id, analyzed_at)), so analysis
        history for a Case is never lost or silently overwritten.

        Phase 4.5 Stage B.5 Schema Alignment: this is the storage-layer
        boundary for the finalized tools.case_enrichment.CaseEnrichmentResult
        contract, and re-validates every primitive value in Python before
        attempting the INSERT (fail closed, matching submit_feedback()'s
        convention) -- never depends on CaseEnrichmentResult.__post_init__
        having already run, since this is a public storage boundary that
        must defend itself regardless of caller. The table's own CHECK
        constraints are the authoritative backstop if this method is ever
        bypassed. Storage layer only validates primitive values/contract
        shape -- it has no notion of CaseEnrichmentResult or any LLM
        concept.

        issue_type/issue_type_confidence/diagnosis/diagnosis_confidence are
        all REQUIRED (never None) -- matching CaseEnrichmentResult, which
        never leaves any of the four as None either. product_model is
        optional; when present, product_source and product_confidence are
        BOTH required, when absent, both must be None -- the same
        consistency rule CaseEnrichmentResult.__post_init__ enforces at the
        dataclass layer, now also enforced here.

        Returns False, never raises, for: an invalid/out-of-range
        issue_type, issue_type_confidence, diagnosis, diagnosis_confidence,
        or product_confidence; an invalid product_source; a product_model/
        product_source/product_confidence combination that violates the
        consistency rule above; an unknown case_id (FK violation); or a
        duplicate (case_id, analyzed_at) pair (UNIQUE violation) -- e.g. a
        caller retrying with the same analyzed_at it already used.

        Does not touch cases.title / cases.product_model -- deciding
        whether/how an analysis result should update the Case's own
        identity fields is deferred to a later stage (Human Override
        policy is explicitly out of scope for this alignment).
        """
        if issue_type not in _ISSUE_TYPE_VALUE_SET:
            return False
        if not _is_valid_confidence(issue_type_confidence):
            return False
        if diagnosis not in _DIAGNOSIS_VALUE_SET:
            return False
        if not _is_valid_confidence(diagnosis_confidence):
            return False
        if product_model is None:
            if product_source is not None or product_confidence is not None:
                return False
        else:
            if product_source is None or product_source not in _PRODUCT_SOURCE_VALUE_SET:
                return False
            if not _is_valid_confidence(product_confidence):
                return False

        with self._connect() as db:
            try:
                db.execute(
                    """
                    INSERT INTO case_analysis (
                        analysis_id, case_id, case_title, issue_summary,
                        issue_type, issue_type_confidence,
                        diagnosis, diagnosis_confidence,
                        product_model, product_source, product_confidence,
                        evidence_json, analysis_version, analyzed_at,
                        source_evidence_watermark
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(analysis_id), str(case_id), case_title, issue_summary,
                        issue_type, issue_type_confidence,
                        diagnosis, diagnosis_confidence,
                        product_model, product_source, product_confidence,
                        evidence_json, str(analysis_version), str(analyzed_at),
                        str(source_evidence_watermark),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def list_cases_needing_analysis(
        self, *, session_id: str | None = None, limit: int | None = None
    ) -> list[sqlite3.Row]:
        """Cases whose evidence has advanced past every analysis on
        record for them (or that have never been analyzed at all), oldest
        evidence first. Each returned row has ``case_id`` and
        ``evidence_watermark`` columns.

        Contract: a case_id is included when
            no case_analysis row exists for it
            OR
            its current evidence watermark (§get_case_evidence_watermark)
            > MAX(source_evidence_watermark) across every case_analysis
              row ever recorded for it.
        Using MAX(source_evidence_watermark) across ALL of a case's past
        analyses (not just its most-recent-by-analyzed_at row) is
        deliberate: source_evidence_watermark only grows across
        successive runs for a real Case (each run computes it fresh from
        current evidence, which never shrinks), so the two are equivalent
        in every normal scenario, but this formulation stays correct
        without depending on that assumption.

        A Case with zero Turns (get_case_evidence_watermark() returns
        None) is never included -- there is no evidence yet to analyze,
        so "needs analysis" would be meaningless for it.

        One query, no per-case follow-up lookups: the per-case evidence
        watermark and the per-case latest-analysis-watermark are each
        computed once via GROUP BY, then LEFT JOINed together.
        ``session_id`` optionally scopes to one session's Cases (matching
        list_cases_for_session's own scoping); ``limit`` bounds one
        caller's batch, applied after ordering so the oldest-evidence
        Cases are always favored first when a caller does not process the
        whole backlog in one run.
        """
        query = """
            SELECT ev.case_id AS case_id, ev.evidence_watermark AS evidence_watermark
            FROM (
                SELECT case_id, MAX(ts) AS evidence_watermark
                FROM (
                    SELECT case_id, created_at AS ts FROM turns
                    UNION ALL
                    SELECT turns.case_id AS case_id, feedback.submitted_at AS ts
                    FROM feedback JOIN turns ON feedback.turn_id = turns.turn_id
                    UNION ALL
                    SELECT turns.case_id AS case_id, retrieval_runs.created_at AS ts
                    FROM retrieval_runs JOIN turns ON retrieval_runs.turn_id = turns.turn_id
                )
                GROUP BY case_id
            ) ev
            LEFT JOIN (
                SELECT case_id, MAX(source_evidence_watermark) AS latest_watermark
                FROM case_analysis
                GROUP BY case_id
            ) la ON ev.case_id = la.case_id
            WHERE (la.latest_watermark IS NULL OR ev.evidence_watermark > la.latest_watermark)
        """
        params: list[Any] = []
        if session_id is not None:
            query += " AND ev.case_id IN (SELECT case_id FROM cases WHERE session_id = ?)"
            params.append(session_id)
        query += " ORDER BY ev.evidence_watermark"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        with self._connect() as db:
            return db.execute(query, params).fetchall()

    # -- case reflection input (Phase 5 Slice 1) -------------------------

    def list_case_ids_created_in_window(
        self,
        *,
        created_from: str | None = None,
        created_before: str | None = None,
    ) -> list[str]:
        """Every case_id whose cases.created_at falls in
        [created_from, created_before) -- a pure Case-identity/creation-time
        filter. Does not touch case_analysis, evidence, or any other table;
        has no notion of "stale", "analyzed", or Reflector.

        Half-open window, matching the conventional [start, end) shape:
        created_from is INCLUSIVE (created_at >= created_from),
        created_before is EXCLUSIVE (created_at < created_before) -- so two
        adjacent windows (e.g. consecutive weekly batches) never overlap
        and never leave a gap at the shared boundary timestamp.

        created_from=None means no lower bound; created_before=None means
        no upper bound; both None returns every Case's id. Comparison is
        lexical TEXT comparison against the same ISO-8601
        (datetime.now(timezone.utc).isoformat(), fixed UTC offset) shape
        every other *_at column in this module already uses -- see
        get_case_evidence_watermark's docstring for why plain string
        comparison is safe for this timestamp shape.

        Deliberately does NOT know about case_analysis: this is the
        Case-occurrence-window half of a two-query composition with
        list_latest_case_analysis (its Case-analysis-snapshot counterpart,
        see that method's docstring) -- a caller wanting "latest analysis
        for Cases created in this window" composes both explicitly. Kept
        this way so this method stays a general-purpose Case-identity
        primitive a future Dashboard (or any other caller with a reason to
        filter Cases by creation time, unrelated to case_analysis) can
        reuse directly, rather than only being reachable through a
        Reflector-shaped combined query.

        Ordering is deterministic: created_at ascending, then case_id
        ascending as the tiebreaker (unlike case_analysis.analyzed_at,
        cases.created_at carries no UNIQUE constraint, so two different
        Cases really can share the same created_at).
        """
        query = "SELECT case_id FROM cases WHERE 1=1"
        params: list[Any] = []
        if created_from is not None:
            query += " AND created_at >= ?"
            params.append(created_from)
        if created_before is not None:
            query += " AND created_at < ?"
            params.append(created_before)
        query += " ORDER BY created_at ASC, case_id ASC"

        with self._connect() as db:
            return [row["case_id"] for row in db.execute(query, params).fetchall()]

    def list_latest_case_analysis(
        self, *, case_ids: Collection[str] | None = None
    ) -> list[sqlite3.Row]:
        """Every Case's most recent case_analysis row (by analyzed_at), one
        row per case_id -- the batch counterpart to get_latest_case_analysis
        (which serves a single case_id). Same "latest" semantics as that
        method: UNIQUE(case_id, analyzed_at) makes MAX(analyzed_at) per
        case_id unambiguous, so a Case's historical (superseded) analysis
        rows can never be double-counted alongside its current one here
        either.

        case_ids=None (the default) returns the latest row for every Case
        that has EVER been analyzed (i.e. every distinct case_id present in
        case_analysis) -- a Case that has never been analyzed contributes no
        row, exactly like get_latest_case_analysis(case_id) returns None for
        it. case_ids=<empty collection> is explicit input, not "no filter":
        it deterministically returns [] without ever querying the database
        -- an empty collection must never be silently reinterpreted as
        "case_ids was omitted". This mirrors the None-vs-explicitly-empty
        distinction tools.retrieval_runtime.persist_turn_and_retrieval
        already establishes for its own candidate_case_ids parameter (see
        that function's docstring): omitted and explicitly-empty are two
        different, both meaningful, inputs.

        Deliberately does NOT know about cases.created_at, any notion of an
        analysis "window", or Reflector -- this is a general-purpose
        "latest snapshot per Case" primitive a future Dashboard can also
        call directly (e.g. "latest analysis for these specific case_ids",
        no time-window concept involved at all). A caller that also wants
        to scope by Case creation time composes this with
        list_case_ids_created_in_window (its Case-identity/creation-time
        counterpart) rather than this method absorbing that filter itself.

        Ordering is deterministic: case_id ascending.
        """
        if case_ids is not None and not case_ids:
            return []

        query = """
            SELECT case_analysis.*
            FROM case_analysis
            INNER JOIN (
                SELECT case_id, MAX(analyzed_at) AS latest_analyzed_at
                FROM case_analysis
                {where}
                GROUP BY case_id
            ) latest
              ON case_analysis.case_id = latest.case_id
             AND case_analysis.analyzed_at = latest.latest_analyzed_at
            ORDER BY case_analysis.case_id ASC
        """
        params: list[Any] = []
        if case_ids is None:
            query = query.format(where="")
        else:
            ids = list(case_ids)
            placeholders = ",".join("?" for _ in ids)
            query = query.format(where=f"WHERE case_id IN ({placeholders})")
            params.extend(ids)

        with self._connect() as db:
            return db.execute(query, params).fetchall()

    # -- reflector proposal domain (Phase 5 Slice 2) ---------------------

    def create_reflection_run(
        self,
        reflection_run_id: str,
        *,
        started_at: str,
        analyzed_case_count: int,
        reflector_version: str,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> bool:
        """Create one reflection_runs row with status='running' -- the
        only legal initial status (never caller-supplied, matching how
        create_case()/create_turn() never accept a caller-chosen initial
        state where only one is legal). complete_reflection_run() makes
        the single legal terminal transition later.

        Returns False, never raises, for: a non-int/negative
        analyzed_case_count, or a duplicate reflection_run_id (PRIMARY KEY
        violation).
        """
        if (
            not isinstance(analyzed_case_count, int)
            or isinstance(analyzed_case_count, bool)
            or analyzed_case_count < 0
        ):
            return False

        with self._connect() as db:
            try:
                db.execute(
                    """
                    INSERT INTO reflection_runs (
                        reflection_run_id, window_start, window_end,
                        started_at, completed_at, status,
                        analyzed_case_count, material_change_detected,
                        run_summary, reflector_version
                    ) VALUES (?, ?, ?, ?, NULL, 'running', ?, NULL, NULL, ?)
                    """,
                    (
                        str(reflection_run_id), window_start, window_end,
                        str(started_at), int(analyzed_case_count), str(reflector_version),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def complete_reflection_run(
        self,
        reflection_run_id: str,
        *,
        status: str,
        completed_at: str,
        material_change_detected: bool | None = None,
        run_summary: str | None = None,
    ) -> bool:
        """Make the single legal terminal UPDATE for a reflection_runs row:
        status must be 'succeeded' or 'failed' -- never 'running' again.
        The WHERE clause only ever matches a row that is CURRENTLY
        status='running', so this can never re-complete an
        already-terminal row.

        Mirrors update_turn_retrieval_observation()'s existing "validated
        UPDATE, rowcount==1" convention -- the only other row-mutation
        precedent already in this module. Returns False, never raises,
        for: an invalid status (not 'succeeded'/'failed'), an unknown
        reflection_run_id, or a reflection_run_id that is not currently
        'running' (already completed once, or never existed).
        """
        if status not in _REFLECTION_RUN_TERMINAL_STATUS_VALUE_SET:
            return False

        with self._connect() as db:
            cur = db.execute(
                """
                UPDATE reflection_runs
                SET status=?, completed_at=?, material_change_detected=?, run_summary=?
                WHERE reflection_run_id=? AND status='running'
                """,
                (
                    status, str(completed_at),
                    None if material_change_detected is None else int(bool(material_change_detected)),
                    run_summary, str(reflection_run_id),
                ),
            )
            return cur.rowcount == 1

    def get_reflection_run(self, reflection_run_id: str) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM reflection_runs WHERE reflection_run_id=?", (reflection_run_id,)
            ).fetchone()

    def create_improvement_proposal(
        self,
        proposal_id: str,
        *,
        improvement_target: str,
        title: str,
        created_at: str,
    ) -> bool:
        """Create one improvement_proposals row with review_status='pending'
        -- the only legal initial review_status (a freshly detected
        Proposal has not yet been reviewed by anyone; never caller-
        supplied, same "only one legal initial value" reasoning as
        create_reflection_run()'s status='running').

        Returns False, never raises, for: an invalid improvement_target, a
        blank title, or a duplicate proposal_id (PRIMARY KEY violation).
        """
        if improvement_target not in _IMPROVEMENT_TARGET_VALUE_SET:
            return False
        if not isinstance(title, str) or not title.strip():
            return False

        with self._connect() as db:
            try:
                db.execute(
                    """
                    INSERT INTO improvement_proposals (
                        proposal_id, improvement_target, title, review_status, created_at
                    ) VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (str(proposal_id), improvement_target, title, str(created_at)),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def get_improvement_proposal(self, proposal_id: str) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM improvement_proposals WHERE proposal_id=?", (proposal_id,)
            ).fetchone()

    def list_improvement_proposals(self, *, review_status: str | None = None) -> list[sqlite3.Row]:
        """Every improvement_proposals row, oldest (created_at) first.
        review_status optionally scopes to one review status -- an invalid
        value returns [] rather than raising, matching list_cases_
        needing_analysis's own optional-filter convention (session_id).
        """
        if review_status is not None and review_status not in _REVIEW_STATUS_VALUE_SET:
            return []
        query = "SELECT * FROM improvement_proposals"
        params: list[Any] = []
        if review_status is not None:
            query += " WHERE review_status=?"
            params.append(review_status)
        query += " ORDER BY created_at"
        with self._connect() as db:
            return db.execute(query, params).fetchall()

    def update_proposal_review_status(self, proposal_id: str, review_status: str) -> bool:
        """The HUMAN review action -- move an improvement_proposals row's
        review_status. This module has no notion of "who" is calling it;
        the process boundary (a human reviewer only, never an AI/
        Reflector code path) is documented, not type-enforced here -- see
        this module's own docstring and tools.reflector_proposals's
        self-improvement boundary section.

        Mirrors update_turn_retrieval_observation()'s "validated UPDATE,
        rowcount==1" convention. Returns False, never raises, for an
        invalid review_status or an unknown proposal_id.
        """
        if review_status not in _REVIEW_STATUS_VALUE_SET:
            return False
        with self._connect() as db:
            cur = db.execute(
                "UPDATE improvement_proposals SET review_status=? WHERE proposal_id=?",
                (review_status, str(proposal_id)),
            )
            return cur.rowcount == 1

    def create_proposal_observation(
        self,
        observation_id: str,
        proposal_id: str,
        reflection_run_id: str,
        *,
        trend: str,
        pattern_summary: str,
        possible_cause: str | None,
        recommended_improvement: str,
        expected_benefit: str | None,
        limitations: str | None,
        supporting_case_ids_json: str,
        supporting_case_count: int,
        confidence: float,
        observed_at: str,
    ) -> bool:
        """Insert one Observation row for a Proposal. Append-only: this
        never updates a prior row -- a Proposal re-observed in a later
        Reflection Run is always a new row (UNIQUE(proposal_id,
        reflection_run_id) enforces at most one Observation per Proposal
        per Run; UNIQUE(proposal_id, observed_at) additionally guarantees
        get_latest_proposal_observation()'s ordering is never ambiguous),
        matching create_case_analysis()'s exact append-only convention.

        Re-validates every primitive value in Python before attempting the
        INSERT (fail closed, matching create_case_analysis()'s own
        convention) -- never depends on tools.reflector_proposals.
        ProposalObservation.__post_init__ having already run, since this
        is a public storage boundary that must defend itself regardless of
        caller. The table's own CHECK constraints are the authoritative
        backstop if this method is ever bypassed.

        Cross-field rules (trend='no_longer_observed' may have
        supporting_case_count=0, every other trend may not) are NOT
        re-checked here -- matching how create_case_analysis() itself
        never re-derives its own product_model/source/confidence
        consistency rule at the SQL/CHECK level either; single-column
        CHECK constraints are the DB-level backstop, cross-field
        consistency is the dataclass layer's job
        (tools.reflector_proposals.ProposalObservation.__post_init__).

        Returns False, never raises, for: an invalid trend, a blank
        pattern_summary/recommended_improvement, an invalid confidence, a
        non-int/negative supporting_case_count, an unknown proposal_id or
        reflection_run_id (FK violation), or a duplicate
        (proposal_id, reflection_run_id) or (proposal_id, observed_at)
        pair (UNIQUE violation).
        """
        if trend not in _PROPOSAL_TREND_VALUE_SET:
            return False
        if not isinstance(pattern_summary, str) or not pattern_summary.strip():
            return False
        if not isinstance(recommended_improvement, str) or not recommended_improvement.strip():
            return False
        if not _is_valid_confidence(confidence):
            return False
        if (
            not isinstance(supporting_case_count, int)
            or isinstance(supporting_case_count, bool)
            or supporting_case_count < 0
        ):
            return False

        with self._connect() as db:
            try:
                db.execute(
                    """
                    INSERT INTO proposal_observations (
                        observation_id, proposal_id, reflection_run_id,
                        trend, pattern_summary, possible_cause,
                        recommended_improvement, expected_benefit, limitations,
                        supporting_case_ids_json, supporting_case_count,
                        confidence, observed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(observation_id), str(proposal_id), str(reflection_run_id),
                        trend, pattern_summary, possible_cause,
                        recommended_improvement, expected_benefit, limitations,
                        str(supporting_case_ids_json), int(supporting_case_count),
                        confidence, str(observed_at),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def get_latest_proposal_observation(self, proposal_id: str) -> sqlite3.Row | None:
        """The most recent proposal_observations row for this proposal_id
        (by observed_at), or None if this Proposal has never been
        observed. Unambiguous by construction, mirroring
        get_latest_case_analysis()'s own reasoning: UNIQUE(proposal_id,
        observed_at) means two rows for the SAME proposal_id can never
        share the same observed_at.
        """
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM proposal_observations WHERE proposal_id=? ORDER BY observed_at DESC LIMIT 1",
                (proposal_id,),
            ).fetchone()

    def list_latest_proposal_observations(
        self, *, proposal_ids: Collection[str] | None = None
    ) -> list[sqlite3.Row]:
        """Every Proposal's most recent proposal_observations row, one row
        per proposal_id -- the batch counterpart to
        get_latest_proposal_observation(), mirroring list_latest_case_
        analysis()'s exact shape and None-vs-empty-collection semantics:
        this is what a future "Proposal + latest Observation" default view
        (see the Phase 5 Slice 2 task instruction, section 2) reads.

        proposal_ids=None returns the latest row for every Proposal that
        has EVER been observed. proposal_ids=<empty collection> is
        explicit input, not "no filter": it deterministically returns []
        without ever querying the database -- an empty collection must
        never be silently reinterpreted as "proposal_ids was omitted".

        Ordering is deterministic: proposal_id ascending.
        """
        if proposal_ids is not None and not proposal_ids:
            return []

        query = """
            SELECT proposal_observations.*
            FROM proposal_observations
            INNER JOIN (
                SELECT proposal_id, MAX(observed_at) AS latest_observed_at
                FROM proposal_observations
                {where}
                GROUP BY proposal_id
            ) latest
              ON proposal_observations.proposal_id = latest.proposal_id
             AND proposal_observations.observed_at = latest.latest_observed_at
            ORDER BY proposal_observations.proposal_id ASC
        """
        params: list[Any] = []
        if proposal_ids is None:
            query = query.format(where="")
        else:
            ids = list(proposal_ids)
            placeholders = ",".join("?" for _ in ids)
            query = query.format(where=f"WHERE proposal_id IN ({placeholders})")
            params.extend(ids)

        with self._connect() as db:
            return db.execute(query, params).fetchall()

    def persist_reflection_proposals(
        self,
        reflection_run_id: str,
        *,
        new_proposals: Collection[Mapping[str, Any]],
        observations: Collection[Mapping[str, Any]],
    ) -> bool:
        """Atomically insert every new Proposal and every Observation for
        one Reflection Run, in a SINGLE transaction -- either every row is
        written, or none is (Phase 5 Slice 6A task instruction, section 4:
        never a Proposal with no Observation, never a half-written
        Observation batch).

        Unlike create_improvement_proposal()/create_proposal_observation()
        (each its own implicit commit-per-call transaction via
        self._connect()'s `with` context), this method opens one explicit
        BEGIN IMMEDIATE / COMMIT / ROLLBACK transaction on a dedicated
        connection, mirroring the exact choreography this module already
        uses for _upgrade_retrieval_runs_execution_statuses() / _upgrade_
        case_analysis_taxonomy() (see those methods' own docstrings) --
        the only other place in this module more than one write needs to
        succeed or fail together. `PRAGMA foreign_keys=ON` is set
        explicitly on this fresh connection (SQLite defaults a brand-new
        connection to foreign_keys=OFF) so the FK-violation-based fail-
        closed behavior below actually fires.

        `new_proposals` items need: proposal_id, improvement_target,
        title, created_at (review_status is never caller-supplied here
        either -- forced to 'pending', matching create_improvement_
        proposal()'s own "only one legal initial value" convention).
        `observations` items need: observation_id, proposal_id, trend,
        pattern_summary, possible_cause, recommended_improvement,
        expected_benefit, limitations, supporting_case_ids_json,
        supporting_case_count, confidence, observed_at -- the exact same
        shape create_proposal_observation() already takes, minus
        reflection_run_id (this method's own `reflection_run_id`
        parameter is used for every row, never read from the mapping, so
        a caller cannot accidentally attach one Observation to a
        different Run than the rest of this same call's batch).

        Plain Mapping[str, Any] rows, not tools.reflector_proposals.
        ImprovementProposal/ProposalObservation instances -- this module
        must not import that module (it would be a circular import: that
        module already imports taxonomy constants FROM this one). A
        caller with ImprovementProposal/ProposalObservation objects (e.g.
        a future persist_reflection_result() glue function) is expected
        to project each into this plain-mapping shape itself, exactly how
        tools.run_case_enrichment already projects a CaseEnrichmentResult
        into create_case_analysis()'s own plain keyword arguments.

        Re-validates every primitive value in Python before opening the
        transaction (fail closed, before touching the DB at all) --
        mirrors create_improvement_proposal()/create_proposal_observation()'s
        own "never depends on the caller's dataclass __post_init__ having
        already run" convention, since this is a public storage boundary
        too. Cross-field rules (trend requires supporting cases except
        'no_longer_observed', etc.) are NOT re-checked here, matching
        create_proposal_observation()'s own reasoning: single-column
        checks are this layer's job, cross-field consistency is the
        dataclass layer's.

        Insertion order within the transaction is always every
        new_proposals row, THEN every observations row -- an Observation
        founding a brand-new Proposal (action=create_new) would otherwise
        violate proposal_observations' own
        FOREIGN KEY (proposal_id) REFERENCES improvement_proposals
        (proposal_id) constraint, since that Proposal row would not exist
        yet.

        The match_existing / create_new guardrails Phase 5 Slice 6A task
        instruction sections 6-7 ask for are NOT re-implemented as
        separate Python checks here -- they fall directly out of the
        schema's own constraints, which this method never bypasses:
          * an Observation whose proposal_id names neither a Proposal in
            `new_proposals` (this call) nor an already-existing
            improvement_proposals row violates the FOREIGN KEY constraint
            above -- exactly "match_existing referencing a Proposal that
            does not exist" (Phase 5 Slice 6A task instruction, section 6:
            "如果不存在：fail closed").
          * a duplicate proposal_id in `new_proposals` (or one that
            collides with an existing row) violates improvement_proposals'
            own PRIMARY KEY -- exactly "duplicate create_new Proposal ID"
            (section 7: "如果 duplicate proposal_id：fail closed / rollback").
        Both surface as sqlite3.IntegrityError, caught below, which rolls
        back the WHOLE transaction and returns False -- never a partial
        write, never a silently-ignored duplicate.

        Returns False, never raises, for: an invalid improvement_target/
        blank title on any new_proposals row; an invalid trend/blank
        pattern_summary/blank recommended_improvement/invalid confidence/
        invalid supporting_case_count on any observations row; or any
        sqlite3.IntegrityError raised while executing the transaction
        (already-rolled-back by this method itself in every such case --
        the caller never needs to issue its own ROLLBACK).
        """
        for proposal in new_proposals:
            if proposal["improvement_target"] not in _IMPROVEMENT_TARGET_VALUE_SET:
                return False
            if not isinstance(proposal["title"], str) or not proposal["title"].strip():
                return False

        for observation in observations:
            if observation["trend"] not in _PROPOSAL_TREND_VALUE_SET:
                return False
            if not isinstance(observation["pattern_summary"], str) or not observation["pattern_summary"].strip():
                return False
            if (
                not isinstance(observation["recommended_improvement"], str)
                or not observation["recommended_improvement"].strip()
            ):
                return False
            if not _is_valid_confidence(observation["confidence"]):
                return False
            count = observation["supporting_case_count"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                return False

        db = sqlite3.connect(self.path)
        try:
            db.row_factory = sqlite3.Row
            db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            try:
                for proposal in new_proposals:
                    db.execute(
                        """
                        INSERT INTO improvement_proposals (
                            proposal_id, improvement_target, title, review_status, created_at
                        ) VALUES (?, ?, ?, 'pending', ?)
                        """,
                        (
                            str(proposal["proposal_id"]), proposal["improvement_target"],
                            proposal["title"], str(proposal["created_at"]),
                        ),
                    )
                for observation in observations:
                    db.execute(
                        """
                        INSERT INTO proposal_observations (
                            observation_id, proposal_id, reflection_run_id,
                            trend, pattern_summary, possible_cause,
                            recommended_improvement, expected_benefit, limitations,
                            supporting_case_ids_json, supporting_case_count,
                            confidence, observed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(observation["observation_id"]), str(observation["proposal_id"]),
                            str(reflection_run_id),
                            observation["trend"], observation["pattern_summary"], observation["possible_cause"],
                            observation["recommended_improvement"], observation["expected_benefit"],
                            observation["limitations"],
                            str(observation["supporting_case_ids_json"]), int(observation["supporting_case_count"]),
                            observation["confidence"], str(observation["observed_at"]),
                        ),
                    )
            except sqlite3.IntegrityError:
                db.execute("ROLLBACK")
                return False
            else:
                db.execute("COMMIT")
            return True
        finally:
            db.close()


__all__ = [
    "FeedbackStoreV2",
    "RetrievalRunInput",
    "EXECUTION_STATUSES",
    "OBSERVATION_STATUSES",
    "REASON_CODES",
    "DIAGNOSIS_VALUES",
    "ISSUE_TYPE_VALUES",
    "PRODUCT_SOURCE_VALUES",
    "IMPROVEMENT_TARGET_VALUES",
    "REVIEW_STATUS_VALUES",
    "PROPOSAL_TREND_VALUES",
    "REFLECTION_RUN_STATUS_VALUES",
]
