"""Dashboard MVP read-projection layer -- pure functions turning
FeedbackStoreV2 rows into plain, JSON-ready dicts. No FastAPI/Starlette
import anywhere in this module: every function here takes a
FeedbackStoreV2 and returns plain dict/list-of-dict, so it is directly
unit-testable without spinning up an HTTP app, and could be reused from a
CLI or a notebook exactly as easily as from dashboard_api.app.

Naming note: "dashboard_api" here is the Curator/Feedback POC's own
read/write API for a FUTURE separate Dashboard frontend (not built in
this slice). It is unrelated to Hermes' own built-in browser dashboard
(`hermes dashboard`, `/opt/hermes/hermes_cli/web_dist`,
`/sandbox/.hermes/dashboard-home` -- see Dockerfile.universal-feedback) --
same kind of naming collision this codebase already documents for
"Curator" (tools.curator_domain) vs. agent.curator.maybe_run_curator().

Every function is a straight read: no INSERT/UPDATE/DELETE anywhere in
this module. The three POST endpoints (proposal review, curator change
review, curator change apply) reuse tools.review_proposal/tools.review_
curator_change/tools.apply_curator_change directly from dashboard_api.app
-- this module has no write-side counterpart to duplicate that logic.

Aggregation philosophy: for every Overview/list number that an EXISTING
single-call FeedbackStoreV2 method can already answer in full (case
diagnosis/product distribution via list_latest_case_analysis(), Proposal
review_status distribution via list_improvement_proposals(), curator_
changes status distribution via list_curator_changes()), this module
fetches the whole result set and aggregates in Python -- POC data scale
makes this the simplest correct option, and it avoids inventing a second,
SQL-side aggregation convention alongside FeedbackStoreV2's existing
row-returning style. Only turns/feedback totals use a dedicated store
method (count_turns()/list_all_feedback()), because no existing method
can answer either without an unreasonable per-Case N+1 scan.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from tools.case_reflection_input import build_case_intelligence_for_ids
from tools.feedback_store_v2 import FeedbackStoreV2

_UNKNOWN_PRODUCT_MODEL_BUCKET = "__unknown__"


def _count_by(values: Iterable[str]) -> dict[str, int]:
    """Deterministic, key-sorted frequency count -- matches tools.
    case_reflection_input._count_by's own reasoning (iteration order must
    never depend on insertion order or dict hash randomization)."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# GET /overview
# ---------------------------------------------------------------------------


def get_overview(store: FeedbackStoreV2) -> dict[str, Any]:
    total_cases = len(store.list_case_ids_created_in_window())
    total_turns = store.count_turns()

    feedback_rows = store.list_all_feedback()
    total_feedback = len(feedback_rows)
    helpful_count = sum(1 for row in feedback_rows if row["helpful"])
    negative_count = total_feedback - helpful_count
    negative_reason_distribution = _count_by(
        row["reason_code"] for row in feedback_rows if not row["helpful"] and row["reason_code"] is not None
    )

    analysis_rows = store.list_latest_case_analysis(case_ids=None)
    case_diagnosis_distribution = _count_by(row["diagnosis"] for row in analysis_rows)
    product_model_distribution = _count_by(
        row["product_model"] if row["product_model"] is not None else _UNKNOWN_PRODUCT_MODEL_BUCKET
        for row in analysis_rows
    )

    proposals = store.list_improvement_proposals()
    curator_changes = store.list_curator_changes()

    return {
        "total_cases": total_cases,
        "total_turns": total_turns,
        "total_feedback": total_feedback,
        "helpful_count": helpful_count,
        "helpful_ratio": round(helpful_count / total_feedback, 4) if total_feedback else 0.0,
        "negative_count": negative_count,
        "negative_ratio": round(negative_count / total_feedback, 4) if total_feedback else 0.0,
        "negative_reason_distribution": negative_reason_distribution,
        "case_diagnosis_distribution": case_diagnosis_distribution,
        "product_model_distribution": product_model_distribution,
        "improvement_proposals": {
            "total": len(proposals),
            "review_status_distribution": _count_by(row["review_status"] for row in proposals),
        },
        "curator_changes": {
            "total": len(curator_changes),
            "status_distribution": _count_by(row["status"] for row in curator_changes),
        },
    }


# ---------------------------------------------------------------------------
# GET /cases
# ---------------------------------------------------------------------------


def list_cases(store: FeedbackStoreV2) -> list[dict[str, Any]]:
    """One row per Case that has at least one case_analysis row (the Case
    universe list_latest_case_analysis(case_ids=None) already defines) --
    a Case with no analysis yet has nothing meaningful to show on this
    page (no title/issue_type/diagnosis), so it is not listed here.

    Deliberately accepts one small, explicit N+1 per Case (list_turns_
    for_case + list_feedback_for_case) rather than adding new aggregate
    store methods for turn_count/feedback_summary -- POC data scale makes
    this the simpler option (see this module's own docstring). turn_count
    and latest_activity both reuse the SAME list_turns_for_case() call
    (no separate query for each).

    latest_feedback_submitted_at is the MAX(feedback.submitted_at) for this
    Case, or None when it has no feedback at all -- read off the same
    feedback_rows already fetched for feedback_summary, no new query. Added
    for the frontend's Cases & Feedback time-range filter: feedback
    submission time is a more meaningful "when did this happen" signal than
    latest_activity (last turn's created_at) when a Case has feedback, since
    it is the actual user action being filtered for; latest_activity remains
    the frontend's fallback for Cases with no feedback yet.
    """
    analysis_rows = store.list_latest_case_analysis(case_ids=None)
    cases: list[dict[str, Any]] = []
    for row in analysis_rows:
        case_id = row["case_id"]
        turns = store.list_turns_for_case(case_id)
        feedback_rows = store.list_feedback_for_case(case_id)
        helpful_count = sum(1 for f in feedback_rows if f["helpful"])
        latest_activity = max((t["created_at"] for t in turns), default=row["analyzed_at"])
        latest_feedback_submitted_at = max(
            (f["submitted_at"] for f in feedback_rows), default=None,
        )

        cases.append({
            "case_id": case_id,
            "case_title": row["case_title"],
            "product_model": row["product_model"],
            "issue_type": row["issue_type"],
            "diagnosis": row["diagnosis"],
            "confidence": row["diagnosis_confidence"],
            "turn_count": len(turns),
            "feedback_summary": {
                "total": len(feedback_rows),
                "helpful_count": helpful_count,
                "negative_count": len(feedback_rows) - helpful_count,
            },
            "latest_activity": latest_activity,
            "latest_feedback_submitted_at": latest_feedback_submitted_at,
        })

    cases.sort(key=lambda c: c["latest_activity"] or "", reverse=True)
    return cases


# ---------------------------------------------------------------------------
# GET /cases/{case_id}
# ---------------------------------------------------------------------------


def get_case_detail(store: FeedbackStoreV2, case_id: str) -> dict[str, Any] | None:
    """None means "no such Case" -- the caller (dashboard_api.app) turns
    that into a 404. A Case with no case_analysis row yet still returns a
    detail payload (with analysis=None), unlike list_cases() which omits
    unanalyzed Cases entirely -- a direct-by-id lookup should never hide
    a Case that genuinely exists just because enrichment hasn't run yet.
    """
    case_row = store.get_case(case_id)
    if case_row is None:
        return None

    analysis_row = store.get_latest_case_analysis(case_id)
    turns = store.list_turns_for_case(case_id)
    feedback_rows = store.list_feedback_for_case(case_id)

    turns_out = []
    for turn in turns:
        retrieval_rows = store.list_retrieval_runs_for_turn(turn["turn_id"])
        turns_out.append({
            "turn_id": turn["turn_id"],
            "question": turn["question_text"],
            "answer": turn["answer_text"],
            "created_at": turn["created_at"],
            "retrieval_observation_status": turn["retrieval_observation_status"],
            "retrieval_summary": [
                {
                    "execution_status": r["execution_status"],
                    "foundry_iq_ok": bool(r["foundry_iq_ok"]) if r["foundry_iq_ok"] is not None else None,
                    "observation_status": r["observation_status"],
                    "result_count": r["result_count"],
                    "reference_count": r["reference_count"],
                    "error_code": r["error_code"],
                }
                for r in retrieval_rows
            ],
        })

    latest_activity = max((t["created_at"] for t in turns), default=case_row["created_at"])

    return {
        "case_id": case_row["case_id"],
        "session_id": case_row["session_id"],
        "latest_activity": latest_activity,
        "analysis": None if analysis_row is None else {
            "case_title": analysis_row["case_title"],
            "issue_summary": analysis_row["issue_summary"],
            "issue_type": analysis_row["issue_type"],
            "issue_type_confidence": analysis_row["issue_type_confidence"],
            "diagnosis": analysis_row["diagnosis"],
            "diagnosis_confidence": analysis_row["diagnosis_confidence"],
            "product_model": analysis_row["product_model"],
            "product_confidence": analysis_row["product_confidence"],
        },
        "turns": turns_out,
        "feedback": [
            {
                "turn_id": f["turn_id"],
                "helpful": bool(f["helpful"]),
                "reason_code": f["reason_code"],
                "suggestion_text": f["suggestion_text"],
                "submitted_at": f["submitted_at"],
            }
            for f in feedback_rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /improvements
# ---------------------------------------------------------------------------


def list_improvements(store: FeedbackStoreV2) -> list[dict[str, Any]]:
    """List-page summary only -- deliberately excludes before_content/
    proposed_content (see get_improvement_detail() for the full payload)
    to keep this page's response small. When a Proposal has more than one
    curator_changes row (a Curator re-run against the same Proposal), the
    most RECENTLY CREATED one is shown here as the summary; the full
    history is only on the detail page.
    """
    proposals = store.list_improvement_proposals()
    if not proposals:
        return []

    proposal_ids = [row["proposal_id"] for row in proposals]
    observations = {
        row["proposal_id"]: row
        for row in store.list_latest_proposal_observations(proposal_ids=proposal_ids)
    }

    latest_change_by_proposal: dict[str, Any] = {}
    for change in store.list_curator_changes():
        existing = latest_change_by_proposal.get(change["proposal_id"])
        if existing is None or change["created_at"] > existing["created_at"]:
            latest_change_by_proposal[change["proposal_id"]] = change

    result: list[dict[str, Any]] = []
    for proposal in proposals:
        observation = observations.get(proposal["proposal_id"])
        change = latest_change_by_proposal.get(proposal["proposal_id"])
        result.append({
            "proposal_id": proposal["proposal_id"],
            "title": proposal["title"],
            "improvement_target": proposal["improvement_target"],
            "review_status": proposal["review_status"],
            "created_at": proposal["created_at"],
            "latest_observation": None if observation is None else {
                "trend": observation["trend"],
                "pattern_summary": observation["pattern_summary"],
                "confidence": observation["confidence"],
                "supporting_case_count": observation["supporting_case_count"],
            },
            "curator_change": None if change is None else {
                "change_id": change["change_id"],
                "change_type": change["change_type"],
                "status": change["status"],
                "confidence": change["confidence"],
            },
        })
    return result


# ---------------------------------------------------------------------------
# GET /improvements/{proposal_id}
# ---------------------------------------------------------------------------


def get_improvement_detail(store: FeedbackStoreV2, proposal_id: str) -> dict[str, Any] | None:
    """Full lineage: Proposal -> latest Observation -> supporting Cases ->
    EVERY curator_changes row for this proposal_id (plural -- a re-run
    Curator against the same Proposal produces another row, never an
    UPDATE; the detail page shows the whole history, unlike the list
    page's single most-recent summary).

    Supporting Cases reuses tools.case_reflection_input.build_case_
    intelligence_for_ids() directly -- the exact same Case-evidence
    resolution tools.curator_analyzer already uses -- rather than
    re-deriving case_analysis lookup logic here.
    """
    proposal_row = store.get_improvement_proposal(proposal_id)
    if proposal_row is None:
        return None

    observation_row = store.get_latest_proposal_observation(proposal_id)

    supporting_cases: list[dict[str, Any]] = []
    if observation_row is not None:
        case_ids = json.loads(observation_row["supporting_case_ids_json"])
        records, _unavailable_case_ids = build_case_intelligence_for_ids(store, case_ids)
        supporting_cases = [
            {
                "case_id": record.case_id,
                "title": record.case_title,
                "product_model": record.product_model,
                "issue_type": record.issue_type,
                "diagnosis": record.diagnosis,
                "confidence": record.diagnosis_confidence,
            }
            for record in records
        ]

    curator_changes = [
        {
            "change_id": row["change_id"],
            "change_type": row["change_type"],
            "rationale": row["rationale"],
            "expected_effect": row["expected_effect"],
            "confidence": row["confidence"],
            "status": row["status"],
            "created_at": row["created_at"],
            "reviewed_at": row["reviewed_at"],
            "applied_at": row["applied_at"],
            "before_content": row["before_content"],
            "proposed_content": row["proposed_content"],
        }
        for row in store.list_curator_changes(proposal_id=proposal_id)
    ]

    return {
        "proposal": {
            "proposal_id": proposal_row["proposal_id"],
            "title": proposal_row["title"],
            "improvement_target": proposal_row["improvement_target"],
            "review_status": proposal_row["review_status"],
            "created_at": proposal_row["created_at"],
        },
        "observation": None if observation_row is None else {
            "trend": observation_row["trend"],
            "pattern_summary": observation_row["pattern_summary"],
            "possible_cause": observation_row["possible_cause"],
            "recommended_improvement": observation_row["recommended_improvement"],
            "expected_benefit": observation_row["expected_benefit"],
            "limitations": observation_row["limitations"],
            "confidence": observation_row["confidence"],
            "supporting_case_ids": json.loads(observation_row["supporting_case_ids_json"]),
            "supporting_case_count": observation_row["supporting_case_count"],
            "observed_at": observation_row["observed_at"],
        },
        "supporting_cases": supporting_cases,
        "curator_changes": curator_changes,
    }


# ---------------------------------------------------------------------------
# GET /curator-changes/{change_id}
# ---------------------------------------------------------------------------


def get_curator_change_detail(store: FeedbackStoreV2, change_id: str) -> dict[str, Any] | None:
    row = store.get_curator_change(change_id)
    if row is None:
        return None
    return {
        "change_id": row["change_id"],
        "proposal_id": row["proposal_id"],
        "target_file": row["target_file"],
        "change_type": row["change_type"],
        "rationale": row["rationale"],
        "before_content": row["before_content"],
        "proposed_content": row["proposed_content"],
        "expected_effect": row["expected_effect"],
        "confidence": row["confidence"],
        "status": row["status"],
        "created_at": row["created_at"],
        "reviewed_at": row["reviewed_at"],
        "applied_at": row["applied_at"],
    }


__all__ = [
    "get_overview",
    "list_cases",
    "get_case_detail",
    "list_improvements",
    "get_improvement_detail",
    "get_curator_change_detail",
]
