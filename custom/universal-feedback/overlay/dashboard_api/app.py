"""Dashboard MVP API -- a minimal, standalone FastAPI app over the
existing FeedbackStoreV2 / Reflector / Curator domain, for a future
Dashboard frontend (not built in this slice).

    FeedbackStoreV2 (unchanged)
     |
     v
    dashboard_api.views    (pure read-projection functions, no FastAPI import)
     |                       tools.review_proposal.review_proposal()
     |                       tools.run_curator.run_curator()              (auto-invoked
     |                       tools.review_curator_change.review_curator_change()  right after an
     |                       tools.apply_curator_change.apply_curator_change()    Accept/Approve)
     v
    dashboard_api.app       (THIS module -- thin HTTP adapter only)
     |
     v
    JSON responses

No new business logic anywhere in this module: every GET endpoint calls a
pure function in dashboard_api.views; every POST endpoint calls the exact
same already-tested Slice 1/2 business function tools/review_proposal.py,
tools/run_curator.py, tools/review_curator_change.py, and tools/
apply_curator_change.py's own CLIs already call. This module's only job is
mapping a domain outcome string/dataclass to an HTTP status code and a
JSON body.

Human-in-the-loop gates are unchanged -- there are still exactly two:
Proposal Accept/Reject and CuratorChange Approve/Reject. What changed is
what happens automatically ONE step after each gate, still synchronously
within the same HTTP request/response:
  * POST /improvements/{id}/review with status="accepted" -> after the
    Proposal is committed accepted, this module immediately calls tools.
    run_curator.run_curator() itself (no subprocess, no queue). A Reject
    NEVER reaches this call. A Curator failure (any CuratorRunOutcome
    status other than "succeeded"/"no_change_recommended") does NOT roll
    back the already-committed Proposal Accept -- it is reported back in
    the same 200 response's nested "curator" object.
  * POST /curator-changes/{id}/review with status="approved" -> after the
    CuratorChange is committed approved, this module immediately calls
    tools.apply_curator_change.apply_curator_change() itself. A Reject
    NEVER reaches this call. Every one of apply_curator_change()'s
    deterministic guards (status=='approved', target_file, non-empty
    content, AGENTS.md readable, before_content match, atomic write) is
    still fully in force -- an apply guard failure (e.g. "source_changed")
    leaves curator_changes.status at 'approved' (never falsely 'applied')
    and is reported as a non-2xx HTTP response, matching this endpoint's
    own dedicated /apply counterpart below exactly.

Naming note: "dashboard_api" is unrelated to Hermes' own built-in browser
dashboard (`hermes dashboard`) -- see dashboard_api.views's own docstring
for the full naming-collision statement.

Not wired into gateway/run.py (the Telegram/messaging gateway daemon) --
an independent, minimal entrypoint, started separately:

    uvicorn dashboard_api.app:app --host 0.0.0.0 --port 8800

No SQLAlchemy, no ORM: FeedbackStoreV2 (raw SQL) is used exactly as every
other consumer in this repo uses it. No router/service/schema package
split beyond this one file plus dashboard_api.views -- nine endpoints does
not justify one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from dashboard_api.views import (  # noqa: E402
    get_case_detail,
    get_curator_change_detail,
    get_improvement_detail,
    get_overview,
    list_cases,
    list_improvements,
)
from tools.apply_curator_change import ApplyOutcome, apply_curator_change  # noqa: E402
from tools.curator_domain import CURATOR_TARGET_FILE  # noqa: E402
from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402
from tools.review_curator_change import review_curator_change  # noqa: E402
from tools.review_proposal import review_proposal  # noqa: E402
from tools.run_curator import CuratorRunOutcome, resolve_default_analyzer, run_curator  # noqa: E402

app = FastAPI(title="Hermes Feedback Dashboard API", version="0.1.0")


def get_store() -> FeedbackStoreV2:
    """A fresh FeedbackStoreV2 per request, pointed at DEFAULT_PATH.
    FeedbackStoreV2 opens its own short-lived sqlite3 connection per
    method call (see its own _connect() docstring) -- there is no long-
    lived connection/pool for this dependency to own, so a new instance
    per request is cheap. Tests override this via
    app.dependency_overrides[get_store] to point at a temp DB.
    """
    return FeedbackStoreV2(DEFAULT_PATH)


class ProposalReviewRequest(BaseModel):
    status: Literal["accepted", "rejected"]


class CuratorReviewRequest(BaseModel):
    status: Literal["approved", "rejected"]


# ---------------------------------------------------------------------------
# GET endpoints -- thin wrappers over dashboard_api.views
# ---------------------------------------------------------------------------


@app.get("/overview")
def get_overview_endpoint(store: FeedbackStoreV2 = Depends(get_store)) -> dict[str, Any]:
    return get_overview(store)


@app.get("/cases")
def list_cases_endpoint(store: FeedbackStoreV2 = Depends(get_store)) -> list[dict[str, Any]]:
    return list_cases(store)


@app.get("/cases/{case_id}")
def get_case_detail_endpoint(case_id: str, store: FeedbackStoreV2 = Depends(get_store)) -> dict[str, Any]:
    detail = get_case_detail(store, case_id)
    if detail is None:
        raise HTTPException(status_code=404, detail={"status": "not_found", "message": f"case {case_id!r} not found"})
    return detail


@app.get("/improvements")
def list_improvements_endpoint(store: FeedbackStoreV2 = Depends(get_store)) -> list[dict[str, Any]]:
    return list_improvements(store)


@app.get("/improvements/{proposal_id}")
def get_improvement_detail_endpoint(
    proposal_id: str, store: FeedbackStoreV2 = Depends(get_store),
) -> dict[str, Any]:
    detail = get_improvement_detail(store, proposal_id)
    if detail is None:
        raise HTTPException(
            status_code=404, detail={"status": "not_found", "message": f"proposal {proposal_id!r} not found"},
        )
    return detail


@app.get("/curator-changes/{change_id}")
def get_curator_change_detail_endpoint(
    change_id: str, store: FeedbackStoreV2 = Depends(get_store),
) -> dict[str, Any]:
    detail = get_curator_change_detail(store, change_id)
    if detail is None:
        raise HTTPException(
            status_code=404, detail={"status": "not_found", "message": f"curator change {change_id!r} not found"},
        )
    return detail


# ---------------------------------------------------------------------------
# POST endpoints -- thin wrappers over tools.review_proposal /
# tools.review_curator_change / tools.apply_curator_change. Every domain
# outcome string is mapped to an HTTP status code here; no guard/state-
# transition logic is re-implemented in this module.
# ---------------------------------------------------------------------------

_PROPOSAL_REVIEW_HTTP_STATUS = {
    "reviewed": 200,
    "not_found": 404,
    "not_pending": 409,
    "update_failed": 500,
}


def _run_curator_after_accept(store: FeedbackStoreV2, proposal_id: str) -> dict[str, Any]:
    """Invoke Curator synchronously, immediately after a Proposal Accept
    has already been committed by the caller. Reused business logic only:
    resolve_default_analyzer()/run_curator() are the exact same functions
    tools/run_curator.py's own CLI calls -- nothing re-implemented here.

    Never raises. The Proposal Accept above is already durably committed
    and must never be rolled back for a Curator failure (LLM runtime
    config unresolved, analyzer error, or any other tools.run_curator.
    CuratorRunOutcome failure status) -- every outcome, success or
    failure, is folded into this same plain dict for the endpoint's 200
    response body's "curator" key.
    """
    try:
        analyzer = resolve_default_analyzer()
    except RuntimeError as exc:
        outcome = CuratorRunOutcome(status="analyzer_failed", proposal_id=proposal_id, error=str(exc))
    else:
        outcome = run_curator(
            store, proposal_id=proposal_id, agents_file=Path(CURATOR_TARGET_FILE), analyzer=analyzer,
        )
    return {
        "status": outcome.status,
        "change_id": outcome.change_id,
        "change_type": outcome.change_type,
        "confidence": outcome.confidence,
        "error": outcome.error,
    }


@app.post("/improvements/{proposal_id}/review")
def review_proposal_endpoint(
    proposal_id: str, body: ProposalReviewRequest, store: FeedbackStoreV2 = Depends(get_store),
) -> dict[str, Any]:
    result = review_proposal(store, proposal_id=proposal_id, review_status=body.status)
    http_status = _PROPOSAL_REVIEW_HTTP_STATUS.get(result, 500)
    payload: dict[str, Any] = {"status": result, "proposal_id": proposal_id}
    if http_status >= 400:
        raise HTTPException(status_code=http_status, detail=payload)

    if body.status == "accepted":
        payload["curator"] = _run_curator_after_accept(store, proposal_id)

    return payload


_CURATOR_REVIEW_HTTP_STATUS = {
    "reviewed": 200,
    "not_found": 404,
    "not_proposed": 409,
    "update_failed": 500,
}

# "not_approved"/"wrong_target"/"empty_content"/"source_changed" are all
# client-visible illegal-state-to-apply-from conditions -> 409 (retrying
# the identical request will never succeed without changing something
# first). "agents_file_unreadable"/"write_failed"/"file_written_db_
# update_failed" are environment/internal problems the client cannot fix
# by changing its request -> 500. Shared by review_curator_change_endpoint
# (auto-apply after Approve) and apply_curator_change_endpoint (the
# dedicated, still-present /apply endpoint) below -- one mapping, reused,
# never duplicated.
_APPLY_HTTP_STATUS = {
    "applied": 200,
    "not_found": 404,
    "not_approved": 409,
    "wrong_target": 409,
    "empty_content": 409,
    "source_changed": 409,
    "agents_file_unreadable": 500,
    "write_failed": 500,
    "file_written_db_update_failed": 500,
}


@app.post("/curator-changes/{change_id}/review")
def review_curator_change_endpoint(
    change_id: str, body: CuratorReviewRequest, store: FeedbackStoreV2 = Depends(get_store),
) -> dict[str, Any]:
    result = review_curator_change(store, change_id=change_id, status=body.status)
    http_status = _CURATOR_REVIEW_HTTP_STATUS.get(result, 500)
    payload: dict[str, Any] = {"status": result, "change_id": change_id}
    if http_status >= 400:
        raise HTTPException(status_code=http_status, detail=payload)

    if body.status != "approved":
        return payload

    # Approve just committed (status='proposed' -> 'approved'). Immediately
    # invoke the existing, unchanged apply pipeline -- same function, same
    # guards, same fail-closed outcomes as the dedicated /apply endpoint
    # below. A guard failure (e.g. "source_changed") leaves curator_changes.
    # status at 'approved' (apply_curator_change() never touches it in that
    # case) and is surfaced here as the identical non-2xx status/detail
    # shape the dedicated /apply endpoint already uses -- never silently
    # claimed as applied.
    outcome: ApplyOutcome = apply_curator_change(
        store, change_id=change_id, agents_file=Path(CURATOR_TARGET_FILE),
    )
    apply_http_status = _APPLY_HTTP_STATUS.get(outcome.status, 500)
    if apply_http_status >= 400:
        detail: dict[str, Any] = {"status": outcome.status, "change_id": change_id}
        if outcome.error:
            detail["message"] = outcome.error
        raise HTTPException(status_code=apply_http_status, detail=detail)

    return {"status": outcome.status, "change_id": change_id, "applied_at": outcome.applied_at}


# Still present as its own endpoint -- deliberately not removed even though
# the frontend's "Apply to Runtime" button is gone (see this module's own
# top docstring): a curator_changes row that auto-apply left at 'approved'
# after a "source_changed" guard failure can legitimately still apply later
# once AGENTS.md is reconciled (see tools.apply_curator_change's own
# docstring for that exact reasoning) -- this endpoint is that manual retry
# path, with no UI wired to it in this POC.
@app.post("/curator-changes/{change_id}/apply")
def apply_curator_change_endpoint(
    change_id: str, store: FeedbackStoreV2 = Depends(get_store),
) -> dict[str, Any]:
    outcome: ApplyOutcome = apply_curator_change(
        store, change_id=change_id, agents_file=Path(CURATOR_TARGET_FILE),
    )
    http_status = _APPLY_HTTP_STATUS.get(outcome.status, 500)
    payload: dict[str, Any] = {"status": outcome.status, "change_id": change_id}
    if outcome.applied_at:
        payload["applied_at"] = outcome.applied_at
    if outcome.error:
        payload["message"] = outcome.error
    if http_status >= 400:
        raise HTTPException(status_code=http_status, detail=payload)
    return payload


__all__ = ["app", "get_store"]
