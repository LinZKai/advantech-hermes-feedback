"""Minimal human-review helper CLI: move one improvement_proposals row
from review_status='pending' to 'accepted' or 'rejected'.

No such CLI/API existed before this -- tools.feedback_store_v2.
FeedbackStoreV2.update_proposal_review_status() has always been the
storage primitive, but every prior call site was a test, not a human-
facing tool. This module is a thin wrapper around that existing method,
not a new approval workflow: no multi-step sign-off, no audit trail table,
no notification, nothing beyond a single validated state transition.

Curator NEVER calls this module and NEVER accepts a Proposal itself --
review_status is a HUMAN lifecycle field (see tools.feedback_store_v2's
own module docstring); tools.run_curator's own deterministic guards
require review_status=='accepted' to already be true before it will do
anything, and there is no code path anywhere in this family of modules
that could set it.

Usage:
    python3 tools/review_proposal.py --proposal-id <proposal_id> --status accepted
    python3 tools/review_proposal.py --proposal-id <proposal_id> --status rejected
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402

# The only two transitions this CLI performs -- 'pending' is never a valid
# --status value here (a Proposal starts 'pending' by construction; this
# tool only ever moves it away from that state, once).
_ALLOWED_TARGET_STATUSES = ("accepted", "rejected")


def review_proposal(store: FeedbackStoreV2, *, proposal_id: str, review_status: str) -> str:
    """Move one improvement_proposals row from 'pending' to `review_status`.

    Returns one of: "reviewed" (the transition succeeded), "not_found" (no
    such proposal_id), "not_pending" (the row exists but its review_status
    is already something other than 'pending' -- refused rather than
    silently overwritten, so an accidental second run can never flip an
    already-reviewed Proposal to a different outcome), "update_failed"
    (the row was 'pending' at the read above but the UPDATE itself
    reported 0 rows changed -- a narrow race window; still fails closed,
    never assumed to have succeeded).

    Never raises for any of the above -- always returns a plain string a
    caller can branch on, matching this whole domain's "no runner-specific
    exception hierarchy for expected terminal states" convention (see
    tools.run_reflector.ReflectorRunOutcome's own docstring for the same
    reasoning stated at length).
    """
    proposal = store.get_improvement_proposal(proposal_id)
    if proposal is None:
        return "not_found"

    if proposal["review_status"] != "pending":
        return "not_pending"

    ok = store.update_proposal_review_status(proposal_id, review_status)
    return "reviewed" if ok else "update_failed"


_RESULT_LABELS = {
    "reviewed": "已更新",
    "not_found": "找不到指定的 Proposal",
    "not_pending": "此 Proposal 已被審查過（非 pending），拒絕覆蓋",
    "update_failed": "更新失敗",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move one improvement_proposals row from review_status='pending' to 'accepted'/'rejected'.",
    )
    parser.add_argument("--proposal-id", type=str, required=True, help="The improvement_proposals.proposal_id to review")
    parser.add_argument("--status", type=str, required=True, choices=_ALLOWED_TARGET_STATUSES, help="The new review_status")
    parser.add_argument("--db", type=Path, default=None, help=f"SQLite DB path (default: {DEFAULT_PATH})")
    args = parser.parse_args(argv)

    store = FeedbackStoreV2(args.db) if args.db is not None else FeedbackStoreV2()

    result = review_proposal(store, proposal_id=args.proposal_id, review_status=args.status)
    print(f"result={result}（{_RESULT_LABELS.get(result, result)}）")
    print(f"proposal_id={args.proposal_id}")
    print(f"status={args.status}")
    return 0 if result == "reviewed" else 1


if __name__ == "__main__":
    sys.exit(main())
