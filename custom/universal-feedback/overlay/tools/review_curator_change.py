"""Minimal human-review helper CLI: move one curator_changes row from
status='proposed' to 'approved' or 'rejected' -- the Curator Slice 2
counterpart to tools.review_proposal, same shape and same reasoning.

No new approval workflow: no multi-step sign-off, no approval_events audit
table, no notification. A thin wrapper around FeedbackStoreV2.
review_curator_change(), whose own WHERE clause already enforces the
proposed-only transition atomically -- this module additionally reads the
row first so a human gets a clear "not_found" vs. "already reviewed"
message, matching tools.review_proposal's own read-then-decide shape.

Only 'proposed' -> 'approved'/'rejected' is legal here. Every other
transition (rejected->approved, approved->rejected, applied->approved,
failed->approved, ...) fails closed -- 'approved'/'rejected'/'applied'/
'failed' are none of them 'proposed', so the guard below refuses all of
them uniformly, without needing to special-case each one.

Usage:
    python3 tools/review_curator_change.py --change-id <change_id> --status approved
    python3 tools/review_curator_change.py --change-id <change_id> --status rejected
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402

_ALLOWED_TARGET_STATUSES = ("approved", "rejected")


def review_curator_change(store: FeedbackStoreV2, *, change_id: str, status: str) -> str:
    """Move one curator_changes row from 'proposed' to `status`.

    Returns one of: "reviewed" (the transition succeeded), "not_found" (no
    such change_id), "not_proposed" (the row exists but its status is
    already something other than 'proposed' -- refused rather than
    silently overwritten, matching tools.review_proposal.review_
    proposal()'s own "not_pending" precedent exactly), "update_failed"
    (the row was 'proposed' at the read above but the UPDATE itself
    reported 0 rows changed -- a narrow race window; still fails closed).

    Never raises -- always returns a plain string a caller can branch on.
    """
    change = store.get_curator_change(change_id)
    if change is None:
        return "not_found"

    if change["status"] != "proposed":
        return "not_proposed"

    reviewed_at = datetime.now(timezone.utc).isoformat()
    ok = store.review_curator_change(change_id, status, reviewed_at=reviewed_at)
    return "reviewed" if ok else "update_failed"


_RESULT_LABELS = {
    "reviewed": "已更新",
    "not_found": "找不到指定的 Curator Change",
    "not_proposed": "此 Curator Change 已被審查過（非 proposed），拒絕覆蓋",
    "update_failed": "更新失敗",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move one curator_changes row from status='proposed' to 'approved'/'rejected'.",
    )
    parser.add_argument("--change-id", type=str, required=True, help="The curator_changes.change_id to review")
    parser.add_argument("--status", type=str, required=True, choices=_ALLOWED_TARGET_STATUSES, help="The new status")
    parser.add_argument("--db", type=Path, default=None, help=f"SQLite DB path (default: {DEFAULT_PATH})")
    args = parser.parse_args(argv)

    store = FeedbackStoreV2(args.db) if args.db is not None else FeedbackStoreV2()

    result = review_curator_change(store, change_id=args.change_id, status=args.status)
    print(f"result={result}（{_RESULT_LABELS.get(result, result)}）")
    print(f"change_id={args.change_id}")
    print(f"status={args.status}")
    return 0 if result == "reviewed" else 1


if __name__ == "__main__":
    sys.exit(main())
