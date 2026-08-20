"""Curator Slice 2: deterministic apply -- one approved curator_changes
row -> overwrite /sandbox/AGENTS.md with its proposed_content.

    FeedbackStoreV2.get_curator_change(change_id)
     |
     v
    deterministic guards               (change exists, status=='approved',
     |                                   target_file==CURATOR_TARGET_FILE,
     |                                   proposed_content non-empty,
     |                                   target_file readable, CURRENT
     |                                   content == change['before_content']
     |                                   -- see this module's own
     |                                   docstring, "Deterministic guards")
     v
    atomic write                        (temp file in the same directory,
     |                                    then os.replace() -- never a
     |                                    partial-write)
     v
    FeedbackStoreV2.mark_curator_change_applied()   [write succeeded]
      or
    FeedbackStoreV2.mark_curator_change_failed()    [write raised]
     v
    ApplyOutcome (human-readable summary on stdout)
     |
     v
    exit

Usage:
    python3 tools/apply_curator_change.py --change-id <change_id> [--agents-file PATH]

No patch/diff engine: target_file is small by design for this POC, so
proposed_content is always the file's COMPLETE replacement content -- this
module never merges, patches, or partially edits.

Runtime AGENTS.md is a disposable working copy (see tools.curator_domain's
own "Self-improvement boundary" docstring and the sibling advantech-
hermes-support-config repo's bootstrap_agents.sh) -- this module never
writes back to the repo AGENTS.md, never touches Git, and never writes to
any file other than exactly CURATOR_TARGET_FILE.

Deterministic guards -- checked BEFORE any write, so an ineligible change
produces no filesystem write and no DB status change of any kind:
  * the change_id exists (get_curator_change)
  * status == 'approved' (never 'proposed'/'rejected'/'applied'/'failed'
    -- Curator itself never approves a change; see tools.
    review_curator_change for the human-facing approve/reject helper)
  * target_file exactly equals tools.curator_domain.CURATOR_TARGET_FILE
    (re-checked here even though CuratorChange.__post_init__ already
    enforced it at creation time -- belt-and-suspenders, matching this
    whole domain's "never trust an upstream layer alone" convention)
  * proposed_content is a non-blank string (same re-check reasoning)
  * `agents_file` exists and is readable
  * the file's CURRENT content equals change['before_content'] EXACTLY --
    this is the guard that prevents a stale proposed_content (generated
    against an OLDER AGENTS.md) from silently clobbering a NEWER one that
    someone else changed in between propose and apply. A mismatch reports
    status="source_changed": no write, status left at 'approved' (not
    'failed' -- the change itself is not broken, its precondition is just
    stale; once AGENTS.md is reconciled, or a fresh Curator run supersedes
    it, this same row could still legitimately apply).

Write/DB-status ordering -- the one invariant this whole module exists to
guarantee: a curator_changes row NEVER says status='applied' unless
target_file NOW actually equals proposed_content. Concretely: the file
write happens FIRST; mark_curator_change_applied() is only ever called
AFTER that write has already succeeded. If the write itself raises,
mark_curator_change_failed() runs instead -- applied_at stays NULL, status
becomes 'failed', and the file is left however the atomic write left it
(either fully untouched, since os.replace() is atomic, or -- in the
vanishingly rare case the OS itself fails mid-replace -- in the state the
OS guarantees for its own atomic rename primitive; this module adds no
weaker guarantee of its own on top of that).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.curator_domain import CURATOR_TARGET_FILE  # noqa: E402
from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically: a temp file in the SAME
    directory (so os.replace() is guaranteed to be an atomic rename on
    the same filesystem, never a cross-filesystem copy), then
    os.replace() onto the real path. On any failure, the temp file is
    removed and the original exception propagates unchanged -- `path`
    itself is left exactly as it was before this call.
    """
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


@dataclass(frozen=True)
class ApplyOutcome:
    """One apply_curator_change() call's outcome.

    `status` is exactly one of:
      * "not_found"                    -- no curator_changes row for the
        given change_id; no write, no DB write.
      * "not_approved"                  -- status is not 'approved'; no
        write, no DB write.
      * "wrong_target"                  -- target_file is not exactly
        CURATOR_TARGET_FILE; no write, no DB write.
      * "empty_content"                 -- proposed_content is missing or
        blank; no write, no DB write.
      * "agents_file_unreadable"         -- `agents_file` does not exist
        or could not be read; no write, no DB write.
      * "source_changed"                 -- the file's current content
        does not equal before_content; no write, status left untouched at
        'approved' (see this module's own docstring for why).
      * "write_failed"                    -- the atomic write itself
        raised; status becomes 'failed', applied_at stays NULL.
      * "file_written_db_update_failed"   -- the write succeeded, but
        mark_curator_change_applied() reported 0 rows changed (the row's
        status was no longer 'approved' by the time this ran -- an
        extremely narrow concurrent-apply race in this single-operator
        POC). The file WAS overwritten; the DB does not yet say so.
        Reported as its own distinct, clearly-named status rather than
        silently claimed as "applied" -- a human must reconcile.
      * "applied"                        -- the write succeeded and the
        row is now status='applied' with a real applied_at.

    Deliberately a plain dataclass with a `status` string, not a runner-
    specific exception hierarchy -- matches every other runner in this
    domain (tools.run_reflector.ReflectorRunOutcome, tools.run_curator.
    CuratorRunOutcome) making the exact same choice for the exact same
    reason.
    """

    status: str
    change_id: str | None = None
    applied_at: str | None = None
    error: str | None = None


def apply_curator_change(
    store: FeedbackStoreV2, *, change_id: str, agents_file: Path,
) -> ApplyOutcome:
    """The reusable apply body, separated from CLI/argv handling so
    main() (or a test) can call it directly against a temp AGENTS.md.
    See this module's own docstring for the full guard sequence and the
    write/DB-status ordering invariant.
    """
    change = store.get_curator_change(change_id)
    if change is None:
        return ApplyOutcome(status="not_found", change_id=change_id)

    if change["status"] != "approved":
        return ApplyOutcome(
            status="not_approved", change_id=change_id,
            error=f"status={change['status']!r}, expected 'approved'",
        )

    if change["target_file"] != CURATOR_TARGET_FILE:
        return ApplyOutcome(
            status="wrong_target", change_id=change_id,
            error=f"target_file={change['target_file']!r}, expected {CURATOR_TARGET_FILE!r}",
        )

    proposed_content = change["proposed_content"]
    if not isinstance(proposed_content, str) or not proposed_content.strip():
        return ApplyOutcome(status="empty_content", change_id=change_id)

    try:
        current_content = agents_file.read_text(encoding="utf-8")
    except OSError as exc:
        return ApplyOutcome(
            status="agents_file_unreadable", change_id=change_id,
            error=f"{type(exc).__name__}: {exc}",
        )

    if current_content != change["before_content"]:
        return ApplyOutcome(status="source_changed", change_id=change_id)

    applied_at = datetime.now(timezone.utc).isoformat()

    try:
        _atomic_write(agents_file, proposed_content)
    except OSError as exc:
        store.mark_curator_change_failed(change_id)
        return ApplyOutcome(
            status="write_failed", change_id=change_id,
            error=f"{type(exc).__name__}: {exc}",
        )

    ok = store.mark_curator_change_applied(change_id, applied_at=applied_at)
    if not ok:
        return ApplyOutcome(
            status="file_written_db_update_failed", change_id=change_id,
            error=(
                f"{agents_file} was overwritten but curator_changes.status could not be "
                "set to 'applied' (row status likely changed concurrently) -- reconcile manually"
            ),
        )

    return ApplyOutcome(status="applied", change_id=change_id, applied_at=applied_at)


# ---------------------------------------------------------------------------
# Human-readable summary -- Traditional Chinese narrative, English
# machine-facing fields.
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    "not_found": "找不到指定的 Curator Change",
    "not_approved": "此 Curator Change 尚未被 approve",
    "wrong_target": "target_file 不是 /sandbox/AGENTS.md",
    "empty_content": "proposed_content 為空",
    "agents_file_unreadable": "無法讀取 AGENTS.md",
    "source_changed": "AGENTS.md 目前內容與 before_content 不一致，拒絕覆寫",
    "write_failed": "寫入 AGENTS.md 失敗",
    "file_written_db_update_failed": "AGENTS.md 已覆寫，但資料庫狀態更新失敗，請人工核對",
    "applied": "成功套用",
}

_SUCCESSFUL_STATUSES = frozenset({"applied"})


def _format_summary(outcome: ApplyOutcome) -> str:
    lines: list[str] = [
        "Curator Change Apply 執行完成",
        f"status={outcome.status}",
        f"（{_STATUS_LABELS.get(outcome.status, outcome.status)}）",
        f"change_id={outcome.change_id}",
    ]
    if outcome.applied_at:
        lines.append(f"applied_at={outcome.applied_at}")
    if outcome.error:
        lines.append(f"error={outcome.error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Curator Slice 2: apply one approved curator_changes row to /sandbox/AGENTS.md.",
    )
    parser.add_argument("--change-id", type=str, required=True, help="The approved curator_changes.change_id to apply")
    parser.add_argument("--db", type=Path, default=None, help=f"SQLite DB path (default: {DEFAULT_PATH})")
    parser.add_argument(
        "--agents-file", type=Path, default=Path(CURATOR_TARGET_FILE),
        help=f"Runtime AGENTS.md path (default: {CURATOR_TARGET_FILE})",
    )
    args = parser.parse_args(argv)

    store = FeedbackStoreV2(args.db) if args.db is not None else FeedbackStoreV2()

    outcome = apply_curator_change(store, change_id=args.change_id, agents_file=args.agents_file)
    print(_format_summary(outcome))
    return 0 if outcome.status in _SUCCESSFUL_STATUSES else 1


if __name__ == "__main__":
    sys.exit(main())
