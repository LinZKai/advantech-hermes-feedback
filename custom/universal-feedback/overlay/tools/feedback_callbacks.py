"""Structured parsing for Universal Feedback Telegram callback_data.

Isolated from tools.feedback_storage.parse_callback_data(), which keeps
serving fb:h/fb:u/fb:y/fb:n as its legacy (run_id, helpful) tuple contract
unchanged. This module adds the fb:r:<run_id>:<reason_code> negative-reason
format on a separate, structured return type without touching that contract.
"""
from __future__ import annotations

from dataclasses import dataclass

REASON_CODES: tuple[str, ...] = (
    "incorrect",
    "incomplete",
    "not_relevant",
    "unclear",
    "other",
)
_REASON_CODE_SET = frozenset(REASON_CODES)

# Telegram inline keyboard callback_data hard limit, in UTF-8 bytes.
MAX_CALLBACK_DATA_BYTES = 64

_NO_REASON_ACTIONS = ("h", "u")


def is_valid_reason_code(reason_code: object) -> bool:
    return isinstance(reason_code, str) and reason_code in _REASON_CODE_SET


@dataclass(frozen=True)
class FeedbackCallback:
    """One parsed fb: callback_data payload (action "h", "u", or "r")."""

    action: str
    run_id: str
    reason_code: str | None = None


def parse_feedback_callback(data: object) -> FeedbackCallback | None:
    """Parse fb:h:<run_id> | fb:u:<run_id> | fb:r:<run_id>:<reason_code>.

    Fails closed: any malformed, oversized, or unrecognized input returns
    None instead of raising, so callers can treat None uniformly as "reject
    this callback".
    """
    if not isinstance(data, str):
        return None
    try:
        encoded = data.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(encoded) > MAX_CALLBACK_DATA_BYTES:
        return None

    parts = data.split(":")
    if len(parts) < 3 or parts[0] != "fb":
        return None
    action = parts[1]

    if action in _NO_REASON_ACTIONS:
        if len(parts) != 3:
            return None
        run_id = parts[2]
        if not run_id.strip():
            return None
        return FeedbackCallback(action=action, run_id=run_id)

    if action == "r":
        if len(parts) != 4:
            return None
        run_id, reason_code = parts[2], parts[3]
        if not run_id.strip() or not is_valid_reason_code(reason_code):
            return None
        return FeedbackCallback(action=action, run_id=run_id, reason_code=reason_code)

    return None
