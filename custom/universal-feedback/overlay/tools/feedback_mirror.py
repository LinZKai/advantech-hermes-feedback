"""Phase 3B: mirror an already-accepted legacy Telegram Feedback outcome
(feedback_runs, tools.feedback_storage) into the Phase 2 v2 ``feedback``
table (tools.feedback_store_v2), keyed by the same ``turn_id`` Phase 3A
already persisted into v2 ``turns``.

This is deliberately a mirror, not a replacement: every function here is
called AFTER the legacy write has already succeeded and the user-visible
Telegram response has already been sent, and every function is fail-closed
-- it catches every exception internally and returns a plain bool, never
raising, because a v2-mirror failure must never surface to the user or
undo/block the legacy outcome that already happened. Callers (adapter.py)
still wrap each call in their own try/except as a second layer, matching
the defense-in-depth convention tools.retrieval_runtime already uses.

Turn linkage: the caller is responsible for supplying the exact same
``turn_id`` string Phase 3A used to create the v2 ``turns`` row for the
Turn the user actually pressed Feedback on -- in practice, the
``feedback_runs.turn_key`` value already stored on the ``feedback_runs``
row the caller just looked up by ``run_id`` (or by
``(chat_id, feedback_message_id)`` for the suggestion-reply flow), which
tools.universal_feedback.turn_key() computed identically to Phase 3A's own
turn_id at the moment the Feedback prompt was first sent
(send_universal_feedback, adapter.py). This module never guesses a turn_id
from question/answer text, timestamps, or "the latest Turn" -- it only
ever accepts one already resolved by the caller from that specific
feedback_runs row, and does nothing at all if that value is missing.

No SQL lives here: every write goes through
tools.feedback_store_v2.FeedbackStoreV2.submit_feedback()/add_suggestion(),
which already own first-write-wins, one-row-per-turn (UNIQUE turn_id), and
the FK relationship to turns -- this module is pure glue around that
existing storage contract.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from tools.feedback_store_v2 import FeedbackStoreV2
from tools.retrieval_runtime import get_store
from tools.universal_feedback import POLICY_VERSION

logger = logging.getLogger(__name__)


def mirror_positive_feedback(turn_id: Any, *, store: FeedbackStoreV2 | None = None) -> bool:
    """Mirror an already-accepted legacy positive ("h") feedback outcome.

    Writes helpful=1, reason_code=NULL, suggestion_text=NULL. Returns True
    only if the v2 row was actually written (including False for a
    duplicate -- turn_id is UNIQUE, so a second positive/negative mirror
    attempt for the same turn is rejected by the storage layer itself, not
    by any logic here). Never raises.
    """
    if not turn_id:
        return False
    try:
        active_store = store if store is not None else get_store()
        if active_store is None:
            return False
        return active_store.submit_feedback(
            uuid.uuid4().hex,
            str(turn_id),
            True,
            feedback_policy_version=POLICY_VERSION,
        )
    except Exception as exc:
        logger.debug("Phase 3B positive feedback mirror failed: %s", type(exc).__name__)
        return False


def mirror_negative_feedback(turn_id: Any, reason_code: Any, *, store: FeedbackStoreV2 | None = None) -> bool:
    """Mirror an already-accepted legacy negative feedback + reason outcome.

    Writes helpful=0, reason_code=<the same code the legacy submission
    just validated and accepted>, suggestion_text=NULL. reason_code is
    re-validated by FeedbackStoreV2.submit_feedback() itself against the
    same tools.feedback_callbacks.REASON_CODES taxonomy the legacy
    fb:r:<run_id>:<reason_code> callback already parsed it against --
    never a new/expanded taxonomy. Returns True only if the v2 row was
    actually written. Never raises.
    """
    if not turn_id:
        return False
    try:
        active_store = store if store is not None else get_store()
        if active_store is None:
            return False
        return active_store.submit_feedback(
            uuid.uuid4().hex,
            str(turn_id),
            False,
            reason_code=reason_code,
            feedback_policy_version=POLICY_VERSION,
        )
    except Exception as exc:
        logger.debug("Phase 3B negative feedback mirror failed: %s", type(exc).__name__)
        return False


def mirror_suggestion(turn_id: Any, suggestion_text: Any, *, store: FeedbackStoreV2 | None = None) -> bool:
    """Mirror an already-accepted legacy optional-suggestion outcome.

    A guarded UPDATE (FeedbackStoreV2.add_suggestion), never a new row:
    only succeeds against an existing negative (helpful=0) v2 feedback row
    for this turn_id that does not already have a suggestion --
    first-write-wins, matching the legacy two-phase reason-then-suggestion
    flow exactly. A positive turn, a turn with no v2 feedback row yet
    (e.g. because the earlier mirror_negative_feedback call itself failed
    or was never reached), or a turn that already has a suggestion all
    correctly return False without creating anything. Never raises.
    """
    if not turn_id:
        return False
    try:
        active_store = store if store is not None else get_store()
        if active_store is None:
            return False
        return active_store.add_suggestion(str(turn_id), suggestion_text)
    except Exception as exc:
        logger.debug("Phase 3B suggestion feedback mirror failed: %s", type(exc).__name__)
        return False


__all__ = [
    "mirror_positive_feedback",
    "mirror_negative_feedback",
    "mirror_suggestion",
]
