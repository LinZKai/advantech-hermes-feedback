"""Gateway-facing glue for Phase 3A: wires tools.retrieval_observer's pure
parser output into tools.feedback_store_v2.FeedbackStoreV2.

This is the only module in the feedback overlay that knows about both the
Hermes gateway's agent-result envelope shape (``messages``,
``history_offset``, ``model``, ``final_response``, and the explicit
``phase3a_boundary_trusted`` flag documented on ``resolve_boundary_trust``
below) AND about FeedbackStoreV2 -- run.py/base.py glue code should only
ever import from here (plus tools.universal_feedback.turn_key for the
shared turn id, and tools.universal_feedback.feedback_eligible for the
policy check they already need for the legacy hook), never call
FeedbackStoreV2 or retrieval_observer directly, and never re-extract
messages/history_offset/model/answer_text from the envelope themselves --
this module owns that extraction so it happens in exactly one place.

Every public entry point here is fail-closed: it catches every exception
internally and returns False/None rather than raising, because Phase 3A
telemetry is a side channel that must never affect the answer, the legacy
feedback_runs flow, or negative-feedback reason/suggestion collection (see
AGENTS.md and docs/current-baseline.md for the layering this preserves).
The two top-level persistence entry points (observe_and_persist_turn,
persist_turn_observation_context) return a real bool -- True only when the
Turn row was actually written -- never None, so a caller can always tell
"attempted" apart from "definitely persisted" without a third ambiguous
state. Nothing here ever logs a question, an answer, raw tool output, a
command, or a secret -- only fixed, safe summaries.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping

from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput
from tools.retrieval_observer import (
    FoundryRetrievalObservation,
    TurnRetrievalObservation,
    observation_to_safe_dict,
    parse_turn_retrieval_observations,
    safe_dict_to_observation,
)
from tools.universal_feedback import safe_feedback_text

logger = logging.getLogger(__name__)

# Recorded on turns.case_assignment_method for every Phase 3A turn: there is
# no Case classifier yet (that is Phase 4+), every turn in a session is
# assigned to that session's one deterministic default case.
DEFAULT_CASE_ASSIGNMENT_METHOD = "phase3_default"

# Fixed namespace for deriving a deterministic default case_id from a
# session_id (uuid5 is a one-way hash of namespace+name -- session_id is
# not recoverable from the resulting case_id). Generated once with
# uuid.uuid4() and frozen here; must never change, or existing sessions
# would silently get a second "different" default case.
_DEFAULT_CASE_NAMESPACE = uuid.UUID("6f6d9f1a-6e0a-4e6a-9f0a-5a1a2b3c4d5e")

_store_singleton: FeedbackStoreV2 | None = None
_store_init_failed = False


def get_store() -> FeedbackStoreV2 | None:
    """Lazily construct the process-wide FeedbackStoreV2 singleton.

    Shared by every gateway/adapter-side glue module that needs
    FeedbackStoreV2 (tools.retrieval_runtime's own Phase 3A entry points,
    and tools.feedback_mirror's Phase 3B ones) so the process never opens
    more than one connection pool to the same DB file for this purpose.

    Caches an init failure (rather than retrying on every turn) so a
    broken DB path does not turn into repeated disk I/O and log noise on
    every message -- if this needs to change (e.g. hot config reload)
    that is a later phase's concern.
    """
    global _store_singleton, _store_init_failed
    if _store_singleton is not None:
        return _store_singleton
    if _store_init_failed:
        return None
    try:
        _store_singleton = FeedbackStoreV2()
    except Exception:
        _store_init_failed = True
        logger.debug("FeedbackStoreV2 init failed", exc_info=True)
        return None
    return _store_singleton


def default_case_id(session_id: str) -> str:
    """Deterministic default Case id for a session: same session_id always
    yields the same case_id, so every Turn in a session shares one Case
    without needing a classifier (Phase 3A has none). Derived via a fixed
    namespace UUID5 hash -- reasonable, bounded length, and does not embed
    session_id (or anything else sensitive) recoverably."""
    return f"case-default-{uuid.uuid5(_DEFAULT_CASE_NAMESPACE, str(session_id)).hex}"


def get_or_create_default_case(store: FeedbackStoreV2, session_id: str) -> str:
    case_id = default_case_id(session_id)
    if store.get_case(case_id) is not None:
        return case_id
    try:
        store.create_case(case_id, session_id)
    except Exception:
        # Most likely a concurrent create for the same session (case_id is
        # deterministic) landing between our get_case() and create_case()
        # calls; re-check once rather than assuming success.
        if store.get_case(case_id) is not None:
            return case_id
        raise
    return case_id


def resolve_boundary_trust(agent_response: Mapping[str, Any]) -> bool:
    """Whether ``agent_response["messages"][agent_response["history_offset"]:]``
    can be trusted as "this Turn only".

    Reads a single explicit, source-computed field --
    ``phase3a_boundary_trusted`` -- that the gateway sets directly on each
    response/agent_result dict it constructs (see run.py's
    ``_run_agent_inner``, at its two normal-completion return points and
    its inactivity-timeout synthetic dict, all three of which set this key
    explicitly from the real ``_session_was_split``/``_compacted_in_place``
    local state available at that exact point).

    This is deliberately NOT reconstructed here by comparing a pre-run and
    post-run session id, or by checking whether ``compacted_in_place`` is
    falsy: a response dict that simply never carries a ``session_id`` key
    (for example the proxy-mode delegation path, or any future code path
    not yet audited to set this field) must never be misread as "session
    split happened" just because ``None != "some-real-session-id"`` would
    evaluate True. A missing ``phase3a_boundary_trusted`` key means
    exactly "the gateway did not assert this boundary is trustworthy",
    which is unconditionally untrusted here -- the same fail-closed
    default as an explicit False.

    Structural validity of ``history_offset`` itself (present, non-negative
    int, in range) is re-checked independently by
    tools.retrieval_observer.parse_turn_retrieval_observations -- this
    function only carries the gateway's own boundary-trust verdict, which
    no amount of offset-value inspection alone could recover.
    """
    if not isinstance(agent_response, Mapping):
        return False
    return bool(agent_response.get("phase3a_boundary_trusted"))


def persist_turn_and_retrieval(
    store: FeedbackStoreV2,
    *,
    session_id: str,
    platform: str,
    platform_chat_id: str | None,
    turn_id: str,
    platform_user_id: str,
    platform_user_message_id: str,
    platform_assistant_message_id: str | None,
    question_text: str,
    answer_text: str,
    feedback_eligible: bool,
    model: str | None,
    provider: str | None,
    hermes_version: str | None,
    support_config_commit: str | None,
    feedback_code_commit: str | None,
    observation: TurnRetrievalObservation,
) -> bool:
    """Create Session/default Case/Turn, then 0..n Retrieval rows, for one
    completed Hermes Turn. Returns True only if the Turn row was actually
    written, False for any failure -- never raises, never returns None. On
    a Retrieval-insert failure, FeedbackStoreV2.add_retrieval_runs already
    downgrades the turn's own retrieval_observation_status to
    'unavailable' via its own savepoint (see feedback_store_v2.py); this
    function does not need to (and must not try to) duplicate that
    recovery, and still returns True in that case -- the Turn row itself
    was written successfully, which is what this return value promises.
    """
    try:
        store.create_or_update_session(session_id, platform, platform_chat_id=platform_chat_id)
        case_id = get_or_create_default_case(store, session_id)
        store.create_turn(
            turn_id,
            case_id,
            platform_user_id=platform_user_id,
            platform_user_message_id=platform_user_message_id,
            platform_assistant_message_id=platform_assistant_message_id,
            question_text=safe_feedback_text(question_text),
            answer_text=safe_feedback_text(answer_text),
            feedback_eligible=feedback_eligible,
            retrieval_observation_status=observation.turn_observation_status,
            retrieval_observation_reason=observation.turn_observation_reason,
            support_config_commit=support_config_commit,
            feedback_code_commit=feedback_code_commit,
            hermes_version=hermes_version,
            model=model,
            provider=provider,
            case_assignment_method=DEFAULT_CASE_ASSIGNMENT_METHOD,
        )
        if observation.turn_observation_status == "complete" and observation.retrievals:
            runs = [_to_retrieval_run_input(r) for r in observation.retrievals]
            store.add_retrieval_runs(turn_id, runs)
        return True
    except Exception as exc:
        logger.debug("Phase 3A turn/retrieval persistence failed: %s", type(exc).__name__)
        return False


def _to_retrieval_run_input(observation: FoundryRetrievalObservation) -> RetrievalRunInput:
    return RetrievalRunInput(
        execution_status=observation.execution_status,
        observation_status=observation.observation_status,
        tool_call_id=observation.tool_call_id,
        request_attempted=observation.request_attempted,
        foundry_iq_ok=observation.foundry_iq_ok,
        observation_reason=observation.observation_reason,
        error_code=observation.error_code,
        http_status=observation.http_status,
        result_count=observation.result_count,
        reference_count=observation.reference_count,
        foundry_schema_version=observation.foundry_schema_version,
    )


def observe_and_persist_turn(
    response: Mapping[str, Any],
    *,
    session_id: str,
    platform: str,
    platform_chat_id: str | None,
    turn_id: str,
    platform_user_id: str,
    platform_user_message_id: str,
    platform_assistant_message_id: str | None,
    question_text: str,
    feedback_eligible: bool,
    provider: str | None = None,
    hermes_version: str | None = None,
    support_config_commit: str | None = None,
    feedback_code_commit: str | None = None,
    store: FeedbackStoreV2 | None = None,
) -> bool:
    """Parse + persist in one call: used where the whole Turn envelope
    (``response`` -- messages, history_offset, model, final_response, the
    boundary-trust verdict) is available in a single scope -- the
    streaming final-answer block in run.py's ``_run_agent_inner``.
    ``messages``/``history_offset``/``model``/``final_response`` are read
    directly from ``response`` here (never pre-extracted by the caller),
    so run.py only has to pass the dict it already has plus the handful of
    values genuinely not part of that envelope (turn/session/platform
    identity, the raw question text, and the feedback-eligibility verdict,
    which needs extra streaming-specific context to compute).

    Returns True only if the Turn row was actually written; False for
    every other outcome (store unavailable, parser exception, storage
    failure) -- never raises, never returns None.
    """
    try:
        active_store = store if store is not None else get_store()
        if active_store is None:
            return False
        observation = parse_turn_retrieval_observations(
            response.get("messages"),
            history_offset=response.get("history_offset"),
            boundary_trusted=resolve_boundary_trust(response),
        )
        return persist_turn_and_retrieval(
            active_store,
            session_id=session_id,
            platform=platform,
            platform_chat_id=platform_chat_id,
            turn_id=turn_id,
            platform_user_id=platform_user_id,
            platform_user_message_id=platform_user_message_id,
            platform_assistant_message_id=platform_assistant_message_id,
            question_text=question_text,
            answer_text=response.get("final_response"),
            feedback_eligible=feedback_eligible,
            model=response.get("model"),
            provider=provider,
            hermes_version=hermes_version,
            support_config_commit=support_config_commit,
            feedback_code_commit=feedback_code_commit,
            observation=observation,
        )
    except Exception:
        logger.debug("Phase 3A observe_and_persist_turn failed", exc_info=True)
        return False


def build_turn_observation_context(
    agent_result: Mapping[str, Any],
    *,
    session_id: str,
    platform: str,
    platform_chat_id: str | None,
    turn_id: str,
    platform_user_id: str,
    platform_user_message_id: str,
) -> dict[str, Any] | None:
    """Parse-only step for the path where the final delivered answer text
    is not known yet (non-streaming: run.py's _handle_message_with_agent,
    before base.py's media/text extraction pipeline runs).
    ``messages``/``history_offset``/``model`` and the boundary-trust
    verdict are read directly from ``agent_result`` here, exactly like
    observe_and_persist_turn above. Returns a plain, JSON-safe dict meant
    to be stashed on MessageEvent.metadata -- never raises; returns None
    on any internal failure so the caller can treat a missing context as
    "nothing to persist" rather than propagating an error into the answer
    path.
    """
    try:
        observation = parse_turn_retrieval_observations(
            agent_result.get("messages"),
            history_offset=agent_result.get("history_offset"),
            boundary_trusted=resolve_boundary_trust(agent_result),
        )
        return {
            "session_id": session_id,
            "platform": platform,
            "platform_chat_id": platform_chat_id,
            "turn_id": turn_id,
            "platform_user_id": platform_user_id,
            "platform_user_message_id": platform_user_message_id,
            "model": agent_result.get("model"),
            "observation": observation_to_safe_dict(observation),
        }
    except Exception:
        logger.debug("Phase 3A build_turn_observation_context failed", exc_info=True)
        return None


def persist_turn_observation_context(
    context: Mapping[str, Any],
    *,
    platform_assistant_message_id: str | None,
    question_text: str,
    answer_text: str,
    feedback_eligible: bool,
    provider: str | None = None,
    hermes_version: str | None = None,
    support_config_commit: str | None = None,
    feedback_code_commit: str | None = None,
    store: FeedbackStoreV2 | None = None,
) -> bool:
    """Persist step for the non-streaming path: called from base.py's
    final-answer block once the delivered text_content is known, using the
    context built earlier by build_turn_observation_context(). Returns
    True only if the Turn row was actually written; False otherwise --
    never raises, never returns None.
    """
    try:
        active_store = store if store is not None else get_store()
        if active_store is None:
            return False
        observation = safe_dict_to_observation(context.get("observation") or {})
        return persist_turn_and_retrieval(
            active_store,
            session_id=context["session_id"],
            platform=context["platform"],
            platform_chat_id=context.get("platform_chat_id"),
            turn_id=context["turn_id"],
            platform_user_id=context["platform_user_id"],
            platform_user_message_id=context["platform_user_message_id"],
            platform_assistant_message_id=platform_assistant_message_id,
            question_text=question_text,
            answer_text=answer_text,
            feedback_eligible=feedback_eligible,
            model=context.get("model"),
            provider=provider,
            hermes_version=hermes_version,
            support_config_commit=support_config_commit,
            feedback_code_commit=feedback_code_commit,
            observation=observation,
        )
    except Exception:
        logger.debug("Phase 3A persist_turn_observation_context failed", exc_info=True)
        return False


__all__ = [
    "DEFAULT_CASE_ASSIGNMENT_METHOD",
    "get_store",
    "default_case_id",
    "get_or_create_default_case",
    "resolve_boundary_trust",
    "persist_turn_and_retrieval",
    "observe_and_persist_turn",
    "build_turn_observation_context",
    "persist_turn_observation_context",
]
