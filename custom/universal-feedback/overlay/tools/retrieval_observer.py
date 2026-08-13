"""Pure parser: turn one Hermes Turn's messages into safe Foundry IQ
retrieval telemetry (Phase 3A).

This module never touches a database, never calls FeedbackStoreV2, and
never sees a MessageEvent or gateway response envelope. Its only inputs
are a message list, a history-offset boundary, and an explicit trust flag
for that boundary; its only output is the dataclasses defined below, which
by construction cannot carry document content, full terminal output, a
full command string, a Query Key, an Authorization header, a SAS token, a
response body, or headers -- there is no field for any of it, so
constructing one with such a keyword argument raises TypeError before the
value reaches anything durable. tools/retrieval_runtime.py is the glue
layer that knows about the gateway's response envelope and FeedbackStoreV2.

Foundry invocation detection reuses tools.universal_feedback.foundry_marker
exactly (assistant tool_calls[] + terminal tool + query_foundry_iq.py in
the machine-readable command argument) -- never the final answer text,
never citation filenames, never [NEW_TOPIC]/[IN_SCOPE] markers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.feedback_store_v2 import EXECUTION_STATUSES, OBSERVATION_STATUSES
from tools.universal_feedback import foundry_marker

# Fixed, safe reasons -- never the underlying exception text, never a
# fragment of the unparseable content itself.
UNTRUSTED_TURN_BOUNDARY_REASON = "untrusted_turn_boundary"
TOOL_RESULT_MISSING_REASON = "tool_result_missing"
OUTER_RESULT_UNPARSEABLE_REASON = "outer_result_unparseable"
INNER_RESULT_UNPARSEABLE_REASON = "foundry_inner_unparseable"
# Outer terminal JSON parsed fine (matches {"output","exit_code","error"}),
# but `output` is empty/absent, so there is no Foundry inner v2 JSON to
# classify from at all. This module deliberately does NOT attempt to infer
# *why* (timeout vs. policy block vs. anything else) from exit_code or the
# outer `error` string: no machine-readable Hermes terminal-tool contract
# for those two signals is available anywhere in this repository (searched
# exhaustively, including AGENTS.md/README/docs in both the feedback and
# support-config repos, and the referenced "Hermes Agent v0.18.0" runtime
# source itself, which is not vendored in either repo) -- guessing from
# exit code conventions or free-form error text would be exactly the kind
# of unverified heuristic this module exists to avoid. Fails closed
# instead: unparseable/partial with this fixed reason, until a real,
# sourced contract is found.
MISSING_INNER_RESULT_REASON = "missing_inner_result"
PARSER_ERROR_REASON = "retrieval_parser_error"

_EXECUTION_STATUS_SET = frozenset(EXECUTION_STATUSES)
_OBSERVATION_STATUS_SET = frozenset(OBSERVATION_STATUSES)

# Foundry IQ inner error_code -> retrieval_runs.execution_status. Codes not
# listed here (invalid_input, missing_query_key, internal_error, or any
# unrecognized/absent value) fall back to the generic "failed" bucket --
# there is no more specific enum slot for those in the v2 schema.
_INNER_ERROR_CODE_TO_EXECUTION_STATUS = {
    "request_timeout": "timed_out",
    "http_error": "http_error",
    "network_error": "network_error",
    "invalid_response": "invalid_response",
    "no_documents": "no_documents",
}


@dataclass(frozen=True)
class FoundryRetrievalObservation:
    """One Foundry IQ invocation's safe, structured telemetry.

    Field set is deliberately identical in spirit to
    tools.feedback_store_v2.RetrievalRunInput (that dataclass is what this
    one is converted into at the storage boundary) -- no field here can
    hold documents/content, full terminal output, a command string, or any
    credential. Constructing this with an unknown keyword argument raises
    TypeError, because a frozen dataclass rejects it structurally.
    """

    tool_call_id: str | None
    execution_status: str
    foundry_iq_ok: bool | None
    observation_status: str
    observation_reason: str | None
    error_code: str | None
    http_status: int | None
    result_count: int | None
    reference_count: int | None
    foundry_schema_version: str | None
    request_attempted: bool | None

    def __post_init__(self) -> None:
        if self.execution_status not in _EXECUTION_STATUS_SET:
            raise ValueError(f"invalid execution_status: {self.execution_status!r}")
        if self.observation_status not in _OBSERVATION_STATUS_SET:
            raise ValueError(f"invalid observation_status: {self.observation_status!r}")


@dataclass(frozen=True)
class TurnRetrievalObservation:
    """The whole current Turn's retrieval telemetry: a turn-level boundary
    verdict plus 0..n per-invocation observations, in assistant-emission
    order."""

    boundary_trusted: bool
    turn_observation_status: str
    turn_observation_reason: str | None
    retrievals: tuple[FoundryRetrievalObservation, ...]

    def __post_init__(self) -> None:
        if self.turn_observation_status not in _OBSERVATION_STATUS_SET:
            raise ValueError(f"invalid turn_observation_status: {self.turn_observation_status!r}")


def _unavailable(reason: str) -> TurnRetrievalObservation:
    return TurnRetrievalObservation(
        boundary_trusted=False,
        turn_observation_status="unavailable",
        turn_observation_reason=reason,
        retrievals=(),
    )


def parse_turn_retrieval_observations(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    history_offset: int | None,
    boundary_trusted: bool,
) -> TurnRetrievalObservation:
    """Parse 0..n Foundry IQ retrieval observations out of one Turn's
    current-turn messages.

    ``boundary_trusted`` must be computed by the caller (the gateway glue
    layer knows about session-split/in-place-compaction, this module does
    not) and is re-validated here defensively: even a caller-asserted
    trusted boundary is rejected if ``history_offset`` is missing, not an
    int, negative, or greater than ``len(messages)`` -- any of those makes
    ``messages[history_offset:]`` unsafe to slice as "this turn only".

    Never raises: any unexpected internal error is caught and reported as
    an unavailable observation with a fixed, safe reason, exactly like an
    untrusted boundary -- a parser bug must never fabricate or misattribute
    a Retrieval row.
    """
    try:
        msgs: list[Mapping[str, Any]] = list(messages) if messages else []

        valid_boundary = (
            boundary_trusted
            and isinstance(history_offset, int)
            and not isinstance(history_offset, bool)
            and 0 <= history_offset <= len(msgs)
        )
        if not valid_boundary:
            return _unavailable(UNTRUSTED_TURN_BOUNDARY_REASON)

        current_messages = msgs[history_offset:]
        tool_results = _index_tool_results(current_messages)

        observations = []
        for tool_call_id in _iter_foundry_tool_call_ids(current_messages):
            raw_content = tool_results.get(tool_call_id)
            observations.append(_parse_single_observation(tool_call_id, raw_content))

        return TurnRetrievalObservation(
            boundary_trusted=True,
            turn_observation_status="complete",
            turn_observation_reason=None,
            retrievals=tuple(observations),
        )
    except Exception:
        return _unavailable(PARSER_ERROR_REASON)


def _iter_foundry_tool_call_ids(current_messages: Sequence[Mapping[str, Any]]):
    """Yield Foundry IQ tool_call_id values in the order the assistant
    emitted them. Detection is exactly tools.universal_feedback.
    foundry_marker() applied to (tool name, machine-readable arguments) --
    never the tool's textual preview, never final answer content."""
    for msg in current_messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id") or call.get("call_id")
            if not call_id:
                continue
            function = call.get("function")
            if isinstance(function, Mapping):
                name = function.get("name") or call.get("name")
                arguments = function.get("arguments")
            else:
                name = call.get("name")
                arguments = call.get("arguments")
            if foundry_marker(name, arguments):
                yield str(call_id)


def _index_tool_results(current_messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Map tool_call_id -> raw tool-role message content, for pairing by
    ID rather than by list position (results can arrive out of order)."""
    index: dict[str, Any] = {}
    for msg in current_messages:
        if not isinstance(msg, Mapping) or msg.get("role") not in ("tool", "function"):
            continue
        call_id = msg.get("tool_call_id") or msg.get("call_id")
        if not call_id:
            continue
        index[str(call_id)] = msg.get("content")
    return index


def _safe_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _unparseable(tool_call_id: str, reason: str) -> FoundryRetrievalObservation:
    return FoundryRetrievalObservation(
        tool_call_id=tool_call_id,
        execution_status="unparseable",
        foundry_iq_ok=None,
        observation_status="partial",
        observation_reason=reason,
        error_code=None,
        http_status=None,
        result_count=None,
        reference_count=None,
        foundry_schema_version=None,
        request_attempted=None,
    )


def _parse_single_observation(tool_call_id: str, raw_content: Any) -> FoundryRetrievalObservation:
    if raw_content is None:
        return FoundryRetrievalObservation(
            tool_call_id=tool_call_id,
            execution_status="unknown",
            foundry_iq_ok=None,
            observation_status="partial",
            observation_reason=TOOL_RESULT_MISSING_REASON,
            error_code=None,
            http_status=None,
            result_count=None,
            reference_count=None,
            foundry_schema_version=None,
            request_attempted=None,
        )

    # Parse order, per contract: tool-role content -> terminal outer JSON
    # -> outer.output -> Foundry inner v2 JSON. Persisted references,
    # truncated JSON, and half-written JSON are all indistinguishable from
    # "not valid JSON matching the expected shape" at this layer, and are
    # deliberately handled identically (unparseable/partial) rather than
    # guessed at.
    outer = _safe_json_object(raw_content)
    if outer is None or "output" not in outer or "exit_code" not in outer:
        return _unparseable(tool_call_id, OUTER_RESULT_UNPARSEABLE_REASON)

    output = outer.get("output")

    if isinstance(output, str) and output.strip():
        # Even when outer.exit_code != 0, a non-empty output is always
        # attempted here: Foundry IQ's own ok:false results legitimately
        # pair with a non-zero process exit code.
        inner = _safe_json_object(output)
        if inner is None or not isinstance(inner.get("ok"), bool):
            return _unparseable(tool_call_id, INNER_RESULT_UNPARSEABLE_REASON)
        return _observation_from_inner(tool_call_id, inner)

    # Outer JSON parsed fine, but there is no inner Foundry JSON to
    # classify (output is empty/absent). Deliberately does NOT inspect
    # outer.exit_code or outer.error to guess why (timeout vs. policy
    # block vs. anything else): no verified, machine-readable Hermes
    # terminal-tool contract for those two signals exists anywhere in this
    # repository (see MISSING_INNER_RESULT_REASON docstring above). Fails
    # closed rather than guessing from a string or an exit-code convention.
    return _unparseable(tool_call_id, MISSING_INNER_RESULT_REASON)


def _observation_from_inner(tool_call_id: str, inner: Mapping[str, Any]) -> FoundryRetrievalObservation:
    """Map Foundry IQ inner v2 JSON (schema_version "foundry-iq-result-v2")
    to one observation. Only ok/request_attempted/error_code/http_status/
    schema_version/documents-length/references-length are ever read --
    question/error/document content are never touched."""
    request_attempted = inner.get("request_attempted")
    if not isinstance(request_attempted, bool):
        request_attempted = None

    schema_version = inner.get("schema_version")
    if not isinstance(schema_version, str):
        schema_version = None

    if inner.get("ok") is True:
        documents = inner.get("documents")
        references = inner.get("references")
        return FoundryRetrievalObservation(
            tool_call_id=tool_call_id,
            execution_status="completed",
            foundry_iq_ok=True,
            observation_status="complete",
            observation_reason=None,
            error_code=None,
            http_status=None,
            result_count=len(documents) if isinstance(documents, list) else None,
            reference_count=len(references) if isinstance(references, list) else None,
            foundry_schema_version=schema_version,
            request_attempted=request_attempted,
        )

    error_code = inner.get("error_code")
    if not isinstance(error_code, str):
        error_code = None
    http_status = inner.get("http_status")
    if not isinstance(http_status, int) or isinstance(http_status, bool):
        http_status = None

    return FoundryRetrievalObservation(
        tool_call_id=tool_call_id,
        execution_status=_INNER_ERROR_CODE_TO_EXECUTION_STATUS.get(error_code, "failed"),
        foundry_iq_ok=False,
        observation_status="complete",
        observation_reason=None,
        error_code=error_code,
        http_status=http_status,
        result_count=None,
        reference_count=None,
        foundry_schema_version=schema_version,
        request_attempted=request_attempted,
    )


def observation_to_safe_dict(observation: TurnRetrievalObservation) -> dict[str, Any]:
    """Flatten to plain JSON-safe primitives, for parking on
    MessageEvent.metadata between the parse step and the later persist
    step (non-streaming path). Round-trips with safe_dict_to_observation."""
    return {
        "boundary_trusted": observation.boundary_trusted,
        "turn_observation_status": observation.turn_observation_status,
        "turn_observation_reason": observation.turn_observation_reason,
        "retrievals": [
            {
                "tool_call_id": r.tool_call_id,
                "execution_status": r.execution_status,
                "foundry_iq_ok": r.foundry_iq_ok,
                "observation_status": r.observation_status,
                "observation_reason": r.observation_reason,
                "error_code": r.error_code,
                "http_status": r.http_status,
                "result_count": r.result_count,
                "reference_count": r.reference_count,
                "foundry_schema_version": r.foundry_schema_version,
                "request_attempted": r.request_attempted,
            }
            for r in observation.retrievals
        ],
    }


def safe_dict_to_observation(data: Mapping[str, Any]) -> TurnRetrievalObservation:
    return TurnRetrievalObservation(
        boundary_trusted=bool(data.get("boundary_trusted")),
        turn_observation_status=data.get("turn_observation_status") or "unavailable",
        turn_observation_reason=data.get("turn_observation_reason"),
        retrievals=tuple(
            FoundryRetrievalObservation(**row) for row in (data.get("retrievals") or [])
        ),
    )


__all__ = [
    "FoundryRetrievalObservation",
    "TurnRetrievalObservation",
    "parse_turn_retrieval_observations",
    "observation_to_safe_dict",
    "safe_dict_to_observation",
    "UNTRUSTED_TURN_BOUNDARY_REASON",
    "TOOL_RESULT_MISSING_REASON",
    "OUTER_RESULT_UNPARSEABLE_REASON",
    "INNER_RESULT_UNPARSEABLE_REASON",
    "MISSING_INNER_RESULT_REASON",
    "PARSER_ERROR_REASON",
]
