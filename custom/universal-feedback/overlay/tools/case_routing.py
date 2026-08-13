"""Phase 4A: pure parser/validator for the Case Routing Control Envelope
protocol.

The protocol asks the main Hermes model to prefix its FINAL non-tool-call
response with a fixed, machine-readable control frame before its normal
user-visible answer::

    <case-routing>{"case_action":"existing","case_id":"case-a","confidence":0.92,"routing_version":"case-router-v1"}</case-routing>
    normal user-visible answer...

This is deliberately an application-level *framed* protocol, not a natural-
language marker to be found by scanning the final answer text: the envelope
only has protocol meaning when it is the absolute prefix of the response
(no leading whitespace, no earlier normal text), with a fixed opening/
closing delimiter pair and a strict JSON schema in between. This module
never searches for the delimiter at an arbitrary position -- doing so would
turn this back into the "regex-guess telemetry from the final answer" shape
Phase 3A already ruled out.

This module never touches a database, never does network I/O, never calls
an LLM, and never reads environment/config. It has no notion of a Case
actually existing in storage -- ``validate_case_routing`` accepts the
caller-supplied set of case ids known to be valid for the current session
(``candidate_case_ids``) as a plain argument, exactly like
tools.retrieval_observer accepts ``boundary_trusted`` as a plain argument
instead of ever asking the gateway/session layer for it. tools/
retrieval_runtime.py (a later Phase 4A stage) is the glue layer that knows
about FeedbackStoreV2 and the gateway's response envelope; it is the only
module that should import from here.

Any failure to parse or validate the envelope is reported as a
``CaseRoutingResult`` with ``status="invalid"`` (envelope was present but
broken) or ``status="absent"`` (no envelope at all) and a fixed, safe
``invalid_reason`` string -- never the raw exception text, the raw model
output, or a guess at what the model "probably meant". Parsing failure must
never fall back to inferring routing from the natural-language answer; the
caller's own policy layer is expected to treat both ``invalid`` and
``absent`` as "no machine-readable routing this turn" and apply its own
conservative fallback (see Future Stage B Notes at the bottom of this file).

Stage A scope: this module only parses, validates, and strips the protocol
envelope. It does not decide what to do with the result -- no Case
creation, no confidence-threshold fallback policy, no persistence. See the
module-level "Future Stage B Notes" docstring at the bottom of this file
for what is intentionally deferred.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Protocol constants (Phase 4A POC policy values -- see AGENTS.md constraint
# against introducing new config surfaces for POC-stage constants. A future
# change to these values ships as a new ROUTING_VERSION, not a config knob;
# see the confidence-threshold discussion in the Phase 4A implementation
# plan for why this module does not read either from config.yaml or an env
# var.)
# ---------------------------------------------------------------------------

ROUTING_VERSION = "case-router-v1"
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
MAX_ENVELOPE_PREFIX_BYTES = 2048

CASE_ROUTING_OPEN_DELIM = "<case-routing>"
CASE_ROUTING_CLOSE_DELIM = "</case-routing>"

_STATUS_SET = frozenset({"valid", "invalid", "absent"})
_CASE_ACTION_SET = frozenset({"existing", "new", "uncertain"})
_REQUIRED_ENVELOPE_KEYS = frozenset({"case_action", "case_id", "confidence", "routing_version"})

# Fixed, safe reasons -- never the underlying exception text, the raw
# envelope body, or a fragment of the model's output. Mirrors the
# tools.retrieval_observer convention of a small, fixed vocabulary of
# reason strings recorded to storage.
REASON_JSON_INVALID = "json_invalid"
REASON_SCHEMA_INVALID = "schema_invalid"
REASON_UNSUPPORTED_ROUTING_VERSION = "unsupported_routing_version"
REASON_UNKNOWN_CASE_ACTION = "unknown_case_action"
REASON_INVALID_CONFIDENCE = "invalid_confidence"
REASON_EXISTING_MISSING_CASE_ID = "existing_missing_case_id"
REASON_EXISTING_UNKNOWN_CASE_ID = "existing_unknown_case_id"
REASON_NEW_WITH_CASE_ID = "new_with_case_id"
REASON_UNCERTAIN_WITH_CASE_ID = "uncertain_with_case_id"
REASON_PREFIX_SIZE_EXCEEDED = "prefix_size_exceeded"
REASON_MISSING_CLOSING_DELIMITER = "missing_closing_delimiter"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseRoutingResult:
    """One Turn's Case Routing Control Envelope outcome.

    ``status`` is exactly one of:

    * ``"valid"``   -- a well-formed envelope was found and passed every
      protocol-level (and, if ``validate_case_routing`` was applied,
      session-scoped) check. ``case_action``/``case_id``/``confidence``/
      ``routing_version`` are all populated and internally consistent.
    * ``"invalid"`` -- an envelope was present (the response started with
      the opening delimiter) but failed some check. ``invalid_reason`` is
      one of the fixed ``REASON_*`` constants; the other fields are not
      populated (constructing a "valid"-shaped ``"invalid"`` result would
      misrepresent an unverified value as trustworthy).
    * ``"absent"``  -- the response did not begin with the opening
      delimiter at all. This is NOT the same as "invalid": a normal
      answer with no routing envelope is the expected common case, not a
      protocol violation. Callers must not conflate the two counters.

    Frozen and structurally validated in ``__post_init__`` -- constructing
    an internally inconsistent result (e.g. ``status="valid"`` with no
    ``case_action``) raises ``ValueError`` before it can reach a caller,
    matching the "structural block, not a runtime filter" convention used
    by tools.feedback_store_v2.RetrievalRunInput and tools.
    retrieval_observer.FoundryRetrievalObservation.
    """

    status: str
    case_action: str | None = None
    case_id: str | None = None
    confidence: float | None = None
    routing_version: str | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUS_SET:
            raise ValueError(f"invalid CaseRoutingResult.status: {self.status!r}")
        if self.status == "valid" and self.case_action not in _CASE_ACTION_SET:
            raise ValueError(
                "CaseRoutingResult.status='valid' requires case_action in "
                f"{sorted(_CASE_ACTION_SET)}, got {self.case_action!r}"
            )


# ---------------------------------------------------------------------------
# Confidence parsing
# ---------------------------------------------------------------------------


def _is_valid_confidence(value: Any) -> bool:
    """True only for a real, finite JSON number in [0.0, 1.0].

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is True -- checked and rejected first, otherwise ``"confidence": true``
    would silently pass as ``1``. Python's ``json`` module also accepts the
    non-standard tokens ``NaN``/``Infinity``/``-Infinity`` as float by
    default (unlike strict JSON), so those are rejected explicitly too --
    without this, a downstream ``0.0 <= confidence <= 1.0`` comparison
    against ``nan`` would silently evaluate False for the wrong reason
    (NaN comparisons are always False) and produce a correct-looking
    rejection for an accidental reason rather than an intentional one.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Envelope body (the JSON between the delimiters) -- protocol-level only
# ---------------------------------------------------------------------------


def _parse_envelope_body(body: str) -> CaseRoutingResult:
    """Parse + protocol-level-validate the JSON strictly between the
    delimiters. No DB/known-case lookup, no confidence-threshold policy --
    see ``validate_case_routing`` for the session-scoped check this
    deliberately does not do."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_JSON_INVALID)

    if not isinstance(parsed, dict):
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_SCHEMA_INVALID)

    # Strict schema: exactly the four required keys, nothing more, nothing
    # less. This is a control protocol, not an extensible payload -- an
    # unrecognized extra field is as much a protocol violation as a missing
    # required one, both collapse to the same fixed reason (see module
    # docstring: fixed, low-cardinality reasons only).
    if set(parsed.keys()) != _REQUIRED_ENVELOPE_KEYS:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_SCHEMA_INVALID)

    routing_version = parsed.get("routing_version")
    if not isinstance(routing_version, str) or routing_version != ROUTING_VERSION:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_UNSUPPORTED_ROUTING_VERSION)

    case_action = parsed.get("case_action")
    if case_action not in _CASE_ACTION_SET:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_UNKNOWN_CASE_ACTION)

    confidence = parsed.get("confidence")
    if not _is_valid_confidence(confidence):
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_INVALID_CONFIDENCE)

    case_id = parsed.get("case_id")
    if case_id is not None and not isinstance(case_id, str):
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_SCHEMA_INVALID)

    if case_action == "existing" and case_id is None:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_EXISTING_MISSING_CASE_ID)
    if case_action == "new" and case_id is not None:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_NEW_WITH_CASE_ID)
    if case_action == "uncertain" and case_id is not None:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_UNCERTAIN_WITH_CASE_ID)

    return CaseRoutingResult(
        status="valid",
        case_action=case_action,
        case_id=case_id,
        confidence=float(confidence),
        routing_version=routing_version,
    )


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------


def parse_case_routing_envelope(raw_envelope_text: str) -> CaseRoutingResult:
    """Parse one Case Routing Control Envelope from the ABSOLUTE PREFIX of
    ``raw_envelope_text``.

    Responsibilities: locate the opening/closing delimiter pair (only at
    the very start of the string -- never searched for at an arbitrary
    position), enforce the ``MAX_ENVELOPE_PREFIX_BYTES`` protocol safety
    boundary, parse the JSON between them, and apply protocol-level schema
    validation (``_parse_envelope_body``). Deliberately does NOT do any
    database/known-case lookup and does NOT apply any confidence-threshold
    policy -- see ``validate_case_routing`` for the one additional,
    session-scoped check this function cannot do on its own, and the
    Phase 4A implementation plan / Future Stage B Notes for why the
    confidence-threshold *decision* belongs to a later stage entirely.

    Returns ``status="absent"`` (not ``"invalid"``) when the text does not
    start with the opening delimiter at all -- a normal answer with no
    routing envelope is the expected common case, not a protocol
    violation.
    """
    if not isinstance(raw_envelope_text, str) or not raw_envelope_text.startswith(CASE_ROUTING_OPEN_DELIM):
        return CaseRoutingResult(status="absent")

    remainder = raw_envelope_text[len(CASE_ROUTING_OPEN_DELIM):]
    close_idx = remainder.find(CASE_ROUTING_CLOSE_DELIM)

    if close_idx == -1:
        # No closing delimiter anywhere in the input. Distinguish a
        # genuinely truncated/incomplete envelope (short remainder, simply
        # never closed) from one that blew past the size safety boundary
        # before ever closing -- both lack a close delimiter, but only the
        # latter should be attributed to size rather than truncation, so a
        # future streaming caller can tell "keep waiting for more chunks"
        # apart from "give up, this is oversized" (see Stage C in the
        # implementation plan; Stage A only needs the classification, not
        # the buffering loop itself).
        if len(remainder.encode("utf-8")) > MAX_ENVELOPE_PREFIX_BYTES:
            return CaseRoutingResult(status="invalid", invalid_reason=REASON_PREFIX_SIZE_EXCEEDED)
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_MISSING_CLOSING_DELIMITER)

    envelope_body = remainder[:close_idx]
    full_span = CASE_ROUTING_OPEN_DELIM + envelope_body + CASE_ROUTING_CLOSE_DELIM
    if len(full_span.encode("utf-8")) > MAX_ENVELOPE_PREFIX_BYTES:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_PREFIX_SIZE_EXCEEDED)

    return _parse_envelope_body(envelope_body)


def validate_case_routing(
    result: CaseRoutingResult,
    *,
    candidate_case_ids: frozenset[str],
    confidence_threshold: float,
) -> CaseRoutingResult:
    """Apply the one check ``parse_case_routing_envelope`` cannot do on its
    own: whether an ``"existing"`` result's ``case_id`` actually names a
    Case known to belong to the current session.

    Also defensively re-checks ``routing_version``/``case_action`` shape
    invariants, in case ``result`` was constructed by a caller other than
    ``parse_case_routing_envelope`` (e.g. hand-built in a test, or
    reconstructed from serialized/stashed state by a later stage) rather
    than a already-trusted ``status="valid"`` value being assumed correct
    unconditionally.

    ``confidence_threshold`` is accepted here to match the Stage B call
    contract, but this function deliberately does NOT use it to downgrade
    ``status`` -- Stage A does not build a Case. An ``"existing"`` result
    with ``confidence`` below the threshold is still returned with
    ``status="valid"``; whether to fall back to creating a new Case for a
    low-confidence match is a POLICY decision that belongs to the Stage B
    persistence layer (see the module-level Future Stage B Notes below and
    the Phase 4A implementation plan, section 14/"Confidence Threshold").
    A non-numeric or out-of-range ``confidence_threshold`` is a caller
    programming error, not a data-quality problem with model output, so it
    raises ``ValueError`` immediately instead of being folded into
    ``invalid_reason``.

    ``candidate_case_ids`` and ``confidence_threshold`` are the ONLY
    external inputs this function reads -- still no database, no network
    I/O, still a pure function.
    """
    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or not (0.0 <= confidence_threshold <= 1.0)
    ):
        raise ValueError(f"invalid confidence_threshold: {confidence_threshold!r}")

    if result.status != "valid":
        return result

    if result.routing_version != ROUTING_VERSION:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_UNSUPPORTED_ROUTING_VERSION)
    if result.case_action not in _CASE_ACTION_SET:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_UNKNOWN_CASE_ACTION)

    if result.case_action == "existing":
        if result.case_id is None:
            return CaseRoutingResult(status="invalid", invalid_reason=REASON_EXISTING_MISSING_CASE_ID)
        # Never "find the closest case" for an unrecognized id -- an
        # unknown case_id is unconditionally invalid, full stop.
        if result.case_id not in candidate_case_ids:
            return CaseRoutingResult(status="invalid", invalid_reason=REASON_EXISTING_UNKNOWN_CASE_ID)
    elif result.case_action == "new" and result.case_id is not None:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_NEW_WITH_CASE_ID)
    elif result.case_action == "uncertain" and result.case_id is not None:
        return CaseRoutingResult(status="invalid", invalid_reason=REASON_UNCERTAIN_WITH_CASE_ID)

    return result


def parse_and_strip_prefix(full_text: str) -> tuple[str, CaseRoutingResult]:
    """Split ``full_text`` into ``(clean_user_text, CaseRoutingResult)``.

    Only the ABSOLUTE PREFIX of ``full_text`` is ever considered for a
    routing envelope -- never a substring found at an arbitrary position.
    A model that writes its normal answer first and appends a
    ``<case-routing>`` block afterwards produces ``status="absent"`` here
    (the prefix rule is intentional protocol design, not a parsing
    limitation -- see module docstring).

    Non-destructive on failure: when no valid envelope is found (whether
    ``"absent"`` or ``"invalid"``), ``clean_user_text`` is ``full_text``
    unchanged. This function never guesses where the "real" answer begins
    inside a malformed/incomplete/oversized control frame -- the caller
    (a later stage) is expected to apply its own fail-closed handling
    (e.g. treating a response that is entirely an invalid control frame as
    equivalent to an empty response) rather than this module fabricating a
    split.
    """
    if not isinstance(full_text, str) or not full_text.startswith(CASE_ROUTING_OPEN_DELIM):
        return full_text, CaseRoutingResult(status="absent")

    result = parse_case_routing_envelope(full_text)
    if result.status != "valid":
        return full_text, result

    close_idx = full_text.find(CASE_ROUTING_CLOSE_DELIM)
    remaining = full_text[close_idx + len(CASE_ROUTING_CLOSE_DELIM):]
    return remaining, result


def strip_case_routing_prefix(content: str) -> str:
    """Pure sanitization helper: return ``content`` with a valid Case
    Routing Control Envelope prefix removed, or ``content`` unchanged if
    none is present.

    Strips only when ``content`` is a fully protocol-valid envelope prefix
    (same definition ``parse_and_strip_prefix``/``parse_case_routing_
    envelope`` use) -- a structurally present-but-malformed prefix (bad
    JSON, wrong schema, unsupported version, ...) is left untouched rather
    than repaired or guessed at, matching the "never destructively rewrite
    invalid data" rule used throughout this module. Intended for the later
    gateway-side history-cleanup step (stripping the envelope from the
    stored assistant message before it re-enters Hermes' own conversation
    history) -- not implemented in this stage.
    """
    clean_text, _ = parse_and_strip_prefix(content)
    return clean_text


def sanitize_case_routing_for_display(text: str) -> str:
    """Remove a Case Routing Control Envelope prefix for USER-VISIBLE
    delivery -- a deliberately SEPARATE, stricter contract from
    ``parse_and_strip_prefix``/``strip_case_routing_prefix``.

    Those two functions only remove a prefix that is fully protocol-valid
    (right JSON, right schema, right ``routing_version``, under
    ``MAX_ENVELOPE_PREFIX_BYTES``); on anything else they are deliberately
    non-destructive, because their job is to hand a caller a trustworthy
    ``CaseRoutingResult`` for a ROUTING decision, and silently repairing or
    guessing at malformed input would misrepresent an unverified value as
    trustworthy (see the module docstring). That non-destructive choice is
    correct for the routing/trust contract, but wrong for display: a model
    that emits a broken, oversized, or unsupported-version control frame
    must never have that raw frame shown to the end user just because it
    failed a trust check that has nothing to do with display safety.

    Whether the envelope is trustworthy for Case assignment is answered
    separately, by ``parse_and_strip_prefix``/``parse_case_routing_
    envelope`` -- a caller needing both facts (e.g. the gateway's
    post-run_conversation cleanup) calls both functions on the same raw
    text, once each, and uses each result for its own concern. This
    function never looks at ``case_action``/``confidence``/
    ``routing_version`` at all.

    Rule: if ``text`` begins with the opening delimiter, strip through the
    FIRST closing delimiter found -- structurally, regardless of whether
    the JSON between them is valid, matches the required schema, names a
    supported ``routing_version``, or fits within
    ``MAX_ENVELOPE_PREFIX_BYTES``. Size/schema/version are trust concerns,
    not display concerns: a malformed-but-delimited control frame is just
    as much "not part of the answer" as a well-formed one. This mirrors
    ``_StreamEnvelopeFilter`` in gateway/run.py, which also hides
    everything between the delimiters purely structurally, without
    checking JSON validity, for the live token stream.

    If NO closing delimiter can be found at all (the response is truncated
    mid-envelope, or the model never closes it), there is no safe way to
    know where internal control metadata ends and the real answer begins.
    Guessing would risk leaking a metadata fragment into the visible
    answer, so this fails closed and returns ``""`` rather than the raw
    text -- matching the streaming filter's oversized-without-close-
    delimiter branch, which also discards its buffer instead of forwarding
    it. Turning an empty result into a user-facing message is the
    gateway's existing empty-response handling's job, not this function's.

    ``text`` that does not start with the opening delimiter at all --
    the common case of a normal answer with no envelope -- is returned
    unchanged.
    """
    if not isinstance(text, str) or not text.startswith(CASE_ROUTING_OPEN_DELIM):
        return text
    remainder = text[len(CASE_ROUTING_OPEN_DELIM):]
    close_idx = remainder.find(CASE_ROUTING_CLOSE_DELIM)
    if close_idx == -1:
        return ""
    return remainder[close_idx + len(CASE_ROUTING_CLOSE_DELIM):]


__all__ = [
    "ROUTING_VERSION",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MAX_ENVELOPE_PREFIX_BYTES",
    "CASE_ROUTING_OPEN_DELIM",
    "CASE_ROUTING_CLOSE_DELIM",
    "REASON_JSON_INVALID",
    "REASON_SCHEMA_INVALID",
    "REASON_UNSUPPORTED_ROUTING_VERSION",
    "REASON_UNKNOWN_CASE_ACTION",
    "REASON_INVALID_CONFIDENCE",
    "REASON_EXISTING_MISSING_CASE_ID",
    "REASON_EXISTING_UNKNOWN_CASE_ID",
    "REASON_NEW_WITH_CASE_ID",
    "REASON_UNCERTAIN_WITH_CASE_ID",
    "REASON_PREFIX_SIZE_EXCEEDED",
    "REASON_MISSING_CLOSING_DELIMITER",
    "CaseRoutingResult",
    "parse_case_routing_envelope",
    "validate_case_routing",
    "parse_and_strip_prefix",
    "strip_case_routing_prefix",
    "sanitize_case_routing_for_display",
]


# ---------------------------------------------------------------------------
# Future Stage B Notes -- documented here, NOT implemented in this stage.
# ---------------------------------------------------------------------------
#
# 1. Idempotent Case creation must not blanket-catch IntegrityError. When a
#    future create_case() call hits a uniqueness-constraint failure, the
#    caller must re-read the existing row and confirm it really is the
#    expected deterministic Case for this session/turn before treating the
#    failure as "already created, continue" -- a bare
#    ``except sqlite3.IntegrityError: pass`` would silently paper over a
#    genuine id collision against an unrelated row.
#
# 2. A duplicate create_turn() (UNIQUE(session_id, platform_user_message_id)
#    already satisfied by an earlier attempt) does NOT imply
#    add_retrieval_runs() for that turn already succeeded. Future retry
#    logic must not skip retrieval persistence just because the turn row
#    already exists -- the two are separate steps that can independently
#    fail and need independent idempotency handling.
