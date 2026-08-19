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

import dataclasses
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.case_routing import (
    CASE_ROUTING_CLOSE_DELIM,
    CASE_ROUTING_OPEN_DELIM,
    CaseRoutingResult,
    DEFAULT_CONFIDENCE_THRESHOLD,
    ROUTING_VERSION,
    validate_case_routing,
)
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

# Phase 4A Stage B case_assignment_method taxonomy. Every Phase-4-routed
# turn (i.e. every call that explicitly supplies candidate_case_ids, see
# persist_turn_and_retrieval) is stamped with exactly one of these -- never
# DEFAULT_CASE_ASSIGNMENT_METHOD, which remains reserved for legacy/
# Phase 3A callers that do not supply Phase 4 routing context at all (see
# module docstring "Phase 4 routing enablement" note below).
CASE_ASSIGNMENT_METHOD_FIRST_TURN = "case_router_v1_first_turn"
CASE_ASSIGNMENT_METHOD_EXISTING = "case_router_v1_existing"
CASE_ASSIGNMENT_METHOD_NEW = "case_router_v1_new"
CASE_ASSIGNMENT_METHOD_FALLBACK_MISSING = "case_router_v1_fallback_missing"
CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID = "case_router_v1_fallback_invalid"
CASE_ASSIGNMENT_METHOD_FALLBACK_UNCERTAIN = "case_router_v1_fallback_uncertain"
CASE_ASSIGNMENT_METHOD_FALLBACK_LOW_CONFIDENCE = "case_router_v1_fallback_low_confidence"
# Phase 4A Stage C: the candidate-Case DB lookup itself failed (raised, or
# the store was unavailable) this turn -- distinct from
# CASE_ASSIGNMENT_METHOD_FIRST_TURN (a successful query that CONFIRMED zero
# candidates). Never conflate the two: a row stamped first_turn is a
# positive claim ("this session had no Cases yet"), while this method
# stamps a turn where that claim could not honestly be made at all. Gated
# on load_candidate_cases()'s own ``unavailable`` return value, never
# inferred from an empty candidate_case_ids alone.
CASE_ASSIGNMENT_METHOD_CANDIDATE_CONTEXT_UNAVAILABLE = "case_router_v1_candidate_context_unavailable"

# Fixed namespace for deriving a deterministic default case_id from a
# session_id (uuid5 is a one-way hash of namespace+name -- session_id is
# not recoverable from the resulting case_id). Generated once with
# uuid.uuid4() and frozen here; must never change, or existing sessions
# would silently get a second "different" default case.
_DEFAULT_CASE_NAMESPACE = uuid.UUID("6f6d9f1a-6e0a-4e6a-9f0a-5a1a2b3c4d5e")

# Fixed namespace for deriving a deterministic REAL (Phase 4) case_id from
# a (session_id, turn_id) pair -- see real_case_id() below. Deliberately a
# DIFFERENT namespace than _DEFAULT_CASE_NAMESPACE: a real Case is scoped
# per Turn, not per session, so sharing the default-case namespace would
# make a Phase 4 "new" case_id collide with the Phase 3A default case_id
# whenever a session's first Case happens to be created via each path.
# Generated once with uuid.uuid4() and frozen here; must never change, or
# existing Phase 4 Turns would silently get a second "different"
# deterministic case on replay.
_REAL_CASE_NAMESPACE = uuid.UUID("6f13814f-df77-4052-985c-48d37e213b7c")

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


def load_candidate_cases(
    session_id: str, *, store: FeedbackStoreV2 | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """Load this session's candidate Cases, plus each one's minimal
    real-time routing context, once before inference.

    Returns ``(candidate_cases, unavailable)``. Each ``candidate_cases``
    entry is ``{"case_id": ..., "first_user_question": ..., ["latest_user_
    question": ...]}`` for one Case known to belong to this session,
    oldest first (matches FeedbackStoreV2.list_cases_for_session's own
    ordering). ``first_user_question``/``latest_user_question`` are
    derived from turns.question_text (the real-time interaction evidence
    already on the case's turns) -- NOT from cases.title/product_model,
    which nothing in this codepath writes (see create_case call sites)
    and which stay reserved for a later Schema Audit decision. Cases
    Enrichment's case_analysis.case_title is a separate, offline/batch
    concept and is never read here. ``latest_user_question`` is omitted
    for a Case with only one turn (first and latest would be identical).
    A Case with zero turns yet (a narrow create-then-query race) yields
    an entry with only ``case_id`` -- never fabricated or fetched via a
    second query.

    ``unavailable=True`` means either DB lookup failed (the cases lookup,
    or -- once cases are known -- the turn-context lookup) -- store
    unavailable, or either query raised. ``candidate_cases`` is always
    ``[]`` in that case, and callers must NEVER read that empty list as
    "confirmed zero Cases" (see CASE_ASSIGNMENT_METHOD_
    CANDIDATE_CONTEXT_UNAVAILABLE): a turn-context lookup failure is a
    lookup failure, not evidence the session has no Cases, and must not
    be conflated with the "genuinely zero Cases" case (which still
    returns ``([], False)``). Never raises -- fail-closed, matching every
    other public entry point in this module.
    """
    active_store = store if store is not None else get_store()
    if active_store is None:
        return [], True
    try:
        case_rows = active_store.list_cases_for_session(session_id)
    except Exception:
        logger.debug("Phase 4 candidate case lookup failed", exc_info=True)
        return [], True
    if not case_rows:
        return [], False

    case_ids = [str(row["case_id"]) for row in case_rows]
    try:
        turn_rows = active_store.list_turn_questions_for_cases(case_ids)
    except Exception:
        logger.debug("Phase 4 candidate turn-context lookup failed", exc_info=True)
        return [], True

    questions_by_case: dict[str, list[str]] = {}
    for turn_row in turn_rows:
        questions_by_case.setdefault(str(turn_row["case_id"]), []).append(turn_row["question_text"])

    candidates: list[dict[str, Any]] = []
    for case_id in case_ids:
        questions = questions_by_case.get(case_id, [])
        candidate: dict[str, Any] = {"case_id": case_id}
        if questions:
            candidate["first_user_question"] = questions[0]
            if len(questions) > 1:
                candidate["latest_user_question"] = questions[-1]
        candidates.append(candidate)
    return candidates, False


def build_case_routing_prompt(candidate_cases: Sequence[Mapping[str, Any]]) -> str:
    """Build the minimal Case Routing Control Envelope prompt contract for
    the current session's known Cases.

    Only meant to be appended to the ephemeral system prompt when
    ``candidate_cases`` is confirmed non-empty (see the Phase 4A Stage C
    plan, sections 2/4) -- an empty-candidate or lookup-unavailable turn
    never gets this prompt at all, so there is nothing for the model to
    route against and no envelope is requested. Lists only case_id/
    first_user_question/latest_user_question per candidate -- a minimal
    semantic anchor derived from turns.question_text so a case_id is
    never just a bare UUID -- deliberately never full turn transcripts,
    assistant answers, feedback, retrieval runs, or case_analysis; the
    model already has normal conversation context, this is routing aid
    only.
    """
    lines = ["Current session Case routing context:", "", "Known cases:"]
    for case in candidate_cases:
        lines.append(f"- case_id: {case['case_id']}")
        if case.get("first_user_question"):
            lines.append(f"  first_user_question: {case['first_user_question']}")
        if case.get("latest_user_question"):
            lines.append(f"  latest_user_question: {case['latest_user_question']}")
    lines += [
        "",
        "For the final user-facing assistant response only, emit exactly "
        "one Case Routing Control Envelope as the absolute first "
        "characters of the response, before any other text:",
        "",
        (
            f'{CASE_ROUTING_OPEN_DELIM}{{"case_action":"existing|new|uncertain",'
            f'"case_id":"<one of the case_id values above, or null>",'
            f'"confidence":0.0,"routing_version":"{ROUTING_VERSION}"}}'
            f"{CASE_ROUTING_CLOSE_DELIM}"
        ),
        "",
        "Rules:",
        '- "existing": this Turn continues one of the cases listed above. '
        "case_id MUST be exactly one of the case_id values shown above.",
        '- "new": this Turn is about a new, distinct technical issue not '
        "covered by any case listed above. case_id MUST be null.",
        '- "uncertain": you are not confident enough to decide. case_id '
        "MUST be null.",
        "- A new topic sentence from the user is NOT automatically a new "
        "Case -- consider whether it is still part of the same underlying "
        'issue as an existing case above before deciding "new".',
        "- Same product/topic/category does not by itself mean the same "
        'Case. "existing" means the new Turn is still working toward '
        "resolving the same primary technical issue represented by that "
        "Case, not merely a similar topic or the same product.",
        "- Never invent a case_id that is not listed above.",
        "- Never emit the envelope in a response that also contains a "
        "tool call.",
        "- After the closing tag, write your normal answer to the user "
        "exactly as you would without this instruction.",
    ]
    return "\n".join(lines)


def default_case_id(session_id: str) -> str:
    """Deterministic default Case id for a session: same session_id always
    yields the same case_id, so every Turn in a session shares one Case
    without needing a classifier (Phase 3A has none). Derived via a fixed
    namespace UUID5 hash -- reasonable, bounded length, and does not embed
    session_id (or anything else sensitive) recoverably."""
    return f"case-default-{uuid.uuid5(_DEFAULT_CASE_NAMESPACE, str(session_id)).hex}"


def real_case_id(session_id: str, turn_id: str) -> str:
    """Deterministic Case id for one REAL (Phase 4) Case, derived from the
    (session_id, turn_id) pair that is deciding to create it.

    Unlike default_case_id() (one deterministic id per session, shared by
    every Turn), a real Case is scoped per Turn: the SAME Turn retried
    always derives the SAME case_id (making a duplicate create_case()
    attempt for that Turn a verifiable idempotent retry -- see
    _get_or_create_real_case), while a DIFFERENT Turn in the same session
    always derives a DIFFERENT case_id (so two distinct new issues in one
    session don't collide onto the same Case id).
    """
    return f"case-router-v1-{uuid.uuid5(_REAL_CASE_NAMESPACE, f'{session_id}:{turn_id}').hex}"


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


def _get_or_create_real_case(store: FeedbackStoreV2, *, session_id: str, turn_id: str) -> str:
    """Idempotently create (or verify) the deterministic REAL Case for one
    Turn deciding "new Case".

    Stricter than get_or_create_default_case(): after a create_case()
    failure, this does not just check "does a row with this id exist" --
    it also verifies that row's session_id actually matches the caller's
    session_id before treating the failure as a safe idempotent retry.
    This is deliberately NOT about tolerating a theoretical uuid5
    collision -- it is about never silently masking a genuine data-
    integrity problem (e.g. a bug that derived the same case_id for two
    different sessions) behind a blanket "already exists, continue"
    assumption. Any exception other than a row that genuinely exists for
    a different session (or does not exist at all) propagates unchanged.
    """
    case_id = real_case_id(session_id, turn_id)
    existing = store.get_case(case_id)
    if existing is not None:
        if existing["session_id"] != session_id:
            raise RuntimeError(
                f"real_case_id collision: case {case_id!r} already exists "
                f"for a different session ({existing['session_id']!r} != "
                f"{session_id!r})"
            )
        return case_id
    try:
        store.create_case(case_id, session_id)
    except sqlite3.IntegrityError:
        existing = store.get_case(case_id)
        if existing is None or existing["session_id"] != session_id:
            raise
        return case_id
    return case_id


def _get_or_create_turn(store: FeedbackStoreV2, turn_id: str, case_id: str, **turn_kwargs: Any) -> None:
    """Idempotently create the turn row for one Turn.

    A duplicate create_turn() (its own UNIQUE(session_id,
    platform_user_message_id) or PRIMARY KEY(turn_id) constraint already
    satisfied by an earlier, possibly-interrupted attempt) must NOT be
    treated as "everything already succeeded, we're done" -- it only
    proves the turn ROW exists. It says nothing about whether
    add_retrieval_runs() for that same turn ever ran or succeeded. This
    function's only job is to make sure the turn row exists (freshly
    created, or verified as the same (turn_id, case_id) pair on retry);
    the caller (persist_turn_and_retrieval) always attempts retrieval
    persistence afterward regardless of which branch this function took.
    """
    try:
        store.create_turn(turn_id, case_id, **turn_kwargs)
    except sqlite3.IntegrityError:
        existing = store.get_turn(turn_id)
        if existing is None or existing["case_id"] != case_id:
            raise
        # Confirmed idempotent retry: same turn_id already attached to the
        # same case_id. Fall through -- caller still runs retrieval
        # persistence.


@dataclass(frozen=True)
class _CaseAssignmentDecision:
    """Stage B's pure policy verdict for one Turn -- no DB access. See
    _resolve_case_assignment()."""

    is_existing: bool
    case_id: str | None
    method: str
    confidence: str | None


def _confidence_str(confidence: float | None) -> str | None:
    """Stringify a Stage A CaseRoutingResult.confidence for storage in the
    turns.case_assignment_confidence TEXT column, preserving the model's
    original value unmodified. None (no model routing decision exists --
    first turn, missing envelope, absent) stays None; a policy fallback
    triggered by a below-threshold or invalid decision still stores the
    real value the model produced, never 0.0 and never a value rewritten
    to reflect the fallback decision."""
    return None if confidence is None else str(confidence)


def _resolve_case_assignment(
    case_routing: CaseRoutingResult | None,
    *,
    candidate_case_ids: frozenset[str],
    confidence_threshold: float,
    candidate_context_unavailable: bool = False,
) -> _CaseAssignmentDecision:
    """Pure Stage B/C routing-decision policy -- no database access, no
    Case/Turn creation. Maps a Stage A CaseRoutingResult plus the caller's
    session-scoped candidate_case_ids onto exactly one case_router_v1_*
    outcome (see module-level CASE_ASSIGNMENT_METHOD_* constants).

    ``candidate_context_unavailable`` (Stage C) is checked FIRST, even
    before the empty-candidate check: it means the gateway's own
    candidate-Case DB lookup failed this turn (see
    load_candidate_cases()), so ``candidate_case_ids`` carries no honest
    information at all -- it must never be read as "confirmed zero
    Cases" (CASE_ASSIGNMENT_METHOD_FIRST_TURN), which is a positive claim
    a failed lookup cannot make. Forces confidence=None, exactly like the
    first-turn short-circuit below, and ignores case_routing entirely
    (the gateway does not even ask the model to route on an unavailable-
    candidate turn -- see plan section 16).

    An empty (but genuinely queried) candidate_case_ids is the next case
    this function short-circuits on before even looking at case_routing
    -- it means the caller has confirmed this session has no Cases yet
    (Phase 4A implementation plan section 5/19: "first turn"), which is a
    deterministic runtime decision, not a model decision or a fallback,
    so it unconditionally forces confidence=None even if a (nonsensical,
    for a first turn) case_routing was somehow supplied.

    "existing" re-runs validate_case_routing() itself (with
    candidate_case_ids) rather than trusting an already-"valid" status
    from an upstream caller -- the persistence layer must never blindly
    trust that a case_id it is about to write really belongs to this
    session (Phase 4A Stage B plan section 9).
    """
    if candidate_context_unavailable:
        return _CaseAssignmentDecision(
            False, None, CASE_ASSIGNMENT_METHOD_CANDIDATE_CONTEXT_UNAVAILABLE, None
        )

    if not candidate_case_ids:
        return _CaseAssignmentDecision(False, None, CASE_ASSIGNMENT_METHOD_FIRST_TURN, None)

    if case_routing is None or case_routing.status == "absent":
        return _CaseAssignmentDecision(False, None, CASE_ASSIGNMENT_METHOD_FALLBACK_MISSING, None)

    if case_routing.status == "invalid":
        return _CaseAssignmentDecision(
            False, None, CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID, _confidence_str(case_routing.confidence)
        )

    # status == "valid" from here on -- case_action is guaranteed to be one
    # of existing/new/uncertain by CaseRoutingResult.__post_init__.
    if case_routing.case_action == "uncertain":
        return _CaseAssignmentDecision(
            False, None, CASE_ASSIGNMENT_METHOD_FALLBACK_UNCERTAIN, _confidence_str(case_routing.confidence)
        )

    if case_routing.case_action == "new":
        return _CaseAssignmentDecision(
            False, None, CASE_ASSIGNMENT_METHOD_NEW, _confidence_str(case_routing.confidence)
        )

    # case_action == "existing"
    validated = validate_case_routing(
        case_routing,
        candidate_case_ids=candidate_case_ids,
        confidence_threshold=confidence_threshold,
    )
    if validated.status != "valid":
        # Unknown case_id (or any other re-validation failure) -- never
        # "find the closest case", always the fixed fallback bucket.
        return _CaseAssignmentDecision(
            False, None, CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID, _confidence_str(case_routing.confidence)
        )

    if case_routing.confidence >= confidence_threshold:
        return _CaseAssignmentDecision(
            True, case_routing.case_id, CASE_ASSIGNMENT_METHOD_EXISTING, _confidence_str(case_routing.confidence)
        )
    return _CaseAssignmentDecision(
        False, None, CASE_ASSIGNMENT_METHOD_FALLBACK_LOW_CONFIDENCE, _confidence_str(case_routing.confidence)
    )


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
    case_routing: CaseRoutingResult | None = None,
    candidate_case_ids: frozenset[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    candidate_context_unavailable: bool = False,
) -> bool:
    """Create Session/Case/Turn, then 0..n Retrieval rows, for one
    completed Hermes Turn. Returns True only if the Turn row was actually
    written, False for any failure -- never raises, never returns None. On
    a Retrieval-insert failure, FeedbackStoreV2.add_retrieval_runs already
    downgrades the turn's own retrieval_observation_status to
    'unavailable' via its own savepoint (see feedback_store_v2.py); this
    function does not need to (and must not try to) duplicate that
    recovery, and still returns True in that case -- the Turn row itself
    was written successfully, which is what this return value promises.

    Phase 4 routing enablement: ``candidate_case_ids`` is the explicit
    opt-in signal, not ``case_routing``. ``candidate_case_ids=None``
    (the default) means the caller supplied no Phase 4 routing context at
    all -- every legacy/Phase 3A caller keeps its exact original
    behavior: every turn in a session shares one deterministic default
    Case (``get_or_create_default_case`` /
    ``DEFAULT_CASE_ASSIGNMENT_METHOD``). ``candidate_case_ids=frozenset()``
    (explicitly empty, as opposed to omitted) is itself meaningful Phase 4
    input -- it asserts "this session genuinely has zero known Cases yet"
    and resolves deterministically to ``case_router_v1_first_turn``, never
    to the Phase 3A default case. This distinction (omitted vs. explicitly
    empty) is why ``candidate_case_ids`` cannot default to ``frozenset()``
    -- doing so would make "old caller, Phase 4 unaware" indistinguishable
    from "Phase 4 caller confirming zero candidates", silently flipping
    every existing Phase 3A caller onto new-Case-per-turn behavior.

    ``candidate_context_unavailable`` (Stage C) is a THIRD, distinct
    signal: pass ``candidate_case_ids=frozenset()`` together with
    ``candidate_context_unavailable=True`` when Phase 4 routing was
    active this turn but the candidate-Case DB lookup itself failed --
    never silently treated as "confirmed zero Cases" (see
    _resolve_case_assignment / CASE_ASSIGNMENT_METHOD_
    CANDIDATE_CONTEXT_UNAVAILABLE).
    """
    try:
        store.create_or_update_session(session_id, platform, platform_chat_id=platform_chat_id)

        if candidate_case_ids is None:
            case_id = get_or_create_default_case(store, session_id)
            case_assignment_method = DEFAULT_CASE_ASSIGNMENT_METHOD
            case_assignment_confidence = None
            case_assignment_classifier_version = None
        else:
            candidate_case_ids = frozenset(candidate_case_ids)
            decision = _resolve_case_assignment(
                case_routing,
                candidate_case_ids=candidate_case_ids,
                confidence_threshold=confidence_threshold,
                candidate_context_unavailable=candidate_context_unavailable,
            )
            if decision.is_existing:
                # Persistence-layer defense in depth: never trust an
                # "existing" case_id blindly, even one
                # validate_case_routing already checked against
                # candidate_case_ids above -- confirm the row is real and
                # genuinely belongs to THIS session before writing a turn
                # against it. create_turn()'s own FK derives turns.
                # session_id from the case row unconditionally (it has no
                # caller-supplied session_id to check against), so it
                # cannot catch a cross-session case_id on its own.
                case_row = store.get_case(decision.case_id)
                if case_row is not None and case_row["session_id"] == session_id:
                    case_id = decision.case_id
                    case_assignment_method = decision.method
                    case_assignment_confidence = decision.confidence
                else:
                    logger.debug(
                        "Phase 4 existing-case assignment rejected at "
                        "persistence layer (case=%r session=%r); falling "
                        "back to a new Case.",
                        decision.case_id, session_id,
                    )
                    case_id = _get_or_create_real_case(store, session_id=session_id, turn_id=turn_id)
                    case_assignment_method = CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID
                    case_assignment_confidence = decision.confidence
            else:
                case_id = _get_or_create_real_case(store, session_id=session_id, turn_id=turn_id)
                case_assignment_method = decision.method
                case_assignment_confidence = decision.confidence
            case_assignment_classifier_version = ROUTING_VERSION

        _get_or_create_turn(
            store,
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
            case_assignment_method=case_assignment_method,
            case_assignment_confidence=case_assignment_confidence,
            case_assignment_classifier_version=case_assignment_classifier_version,
        )
        # A duplicate turn (idempotent retry, see _get_or_create_turn) does
        # NOT imply retrieval persistence for this turn already succeeded
        # -- always attempt it.
        if observation.turn_observation_status == "complete" and observation.retrievals:
            runs = [_to_retrieval_run_input(r) for r in observation.retrievals]
            store.add_retrieval_runs(turn_id, runs)
        return True
    except Exception as exc:
        logger.debug("Phase 3A/4A turn/retrieval persistence failed: %s", type(exc).__name__)
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
    case_routing: CaseRoutingResult | None = None,
    candidate_case_ids: frozenset[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    candidate_context_unavailable: bool = False,
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

    ``case_routing``/``candidate_case_ids``/``confidence_threshold``/
    ``candidate_context_unavailable`` are Phase 4A Stage B/C passthroughs
    to ``persist_turn_and_retrieval`` -- this function does not parse a
    routing envelope out of ``response`` itself and does not decide
    candidate availability; the gateway (Stage C) derives
    ``case_routing`` authoritatively from ``response["final_response"]``
    (via ``tools.case_routing.parse_and_strip_prefix``) once
    ``run_conversation()`` has returned, and derives ``candidate_case_ids``
    /``candidate_context_unavailable`` from ``load_candidate_cases()``
    called before inference -- both are passed here already resolved.
    Omitting ``candidate_case_ids`` (the default) keeps the exact
    Phase 3A behavior -- see persist_turn_and_retrieval's docstring for
    the full enablement rule.

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
            case_routing=case_routing,
            candidate_case_ids=candidate_case_ids,
            confidence_threshold=confidence_threshold,
            candidate_context_unavailable=candidate_context_unavailable,
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
    candidate_case_ids: frozenset[str] | None = None,
    candidate_context_unavailable: bool = False,
    case_routing: CaseRoutingResult | None = None,
) -> dict[str, Any] | None:
    """Parse-only step for the path where the final delivered text_content
    (post media/TTS extraction) is not known yet (non-streaming: run.py's
    _handle_message_with_agent, before base.py's media/text extraction
    pipeline runs). ``messages``/``history_offset``/``model`` and the
    boundary-trust verdict are read directly from ``agent_result`` here,
    exactly like observe_and_persist_turn above. Returns a plain,
    JSON-safe dict meant to be stashed on MessageEvent.metadata -- never
    raises; returns None on any internal failure so the caller can treat
    a missing context as "nothing to persist" rather than propagating an
    error into the answer path.

    ``candidate_case_ids``/``candidate_context_unavailable`` are known
    this early (queried before inference, per the Phase 4A implementation
    plan) so they are safe to stash on the context as JSON-safe values.
    ``case_routing`` is ALSO accepted here in Stage C: unlike
    ``text_content`` (base.py's media/TTS-processed delivery text, not
    yet known here), the actual Case Routing Control Envelope is parsed
    from ``agent_result["final_response"]`` directly by the gateway right
    after ``run_conversation()`` returns -- before this function is even
    called -- so it IS already available and is stashed here (as a plain
    dict via ``dataclasses.asdict``, JSON-safe) rather than re-derived
    later. ``persist_turn_observation_context`` below still accepts a
    direct ``case_routing`` override for callers/tests that have one in
    hand without going through this stashing step.
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
            "candidate_case_ids": (
                sorted(candidate_case_ids) if candidate_case_ids is not None else None
            ),
            "candidate_context_unavailable": candidate_context_unavailable,
            "case_routing": (
                dataclasses.asdict(case_routing) if case_routing is not None else None
            ),
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
    case_routing: CaseRoutingResult | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Persist step for the non-streaming path: called from base.py's
    final-answer block once the delivered text_content is known, using the
    context built earlier by build_turn_observation_context(). Returns
    True only if the Turn row was actually written; False otherwise --
    never raises, never returns None.

    ``case_routing`` passed directly here always wins (Stage B tests, and
    any caller that has a fresher/better value than what was stashed on
    ``context``). When omitted, falls back to whatever
    ``build_turn_observation_context`` stashed on ``context["case_routing"]``
    (reconstructed via ``CaseRoutingResult(**dict)``) -- this is the normal
    Stage C path, since base.py's own delivery pipeline never re-parses
    the envelope itself (see plan section 9: single parse point).
    ``candidate_case_ids``/``candidate_context_unavailable`` are always
    read back from ``context`` (there is no direct-override parameter for
    either -- they are known before inference and never change between
    build and persist); a missing/None ``candidate_case_ids`` entry
    preserves the exact legacy behavior of persist_turn_and_retrieval.
    """
    try:
        active_store = store if store is not None else get_store()
        if active_store is None:
            return False
        observation = safe_dict_to_observation(context.get("observation") or {})
        raw_candidate_case_ids = context.get("candidate_case_ids")
        candidate_case_ids = (
            frozenset(raw_candidate_case_ids) if raw_candidate_case_ids is not None else None
        )
        candidate_context_unavailable = bool(context.get("candidate_context_unavailable", False))
        effective_case_routing = case_routing
        if effective_case_routing is None:
            raw_case_routing = context.get("case_routing")
            if raw_case_routing is not None:
                effective_case_routing = CaseRoutingResult(**raw_case_routing)
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
            case_routing=effective_case_routing,
            candidate_case_ids=candidate_case_ids,
            confidence_threshold=confidence_threshold,
            candidate_context_unavailable=candidate_context_unavailable,
        )
    except Exception:
        logger.debug("Phase 3A persist_turn_observation_context failed", exc_info=True)
        return False


__all__ = [
    "DEFAULT_CASE_ASSIGNMENT_METHOD",
    "CASE_ASSIGNMENT_METHOD_FIRST_TURN",
    "CASE_ASSIGNMENT_METHOD_EXISTING",
    "CASE_ASSIGNMENT_METHOD_NEW",
    "CASE_ASSIGNMENT_METHOD_FALLBACK_MISSING",
    "CASE_ASSIGNMENT_METHOD_FALLBACK_INVALID",
    "CASE_ASSIGNMENT_METHOD_FALLBACK_UNCERTAIN",
    "CASE_ASSIGNMENT_METHOD_FALLBACK_LOW_CONFIDENCE",
    "CASE_ASSIGNMENT_METHOD_CANDIDATE_CONTEXT_UNAVAILABLE",
    "get_store",
    "load_candidate_cases",
    "build_case_routing_prompt",
    "default_case_id",
    "real_case_id",
    "get_or_create_default_case",
    "resolve_boundary_trust",
    "persist_turn_and_retrieval",
    "observe_and_persist_turn",
    "build_turn_observation_context",
    "persist_turn_observation_context",
]
