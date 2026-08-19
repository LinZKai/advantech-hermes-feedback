"""The Existing-Proposal candidate + match/new resolution contract.

    FeedbackStoreV2
     |
     v
    list_improvement_proposals(review_status="pending")   (tools.feedback_store_v2, unchanged)
     |
     v
    list_latest_proposal_observations(proposal_ids=...)   (tools.feedback_store_v2, unchanged)
     |
     v
    build_proposal_candidates()   (this module -- merge, or raise on a
     |                              structurally-impossible gap)
     v
    tuple[ProposalCandidate, ...]
     |
     v
    [ future Reflector reasoning -- NOT implemented here ]
     |
     v
    ProposalResolution            (this module -- one match/new decision)
     |
     v
    validate_proposal_resolution()  (this module -- deterministic,
                                       candidate-bound, no LLM)

The minimal projection a future Reflector step needs to decide "is this
finding an existing Proposal re-observed, or a genuinely new one", and the
deterministic guardrails around that decision. No LLM call, no prompt, no
semantic/fuzzy/embedding matching (the actual identity DECISION is a
future Reflector reasoning step's job -- see "Proposal Identity Rule"
below), no ReflectionResult persistence, no runner, no scheduler. Storage
queries stay in tools.feedback_store_v2 -- this module issues no raw SQL
of its own.

---------------------------------------------------------------------------
Proposal Identity Rule (documented here, NOT implemented)
---------------------------------------------------------------------------

Proposal identity is NOT title string equality, and NOT bare topic
similarity. The same Proposal represents THE SAME UNDERLYING IMPROVEMENT
OPPORTUNITY, which in practice requires weighing together:

    improvement_target
    + problem/pattern semantics
    + recommended remediation direction

Example -- likely the SAME Proposal (same knowledge-gap opportunity, two
different phrasings of the same underlying issue):

    "SNMP disable FAQ missing"
    vs.
    "SNMP configuration documentation incomplete"

Example -- likely DIFFERENT Proposals, even with the same product/topic,
because the improvement_target differs:

    "KB content missing"                                    (knowledge)
    vs.
    "KB retrieved successfully but Hermes omits the command" (agent_behavior)

This module only builds the CANDIDATE PROJECTION a future reasoning step
would use to apply this rule, and a VALIDATOR that rejects a resolution
that could not possibly be consistent with it (an unknown proposal_id, or
-- see validate_proposal_resolution's docstring -- a target mismatch). It
does not, and this slice must not, implement the actual semantic judgment:
no title similarity, no Levenshtein/string-distance, no embedding, no
cosine similarity, no semantic hashing, no LLM call. That reasoning is a
future Reflector step's job. This slice's contribution is narrower and
purely structural: "the AI may choose a candidate, but only a real one,
and only a target-consistent one."

---------------------------------------------------------------------------
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Collection

from tools._validation import is_valid_confidence as _is_valid_confidence
from tools._validation import require_nonblank_str as _require_nonblank_str
from tools.feedback_store_v2 import (
    FeedbackStoreV2,
    IMPROVEMENT_TARGET_VALUES,
    PROPOSAL_TREND_VALUES,
    REVIEW_STATUS_VALUES,
)

logger = logging.getLogger(__name__)

_IMPROVEMENT_TARGET_VALUE_SET = frozenset(IMPROVEMENT_TARGET_VALUES)
_REVIEW_STATUS_VALUE_SET = frozenset(REVIEW_STATUS_VALUES)
_PROPOSAL_TREND_VALUE_SET = frozenset(PROPOSAL_TREND_VALUES)
_TRENDS_REQUIRING_SUPPORTING_CASES = frozenset(PROPOSAL_TREND_VALUES) - {"no_longer_observed"}

# match_existing / create_new is a transient IN-MEMORY reasoning-step
# concept, not a stored value anywhere in tools.feedback_store_v2's schema
# (no column represents it), so -- unlike IMPROVEMENT_TARGET_VALUES/
# REVIEW_STATUS_VALUES/PROPOSAL_TREND_VALUES, which back real CHECK
# constraints and live in that module as the taxonomy authority -- this
# taxonomy is local to this module: a decision taxonomy with no DB column
# of its own belongs next to the code that makes the decision.
PROPOSAL_RESOLUTION_ACTION_VALUES: tuple[str, ...] = ("match_existing", "create_new")
_RESOLUTION_ACTION_VALUE_SET = frozenset(PROPOSAL_RESOLUTION_ACTION_VALUES)

DEFAULT_CANDIDATE_REVIEW_STATUS = "pending"


# ---------------------------------------------------------------------------
# ProposalCandidate -- the minimal projection for a match/new decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalCandidate:
    """The minimal projection of one existing Proposal a future Reflector
    step needs to decide "does this finding belong to this Proposal?".

    Deliberately NOT the full Proposal + Observation history: no
    created_at, no reflection_run_id, no observation history, and no
    supporting_case_ids (only the COUNT) -- carrying more than this
    minimal projection would let a future matching step accidentally key
    off details (e.g. exact historical wording) it should not need for an
    identity decision, and bloats context for no benefit.

    review_status is included as explicit context even though every
    Candidate returned by build_proposal_candidates()'s current default
    scope shares the same value ('pending') -- this dataclass itself does
    NOT assume or enforce 'pending'-only (see this module's docstring on
    candidate scope): a Candidate carrying review_status='accepted' is
    structurally valid here, so a future caller with a broader scope need
    (e.g. re-checking an already-accepted Proposal) is not blocked by this
    contract, only by build_proposal_candidates()'s own default filter.

    The latest_* fields are a flat projection of ONE ProposalObservation
    (the Proposal's most recent one) -- never a nested
    tools.reflector_proposals.ProposalObservation object and never more
    than one Observation's worth of data.
    """

    proposal_id: str
    improvement_target: str
    title: str
    review_status: str

    latest_pattern_summary: str
    latest_recommended_improvement: str
    latest_trend: str
    latest_supporting_case_count: int
    latest_confidence: float

    def __post_init__(self) -> None:
        _require_nonblank_str(self.proposal_id, "proposal_id")

        if self.improvement_target not in _IMPROVEMENT_TARGET_VALUE_SET:
            raise ValueError(f"invalid improvement_target: {self.improvement_target!r}")

        _require_nonblank_str(self.title, "title")

        if self.review_status not in _REVIEW_STATUS_VALUE_SET:
            raise ValueError(f"invalid review_status: {self.review_status!r}")

        _require_nonblank_str(self.latest_pattern_summary, "latest_pattern_summary")
        _require_nonblank_str(self.latest_recommended_improvement, "latest_recommended_improvement")

        if self.latest_trend not in _PROPOSAL_TREND_VALUE_SET:
            raise ValueError(f"invalid latest_trend: {self.latest_trend!r}")

        if (
            not isinstance(self.latest_supporting_case_count, int)
            or isinstance(self.latest_supporting_case_count, bool)
            or self.latest_supporting_case_count < 0
        ):
            raise ValueError(
                f"invalid latest_supporting_case_count: {self.latest_supporting_case_count!r}"
            )
        if self.latest_supporting_case_count == 0 and self.latest_trend in _TRENDS_REQUIRING_SUPPORTING_CASES:
            raise ValueError(
                f"latest_trend={self.latest_trend!r} requires latest_supporting_case_count >= 1 "
                "(only trend='no_longer_observed' may have zero) -- mirrors tools.reflector_"
                "proposals.ProposalObservation's own cross-field rule, re-checked here as this "
                "module's own read-side boundary"
            )

        if not _is_valid_confidence(self.latest_confidence):
            raise ValueError(f"invalid latest_confidence: {self.latest_confidence!r}")


class ProposalCandidateBuildError(RuntimeError):
    """Raised by build_proposal_candidates() when a Proposal in scope has
    NO latest Observation at all.

    A single, minimal exception type on purpose, matching tools.
    case_enrichment_analyzer.CaseEnrichmentAnalyzerError's own "one
    exception type, message carries the detail" convention.

    This is deliberately NOT handled the way tools.case_reflection_input.
    build_reflector_input() handles a Case with no case_analysis yet (an
    explicit cases_missing_analysis gap field, no exception) -- the two
    situations are not analogous. A Case naturally exists before its own
    Case Enrichment completes; that is the NORMAL, EXPECTED, common
    pipeline order. A pending Improvement Proposal with zero Observations
    is different: tools.reflector_proposals.ReflectionResult.__post_init__
    already structurally guarantees every new_proposals entry has a
    founding proposal_observations entry at the moment a Reflection Run's
    result is constructed -- so encountering a Proposal with no
    Observation here means something has already gone wrong upstream (a
    partial/crashed write, or direct DB tampering), not a normal transient
    state this function should quietly tolerate. Failing loudly is the
    correct "fail closed" response for a genuine invariant violation.
    """


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------


def build_proposal_candidates(
    store: FeedbackStoreV2, *, review_status: str = DEFAULT_CANDIDATE_REVIEW_STATUS,
) -> tuple[ProposalCandidate, ...]:
    """Build the current candidate set a future Reflector step can choose
    from when deciding match_existing vs. create_new for one finding.

        list_improvement_proposals(review_status=review_status)
                |
                v
        list_latest_proposal_observations(proposal_ids=...)   (one batch
                |                                                query --
                |                                                no N+1)
                v
        merge each Proposal + its latest Observation -> ProposalCandidate
                |
                v
        deterministic ordering (proposal_id ascending, matching every
        other batch read in this domain: tools.feedback_store_v2.
        list_latest_case_analysis / list_latest_proposal_observations)

    review_status defaults to 'pending' (the set most in need of avoiding
    a duplicate-Proposal re-creation, since it is the only status not yet
    resolved by a human). Exposed as a keyword parameter, never hard-coded
    inside this function body, so a future caller with a genuine reason to
    widen the candidate scope is not blocked by this contract.

    Uses only tools.feedback_store_v2.FeedbackStoreV2 query methods -- no
    raw SQL here.

    Raises ProposalCandidateBuildError (never silently drops the Proposal,
    never silently builds a title-only/placeholder Candidate) if any
    Proposal in scope has no latest Observation at all -- see that
    exception's own docstring for why.
    """
    proposal_rows = store.list_improvement_proposals(review_status=review_status)
    if not proposal_rows:
        return ()

    proposal_ids = [row["proposal_id"] for row in proposal_rows]
    observation_rows = store.list_latest_proposal_observations(proposal_ids=proposal_ids)
    latest_observation_by_proposal_id = {row["proposal_id"]: row for row in observation_rows}

    missing = [pid for pid in proposal_ids if pid not in latest_observation_by_proposal_id]
    if missing:
        raise ProposalCandidateBuildError(
            f"{len(missing)} Proposal(s) in scope (review_status={review_status!r}) have no "
            f"latest Observation, which should never happen given ReflectionResult's own "
            f"founding-Observation invariant: {sorted(missing)}"
        )

    candidates = [
        ProposalCandidate(
            proposal_id=row["proposal_id"],
            improvement_target=row["improvement_target"],
            title=row["title"],
            review_status=row["review_status"],
            latest_pattern_summary=latest_observation_by_proposal_id[row["proposal_id"]]["pattern_summary"],
            latest_recommended_improvement=latest_observation_by_proposal_id[row["proposal_id"]]["recommended_improvement"],
            latest_trend=latest_observation_by_proposal_id[row["proposal_id"]]["trend"],
            latest_supporting_case_count=latest_observation_by_proposal_id[row["proposal_id"]]["supporting_case_count"],
            latest_confidence=latest_observation_by_proposal_id[row["proposal_id"]]["confidence"],
        )
        for row in proposal_rows
    ]
    candidates.sort(key=lambda c: c.proposal_id)
    return tuple(candidates)


# ---------------------------------------------------------------------------
# ProposalResolution -- one match/new decision for one finding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalResolution:
    """One AI decision for one finding from a Reflection Run: does it
    belong to an existing Proposal, or is it genuinely new?

    action='match_existing' requires proposal_id to be a real, non-None
    string -- but this dataclass alone cannot verify it names a real
    offered candidate (it has no candidate list to check against); that
    is validate_proposal_resolution()'s job. action='create_new' requires
    proposal_id=None -- a finding cannot simultaneously claim to be new
    and reference an existing Proposal id.

    improvement_target is required for BOTH actions: a create_new finding
    needs it to know what kind of Proposal it will become, and a
    match_existing finding needs it to state what target the finding
    itself belongs to, independent of whatever Proposal it may or may not
    turn out to match -- this also feeds validate_proposal_resolution()'s
    target-consistency check. Deliberately minimal beyond that -- no
    title/pattern_summary/trend/etc. here; this module models the
    match/new DECISION only, not the finding's full content.
    """

    action: str
    proposal_id: str | None
    improvement_target: str

    def __post_init__(self) -> None:
        if self.action not in _RESOLUTION_ACTION_VALUE_SET:
            raise ValueError(f"invalid action: {self.action!r}")
        if self.improvement_target not in _IMPROVEMENT_TARGET_VALUE_SET:
            raise ValueError(f"invalid improvement_target: {self.improvement_target!r}")

        if self.action == "match_existing":
            if self.proposal_id is None:
                raise ValueError("action='match_existing' requires a non-None proposal_id")
            _require_nonblank_str(self.proposal_id, "proposal_id")
        else:  # create_new
            if self.proposal_id is not None:
                raise ValueError("action='create_new' requires proposal_id=None")


# ---------------------------------------------------------------------------
# Deterministic, candidate-bound validation -- no LLM, no matching
# ---------------------------------------------------------------------------


def validate_proposal_resolution(
    resolution: ProposalResolution, *, candidates: Collection[ProposalCandidate],
) -> bool:
    """Whether `resolution` is a legal decision given the `candidates`
    actually offered -- deterministic, no LLM, no semantic reasoning of
    any kind (see this module's "Proposal Identity Rule" for what this
    function deliberately does NOT attempt to judge).

    action='create_new': always valid regardless of `candidates` (even an
    empty candidate set) -- ProposalResolution.__post_init__ already
    guarantees proposal_id is None for this action; re-checked here too
    (defense in depth) rather than trusting the dataclass invariant
    unconditionally.

    action='match_existing': valid ONLY if proposal_id names one of the
    OFFERED candidates -- an unknown/hallucinated proposal_id is
    unconditionally rejected (False), never "find the closest candidate".
    An empty `candidates` collection therefore makes every match_existing
    resolution False by construction -- there is nothing to match against.

    improvement_target consistency: when a match IS found, this function
    additionally requires `resolution.improvement_target ==
    matched_candidate.improvement_target` -- an agent_behavior finding may
    never resolve to a knowledge Candidate, even if the referenced
    proposal_id is real. This mismatch also returns False, not a separate
    exception type -- from a caller's perspective both failure reasons
    ("hallucinated id" and "real id, wrong target") mean the same thing:
    this resolution must not be accepted as-is.

    Raises ValueError -- a caller-programming-error signal, distinct from
    the substantive True/False decision above -- if `candidates` itself
    contains a duplicate proposal_id (a malformed candidate set should
    never silently produce an ambiguous "match" verdict).
    """
    candidate_ids = [c.proposal_id for c in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidates must not contain duplicate proposal_id values")

    if resolution.action == "create_new":
        return resolution.proposal_id is None

    # action == "match_existing"
    if resolution.proposal_id is None:
        return False
    matched = next((c for c in candidates if c.proposal_id == resolution.proposal_id), None)
    if matched is None:
        return False
    return matched.improvement_target == resolution.improvement_target


__all__ = [
    "PROPOSAL_RESOLUTION_ACTION_VALUES",
    "DEFAULT_CANDIDATE_REVIEW_STATUS",
    "ProposalCandidate",
    "ProposalCandidateBuildError",
    "build_proposal_candidates",
    "ProposalResolution",
    "validate_proposal_resolution",
]
