"""Phase 5 Slice 2: the Cross-case Reflector proposal domain contract.

    ReflectorInput                    (tools.case_reflection_input, unchanged)
     |
     v
    [ future Reflector LLM -- NOT implemented here ]
     |
     v
    ReflectionResult                 (this module)
     |                                 .new_proposals: ImprovementProposal
     |                                 .proposal_observations: ProposalObservation
     v
    structural validation             (each dataclass's own __post_init__)

Slice 2 scope only: the typed domain contracts a future Reflector step
produces, and the deterministic helper(s) that shape does not need an LLM
for. No LLM call, no prompt, no matching algorithm (embedding/fuzzy/LLM),
no persistence orchestration, no runner, no scheduler -- storage queries
for the three new tables these contracts mirror
(reflection_runs/improvement_proposals/proposal_observations) stay in
tools.feedback_store_v2, matching tools.case_enrichment's own "glue, not
storage" layering (see that module's docstring). This module issues no raw
SQL of its own.

Three distinct concepts, matching the three storage tables
tools.feedback_store_v2 now provides (see that module's docstring for the
full schema/lifecycle reasoning):

    ReflectionRun        = one cross-Case analysis execution
                            (reflection_runs -- not modeled as a dataclass
                            in this module; see this docstring's note below)
    ImprovementProposal  = one persistently-tracked improvement issue,
                            awaiting human handling (identity + human
                            lifecycle only)
    ProposalObservation  = one ReflectionRun's latest snapshot/observation
                            of a Proposal (append-only history)

A Proposal is re-observed across runs, not re-created:

    Reflection Run #1 -> Proposal A -> Observation A1
    Reflection Run #2 -> Proposal A -> Observation A2
                       -> new Proposal B -> Observation B1

The intended default view is ImprovementProposal + its LATEST
ProposalObservation (see tools.feedback_store_v2.list_latest_proposal_
observations()) -- never a growing pile of duplicate cards, one per
Reflection Run, for the same underlying issue.

No ReflectionRun dataclass here: reflection_runs (see
tools.feedback_store_v2) is a mutable batch-run bookkeeping record (started/
completed/status/reflector_version, ...) with no independent "result
shape" of its own to validate beyond what FeedbackStoreV2.
create_reflection_run()/complete_reflection_run() already accept as plain
keyword arguments -- there is no equivalent of CaseEnrichmentResult's
structural-validation need for it. ReflectionResult below is that
equivalent for the actual ANALYTICAL OUTPUT of a run.

Self-improvement boundary (Phase 5 task instruction, section 17) --
stated explicitly, once, here, and never contradicted anywhere else in
this module or in tools.feedback_store_v2's proposal-domain tables:

    An ImprovementProposal, regardless of improvement_target, is NEVER
    permission for any code path to directly modify Hermes production
    behavior, the KB, a Skill, or SOUL. improvement_target='agent_behavior'
    only CLASSIFIES a proposal as being about Hermes' own response
    strategy/instructions/reasoning pattern -- it is not a self-edit
    trigger. No function in this module writes to any production
    configuration, prompt, KB, or Skill file, and none ever should. The
    only lifecycle this module models is: proposal -> human review
    (pending/accepted/rejected) -- draft patch, test, approval, and deploy
    are explicitly out of scope for this slice and are not represented by
    any field here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from tools.feedback_store_v2 import (
    IMPROVEMENT_TARGET_VALUES,
    PROPOSAL_TREND_VALUES,
    REVIEW_STATUS_VALUES,
)

_IMPROVEMENT_TARGET_VALUE_SET = frozenset(IMPROVEMENT_TARGET_VALUES)
_REVIEW_STATUS_VALUE_SET = frozenset(REVIEW_STATUS_VALUES)
_PROPOSAL_TREND_VALUE_SET = frozenset(PROPOSAL_TREND_VALUES)

# trend values that structurally REQUIRE at least one supporting Case --
# every value except 'no_longer_observed' (see ProposalObservation's
# docstring: you cannot claim a pattern is new/growing/stable/declining
# without pointing at concrete supporting evidence; 'no_longer_observed'
# is the one trend that legitimately means "the evidence is now gone").
_TRENDS_REQUIRING_SUPPORTING_CASES = frozenset(PROPOSAL_TREND_VALUES) - {"no_longer_observed"}


def _is_valid_confidence(value: Any) -> bool:
    """True only for a real, finite number in [0.0, 1.0].

    Deliberately its own local copy rather than importing tools.
    feedback_store_v2._is_valid_confidence, tools.case_enrichment.
    _is_valid_confidence, or tools.case_reflection_input.
    _is_valid_confidence -- none of those modules export it, and each of
    them already independently keeps its own copy of this exact check
    rather than reaching across a module boundary at a private,
    leading-underscore helper (see each of their own docstrings for this
    same reasoning). This module follows the same established convention.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return 0.0 <= value <= 1.0


def _require_nonblank_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string, got {value!r}")


def _require_optional_str(value: Any, field_name: str) -> None:
    """None is always allowed; a non-None value must be a non-blank str --
    an empty string is never accepted as a stand-in for "no value", so
    None vs. a real narrative string is never ambiguous (see
    ProposalObservation's docstring for which fields use this)."""
    if value is not None:
        _require_nonblank_str(value, field_name)


# ---------------------------------------------------------------------------
# Deterministic helper -- no LLM, no matching
# ---------------------------------------------------------------------------


def build_supporting_case_ids(case_ids: Iterable[str]) -> tuple[str, ...]:
    """Dedupe and sort case_ids into the canonical, deterministic shape
    ProposalObservation.supporting_case_ids requires.

    Purely deterministic set-then-sort -- no similarity/matching/embedding
    logic of any kind (explicitly out of scope for this slice; see this
    module's docstring). A future matching step decides WHICH case_ids
    belong together; this helper only normalizes an already-decided
    collection into the contract's required shape once it does.
    """
    return tuple(sorted(set(case_ids)))


# ---------------------------------------------------------------------------
# ImprovementProposal -- identity + human lifecycle only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImprovementProposal:
    """One persistently-tracked improvement issue awaiting human handling.

    Deliberately narrow: identity (proposal_id), classification
    (improvement_target), a short identifying title, and the HUMAN review
    lifecycle (review_status) -- nothing that changes on every
    re-observation (pattern_summary, recommended_improvement, confidence,
    trend, supporting Cases, ...) belongs here; all of that lives on
    ProposalObservation instead (Phase 5 task instruction, section 6).

    Only one timestamp (created_at), not a separate first_detected_at:
    within this slice's actual code paths, a Proposal is only ever created
    once, at the exact moment its first Observation is produced, so the
    two would always hold the identical value -- carrying both here would
    be a field with no semantic content yet. A future proposal-merge/
    matching step (explicitly out of scope for this slice -- see this
    module's docstring) is the concrete future moment that could need a
    "first observed" timestamp genuinely distinct from this row's own
    creation time; that is the right time to add it back, not now.

    review_status is NEVER set to 'accepted'/'rejected' by this dataclass
    or by anything in tools.reflector_proposals -- only a human reviewer,
    via tools.feedback_store_v2.FeedbackStoreV2.
    update_proposal_review_status(), moves it away from 'pending'. This
    dataclass structurally allows constructing any of the three values
    (e.g. to represent an already-reviewed Proposal freshly read back from
    storage) -- it does not itself enforce WHO may choose one.
    """

    proposal_id: str
    improvement_target: str
    title: str
    review_status: str
    created_at: str

    def __post_init__(self) -> None:
        _require_nonblank_str(self.proposal_id, "proposal_id")

        if self.improvement_target not in _IMPROVEMENT_TARGET_VALUE_SET:
            raise ValueError(f"invalid improvement_target: {self.improvement_target!r}")

        _require_nonblank_str(self.title, "title")

        if self.review_status not in _REVIEW_STATUS_VALUE_SET:
            raise ValueError(f"invalid review_status: {self.review_status!r}")

        _require_nonblank_str(self.created_at, "created_at")


# ---------------------------------------------------------------------------
# ProposalObservation -- one Reflection Run's snapshot of a Proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalObservation:
    """One Reflection Run's observation of a Proposal -- append-only
    history, never updated in place (a Proposal re-observed in a later run
    is always a new ProposalObservation, matching tools.feedback_store_v2.
    create_proposal_observation()'s exact append-only storage contract).

    Five narrative fields carry the actual "small improvement report"
    (Phase 5 task instruction, section 8) -- each a full-paragraph string,
    never an artificially short label:

        pattern_summary          - REQUIRED. What was observed.
        possible_cause            - OPTIONAL (nullable). See below.
        recommended_improvement  - REQUIRED. What might be done about it.
        expected_benefit          - OPTIONAL (nullable). What improving it
                                     might achieve, when the AI can state
                                     one with any confidence.
        limitations                - OPTIONAL (nullable). Known caveats on
                                     this observation, when there are any
                                     worth stating.

    pattern_summary and recommended_improvement are required because an
    Observation without either carries no actionable content at all -- it
    would not describe an improvement to propose. possible_cause/
    expected_benefit/limitations are optional because the AI may
    legitimately have nothing supportable to state for one of them on a
    given Observation (e.g. a fresh 'new'-trend Observation with thin
    evidence may have no defensible expected_benefit yet) -- None must
    never be forced into a fabricated one-line placeholder just to satisfy
    a NOT NULL-style requirement.

    possible_cause is an EVIDENCE-SUPPORTED HYPOTHESIS ONLY -- never a
    claim of proven root cause. This is a semantic contract enforced by
    definition/documentation (matching tools.case_enrichment.
    CaseEnrichmentResult's own "diagnosis is an AI diagnosis, never a
    proven root cause" convention), not by a runtime content classifier --
    there is no way to verify a hypothesis-shaped string is genuinely
    hypothesis-shaped rather than overclaiming certainty, the same
    structural-only limitation tools.case_enrichment already documents for
    its own Evidence Facts vs. AI Inference boundary.

    trend is the AI's observation of how this Proposal's supporting
    evidence has changed since its PREVIOUS Observation (if any) --
    'no_longer_observed' is NOT the same as review_status='rejected': it
    only means current evidence no longer supports the pattern as active,
    never a human decision (see tools.feedback_store_v2's REVIEW_STATUS_
    VALUES/PROPOSAL_TREND_VALUES docstring comments for the full
    trend-vs-review_status independence statement).

    supporting_case_ids is deterministic by construction: __post_init__
    requires it already be dedupe-and-sorted (use build_supporting_
    case_ids() to normalize a raw collection into this shape) and rejects
    a non-empty duplicate. supporting_case_count must equal
    len(supporting_case_ids) exactly.

    Zero supporting Cases (Phase 5 task instruction, section 9): allowed
    ONLY when trend='no_longer_observed' (the one trend that legitimately
    means "no current evidence") -- every other trend value fails closed
    and requires at least one supporting_case_id, since claiming a pattern
    is new/growing/stable/declining with zero cited evidence would not be
    a defensible observation.
    """

    observation_id: str
    proposal_id: str
    reflection_run_id: str

    trend: str

    pattern_summary: str
    possible_cause: str | None
    recommended_improvement: str
    expected_benefit: str | None
    limitations: str | None

    supporting_case_ids: tuple[str, ...]
    supporting_case_count: int

    confidence: float
    observed_at: str

    def __post_init__(self) -> None:
        _require_nonblank_str(self.observation_id, "observation_id")
        _require_nonblank_str(self.proposal_id, "proposal_id")
        _require_nonblank_str(self.reflection_run_id, "reflection_run_id")

        if self.trend not in _PROPOSAL_TREND_VALUE_SET:
            raise ValueError(f"invalid trend: {self.trend!r}")

        _require_nonblank_str(self.pattern_summary, "pattern_summary")
        _require_optional_str(self.possible_cause, "possible_cause")
        _require_nonblank_str(self.recommended_improvement, "recommended_improvement")
        _require_optional_str(self.expected_benefit, "expected_benefit")
        _require_optional_str(self.limitations, "limitations")

        if not isinstance(self.supporting_case_ids, tuple) or not all(
            isinstance(c, str) and c.strip() for c in self.supporting_case_ids
        ):
            raise ValueError("supporting_case_ids must be a tuple of non-blank strings")
        if len(set(self.supporting_case_ids)) != len(self.supporting_case_ids):
            raise ValueError("supporting_case_ids must not contain duplicates")
        if tuple(sorted(self.supporting_case_ids)) != self.supporting_case_ids:
            raise ValueError("supporting_case_ids must be sorted (see build_supporting_case_ids)")

        if self.supporting_case_count != len(self.supporting_case_ids):
            raise ValueError(
                f"supporting_case_count ({self.supporting_case_count}) must equal "
                f"len(supporting_case_ids) ({len(self.supporting_case_ids)})"
            )

        if not self.supporting_case_ids and self.trend in _TRENDS_REQUIRING_SUPPORTING_CASES:
            raise ValueError(
                f"trend={self.trend!r} requires at least one supporting_case_id "
                "(only trend='no_longer_observed' may have zero)"
            )

        if not _is_valid_confidence(self.confidence):
            raise ValueError(f"invalid confidence: {self.confidence!r}")

        _require_nonblank_str(self.observed_at, "observed_at")


# ---------------------------------------------------------------------------
# ReflectionResult -- one Reflection Run's full analytical output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectionResult:
    """One Reflection Run's complete analytical output.

    A Reflection Run finding NOTHING worth proposing is a fully legal,
    successful result -- this dataclass never forces "at least one
    proposal": material_change_detected=False, new_proposals=(),
    proposal_observations=() together are valid (Phase 5 task instruction,
    section 10), with run_summary carrying a real sentence either way
    (e.g. "本次 analysis horizon 未發現具有實質新意義的 recurring pattern。") --
    run_summary is REQUIRED regardless of outcome, since even a "nothing
    found" result needs a real, human-readable summary of that fact.

    new_proposals holds ImprovementProposal identities detected FOR THE
    FIRST TIME this run; proposal_observations holds EVERY Observation
    produced this run, for both new and previously-existing Proposals (a
    continuing Proposal re-observed this run appears only in
    proposal_observations, not in new_proposals). Every new_proposals
    entry must have exactly one corresponding proposal_observations entry
    (a newly detected Proposal is never declared without the Observation
    that founded it) -- checked below, not just documented. At most one
    Observation per proposal_id in proposal_observations (mirrors
    tools.feedback_store_v2's UNIQUE(proposal_id, reflection_run_id)
    constraint at this Python-object layer too).

    material_change_detected=True requires at least one new_proposals or
    proposal_observations entry -- claiming a material change occurred
    with literally nothing recorded to support that claim would be
    internally inconsistent. The reverse is NOT required:
    material_change_detected=False may still carry proposal_observations
    entries (e.g. routine trend='stable' re-observations recorded every
    run purely for continuity, which do not themselves constitute a
    material change) -- see this module's docstring, "Zero-change
    Semantics" is one-directional by design, not a blanket "empty iff
    False" rule.

    No LLM behavior, no persistence, no scheduler concept -- exactly the
    same "contract only, no side effects" shape as tools.case_enrichment.
    CaseEnrichmentResult / tools.case_reflection_input.ReflectorInput.
    Persisting a ReflectionResult (writing its reflection_runs/
    improvement_proposals/proposal_observations rows via
    tools.feedback_store_v2) is a future runner's job, not built here.
    """

    reflection_run_id: str
    run_summary: str
    material_change_detected: bool
    new_proposals: tuple[ImprovementProposal, ...]
    proposal_observations: tuple[ProposalObservation, ...]

    def __post_init__(self) -> None:
        _require_nonblank_str(self.reflection_run_id, "reflection_run_id")
        _require_nonblank_str(self.run_summary, "run_summary")

        if not isinstance(self.material_change_detected, bool):
            raise ValueError("material_change_detected must be a bool")

        if not isinstance(self.new_proposals, tuple) or not all(
            isinstance(p, ImprovementProposal) for p in self.new_proposals
        ):
            raise ValueError("new_proposals must be a tuple of ImprovementProposal")
        if not isinstance(self.proposal_observations, tuple) or not all(
            isinstance(o, ProposalObservation) for o in self.proposal_observations
        ):
            raise ValueError("proposal_observations must be a tuple of ProposalObservation")

        observation_proposal_ids = [o.proposal_id for o in self.proposal_observations]
        if len(set(observation_proposal_ids)) != len(observation_proposal_ids):
            raise ValueError(
                "proposal_observations must contain at most one Observation per proposal_id"
            )

        for observation in self.proposal_observations:
            if observation.reflection_run_id != self.reflection_run_id:
                raise ValueError(
                    f"proposal_observations entry {observation.observation_id!r} has "
                    f"reflection_run_id {observation.reflection_run_id!r}, expected "
                    f"{self.reflection_run_id!r}"
                )

        new_proposal_ids = {p.proposal_id for p in self.new_proposals}
        if len(new_proposal_ids) != len(self.new_proposals):
            raise ValueError("new_proposals must not contain duplicate proposal_id values")
        observed_proposal_id_set = set(observation_proposal_ids)
        missing_observations = new_proposal_ids - observed_proposal_id_set
        if missing_observations:
            raise ValueError(
                "every new_proposals entry requires a corresponding "
                f"proposal_observations entry; missing for: {sorted(missing_observations)}"
            )

        if self.material_change_detected and not self.new_proposals and not self.proposal_observations:
            raise ValueError(
                "material_change_detected=True requires at least one new proposal or observation"
            )


__all__ = [
    "build_supporting_case_ids",
    "ImprovementProposal",
    "ProposalObservation",
    "ReflectionResult",
]
