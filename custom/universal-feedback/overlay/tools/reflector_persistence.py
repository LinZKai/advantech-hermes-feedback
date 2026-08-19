"""ReflectionResult persistence: turns an already-produced, already-
validated ReflectionResult into durable rows. No LLM call and no semantic
reasoning anywhere in this module -- Reflector semantic reasoning and DB
persistence are two different responsibilities, and this module is only
the second one. Storage queries stay in tools.feedback_store_v2 -- this
module issues no raw SQL of its own.

    ReflectionResult            (tools.reflector_proposals -- produced by
     |                            tools.reflector_analyzer.
     |                            analyze_reflection_with_llm())
     v
    persist_reflection_result() (this module -- the persistence boundary)
                v
    reflection_runs / improvement_proposals / proposal_observations
    (tools.feedback_store_v2)

---------------------------------------------------------------------------
Lifecycle
---------------------------------------------------------------------------

    1. create_reflection_run(status='running') -- its OWN commit boundary,
       deliberately SEPARATE from step 2's transaction. This is the audit
       record ("a Reflection attempt happened here") and it must survive
       even if everything after it fails; a reflection_runs row is never
       rolled back by a later Proposal/Observation persistence failure.

    2. ATOMIC transaction: every new_proposals row, then every proposal_
       observations row, via FeedbackStoreV2.persist_reflection_proposals()
       -- either every row for this Run is written, or none is.

    3a. success -> complete_reflection_run(status='succeeded', ...)
    3b. failure -> complete_reflection_run(status='failed') -- step 2 has
        already rolled back its own transaction by the time this runs;
        this call only marks the audit record's own terminal state.

Deliberately NOT a single BEGIN...COMMIT wrapping all three steps: a
reflection_runs row created at step 1 and a failed step 2 must BOTH remain
visible afterward -- the row proves the attempt happened, and its
status='failed' proves what happened to it. A single all-or-nothing
transaction across all three steps would make a failed attempt
indistinguishable from one that never ran at all.

---------------------------------------------------------------------------
Zero-result / no-material-change Reflection Runs
---------------------------------------------------------------------------

new_proposals=() and proposal_observations=() is a fully legal
ReflectionResult. Persisting one is not a special case: step 2's
transaction simply has nothing to insert, and step 3a still records
run_summary and material_change_detected=False. material_change_detected
is one-directional: True requires at least one entry, but False may still
carry proposal_observations entries (e.g. routine trend='stable'
re-observations recorded purely for continuity) -- this module never
re-derives or overrides that rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tools.feedback_store_v2 import FeedbackStoreV2
from tools.reflector_proposals import ImprovementProposal, ProposalObservation, ReflectionResult


class ReflectionPersistenceError(RuntimeError):
    """Raised by persist_reflection_result() only when create_reflection_run()
    itself fails -- the one failure mode step 2/3 above cannot recover
    from, since without a reflection_runs row there is nothing to attach a
    'failed' status to. Never raised for a step 2 (Proposal/Observation)
    failure -- that failure is instead reported through this function's
    own return value (ReflectionPersistenceOutcome.status='failed'),
    because a reflection_runs row DOES exist in that case and IS
    successfully marked 'failed'; raising there would hide that this
    function still did useful, durable work before returning.
    """


@dataclass(frozen=True)
class ReflectionPersistenceOutcome:
    """The result of one persist_reflection_result() call. Deliberately
    minimal -- just enough for a caller to know what happened and to look
    up the full row via FeedbackStoreV2.get_reflection_run(reflection_run_id)
    if it needs more.
    """

    reflection_run_id: str
    status: str  # "succeeded" | "failed" -- mirrors REFLECTION_RUN_STATUS_VALUES's own terminal values

    def __post_init__(self) -> None:
        if self.status not in ("succeeded", "failed"):
            raise ValueError(f"invalid ReflectionPersistenceOutcome.status: {self.status!r}")


# ---------------------------------------------------------------------------
# Dataclass -> plain-mapping projection. tools.feedback_store_v2 must not
# import tools.reflector_proposals (that module already imports taxonomy
# constants FROM tools.feedback_store_v2, so the reverse import would be
# circular), so this module does the projection instead.
# ---------------------------------------------------------------------------


def _proposal_row(proposal: ImprovementProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "improvement_target": proposal.improvement_target,
        "title": proposal.title,
        "created_at": proposal.created_at,
    }


def _observation_row(observation: ProposalObservation) -> dict[str, Any]:
    """observation.supporting_case_ids is already a deterministically
    deduped, sorted tuple (enforced by ProposalObservation.__post_init__),
    so serializing it here is a direct, order-preserving projection."""
    return {
        "observation_id": observation.observation_id,
        "proposal_id": observation.proposal_id,
        "trend": observation.trend,
        "pattern_summary": observation.pattern_summary,
        "possible_cause": observation.possible_cause,
        "recommended_improvement": observation.recommended_improvement,
        "expected_benefit": observation.expected_benefit,
        "limitations": observation.limitations,
        "supporting_case_ids_json": json.dumps(list(observation.supporting_case_ids), ensure_ascii=False),
        "supporting_case_count": observation.supporting_case_count,
        "confidence": observation.confidence,
        "observed_at": observation.observed_at,
    }


# ---------------------------------------------------------------------------
# Persistence boundary
# ---------------------------------------------------------------------------


def persist_reflection_result(
    store: FeedbackStoreV2,
    result: ReflectionResult,
    *,
    started_at: str,
    completed_at: str,
    analyzed_case_count: int,
    reflector_version: str,
    window_start: str | None = None,
    window_end: str | None = None,
) -> ReflectionPersistenceOutcome:
    """Persist one already-computed ReflectionResult, following the
    lifecycle documented at the top of this module.

    `result.reflection_run_id` is used as the reflection_runs primary key
    -- this function never generates its own id; the caller is expected to
    pass through the same id it used to produce `result`, so the
    ReflectionResult's identity and its reflection_runs row's identity are
    always the same value.

    `started_at`/`completed_at`/`analyzed_case_count`/`reflector_version`/
    `window_start`/`window_end` are plain caller-supplied values -- this
    function performs no timestamp generation, no version resolution, and
    no re-derivation of analyzed_case_count from `result` (ReflectionResult
    carries no case-count field of its own).

    Raises ReflectionPersistenceError only if create_reflection_run()
    itself fails (a duplicate reflection_run_id, or an invalid
    analyzed_case_count). Never raises for a step 2 failure:
    FeedbackStoreV2.persist_reflection_proposals() never raises, so this
    function's own control flow is a plain if/else on its boolean return
    value.
    """
    created = store.create_reflection_run(
        result.reflection_run_id,
        started_at=started_at,
        analyzed_case_count=analyzed_case_count,
        reflector_version=reflector_version,
        window_start=window_start,
        window_end=window_end,
    )
    if not created:
        raise ReflectionPersistenceError(
            f"create_reflection_run failed for reflection_run_id={result.reflection_run_id!r} "
            "(duplicate reflection_run_id, or invalid analyzed_case_count)"
        )

    proposal_rows = [_proposal_row(proposal) for proposal in result.new_proposals]
    observation_rows = [_observation_row(observation) for observation in result.proposal_observations]

    persisted = store.persist_reflection_proposals(
        result.reflection_run_id,
        new_proposals=proposal_rows,
        observations=observation_rows,
    )

    if persisted:
        store.complete_reflection_run(
            result.reflection_run_id,
            status="succeeded",
            completed_at=completed_at,
            material_change_detected=result.material_change_detected,
            run_summary=result.run_summary,
        )
        return ReflectionPersistenceOutcome(reflection_run_id=result.reflection_run_id, status="succeeded")

    store.complete_reflection_run(
        result.reflection_run_id,
        status="failed",
        completed_at=completed_at,
    )
    return ReflectionPersistenceOutcome(reflection_run_id=result.reflection_run_id, status="failed")


__all__ = [
    "ReflectionPersistenceError",
    "ReflectionPersistenceOutcome",
    "persist_reflection_result",
]
