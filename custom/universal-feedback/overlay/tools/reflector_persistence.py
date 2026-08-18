"""Phase 5 Slice 6A: ReflectionResult persistence.

    ReflectionResult                   (tools.reflector_proposals -- produced
     |                                   by tools.reflector_analyzer.
     |                                   analyze_reflection_with_llm(),
     |                                   Slice 5B; unchanged here)
     v
    persist_reflection_result()        (this module -- the persistence
                |                        boundary: deterministic only, no
                |                        semantic reasoning, no LLM call)
                v
    reflection_runs / improvement_proposals / proposal_observations
    (tools.feedback_store_v2 -- unchanged existing storage API, plus one
     new atomic batch-insert method, persist_reflection_proposals(),
     added in this slice)

Slice 6A scope only: turn an already-produced, already-validated
ReflectionResult into durable rows. No LLM call and no semantic reasoning
anywhere in this module (it never imports tools.reflector_analyzer or
agent.auxiliary_client, and never builds a prompt) -- Reflector semantic
reasoning and DB persistence are two different responsibilities, and this
module is only the second one (Phase 5 Slice 6A task instruction, section
16: "Reflector semantic reasoning != DB persistence"). No runner, no
scheduler, no trigger policy, no human-review workflow -- this module's
only job is "given one already-computed ReflectionResult, make it
durable"; deciding WHEN to run a Reflection, and calling this function
after doing so, is a future Slice 6B runner's job. Storage queries stay in
tools.feedback_store_v2 -- this module issues no raw SQL of its own,
matching every other module in this domain's own "glue, not storage"
layering (see e.g. tools.reflector_prompt_context's own docstring for the
precedent this mirrors).

---------------------------------------------------------------------------
Lifecycle (Phase 5 Slice 6A task instruction, section 3)
---------------------------------------------------------------------------

    1. create_reflection_run(status='running')
       -- its OWN commit boundary (tools.feedback_store_v2.FeedbackStoreV2.
       create_reflection_run() already commits via its own `with self.
       _connect() as db:` block), deliberately SEPARATE from step 2's
       transaction. This is the audit record -- "a Reflection attempt
       happened here, covering N analyzed Cases" -- and it must survive
       even if everything after it fails; a reflection_runs row is never
       rolled back by a later Proposal/Observation persistence failure.

    2. ATOMIC transaction: every new_proposals row, then every proposal_
       observations row, via tools.feedback_store_v2.FeedbackStoreV2.
       persist_reflection_proposals() -- either every row for this Run is
       written, or none is. See that method's own docstring for exactly
       how it enforces "no Proposal without its founding Observation" and
       "no half-written Observation batch", and how the match_existing/
       create_new guardrails (sections 6-7) fall directly out of the
       schema's FOREIGN KEY/PRIMARY KEY constraints rather than being
       re-implemented as separate Python checks here.

    3a. success -> complete_reflection_run(status='succeeded',
        material_change_detected=result.material_change_detected,
        run_summary=result.run_summary)
    3b. failure -> complete_reflection_run(status='failed') -- step 2 has
        already rolled back its own transaction by the time this runs; this
        call only marks the audit record's own terminal state. No Proposal/
        Observation row from a failed run is ever left behind (step 2's own
        atomicity already guarantees that); what step 3b guarantees is that
        the reflection_runs row itself never has to be silently deleted or
        left stuck at status='running' to represent that failure.

Deliberately NOT a single BEGIN...COMMIT wrapping all three steps (Phase 5
Slice 6A task instruction, section 3: "不要單純 BEGIN insert reflection_run
insert proposal insert observations COMMIT 然後任何失敗全部 rollback，因為這樣
failed run 可能完全消失") -- a reflection_runs row created at step 1 and a
failed step 2 must BOTH be visible afterward: the row proves the attempt
happened, and its status='failed' proves what happened to it. A single
all-or-nothing transaction across all three steps would make a failed
attempt structurally indistinguishable from one that never ran at all.

---------------------------------------------------------------------------
Zero-result / no-material-change Reflection Runs (Phase 5 Slice 6A task
instruction, section 14)
---------------------------------------------------------------------------

new_proposals=() and proposal_observations=() is a fully legal
ReflectionResult (tools.reflector_proposals.ReflectionResult already
allows this -- see that module's own docstring). Persisting one is not a
special case here: step 2's transaction simply has nothing to insert
(trivially succeeds), and step 3a still records run_summary and
material_change_detected=False on the reflection_runs row. This module
never assumes material_change_detected=False implies zero observations,
or vice versa -- ReflectionResult's own "Zero-change Semantics" is
one-directional by design (material_change_detected=True requires at
least one entry; material_change_detected=False may still carry
proposal_observations entries, e.g. routine trend='stable'
re-observations recorded purely for continuity) and this module never
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
    'failed' status to. A single, minimal exception type, matching tools.
    case_enrichment_analyzer.CaseEnrichmentAnalyzerError's and tools.
    reflector_analyzer.ReflectorAnalyzerError's own "one exception type,
    message carries the detail" convention. Never raised for a step 2
    (Proposal/Observation) failure -- that failure is instead reported
    through this function's own return value (see
    ReflectionPersistenceOutcome.status='failed'), because a reflection_runs
    row DOES exist in that case and IS successfully marked 'failed';
    raising there would hide that this function still did useful, durable
    work (recording the failed attempt) before returning.
    """


@dataclass(frozen=True)
class ReflectionPersistenceOutcome:
    """The result of one persist_reflection_result() call.

    Deliberately minimal -- just enough for a caller (a future Slice 6B
    runner) to know what happened and to look up the full row via
    FeedbackStoreV2.get_reflection_run(reflection_run_id) if it needs more.
    Not a new domain type competing with ReflectionResult/ReflectionRun --
    this is a persistence-call receipt, nothing else.
    """

    reflection_run_id: str
    status: str  # "succeeded" | "failed" -- mirrors REFLECTION_RUN_STATUS_VALUES's own terminal values

    def __post_init__(self) -> None:
        if self.status not in ("succeeded", "failed"):
            raise ValueError(f"invalid ReflectionPersistenceOutcome.status: {self.status!r}")


# ---------------------------------------------------------------------------
# Dataclass -> plain-mapping projection (tools.feedback_store_v2 must not
# import tools.reflector_proposals -- that module already imports taxonomy
# constants FROM tools.feedback_store_v2, so the reverse import would be
# circular; this module does that projection instead, exactly how tools.
# run_case_enrichment already projects a CaseEnrichmentResult into tools.
# feedback_store_v2.create_case_analysis()'s own plain keyword arguments).
# ---------------------------------------------------------------------------


def _proposal_row(proposal: ImprovementProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "improvement_target": proposal.improvement_target,
        "title": proposal.title,
        "created_at": proposal.created_at,
    }


def _observation_row(observation: ProposalObservation) -> dict[str, Any]:
    """supporting_case_ids_json mirrors tools.run_case_enrichment.
    _evidence_to_json's exact serialization convention
    (json.dumps(..., ensure_ascii=False), no sort_keys -- there are no
    dict keys here to sort, only a list) -- observation.supporting_case_ids
    is already a deterministically deduped, sorted tuple (tools.
    reflector_proposals.ProposalObservation.__post_init__ enforces this),
    so this is a direct, order-preserving projection, never a re-sort."""
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
    -- this function never generates its own id, matching how tools.
    reflector_analyzer.analyze_reflection_with_llm() already requires
    `reflection_run_id` as a caller-supplied argument (Slice 5B) rather
    than generating one internally; the SAME id a caller passed to that
    function is expected to be passed through to this one, so the
    ReflectionResult's own identity and its reflection_runs row's identity
    are always the same value, never two independently-generated ids for
    the same Run.

    `started_at`/`completed_at`/`analyzed_case_count`/`reflector_version`/
    `window_start`/`window_end` are plain caller-supplied values, exactly
    like tools.feedback_store_v2.FeedbackStoreV2.create_reflection_run()'s
    own parameters -- this function performs no timestamp generation, no
    version resolution, and no re-derivation of analyzed_case_count from
    `result` (ReflectionResult carries no case-count field of its own; the
    ReflectorPromptContext/ReflectorInput that produced it does -- a
    future Slice 6B runner is expected to carry that number through, the
    same way it is expected to generate reflection_run_id/started_at/
    observed_at before calling analyze_reflection_with_llm() in the first
    place).

    Raises ReflectionPersistenceError only if create_reflection_run()
    itself fails (a duplicate reflection_run_id, or an invalid
    analyzed_case_count) -- see that exception's own docstring for why a
    step 2 (Proposal/Observation) failure is reported differently, through
    this function's return value instead.

    Never raises for a step 2 failure: this function catches nothing
    itself in that branch because there is nothing to catch --
    FeedbackStoreV2.persist_reflection_proposals() already never raises
    (see that method's own "returns False, never raises" contract), so
    this function's own control flow is a plain if/else on its boolean
    return value.
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
