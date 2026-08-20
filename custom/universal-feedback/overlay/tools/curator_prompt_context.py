"""The Curator Prompt Context projection -- Curator Slice 1's LLM-facing
input.

    ImprovementProposal row + latest ProposalObservation row
     + CaseIntelligenceRecord evidence (tools.case_reflection_input)
     + current /sandbox/AGENTS.md content
     |
     v
    build_curator_prompt_context()   (this module -- pure projection, no
     |                                 DB, no LLM)
     v
    CuratorPromptContext              (this module -- the LLM-facing
     |                                 contract)
     v
    serialize_curator_prompt_context()  (this module -- deterministic
     |                                    JSON string)
     v
    tools.curator_analyzer -- one call_llm() invocation

Deliberately narrower than tools.reflector_prompt_context.
ReflectorPromptContext: a Curator run reasons about exactly ONE already-
accepted Proposal, not a whole Case window plus a set of existing-Proposal
candidates, so there is no analysis_window/summary/existing_proposals
concept here -- only the one Proposal's narrative fields, its supporting
Case evidence, and the one file it may propose a change to.

This module takes no store/DB argument and issues no raw SQL -- it only
transforms already-fetched, already-typed values (a caller, tools.
run_curator, is responsible for turning sqlite3.Row objects into the plain
values/CaseIntelligenceRecord tuples this module's builder accepts).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Collection

from tools._validation import is_valid_confidence as _is_valid_confidence
from tools._validation import require_nonblank_str as _require_nonblank_str
from tools.case_reflection_input import CaseIntelligenceRecord
from tools.curator_domain import CURATOR_TARGET_FILE
from tools.reflector_prompt_context import CaseIntelligenceProjection

# ---------------------------------------------------------------------------
# CaseIntelligenceRecord -> CaseIntelligenceProjection
# ---------------------------------------------------------------------------


def _project_case(record: CaseIntelligenceRecord) -> CaseIntelligenceProjection:
    """CaseIntelligenceRecord -> CaseIntelligenceProjection: drop analysis_
    version/analyzed_at/source_evidence_watermark (audit/staleness
    metadata, not reasoning evidence -- same exclusion rationale as
    tools.reflector_prompt_context's own identically-named helper), carry
    everything else unchanged. A small, intentional duplication of that
    helper's trivial field mapping (not its validation logic, which stays
    centralized in CaseIntelligenceProjection.__post_init__) -- this
    module does not import a private symbol across a file boundary for an
    11-field constructor call.
    """
    return CaseIntelligenceProjection(
        case_id=record.case_id,
        case_title=record.case_title,
        issue_summary=record.issue_summary,
        product_model=record.product_model,
        product_source=record.product_source,
        product_confidence=record.product_confidence,
        issue_type=record.issue_type,
        issue_type_confidence=record.issue_type_confidence,
        diagnosis=record.diagnosis,
        diagnosis_confidence=record.diagnosis_confidence,
        evidence=record.evidence,
    )


# ---------------------------------------------------------------------------
# CuratorPromptContext -- the top-level LLM-facing contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CuratorPromptContext:
    """The complete, deterministic, LLM-facing projection tools.
    curator_analyzer reads: one accepted, agent_behavior Improvement
    Proposal + its latest Observation + the Case evidence behind that
    Observation + the current complete content of the one file Curator v1
    may propose a change to.

    improvement_target is deliberately NOT a field here: by the time this
    context is built, tools.run_curator's own deterministic guards have
    already confirmed improvement_target='agent_behavior' -- re-stating it
    as a context field would invite a future caller to imagine Curator
    branches on it, when in this slice it never does.

    unavailable_case_ids makes a resolution gap impossible to miss (same
    reasoning as ReflectorInput's cases_missing_analysis/cases_with_
    unparseable_analysis, collapsed into one field here since a Curator
    run's case_id list is small and explicit rather than a whole window --
    see tools.case_reflection_input.build_case_intelligence_for_ids's own
    docstring for why the two reasons are not split here either).
    """

    proposal_id: str
    title: str

    pattern_summary: str
    possible_cause: str | None
    recommended_improvement: str
    expected_benefit: str | None
    limitations: str | None
    observation_confidence: float
    observed_at: str

    supporting_cases: tuple[CaseIntelligenceProjection, ...]
    unavailable_case_ids: tuple[str, ...]

    target_file: str
    current_content: str

    def __post_init__(self) -> None:
        _require_nonblank_str(self.proposal_id, "proposal_id")
        _require_nonblank_str(self.title, "title")

        _require_nonblank_str(self.pattern_summary, "pattern_summary")
        if self.possible_cause is not None:
            _require_nonblank_str(self.possible_cause, "possible_cause")
        _require_nonblank_str(self.recommended_improvement, "recommended_improvement")
        if self.expected_benefit is not None:
            _require_nonblank_str(self.expected_benefit, "expected_benefit")
        if self.limitations is not None:
            _require_nonblank_str(self.limitations, "limitations")
        if not _is_valid_confidence(self.observation_confidence):
            raise ValueError(f"invalid observation_confidence: {self.observation_confidence!r}")
        _require_nonblank_str(self.observed_at, "observed_at")

        if not isinstance(self.supporting_cases, tuple) or not all(
            isinstance(c, CaseIntelligenceProjection) for c in self.supporting_cases
        ):
            raise ValueError("supporting_cases must be a tuple of CaseIntelligenceProjection")
        if not isinstance(self.unavailable_case_ids, tuple) or not all(
            isinstance(c, str) for c in self.unavailable_case_ids
        ):
            raise ValueError("unavailable_case_ids must be a tuple of str")

        if self.target_file != CURATOR_TARGET_FILE:
            raise ValueError(
                f"invalid target_file: {self.target_file!r}; Curator v1 may only "
                f"target {CURATOR_TARGET_FILE!r}"
            )
        if not isinstance(self.current_content, str):
            raise ValueError("current_content must be a string")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_curator_prompt_context(
    *,
    proposal_id: str,
    title: str,
    pattern_summary: str,
    possible_cause: str | None,
    recommended_improvement: str,
    expected_benefit: str | None,
    limitations: str | None,
    observation_confidence: float,
    observed_at: str,
    supporting_case_records: Collection[CaseIntelligenceRecord],
    unavailable_case_ids: Collection[str],
    current_content: str,
) -> CuratorPromptContext:
    """Project already-fetched, already-typed proposal/observation fields
    and Case evidence into one CuratorPromptContext.

    Pure function: no DB access, no raw SQL, no LLM call, no store
    argument -- the caller (tools.run_curator) is responsible for
    resolving the accepted Proposal row, its latest Observation row, and
    supporting_case_records (via tools.case_reflection_input.
    build_case_intelligence_for_ids) before calling this.

    target_file is never a parameter -- always tools.curator_domain.
    CURATOR_TARGET_FILE, the one constant this whole family of modules
    treats as fixed.

    Ordering: `supporting_cases` is sorted by case_id ascending (mirrors
    tools.reflector_prompt_context.build_reflector_prompt_context's own
    determinism guarantee); `unavailable_case_ids` is deduplicated and
    sorted.
    """
    supporting_cases = tuple(
        sorted((_project_case(record) for record in supporting_case_records), key=lambda c: c.case_id)
    )

    return CuratorPromptContext(
        proposal_id=proposal_id,
        title=title,
        pattern_summary=pattern_summary,
        possible_cause=possible_cause,
        recommended_improvement=recommended_improvement,
        expected_benefit=expected_benefit,
        limitations=limitations,
        observation_confidence=observation_confidence,
        observed_at=observed_at,
        supporting_cases=supporting_cases,
        unavailable_case_ids=tuple(sorted(set(unavailable_case_ids))),
        target_file=CURATOR_TARGET_FILE,
        current_content=current_content,
    )


# ---------------------------------------------------------------------------
# Deterministic JSON serialization
# ---------------------------------------------------------------------------


def serialize_curator_prompt_context(context: CuratorPromptContext) -> str:
    """Deterministic JSON string for one CuratorPromptContext.

    Unlike tools.reflector_prompt_context.serialize_reflector_prompt_
    context, this context carries no types.MappingProxyType field, so
    plain dataclasses.asdict() (which recurses into nested dataclasses,
    including the CaseAnalysisEvidence tuples nested inside each
    CaseIntelligenceProjection) is sufficient -- no hand-rolled `_to_json_
    ready` needed. sort_keys=True normalizes dict key order; ensure_ascii=
    False preserves non-ASCII text (e.g. Chinese pattern_summary content)
    as real Unicode rather than \\uXXXX escapes.
    """
    return json.dumps(asdict(context), ensure_ascii=False, sort_keys=True)


__all__ = [
    "CaseIntelligenceProjection",
    "CuratorPromptContext",
    "build_curator_prompt_context",
    "serialize_curator_prompt_context",
]
