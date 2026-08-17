"""Phase 5 Slice 4: the Reflector Prompt Context projection.

    ReflectorInput                          Collection[ProposalCandidate]
    (tools.case_reflection_input)           (tools.proposal_matching)
     |                                        |
     v                                        v
    build_reflector_prompt_context()   (this module -- pure projection,
                |                        no DB, no LLM)
                v
    ReflectorPromptContext             (this module -- the LLM-facing
                |                        contract)
                v
    serialize_reflector_prompt_context()  (this module -- deterministic
                |                           JSON string)
                v
    [ future Reflector prompt / LLM call -- NOT implemented here ]

Slice 4 scope only: project two already-built internal domain objects
(ReflectorInput, the Case Intelligence input Slice 1 assembles; and
Collection[ProposalCandidate], the existing-Proposal candidates Slice 3
assembles) into a single, minimal, deterministic, LLM-facing contract, and
serialize that contract to a JSON string. No LLM call, no prompt text, no
structured-output parser, no semantic/fuzzy/embedding matching, no
ReflectionResult generation, no persistence, no runner, no scheduler.

    internal domain model  (ReflectorInput / ProposalCandidate / ...)
            |
            v
    LLM-facing projection  (ReflectorPromptContext -- THIS module)
            |
            v
    future Reflector prompt  (NOT built here)

ReflectorPromptContext is explicitly NOT any of: a DB schema, a raw
sqlite3.Row, a raw ReflectorInput dump, a ReflectionResult, a
ReflectionRun, or an ImprovementProposal persistence model -- it is a
narrower, LLM-facing VIEW derived from those, built fresh every time by
build_reflector_prompt_context() and never itself persisted.

This module takes no store/DB argument and issues no raw SQL -- it only
transforms already-built in-memory objects, matching tools.case_
reflection_input's and tools.proposal_matching's own "glue, not storage"
layering (see each module's docstring) one level further removed from
storage than either of them.

---------------------------------------------------------------------------
Operational metadata decision (Phase 5 task instruction, section 9)
---------------------------------------------------------------------------

ReflectorInput carries five window-level numbers: window_case_count,
analyzed_case_count, coverage_ratio, cases_missing_analysis, cases_with_
unparseable_analysis. None of ReflectorInput's own fields are removed or
changed by this decision -- this module only decides what to PROJECT into
the LLM-facing context:

    INCLUDED: analyzed_case_count (as summary.analyzed_case_count)
    EXCLUDED: window_case_count, coverage_ratio,
              cases_missing_analysis, cases_with_unparseable_analysis

Reasoning: the Reflector's actual reasoning job is to find recurring
patterns across Cases that HAVE usable Case Intelligence -- exactly the
Cases already present in `cases`. analyzed_case_count is semantic evidence
context (it tells the Reflector how many Cases its own `cases` list
represents, directly supporting pattern-strength judgment -- "3 out of 3
analyzed Cases" reads very differently from "3 out of 300"... except that
second number, window_case_count, is deliberately NOT given either,
because window_case_count/coverage_ratio/the two gap-id lists are pipeline
HEALTH/OBSERVABILITY metadata: whether Case Enrichment kept up with the
window, not a fact about the recurring-pattern evidence itself. Mixing
pipeline-health signals into the same context the Reflector reasons over
about PATTERNS risks the Reflector conflating "enrichment coverage was
incomplete this window" with "the underlying issue is rare" -- two
unrelated claims. A future batch runner (not built in this slice) is the
right place to read window_case_count/coverage_ratio/the gap lists and
decide operational things (retry Case Enrichment, alert on low coverage,
delay the run) -- never the Reflector's own reasoning input.

---------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Collection, Mapping

from tools.case_enrichment import CaseAnalysisEvidence
from tools.case_reflection_input import CaseIntelligenceRecord, ReflectorInput
from tools.feedback_store_v2 import DIAGNOSIS_VALUES, ISSUE_TYPE_VALUES, PRODUCT_SOURCE_VALUES
from tools.proposal_matching import ProposalCandidate

_ISSUE_TYPE_VALUE_SET = frozenset(ISSUE_TYPE_VALUES)
_DIAGNOSIS_VALUE_SET = frozenset(DIAGNOSIS_VALUES)
_PRODUCT_SOURCE_VALUE_SET = frozenset(PRODUCT_SOURCE_VALUES)


def _is_valid_confidence(value: Any) -> bool:
    """True only for a real, finite number in [0.0, 1.0].

    Deliberately its own local copy -- see tools.case_reflection_input.
    _is_valid_confidence's docstring for why every module in this domain
    keeps its own copy rather than importing a sibling module's private
    helper.
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


# ---------------------------------------------------------------------------
# AnalysisWindow -- the Case occurrence horizon, unchanged semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisWindow:
    """The Case occurrence horizon this context analyzes -- a direct,
    unchanged projection of ReflectorInput.window_start/window_end.

    `start` is INCLUSIVE, `end` is EXCLUSIVE -- identical semantics to
    ReflectorInput's own window_start/window_end (see that class's
    docstring); this dataclass does not re-derive or reinterpret the
    boundary rule, only renames the two fields to match this context's
    JSON shape (Phase 5 task instruction, section 8: "analysis_window":
    {"start": ..., "end": ...}).

    This is explicitly NOT: a ReflectionRun's started_at/completed_at
    (execution time), a case_analysis.analyzed_at value, or a
    source_evidence_watermark -- see this module's and ReflectorInput's
    own docstrings for why those are different concepts entirely.
    """

    start: str | None
    end: str | None

    def __post_init__(self) -> None:
        if self.start is not None and not isinstance(self.start, str):
            raise ValueError("start must be a string or None")
        if self.end is not None and not isinstance(self.end, str):
            raise ValueError("end must be a string or None")


# ---------------------------------------------------------------------------
# ReflectorContextSummary -- operational count + deterministic aggregates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectorContextSummary:
    """Deterministic, program-computed facts about `cases` -- the program
    counts, the Reflector interprets (Phase 5 task instruction, section
    10). Never itself a claim about which pattern matters; that judgment
    is explicitly deferred to a future Reflector reasoning step.

    analyzed_case_count is the ONE operational-metadata number projected
    into this context -- see this module's own docstring, "Operational
    metadata decision", for why window_case_count/coverage_ratio/the two
    gap-id lists are deliberately excluded.

    by_product_model/by_issue_type/by_diagnosis are plain dict, not
    types.MappingProxyType -- ReflectorInput's own equivalents are
    MappingProxyType (an intentionally immutable view at THAT layer, see
    ReflectorInput's docstring), but a mappingproxy is not directly
    JSON-serializable (json.dumps() raises TypeError on one) and this
    context's whole purpose is to become JSON. build_reflector_prompt_
    context() converts each via dict(...) at projection time (Phase 5 task
    instruction, section 10: "Projection 時 MappingProxyType 必須轉成普通
    dict"); serialize_reflector_prompt_context() independently converts
    any Mapping it encounters too, so this is never assumed to have
    already happened by the time serialization runs.
    """

    analyzed_case_count: int
    by_product_model: Mapping[str, int]
    by_issue_type: Mapping[str, int]
    by_diagnosis: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.analyzed_case_count, int)
            or isinstance(self.analyzed_case_count, bool)
            or self.analyzed_case_count < 0
        ):
            raise ValueError(f"invalid analyzed_case_count: {self.analyzed_case_count!r}")

        for name, mapping in (
            ("by_product_model", self.by_product_model),
            ("by_issue_type", self.by_issue_type),
            ("by_diagnosis", self.by_diagnosis),
        ):
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{name} must be a Mapping")
            total = sum(mapping.values())
            if total != self.analyzed_case_count:
                raise ValueError(
                    f"{name} counts ({total}) must sum to analyzed_case_count "
                    f"({self.analyzed_case_count})"
                )


# ---------------------------------------------------------------------------
# CaseIntelligenceProjection -- one Case's LLM-facing view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseIntelligenceProjection:
    """One Case's LLM-facing projection of its CaseIntelligenceRecord.

    Deliberately excludes analysis_version, analyzed_at, and source_
    evidence_watermark (Phase 5 task instruction, section 11) -- all three
    are audit/stale-detection/internal-traceability metadata, not
    recurring-pattern reasoning evidence; no concrete Reflector reasoning
    use case was found for any of them during this slice's read-only
    audit, so per the task instruction's own default preference ("exclude
    all three" unless a specific use case is found), they are excluded.

    `evidence` reuses tools.case_enrichment.CaseAnalysisEvidence directly
    (Phase 5 task instruction, section 5: "如果 CaseAnalysisEvidence 已有
    明確 semantic fields，優先重用") -- that dataclass is already exactly
    the typed, immutable, minimal (type/turn_id/fact) shape a Reflector
    needs to see WHY a Case was judged the way it was, not just its final
    issue_type/diagnosis verdict. No new evidence type was created.

    Structurally re-validates every field against the same taxonomies/
    ranges tools.case_reflection_input.CaseIntelligenceRecord already
    enforces -- belt-and-suspenders, matching this domain's established
    convention that every boundary defends itself rather than trusting an
    upstream layer unconditionally (see CaseIntelligenceRecord's own
    docstring for the precedent this mirrors).
    """

    case_id: str
    case_title: str | None
    issue_summary: str | None
    product_model: str | None
    product_source: str | None
    product_confidence: float | None
    issue_type: str
    issue_type_confidence: float
    diagnosis: str
    diagnosis_confidence: float
    evidence: tuple[CaseAnalysisEvidence, ...]

    def __post_init__(self) -> None:
        _require_nonblank_str(self.case_id, "case_id")

        if self.case_title is not None and not isinstance(self.case_title, str):
            raise ValueError("case_title must be a string or None")
        if self.issue_summary is not None and not isinstance(self.issue_summary, str):
            raise ValueError("issue_summary must be a string or None")

        if self.issue_type not in _ISSUE_TYPE_VALUE_SET:
            raise ValueError(f"invalid issue_type: {self.issue_type!r}")
        if not _is_valid_confidence(self.issue_type_confidence):
            raise ValueError(f"invalid issue_type_confidence: {self.issue_type_confidence!r}")

        if self.diagnosis not in _DIAGNOSIS_VALUE_SET:
            raise ValueError(f"invalid diagnosis: {self.diagnosis!r}")
        if not _is_valid_confidence(self.diagnosis_confidence):
            raise ValueError(f"invalid diagnosis_confidence: {self.diagnosis_confidence!r}")

        if self.product_model is None:
            if self.product_source is not None:
                raise ValueError("product_source must be None when product_model is None")
            if self.product_confidence is not None:
                raise ValueError("product_confidence must be None when product_model is None")
        else:
            _require_nonblank_str(self.product_model, "product_model")
            if self.product_source not in _PRODUCT_SOURCE_VALUE_SET:
                raise ValueError(f"invalid product_source: {self.product_source!r}")
            if not _is_valid_confidence(self.product_confidence):
                raise ValueError(f"invalid product_confidence: {self.product_confidence!r}")

        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, CaseAnalysisEvidence) for item in self.evidence
        ):
            raise ValueError("evidence must be a tuple of CaseAnalysisEvidence")


def _project_case(record: CaseIntelligenceRecord) -> CaseIntelligenceProjection:
    """CaseIntelligenceRecord -> CaseIntelligenceProjection: drop analysis_
    version/analyzed_at/source_evidence_watermark, carry everything else
    unchanged. evidence ordering is inherited as-is (already deterministic
    -- a fixed JSON array order recovered from storage, see
    CaseIntelligenceRecord's own docstring), never re-sorted here."""
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
# ReflectorPromptContext -- the top-level LLM-facing contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectorPromptContext:
    """The complete, deterministic, LLM-facing projection a future
    Reflector prompt reads: Current Case Intelligence + Existing Proposal
    Candidates + the analysis window they both belong to + a deterministic
    summary (Phase 5 task instruction, section 3).

    Together `cases` and `existing_proposals` are what let a future
    Reflector reasoning step judge, for each recurring pattern it finds,
    whether it is the SAME underlying improvement opportunity as an
    existing Candidate or a genuinely new one -- see tools.
    proposal_matching's module docstring for the full Proposal Identity
    Rule this context exists to support (title equality is explicitly NOT
    the rule; that judgment is not made here or anywhere in this slice).

    No self-improvement EXECUTION fields exist here or anywhere in this
    module (Phase 5 task instruction, section 16) -- no change_strategy,
    validation_plan, candidate_patch, test_result, or deployment_plan.
    Each CaseIntelligenceProjection/ProposalCandidate still carries
    improvement_target (product_model/... for Cases is Case-level, not
    improvement-level; ProposalCandidate.improvement_target is the
    relevant field here) so a future pipeline CAN route different targets
    differently (knowledge -> KB draft, agent_behavior -> Skill/behavior
    candidate change, retrieval -> retrieval improvement, workflow ->
    workflow improvement, other -> manual investigation) -- but no such
    routing, patch generation, or execution flow is built in this slice.

    Cross-object invariants (checked here, not in any single nested
    dataclass, since no single nested object can see both `cases` and
    `summary` together):
      * summary.analyzed_case_count == len(cases)
      * no duplicate case_id across `cases`
      * no duplicate proposal_id across `existing_proposals` (mirrors
        tools.reflector_proposals.ReflectionResult's own "at most one
        Observation per proposal_id" precedent for the same reason: a
        collection-level invariant no single element's own __post_init__
        can enforce on its own)
    """

    analysis_window: AnalysisWindow
    summary: ReflectorContextSummary
    cases: tuple[CaseIntelligenceProjection, ...]
    existing_proposals: tuple[ProposalCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_window, AnalysisWindow):
            raise ValueError("analysis_window must be an AnalysisWindow")
        if not isinstance(self.summary, ReflectorContextSummary):
            raise ValueError("summary must be a ReflectorContextSummary")

        if not isinstance(self.cases, tuple) or not all(
            isinstance(c, CaseIntelligenceProjection) for c in self.cases
        ):
            raise ValueError("cases must be a tuple of CaseIntelligenceProjection")
        if not isinstance(self.existing_proposals, tuple) or not all(
            isinstance(p, ProposalCandidate) for p in self.existing_proposals
        ):
            raise ValueError("existing_proposals must be a tuple of ProposalCandidate")

        if self.summary.analyzed_case_count != len(self.cases):
            raise ValueError(
                f"summary.analyzed_case_count ({self.summary.analyzed_case_count}) must equal "
                f"len(cases) ({len(self.cases)})"
            )

        case_ids = [c.case_id for c in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("cases must not contain duplicate case_id values")

        proposal_ids = [p.proposal_id for p in self.existing_proposals]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("existing_proposals must not contain duplicate proposal_id values")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_reflector_prompt_context(
    reflector_input: ReflectorInput,
    proposal_candidates: Collection[ProposalCandidate],
) -> ReflectorPromptContext:
    """Project an already-built ReflectorInput and an already-built
    Collection of ProposalCandidate into one ReflectorPromptContext.

    Pure function: no DB access, no raw SQL, no storage call, no LLM call,
    and never mutates either input (both are read-only here -- `cases` is
    read via ReflectorInput.cases, never re-queried; `proposal_candidates`
    is only iterated/sorted into a new tuple, its own elements untouched).

    Ordering:
      * `cases` inherits ReflectorInput.cases's own order UNCHANGED --
        that field is a concrete, already-deterministic tuple on a single
        typed input object (case_id ascending, guaranteed by tools.
        case_reflection_input.build_reflector_input -- see that module's
        docstring), so re-sorting here would be redundant, not safer.
      * `existing_proposals` is explicitly re-sorted by proposal_id here,
        NEVER trusting `proposal_candidates`'s own incidental iteration
        order -- unlike `reflector_input.cases`, this parameter's type is
        a generic Collection[ProposalCandidate] with no ordering guarantee
        at all (a caller could pass a set, or a reversed list, and still
        get identical, deterministic JSON out of this function).

    Never catches exceptions from constructing the nested dataclasses --
    a genuinely malformed input (e.g. duplicate proposal_id in
    `proposal_candidates`) is a real bug worth surfacing immediately as a
    raised ValueError, matching how tools.case_reflection_input.
    build_reflector_input and tools.proposal_matching.
    build_proposal_candidates both let their own real data-integrity
    failures propagate rather than inventing a new silent-failure mode.
    """
    analysis_window = AnalysisWindow(start=reflector_input.window_start, end=reflector_input.window_end)

    summary = ReflectorContextSummary(
        analyzed_case_count=reflector_input.analyzed_case_count,
        by_product_model=dict(reflector_input.by_product_model),
        by_issue_type=dict(reflector_input.by_issue_type),
        by_diagnosis=dict(reflector_input.by_diagnosis),
    )

    cases = tuple(_project_case(record) for record in reflector_input.cases)

    existing_proposals = tuple(sorted(proposal_candidates, key=lambda c: c.proposal_id))

    return ReflectorPromptContext(
        analysis_window=analysis_window,
        summary=summary,
        cases=cases,
        existing_proposals=existing_proposals,
    )


# ---------------------------------------------------------------------------
# Deterministic JSON serialization
# ---------------------------------------------------------------------------


def _to_json_ready(value: Any) -> Any:
    """Recursively convert a ReflectorPromptContext (or any value inside
    it) into a plain, json.dumps()-ready structure: dataclass -> dict
    (recursing into each field), Mapping (including types.MappingProxyType
    -- never assumed to have already been converted upstream, see
    ReflectorContextSummary's own docstring) -> plain dict, list/tuple ->
    list, everything else (str/int/float/bool/None) -> itself unchanged,
    so None round-trips to JSON null exactly as json.dumps() already
    handles natively.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json_ready(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(v) for v in value]
    return value


def serialize_reflector_prompt_context(context: ReflectorPromptContext) -> str:
    """Deterministic JSON string for one ReflectorPromptContext.

    json.dumps(..., ensure_ascii=False, sort_keys=True) -- the exact
    convention tools.case_enrichment_analyzer._build_messages already
    established for the equivalent Phase 4.5 evidence-serialization step
    (see that function's docstring): sort_keys=True normalizes dict *key*
    order; ensure_ascii=False preserves non-ASCII text (e.g. Chinese
    pattern_summary content) as real Unicode rather than \\uXXXX escapes.
    No separators/indent/default override -- none of those are used by
    the precedent this mirrors, and none are needed here either.

    Every list-valued field this context carries (`cases`,
    `existing_proposals`, and each Case's `evidence`) is already
    deterministically ordered by the time it reaches this function (see
    build_reflector_prompt_context's own ordering guarantees) -- this
    function does not re-sort anything; combined with sort_keys=True for
    dict keys, the same semantic ReflectorPromptContext always serializes
    to the exact same string, regardless of how its `cases`/
    `existing_proposals` tuples were originally constructed by the caller.
    """
    return json.dumps(_to_json_ready(context), ensure_ascii=False, sort_keys=True)


__all__ = [
    "AnalysisWindow",
    "ReflectorContextSummary",
    "CaseIntelligenceProjection",
    "ReflectorPromptContext",
    "build_reflector_prompt_context",
    "serialize_reflector_prompt_context",
]
