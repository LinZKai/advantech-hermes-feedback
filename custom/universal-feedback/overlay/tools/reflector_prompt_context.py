"""The Reflector Prompt Context projection.

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

Projects two already-built internal domain objects (ReflectorInput and
Collection[ProposalCandidate]) into a single, minimal, deterministic,
LLM-facing contract, and serializes that contract to a JSON string. No LLM
call, no prompt text, no structured-output parser, no semantic/fuzzy/
embedding matching, no ReflectionResult generation, no persistence.

ReflectorPromptContext is explicitly NOT a DB schema, a raw sqlite3.Row, a
raw ReflectorInput dump, a ReflectionResult, a ReflectionRun, or an
ImprovementProposal persistence model -- it is a narrower, LLM-facing VIEW
derived from those, built fresh every time and never itself persisted.

This module takes no store/DB argument and issues no raw SQL -- it only
transforms already-built in-memory objects.

Operational metadata: ReflectorInput carries five window-level numbers
(window_case_count, analyzed_case_count, coverage_ratio, cases_missing_
analysis, cases_with_unparseable_analysis). Only analyzed_case_count is
projected into the LLM-facing context (as summary.analyzed_case_count);
the other four are pipeline health/observability metadata (whether Case
Enrichment kept up with the window), not evidence about the recurring
patterns the Reflector reasons over, so mixing them in risks the Reflector
conflating "enrichment coverage was incomplete" with "the issue is rare."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Collection, Mapping

from tools._validation import is_valid_confidence as _is_valid_confidence
from tools._validation import require_nonblank_str as _require_nonblank_str
from tools.case_enrichment import CaseAnalysisEvidence
from tools.case_reflection_input import CaseIntelligenceRecord, ReflectorInput
from tools.feedback_store_v2 import DIAGNOSIS_VALUES, ISSUE_TYPE_VALUES, PRODUCT_SOURCE_VALUES
from tools.proposal_matching import ProposalCandidate

_ISSUE_TYPE_VALUE_SET = frozenset(ISSUE_TYPE_VALUES)
_DIAGNOSIS_VALUE_SET = frozenset(DIAGNOSIS_VALUES)
_PRODUCT_SOURCE_VALUE_SET = frozenset(PRODUCT_SOURCE_VALUES)


# ---------------------------------------------------------------------------
# AnalysisWindow -- the Case occurrence horizon, unchanged semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisWindow:
    """The Case occurrence horizon this context analyzes -- a direct,
    unchanged projection of ReflectorInput.window_start/window_end.

    `start` is INCLUSIVE, `end` is EXCLUSIVE -- identical semantics to
    ReflectorInput's own window_start/window_end; this dataclass only
    renames the two fields to match this context's JSON shape.
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
    into this context -- see this module's own docstring for why
    window_case_count/coverage_ratio/the two gap-id lists are excluded.

    by_product_model/by_issue_type/by_diagnosis are plain dict, not
    types.MappingProxyType -- ReflectorInput's own equivalents are
    MappingProxyType, but that is not directly JSON-serializable
    (json.dumps() raises TypeError on one) and this context's whole
    purpose is to become JSON. build_reflector_prompt_context() converts
    each via dict(...) at projection time; serialize_reflector_prompt_
    context() independently converts any Mapping it encounters too.
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
    evidence_watermark -- all three are audit/stale-detection/internal-
    traceability metadata, not recurring-pattern reasoning evidence.

    `evidence` reuses tools.case_enrichment.CaseAnalysisEvidence directly:
    that dataclass is already exactly the typed, immutable, minimal
    (type/turn_id/fact) shape a Reflector needs to see WHY a Case was
    judged the way it was, not just its final issue_type/diagnosis verdict.

    Structurally re-validates every field against the same taxonomies/
    ranges tools.case_reflection_input.CaseIntelligenceRecord already
    enforces -- every boundary in this domain defends itself rather than
    trusting an upstream layer unconditionally.
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
    unchanged. evidence ordering is inherited as-is, never re-sorted."""
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
    summary.

    Together `cases` and `existing_proposals` are what let a future
    Reflector reasoning step judge, for each recurring pattern it finds,
    whether it is the SAME underlying improvement opportunity as an
    existing Candidate or a genuinely new one -- see tools.
    proposal_matching's module docstring for the full Proposal Identity
    Rule this context exists to support (title equality is NOT the rule).

    No self-improvement EXECUTION fields exist here (no change_strategy,
    validation_plan, candidate_patch, test_result, deployment_plan) -- only
    ProposalCandidate.improvement_target, so a future pipeline CAN route
    different targets differently, but no such routing is built here.

    Cross-object invariants (checked here, not in any single nested
    dataclass, since no single nested object can see both `cases` and
    `summary` together):
      * summary.analyzed_case_count == len(cases)
      * no duplicate case_id across `cases`
      * no duplicate proposal_id across `existing_proposals`
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
    and never mutates either input.

    Ordering: `cases` inherits ReflectorInput.cases's own order UNCHANGED
    (already deterministic, case_id ascending). `existing_proposals` is
    explicitly re-sorted by proposal_id here, since `proposal_candidates`
    is a generic Collection with no ordering guarantee of its own -- a
    caller could pass a set, or a reversed list, and still get identical,
    deterministic JSON out of this function.

    Never catches exceptions from constructing the nested dataclasses --
    a genuinely malformed input (e.g. duplicate proposal_id in
    `proposal_candidates`) is a real bug worth surfacing immediately as a
    raised ValueError.
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
    (recursing into each field), Mapping (including types.MappingProxyType)
    -> plain dict, list/tuple -> list, everything else unchanged.
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

    sort_keys=True normalizes dict key order; ensure_ascii=False preserves
    non-ASCII text (e.g. Chinese pattern_summary content) as real Unicode
    rather than \\uXXXX escapes. Every list-valued field this context
    carries is already deterministically ordered by the time it reaches
    this function (see build_reflector_prompt_context's own ordering
    guarantees), so combined with sort_keys=True the same semantic
    ReflectorPromptContext always serializes to the exact same string.
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
