"""The Cross-case Reflector input contract.

    FeedbackStoreV2
     |
     v
    list_case_ids_created_in_window()   (tools.feedback_store_v2, unchanged)
     |
     v
    list_latest_case_analysis()         (tools.feedback_store_v2, unchanged)
     |
     v
    _build_case_intelligence_record()   (this module -- parse/validate each
     |                                    row, or classify the gap)
     v
    deterministic aggregation           (this module -- pure counting)
     |
     v
    ReflectorInput

Assembles the deterministic input a future Reflector LLM call needs. No
LLM call, no prompt, no ImprovementProposal, no persistence, no scheduler.
Storage queries stay in tools.feedback_store_v2 -- this module issues no
raw SQL of its own.

Case identity is session-scoped: Case Routing never merges Cases across
Sessions, and cases.created_at is set once, at Case creation, and never
rewritten -- so it reliably means "this independent Case occurrence began
at this time". This module's analysis window is always a cases.created_at
range -- never cases.updated_at (that column is never actually updated by
anything) and never case_analysis.source_evidence_watermark (that field's
responsibility stays Case Enrichment staleness, not Case occurrence time).

Cross-Case pattern detection itself (similarity, recurrence, proposals) is
explicitly NOT this module's job -- this module only assembles the
deterministic facts a future Reflector LLM step will reason over.

Serialization boundary: ReflectorInput.by_product_model/by_issue_type/
by_diagnosis are types.MappingProxyType, an intentionally immutable view --
NOT a JSON-ready shape. json.dumps() raises TypeError on a mappingproxy
directly; a future serializer must convert each via dict(...) first. This
module deliberately builds no such serializer.

Eligibility: is_reflection_eligible() below is a deliberately separate,
tiny, LLM-free function -- ReflectorInput is a pure evidence/data contract
with no opinion on whether a Reflector run should actually proceed;
"should we run Reflector now" is an orchestration POLICY decision, so the
threshold is a caller-supplied parameter, never a ReflectorInput field.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Collection, Mapping

from tools._validation import is_valid_confidence as _is_valid_confidence
from tools._validation import require_nonblank_str as _require_nonblank_str
from tools.case_enrichment import CaseAnalysisEvidence
from tools.feedback_store_v2 import (
    DIAGNOSIS_VALUES,
    FeedbackStoreV2,
    ISSUE_TYPE_VALUES,
    PRODUCT_SOURCE_VALUES,
)

logger = logging.getLogger(__name__)

_ISSUE_TYPE_VALUE_SET = frozenset(ISSUE_TYPE_VALUES)
_DIAGNOSIS_VALUE_SET = frozenset(DIAGNOSIS_VALUES)
_PRODUCT_SOURCE_VALUE_SET = frozenset(PRODUCT_SOURCE_VALUES)

# Deterministic bucket key for a Case whose latest case_analysis has
# product_model=None -- chosen to be a value no real product model string
# could plausibly collide with (dunder-wrapped), matching the sentinel
# style already used elsewhere in this codebase for "this is a fixed marker,
# never a real observed value" (see e.g. tools.feedback_store_v2's fixed
# _RETRIEVAL_INSERT_FAILED_REASON reason string).
UNKNOWN_PRODUCT_MODEL_BUCKET = "__unknown__"

_EVIDENCE_REQUIRED_KEYS = frozenset({"type", "turn_id", "fact"})


# ---------------------------------------------------------------------------
# Case Intelligence record -- one Case's latest case_analysis, typed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseIntelligenceRecord:
    """One Case's latest case_analysis row, reconstructed into a typed,
    validated shape -- never a raw sqlite3.Row exposed as part of the
    ReflectorInput public contract.

    Structurally re-validates every field against the same taxonomies/
    ranges enforced at WRITE time (create_case_analysis() /
    CaseEnrichmentResult) -- every boundary in this codebase defends
    itself rather than trusting an upstream layer unconditionally.

    Unlike CaseEnrichmentResult (the write-time contract, which requires
    at least one evidence entry), `evidence` here MAY be empty:
    case_analysis.evidence_json is a nullable TEXT column with no
    non-empty constraint at the storage layer, so a row written directly
    through FeedbackStoreV2.create_case_analysis() (bypassing the
    analyzer, e.g. in a test) can legitimately have no evidence recorded.
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
    analysis_version: str
    analyzed_at: str
    source_evidence_watermark: str

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

        _require_nonblank_str(self.analysis_version, "analysis_version")
        _require_nonblank_str(self.analyzed_at, "analyzed_at")
        _require_nonblank_str(self.source_evidence_watermark, "source_evidence_watermark")


def _parse_evidence_json(evidence_json: str | None) -> tuple[CaseAnalysisEvidence, ...] | None:
    """Parse case_analysis.evidence_json into a tuple of CaseAnalysisEvidence,
    or None if it is present but malformed.

    None (the column's own "no evidence recorded" state) maps to an empty
    tuple, never to a parse failure -- a missing value is not the same as
    an invalid one. Any other problem (invalid JSON, wrong root type, an
    item with the wrong keys, or a __post_init__ rejection) returns None
    so the caller can classify this Case's latest analysis as unparseable
    rather than silently dropping or guessing at its evidence.
    """
    if evidence_json is None:
        return ()
    try:
        parsed = json.loads(evidence_json)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None

    items: list[CaseAnalysisEvidence] = []
    try:
        for item in parsed:
            if not isinstance(item, Mapping) or set(item.keys()) != _EVIDENCE_REQUIRED_KEYS:
                return None
            items.append(CaseAnalysisEvidence(
                type=item.get("type"), turn_id=item.get("turn_id"), fact=item.get("fact"),
            ))
    except (ValueError, TypeError):
        return None
    return tuple(items)


def _build_case_intelligence_record(row: Any) -> CaseIntelligenceRecord | None:
    """Reconstruct one CaseIntelligenceRecord from one
    list_latest_case_analysis() row, or None if it fails to parse/validate.

    Never raises -- fail closed. The caller (build_reflector_input) is
    responsible for recording which case_id this was for, in
    ReflectorInput.cases_with_unparseable_analysis -- this function itself
    has no case_id-tracking responsibility beyond logging.
    """
    evidence = _parse_evidence_json(row["evidence_json"])
    if evidence is None:
        logger.debug(
            "case_analysis row for case %r has unparseable evidence_json", row["case_id"],
        )
        return None
    try:
        return CaseIntelligenceRecord(
            case_id=row["case_id"],
            case_title=row["case_title"],
            issue_summary=row["issue_summary"],
            product_model=row["product_model"],
            product_source=row["product_source"],
            product_confidence=row["product_confidence"],
            issue_type=row["issue_type"],
            issue_type_confidence=row["issue_type_confidence"],
            diagnosis=row["diagnosis"],
            diagnosis_confidence=row["diagnosis_confidence"],
            evidence=evidence,
            analysis_version=row["analysis_version"],
            analyzed_at=row["analyzed_at"],
            source_evidence_watermark=row["source_evidence_watermark"],
        )
    except (ValueError, TypeError):
        logger.debug(
            "case_analysis row for case %r failed CaseIntelligenceRecord validation",
            row["case_id"], exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# ReflectorInput -- the Slice 1 output contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectorInput:
    """One Reflector analysis window's full deterministic input.

    `window_start`/`window_end` are carried through unchanged for
    traceability -- window_start is INCLUSIVE, window_end is EXCLUSIVE,
    matching FeedbackStoreV2.list_case_ids_created_in_window's own
    [start, end) contract exactly.

    `cases` holds every window Case's CaseIntelligenceRecord that was
    successfully reconstructed -- the actual Case Intelligence a future
    Reflector LLM step reads, not just aggregate counts.

    Two gap-tracking fields make missing/broken data structurally
    impossible to miss. Neither is a permanent verdict on the Case -- a
    later batch's Case Enrichment retry can still resolve either gap:
      * `cases_missing_analysis` -- case_id of every window Case with NO
        case_analysis row at all yet.
      * `cases_with_unparseable_analysis` -- case_id of every window Case
        whose latest case_analysis row exists but failed to reconstruct
        into a valid CaseIntelligenceRecord.
    Neither bucket is included in `cases`, `analyzed_case_count`, or any
    by_* aggregation.

    Count semantics:
      * `window_case_count` -- how many Cases actually occurred in this
        window, regardless of analysis status.
      * `analyzed_case_count` -- how many of those Cases are actually
        USABLE by a Reflector step right now, i.e. exactly `len(cases)`.
    These are deliberately two different numbers -- reading
    `analyzed_case_count` alone as "how many Cases happened this window"
    would understate the window whenever coverage is incomplete. Exact
    invariant (checked below, not just documented):

        window_case_count
        == analyzed_case_count
         + len(cases_missing_analysis)
         + len(cases_with_unparseable_analysis)

    `coverage_ratio` is a deterministic, purely observational metric --
    analyzed_case_count / window_case_count, or 0.0 when
    window_case_count is 0 (an empty window has nothing to cover, not
    "100% coverage of nothing"). NOT used as a hard gate anywhere in this
    module -- see is_reflection_eligible() for the actual eligibility
    decision, which deliberately does NOT read coverage_ratio at all.

    `by_product_model`/`by_issue_type`/`by_diagnosis` are computed ONLY
    over `cases` and sum to `analyzed_case_count` -- NEVER
    `window_case_count`. by_* values are immutable (types.MappingProxyType
    -- a serialization boundary, not a JSON-ready shape) and key-sorted
    for deterministic iteration; by_product_model uses
    UNKNOWN_PRODUCT_MODEL_BUCKET for a Case with product_model=None.
    """

    window_start: str | None
    window_end: str | None
    cases: tuple[CaseIntelligenceRecord, ...]
    cases_missing_analysis: tuple[str, ...]
    cases_with_unparseable_analysis: tuple[str, ...]
    window_case_count: int
    analyzed_case_count: int
    coverage_ratio: float
    by_product_model: Mapping[str, int]
    by_issue_type: Mapping[str, int]
    by_diagnosis: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.window_start is not None and not isinstance(self.window_start, str):
            raise ValueError("window_start must be a string or None")
        if self.window_end is not None and not isinstance(self.window_end, str):
            raise ValueError("window_end must be a string or None")

        if not isinstance(self.cases, tuple) or not all(
            isinstance(c, CaseIntelligenceRecord) for c in self.cases
        ):
            raise ValueError("cases must be a tuple of CaseIntelligenceRecord")
        if not isinstance(self.cases_missing_analysis, tuple) or not all(
            isinstance(c, str) for c in self.cases_missing_analysis
        ):
            raise ValueError("cases_missing_analysis must be a tuple of str")
        if not isinstance(self.cases_with_unparseable_analysis, tuple) or not all(
            isinstance(c, str) for c in self.cases_with_unparseable_analysis
        ):
            raise ValueError("cases_with_unparseable_analysis must be a tuple of str")

        if self.analyzed_case_count != len(self.cases):
            raise ValueError(
                f"analyzed_case_count ({self.analyzed_case_count}) must equal "
                f"len(cases) ({len(self.cases)})"
            )

        expected_window_case_count = (
            self.analyzed_case_count
            + len(self.cases_missing_analysis)
            + len(self.cases_with_unparseable_analysis)
        )
        if self.window_case_count != expected_window_case_count:
            raise ValueError(
                f"window_case_count ({self.window_case_count}) must equal "
                f"analyzed_case_count + len(cases_missing_analysis) + "
                f"len(cases_with_unparseable_analysis) ({expected_window_case_count})"
            )

        expected_coverage_ratio = (
            self.analyzed_case_count / self.window_case_count if self.window_case_count > 0 else 0.0
        )
        if (
            isinstance(self.coverage_ratio, bool)
            or not isinstance(self.coverage_ratio, (int, float))
            or self.coverage_ratio != expected_coverage_ratio
        ):
            raise ValueError(
                f"coverage_ratio ({self.coverage_ratio!r}) must equal "
                f"analyzed_case_count / window_case_count ({expected_coverage_ratio!r})"
            )
        if not (0.0 <= self.coverage_ratio <= 1.0):
            raise ValueError(f"coverage_ratio out of range [0.0, 1.0]: {self.coverage_ratio!r}")

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


def _count_by(
    records: list[CaseIntelligenceRecord], key_fn: Callable[[CaseIntelligenceRecord], str]
) -> Mapping[str, int]:
    """Deterministic, immutable frequency count of key_fn(record) across
    records -- key-sorted so iteration order never depends on insertion
    order or dict hash randomization."""
    counts: dict[str, int] = {}
    for record in records:
        key = key_fn(record)
        counts[key] = counts.get(key, 0) + 1
    return MappingProxyType(dict(sorted(counts.items())))


def build_reflector_input(
    store: FeedbackStoreV2,
    *,
    window_start: str | None = None,
    window_end: str | None = None,
) -> ReflectorInput:
    """Assemble one Reflector analysis window's full deterministic input.

        list_case_ids_created_in_window()
                |
                v
        list_latest_case_analysis(case_ids=...)
                |
                v
        parse/validate each row -> CaseIntelligenceRecord, or classify the
        gap (missing analysis / unparseable analysis)
                |
                v
        deterministic aggregation (window_case_count / analyzed_case_count /
        coverage_ratio / by_product_model / by_issue_type / by_diagnosis)
                |
                v
        ReflectorInput

    Uses only tools.feedback_store_v2.FeedbackStoreV2 query methods -- no
    raw SQL in this module.

    Missing-analysis handling: a Case whose creation time falls inside
    [window_start, window_end) but that has never been analyzed (no
    case_analysis row at all) is NEVER silently dropped -- its case_id is
    recorded in ReflectorInput.cases_missing_analysis. Likewise a Case
    whose latest case_analysis row exists but fails to reconstruct into a
    valid CaseIntelligenceRecord (malformed evidence_json, or a value that
    no longer matches the current taxonomy) is recorded in
    ReflectorInput.cases_with_unparseable_analysis, never merged into
    `cases` and never dropped without a trace.

    This function deliberately does NOT hard-fail the whole build when
    some window Cases lack analysis -- a real production batch can
    legitimately leave some Cases unanalyzed even in an otherwise-healthy
    run; aborting the entire Reflector input for one such gap would make
    the whole pipeline fragile for no real safety benefit. Instead, the
    gap is made a first-class, impossible-to-miss field on the returned
    contract.

    Does not catch exceptions raised by the store.* calls themselves -- a
    real database/connectivity failure propagates to the caller unchanged.
    There is no meaningful "empty ReflectorInput" fallback for a DB
    failure that would not misrepresent an error as "zero Cases in this
    window".
    """
    case_ids = store.list_case_ids_created_in_window(
        created_from=window_start, created_before=window_end,
    )
    analysis_rows = store.list_latest_case_analysis(case_ids=case_ids)

    records: list[CaseIntelligenceRecord] = []
    unparseable: list[str] = []
    analyzed_case_ids: set[str] = set()
    for row in analysis_rows:
        analyzed_case_ids.add(row["case_id"])
        record = _build_case_intelligence_record(row)
        if record is None:
            unparseable.append(row["case_id"])
        else:
            records.append(record)

    missing = sorted(cid for cid in case_ids if cid not in analyzed_case_ids)
    records.sort(key=lambda r: r.case_id)
    unparseable.sort()

    by_product_model = _count_by(
        records,
        lambda r: r.product_model if r.product_model is not None else UNKNOWN_PRODUCT_MODEL_BUCKET,
    )
    by_issue_type = _count_by(records, lambda r: r.issue_type)
    by_diagnosis = _count_by(records, lambda r: r.diagnosis)

    window_case_count = len(case_ids)
    analyzed_case_count = len(records)
    coverage_ratio = analyzed_case_count / window_case_count if window_case_count > 0 else 0.0

    return ReflectorInput(
        window_start=window_start,
        window_end=window_end,
        cases=tuple(records),
        cases_missing_analysis=tuple(missing),
        cases_with_unparseable_analysis=tuple(unparseable),
        window_case_count=window_case_count,
        analyzed_case_count=analyzed_case_count,
        coverage_ratio=coverage_ratio,
        by_product_model=by_product_model,
        by_issue_type=by_issue_type,
        by_diagnosis=by_diagnosis,
    )


# ---------------------------------------------------------------------------
# Case-id-scoped assembly -- the Curator's own entry point into this same
# case_analysis row -> CaseIntelligenceRecord reconstruction (Curator
# Slice 1). Deliberately separate from build_reflector_input() above: a
# Curator run resolves ONE Proposal's/Observation's explicit, already-known
# supporting_case_ids, never a Case-creation-time window.
# ---------------------------------------------------------------------------


def build_case_intelligence_for_ids(
    store: FeedbackStoreV2, case_ids: Collection[str],
) -> tuple[tuple[CaseIntelligenceRecord, ...], tuple[str, ...]]:
    """Reconstruct CaseIntelligenceRecord for an explicit, caller-supplied
    collection of case_ids (e.g. one ImprovementProposal Observation's
    supporting_case_ids) -- the case_id-scoped counterpart to
    build_reflector_input()'s window-scoped assembly. Reuses the exact same
    case_analysis row -> CaseIntelligenceRecord reconstruction
    (_build_case_intelligence_record) so evidence_json parsing/validation
    is never duplicated between the two callers.

    case_ids=() returns ((), ()) without querying the store, matching
    FeedbackStoreV2.list_latest_case_analysis()'s own empty-collection
    contract (an empty collection is explicit input, not "no filter").

    Returns (records, unusable_case_ids). `records` holds every case_id
    that reconstructed successfully, ordered case_id ascending (same
    ordering convention as build_reflector_input()). `unusable_case_ids`
    (sorted, deduplicated) holds every given case_id that has no
    case_analysis row at all, OR whose latest row failed to reconstruct --
    unlike ReflectorInput, this function does not split those two reasons
    into separate fields: a caller consuming one small, explicit case_id
    list only needs to know which ones it cannot use, not a window-level
    coverage/health breakdown.
    """
    ids = list(case_ids)
    if not ids:
        return (), ()

    rows = store.list_latest_case_analysis(case_ids=ids)
    found_case_ids: set[str] = set()
    records: list[CaseIntelligenceRecord] = []
    unusable: list[str] = []
    for row in rows:
        found_case_ids.add(row["case_id"])
        record = _build_case_intelligence_record(row)
        if record is None:
            unusable.append(row["case_id"])
        else:
            records.append(record)

    unusable.extend(cid for cid in ids if cid not in found_case_ids)
    records.sort(key=lambda r: r.case_id)
    return tuple(records), tuple(sorted(set(unusable)))


# ---------------------------------------------------------------------------
# Reflection eligibility -- orchestration POLICY, deliberately separate from
# the ReflectorInput data contract above (see this module's docstring)
# ---------------------------------------------------------------------------

# First-cut POC default -- a plain, unenforced default a caller may
# override, never a value baked into ReflectorInput itself.
DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION = 5


def is_reflection_eligible(
    reflector_input: ReflectorInput,
    *,
    min_analyzed_cases: int = DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION,
) -> bool:
    """Whether this ReflectorInput has enough usable Case Intelligence for
    a Reflector run to proceed -- deterministic, no LLM call.

    Reads ONLY analyzed_case_count -- never window_case_count and never
    coverage_ratio. window_case_count also counts Cases this ReflectorInput
    cannot actually give the Reflector any evidence for (missing/
    unparseable), so gating on it would let a window with many occurrences
    but few usable analyses pass eligibility on the strength of Cases the
    Reflector will never actually see. coverage_ratio is deliberately kept
    observational-only, not read here either -- coverage-aware gating is
    an explicit non-goal.

    `min_analyzed_cases` is a caller-supplied ORCHESTRATION POLICY
    parameter, never a ReflectorInput field. Fails closed on an invalid
    threshold: raises ValueError immediately for a non-int or non-positive
    min_analyzed_cases -- an invalid threshold is the caller's bug, not
    something to fold into a False return.
    """
    if (
        isinstance(min_analyzed_cases, bool)
        or not isinstance(min_analyzed_cases, int)
        or min_analyzed_cases < 1
    ):
        raise ValueError(f"invalid min_analyzed_cases: {min_analyzed_cases!r}")

    return reflector_input.analyzed_case_count >= min_analyzed_cases


__all__ = [
    "DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION",
    "UNKNOWN_PRODUCT_MODEL_BUCKET",
    "CaseIntelligenceRecord",
    "ReflectorInput",
    "build_reflector_input",
    "build_case_intelligence_for_ids",
    "is_reflection_eligible",
]
