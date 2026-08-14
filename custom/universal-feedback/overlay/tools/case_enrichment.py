"""Phase 4.5 Stage B: the Case enrichment analysis contract.

    DB
     |
     v
    CaseEnrichmentInput      (this module, build_case_enrichment_input)
     |
     v
    [ future LLM -- NOT implemented here ]
     |
     v
    CaseEnrichmentResult     (this module, parse_case_enrichment_result)
     |
     v
    structural validation    (CaseEnrichmentResult.__post_init__)

Stage B scope only: assemble the input a future analysis step needs, and
validate the shape of whatever structured result it produces. No LLM call,
no persistence (create_case_analysis is never called from here), and no
Case-identity write (cases.title / cases.product_model are read, never
written). Storage queries stay in tools.feedback_store_v2 -- this module
issues no raw SQL of its own, matching tools.retrieval_runtime's own
"glue, not storage" layering (see that module's docstring).

Case is the analysis boundary throughout: every function here takes a
single case_id, never a session_id, matching the Phase 4.5 planning
audit's "Case is the analysis unit" principle.

Evidence Facts vs. AI Inference (see the Phase 4.5 planning audit): this
module enforces the STRUCTURE of that boundary but cannot verify the
CONTENT of it. A CaseAnalysisEvidence.fact string that is itself an
inference dressed up as a fact (e.g. "The KB is definitely incomplete"
instead of "Negative feedback reason_code=incomplete") passes structural
validation -- there is no content classifier here, and building one is
explicitly out of scope for Stage B.

Taxonomy authority (Stage B.5 Schema Alignment): tools.feedback_store_v2 is
the single authority for DIAGNOSIS_VALUES/ISSUE_TYPE_VALUES/
PRODUCT_SOURCE_VALUES -- all three are imported from there, never
re-declared here. (An earlier Stage B draft kept a local, deliberately
diverging DIAGNOSIS_VALUES while the storage-layer schema had not caught up
yet; that divergence is resolved now that tools.feedback_store_v2's
case_analysis schema and CHECK constraints match this contract exactly --
see that module's docstring for the migration that aligned them.) The
per-value SEMANTICS below (what each taxonomy value practically means) are
still documented here, next to CaseEnrichmentResult, since that is where a
future prompt/analyzer implementation will actually need them -- only the
authoritative value LIST moved.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

from tools.feedback_store_v2 import (
    DIAGNOSIS_VALUES,
    ISSUE_TYPE_VALUES,
    PRODUCT_SOURCE_VALUES,
    FeedbackStoreV2,
)

logger = logging.getLogger(__name__)

_DIAGNOSIS_VALUE_SET = frozenset(DIAGNOSIS_VALUES)
_ISSUE_TYPE_VALUE_SET = frozenset(ISSUE_TYPE_VALUES)
_PRODUCT_SOURCE_VALUE_SET = frozenset(PRODUCT_SOURCE_VALUES)

# Diagnosis semantics (values themselves: see DIAGNOSIS_VALUES, imported
# above from tools.feedback_store_v2, the taxonomy authority).
#
# diagnosis is a CASE-LEVEL analysis result, produced for every analyzed
# Case, not a Negative-Feedback-only field. Feedback (CaseFeedbackEvidence)
# is important EVIDENCE for deciding a diagnosis, never a prerequisite for
# having one, and never a shortcut to one: helpful=true does not mean
# "skip diagnosis" (a Case with only positive feedback can and should still
# get diagnosis="no_issue_detected", explicitly, not an omitted/None
# value), and helpful=false + reason_code="incomplete" does not mean
# "diagnosis is automatically knowledge_gap" -- the same negative-feedback
# Case could just as validly turn out to be retrieval_issue,
# answer_quality_issue, or other_or_unclear once retrieval telemetry and
# the assistant's actual answer are weighed too. Stage B intentionally
# does NOT encode either shortcut as a validation rule (no
# "positive feedback -> force no_issue_detected", no
# "negative feedback -> force an issue diagnosis") -- that judgment is
# exactly the semantic reasoning a future analyzer does, by weighing
# Turns + the assistant's answer + retrieval telemetry + feedback
# together; Stage B only checks that whatever value comes out is
# structurally one of the five below.
#
#   knowledge_gap                - existing knowledge/FAQ/KB may be
#                                   insufficient to support this Case (an
#                                   AI diagnosis, not a proven KB gap).
#   retrieval_issue               - relevant knowledge may exist, but this
#                                   turn's retrieval didn't successfully
#                                   surface it, surfaced too little, or
#                                   retrieval telemetry points at this
#                                   layer as the likely cause.
#   answer_quality_issue          - knowledge/retrieval may not be the
#                                   main problem, but Hermes' final answer
#                                   may be incomplete, unclear, an
#                                   incorrect use of the evidence it had,
#                                   or otherwise did not effectively
#                                   resolve the user's question.
#   other_or_unclear               - evidence suggests a possible issue, but
#                                   not enough to attribute it to one of
#                                   the three causes above.
#   no_issue_detected              - this Case has been analyzed, and
#                                   current evidence does not show an
#                                   AI-support issue worth improving.

# Issue Type semantics (values themselves: see ISSUE_TYPE_VALUES, imported
# above from tools.feedback_store_v2, the taxonomy authority).
#
# Issue Type: what KIND of question this Case is about -- i.e. what the
# USER was asking. This is orthogonal to `diagnosis` (which is about where
# the AI SUPPORT PROCESS may have fallen short, never about what the user
# asked) -- see CaseEnrichmentResult's docstring for the full orthogonality
# contract (no hard-coded issue_type -> diagnosis mapping exists or should
# ever exist here). issue_type applies to EVERY analyzed Case, regardless
# of Feedback polarity: a Case with helpful=true still gets a real
# issue_type (and, separately, a real diagnosis -- see the diagnosis
# semantics above for why "positive feedback" must never be treated as
# "skip diagnosis" either). Never None: a model unable to classify must
# emit "other_or_unclear" explicitly rather than omit the field, so nothing
# downstream (a future dashboard, in particular) has to treat a missing
# issue_type as a special case.
#
#   product_usage_or_application         - the user is asking HOW to use,
#                                           configure, apply, or integrate
#                                           the product: setup, operation,
#                                           installation, commands,
#                                           enabling/disabling a feature,
#                                           an application scenario, or an
#                                           integration procedure (e.g.
#                                           "how do I disable SNMP",
#                                           "how do I set the IP",
#                                           "how do I wire this to a PLC
#                                           over Modbus", "how do I
#                                           integrate with SCADA").
#   product_capability_or_compatibility  - the user is asking whether the
#                                           product CAN do something,
#                                           DOES support something, or IS
#                                           compatible with something:
#                                           specs, supported
#                                           features/protocols/OSes,
#                                           firmware/driver compatibility,
#                                           capability limits (e.g. "does
#                                           it support Windows 11", "what
#                                           protocols does it support",
#                                           "what's the max baud rate",
#                                           "is it compatible with this
#                                           PLC model").
#   product_issue                        - the Case describes what looks
#                                           like a product malfunction,
#                                           defect, failure, or unexpected
#                                           behavior (e.g. constant
#                                           reboots, an unresponsive port,
#                                           a feature that broke after a
#                                           firmware update). This only
#                                           characterizes how the Case
#                                           READS -- it is never a proven
#                                           defect, and never implies the
#                                           root cause has been confirmed
#                                           to be the product itself.
#   other_or_unclear                      - not enough information,
#                                           unclear, or does not reasonably
#                                           fit the three categories above.
#                                           Never force a guess into one of
#                                           the other three.

# Evidence source taxonomy: one entry per evidence-bearing data source
# actually present on CaseEnrichmentInput today (turns.question_text ->
# user_text, turns.answer_text -> assistant_text, retrievals -> retrieval,
# feedback -> feedback). "assistant_text" was added in this Contract
# Refinement specifically so a diagnosis of "answer_quality_issue" can cite
# a fact drawn from Hermes' own prior answer, not just from what the user
# said. Evidence type is a SOURCE taxonomy only -- it says where a fact
# came from, never what kind of diagnosis/issue it supports.
EVIDENCE_TYPES: tuple[str, ...] = ("user_text", "assistant_text", "retrieval", "feedback")
_EVIDENCE_TYPE_SET = frozenset(EVIDENCE_TYPES)


def _is_valid_confidence(value: Any) -> bool:
    """True only for a real, finite JSON number in [0.0, 1.0].

    Deliberately mirrors tools.case_routing._is_valid_confidence's and
    tools.feedback_store_v2._is_valid_confidence's exact defensive logic
    (bool-before-int check, NaN/Infinity rejection) rather than importing
    either of those private, leading-underscore helpers across a module
    boundary -- neither module exports it, so re-deriving the same small,
    self-contained numeric check here respects each module's own
    encapsulation choice instead of reaching past it. This is a validation
    PATTERN, not a taxonomy-literal duplication -- DIAGNOSIS_VALUES/
    ISSUE_TYPE_VALUES/PRODUCT_SOURCE_VALUES are all imported from
    tools.feedback_store_v2 above, never re-declared.
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
# Input contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseTurnEvidence:
    """One Turn's observed data -- exactly the columns that exist on
    `turns` today. No document content anywhere (there is none to have)."""

    turn_id: str
    question_text: str
    answer_text: str
    created_at: str
    retrieval_observation_status: str
    case_assignment_method: str | None
    case_assignment_confidence: str | None


@dataclass(frozen=True)
class CaseRetrievalEvidence:
    """One Foundry IQ invocation's telemetry for a Turn in this Case.

    Deliberately has no field for retrieved document content, titles, or
    query text -- tools.retrieval_observer/tools.feedback_store_v2 never
    persist any of that (confirmed by the Phase 4.5 planning audit), so
    there is nothing here to carry even if this dataclass had a slot for
    it. A future analysis step cannot infer a product from *what* was
    retrieved, only from whether/how retrieval succeeded.
    """

    turn_id: str
    execution_status: str
    foundry_iq_ok: bool | None
    result_count: int | None
    reference_count: int | None
    observation_status: str
    observation_reason: str | None


@dataclass(frozen=True)
class CaseFeedbackEvidence:
    """One Turn's Feedback row, if any."""

    turn_id: str
    helpful: bool
    reason_code: str | None
    suggestion_text: str | None


@dataclass(frozen=True)
class CaseEnrichmentInput:
    """Everything a future analysis step is given for one Case -- the
    Case boundary, never the whole Session (Phase 4.5 planning audit,
    "Case is the analysis unit").

    `current_title`/`current_product_model` are the Case's EXISTING
    `cases.title`/`cases.product_model` values, carried through unchanged
    -- almost always None today, since nothing writes them yet (Stage A).
    This is Evidence, not Inference: build_case_enrichment_input() never
    guesses a product from question_text or anywhere else to populate
    these -- that guess is exactly the job Stage D's LLM call has not been
    built yet to do.

    Structurally requires at least one turn (see __post_init__) -- an
    empty-turns input would have nothing for any future analysis step to
    analyze. build_case_enrichment_input() already fails closed (returns
    None) before ever attempting to construct one this way; this is a
    second, structural line of defense against any other future caller
    constructing one directly with zero turns.
    """

    case_id: str
    session_id: str
    current_title: str | None
    current_product_model: str | None
    turns: tuple[CaseTurnEvidence, ...]
    retrievals: tuple[CaseRetrievalEvidence, ...]
    feedback: tuple[CaseFeedbackEvidence, ...]
    evidence_watermark: str

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("CaseEnrichmentInput requires at least one turn")


def _turn_evidence_from_row(row: Any) -> CaseTurnEvidence:
    return CaseTurnEvidence(
        turn_id=row["turn_id"],
        question_text=row["question_text"],
        answer_text=row["answer_text"],
        created_at=row["created_at"],
        retrieval_observation_status=row["retrieval_observation_status"],
        case_assignment_method=row["case_assignment_method"],
        case_assignment_confidence=row["case_assignment_confidence"],
    )


def _retrieval_evidence_from_row(row: Any) -> CaseRetrievalEvidence:
    foundry_iq_ok = row["foundry_iq_ok"]
    return CaseRetrievalEvidence(
        turn_id=row["turn_id"],
        execution_status=row["execution_status"],
        foundry_iq_ok=None if foundry_iq_ok is None else bool(foundry_iq_ok),
        result_count=row["result_count"],
        reference_count=row["reference_count"],
        observation_status=row["observation_status"],
        observation_reason=row["observation_reason"],
    )


def _feedback_evidence_from_row(row: Any) -> CaseFeedbackEvidence:
    return CaseFeedbackEvidence(
        turn_id=row["turn_id"],
        helpful=bool(row["helpful"]),
        reason_code=row["reason_code"],
        suggestion_text=row["suggestion_text"],
    )


def build_case_enrichment_input(store: FeedbackStoreV2, case_id: str) -> CaseEnrichmentInput | None:
    """Assemble one Case's full analysis input from the store.

    Uses only existing tools.feedback_store_v2.FeedbackStoreV2 query
    methods (get_case, list_turns_for_case, list_retrieval_runs_for_turn,
    list_feedback_for_case, get_case_evidence_watermark) -- no raw SQL
    here; storage query responsibility stays in feedback_store_v2.py.

    Retrieval evidence is gathered by looping list_retrieval_runs_for_turn
    per Turn (an N+1 query pattern) rather than one Case-level batch
    query, because adding such a batch query would mean touching
    feedback_store_v2.py, which is out of scope for Stage B. This is a
    manual-batch POC assembly step outside the Telegram request path, not
    a latency-sensitive hot path, so the N+1 pattern is an accepted
    simplification here, not an oversight.

    Ordering is fully deterministic without any additional sorting in this
    function: list_turns_for_case already orders by created_at,
    list_retrieval_runs_for_turn by invocation_order, and
    list_feedback_for_case by (submitted_at, feedback_id) -- turns are
    iterated in their already-deterministic order and each turn's
    retrievals are appended in their own already-deterministic order, so
    the combined retrievals tuple is deterministic too.

    Returns None (fail closed, never raises) for:
      * an unknown case_id (store.get_case returns None)
      * a Case with zero Turns (schema allows it -- see
        FeedbackStoreV2.get_case_evidence_watermark's docstring for how --
        but there is nothing to analyze, so this is not a valid input)
      * any unexpected internal failure (DB error, etc.)
    """
    try:
        case_row = store.get_case(case_id)
        if case_row is None:
            return None

        turn_rows = store.list_turns_for_case(case_id)
        if not turn_rows:
            return None

        turns = tuple(_turn_evidence_from_row(r) for r in turn_rows)

        retrievals: list[CaseRetrievalEvidence] = []
        for turn_row in turn_rows:
            for retrieval_row in store.list_retrieval_runs_for_turn(turn_row["turn_id"]):
                retrievals.append(_retrieval_evidence_from_row(retrieval_row))

        feedback = tuple(
            _feedback_evidence_from_row(r) for r in store.list_feedback_for_case(case_id)
        )

        evidence_watermark = store.get_case_evidence_watermark(case_id)
        if evidence_watermark is None:
            # Should be unreachable given turn_rows is non-empty (any Case
            # with >=1 turn has at least that turn's created_at to anchor
            # the watermark on) -- fail closed anyway rather than build an
            # input with a nonsensical missing watermark.
            logger.debug(
                "Case %r has turns but no evidence watermark; refusing to build input", case_id,
            )
            return None

        return CaseEnrichmentInput(
            case_id=str(case_id),
            session_id=case_row["session_id"],
            current_title=case_row["title"],
            current_product_model=case_row["product_model"],
            turns=turns,
            retrievals=tuple(retrievals),
            feedback=feedback,
            evidence_watermark=evidence_watermark,
        )
    except Exception:
        logger.debug("build_case_enrichment_input failed for case %r", case_id, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseAnalysisEvidence:
    """One supporting fact for a CaseEnrichmentResult -- an observed data
    point, never a conclusion. "Negative feedback reason_code=incomplete"
    is a fact; "the KB is definitely incomplete" is an inference and does
    not belong here. This dataclass can only check STRUCTURE (type is one
    of EVIDENCE_TYPES, fact is a non-blank string) -- it cannot verify that
    a given fact string is genuinely fact-shaped rather than conclusion-
    shaped prose; see this module's docstring.

    turn_id is REQUIRED (Stage B Contract Refinement -- previously
    optional): all four EVIDENCE_TYPES today (user_text, assistant_text,
    retrieval, feedback) are Turn-level evidence, so there is currently no
    legitimate Case-level fact that would need a null turn_id. If a real
    Case-level evidence source is added later, extend this contract
    explicitly then -- do not pre-reserve a nullable turn_id for a
    hypothetical source that does not exist yet.
    """

    type: str
    turn_id: str
    fact: str

    def __post_init__(self) -> None:
        if self.type not in _EVIDENCE_TYPE_SET:
            raise ValueError(f"invalid evidence type: {self.type!r}")
        _require_nonblank_str(self.fact, "fact")
        _require_nonblank_str(self.turn_id, "turn_id")


@dataclass(frozen=True)
class CaseEnrichmentResult:
    """A validated Case analysis result.

    issue_type/issue_type_confidence (Stage B Contract Refinement) are a
    dimension SEPARATE from diagnosis, and the two are ORTHOGONAL by
    design:

        issue_type = what is the user asking about?      (see ISSUE_TYPE_VALUES)
        diagnosis  = did the AI support process fall short, and where?
                                                           (see DIAGNOSIS_VALUES)

    Both are always required and never None -- a model unable to classify
    the issue type must emit "other_or_unclear" explicitly, matching how
    diagnosis already has its own "no clear cause" bucket. Both apply to
    EVERY analyzed Case regardless of Feedback polarity -- see
    DIAGNOSIS_VALUES's comment block for why "positive feedback" and
    "negative feedback" must never be treated as shortcuts to a particular
    diagnosis, which applies identically to issue_type.

    No hard-coded issue_type -> diagnosis mapping exists here, and none
    should ever be added: any issue_type may legitimately pair with any
    diagnosis. "product_usage_or_application" + "knowledge_gap",
    "product_usage_or_application" + "answer_quality_issue", and
    "product_usage_or_application" + "no_issue_detected" are all equally
    valid outcomes for a Case about how to use a product -- which one is
    correct depends entirely on the Case's actual evidence, not on which
    issue_type was assigned. __post_init__ below validates issue_type and
    diagnosis independently against their own taxonomies and never
    cross-checks one against the other.

    Product identity rule (see this module's report / product semantics
    section): product_model is optional. When present, product_source and
    product_confidence are BOTH required; when absent, both must be None.
    There is no in-between state where a product is named without a
    source, or a source is given without a product.

    product_source == "explicit_user_text" means the product model string
    can be found directly in one of CaseEnrichmentInput.turns[*].
    question_text -- the USER's own words, never the assistant's answer,
    never retrieval telemetry, never an existing cases.title. Anything
    else (the model synthesizing a product from Case context, retrieval
    result counts, or its own prior answers) is "inference". Stage B does
    not verify the product string is actually a substring of any
    question_text -- there is no product master/parser in this repository
    to check against, and the task that authorized this module was
    explicit that building one is out of scope. This is a semantic
    contract enforced by definition/documentation, not by a runtime check.

    Requires at least one evidence entry: a CaseEnrichmentResult with zero
    evidence would have no traceable support for issue_summary/diagnosis
    at all, which defeats the entire Evidence Facts vs. AI Inference
    separation this contract exists to enforce.
    """

    case_title: str
    issue_summary: str
    issue_type: str
    issue_type_confidence: float
    diagnosis: str
    diagnosis_confidence: float
    product_model: str | None
    product_source: str | None
    product_confidence: float | None
    evidence: tuple[CaseAnalysisEvidence, ...]

    def __post_init__(self) -> None:
        _require_nonblank_str(self.case_title, "case_title")
        _require_nonblank_str(self.issue_summary, "issue_summary")

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
        if len(self.evidence) < 1:
            raise ValueError("evidence must contain at least one fact")


_RESULT_REQUIRED_KEYS = frozenset({
    "case_title", "issue_summary", "issue_type", "issue_type_confidence",
    "diagnosis", "diagnosis_confidence",
    "product_model", "product_source", "product_confidence", "evidence",
})
_EVIDENCE_REQUIRED_KEYS = frozenset({"type", "turn_id", "fact"})


def parse_case_enrichment_result(data: Any) -> CaseEnrichmentResult | None:
    """Validate an already-decoded Python mapping (e.g. json.loads() output)
    into a CaseEnrichmentResult, or return None if it is not one.

    Only handles an already-decoded mapping -- no markdown code-fence
    stripping, no XML envelope, no streaming-chunk buffering (Stage B is
    not wired to any model call; that is a later stage's concern).

    Unknown top-level keys are REJECTED, not ignored: `set(data.keys())`
    must equal exactly _RESULT_REQUIRED_KEYS, and the same rule applies to
    each evidence item's own keys. This mirrors the one directly relevant
    precedent already in this codebase -- tools.case_routing.
    _parse_envelope_body's `if set(parsed.keys()) != _REQUIRED_ENVELOPE_KEYS`
    check on the Case Routing Control Envelope -- rather than silently
    accepting (and discarding) an unexpected field from a model that may
    be trying to smuggle extra, unvalidated content through a structured
    output channel.

    Never raises: any structural problem (wrong root type, missing/extra
    keys, wrong evidence shape, or any CaseEnrichmentResult/
    CaseAnalysisEvidence __post_init__ rejection) returns None.
    """
    if not isinstance(data, Mapping):
        return None
    if set(data.keys()) != _RESULT_REQUIRED_KEYS:
        return None

    raw_evidence = data.get("evidence")
    if not isinstance(raw_evidence, (list, tuple)):
        return None

    evidence_items: list[CaseAnalysisEvidence] = []
    try:
        for item in raw_evidence:
            if not isinstance(item, Mapping) or set(item.keys()) != _EVIDENCE_REQUIRED_KEYS:
                return None
            evidence_items.append(CaseAnalysisEvidence(
                type=item.get("type"),
                turn_id=item.get("turn_id"),
                fact=item.get("fact"),
            ))

        return CaseEnrichmentResult(
            case_title=data.get("case_title"),
            issue_summary=data.get("issue_summary"),
            issue_type=data.get("issue_type"),
            issue_type_confidence=data.get("issue_type_confidence"),
            diagnosis=data.get("diagnosis"),
            diagnosis_confidence=data.get("diagnosis_confidence"),
            product_model=data.get("product_model"),
            product_source=data.get("product_source"),
            product_confidence=data.get("product_confidence"),
            evidence=tuple(evidence_items),
        )
    except (ValueError, TypeError):
        return None


__all__ = [
    "DIAGNOSIS_VALUES",
    "ISSUE_TYPE_VALUES",
    "EVIDENCE_TYPES",
    "CaseTurnEvidence",
    "CaseRetrievalEvidence",
    "CaseFeedbackEvidence",
    "CaseEnrichmentInput",
    "build_case_enrichment_input",
    "CaseAnalysisEvidence",
    "CaseEnrichmentResult",
    "parse_case_enrichment_result",
]
