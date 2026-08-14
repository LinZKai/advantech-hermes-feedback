"""Phase 4.5 Stage D1: the real LLM Case Enrichment analyzer.

    CaseEnrichmentInput
     |
     v
    _build_messages()            (this module -- dedicated enrichment prompt,
     |                             no SOUL/memory/Skills/tools/session history)
     v
    call_llm()                   (agent.auxiliary_client -- Hermes' existing
     |                             side-task inference boundary, injected via
     |                             `llm_call` for testability)
     v
    _extract_text()              (this module -- minimal OpenAI-compatible
     |                             response.choices[0].message.content reader)
     v
    json.loads()
     |
     v
    parse_case_enrichment_result()  (tools.case_enrichment -- unchanged,
     |                                 frozen Stage B contract)
     v
    CaseEnrichmentResult   or   CaseEnrichmentAnalyzerError

Stage D1 scope only: turn a CaseEnrichmentInput into a real, validated
CaseEnrichmentResult using Hermes' existing inference infrastructure. No
persistence anywhere in this module (create_case_analysis is never called
here, analysis_id/analyzed_at are never generated here -- that is Stage D2's
job), and no retry (agent.auxiliary_client.call_llm already retries/falls
back at the provider layer -- see the Stage D Hermes LLM Integration Audit).

Why `call_llm()` and not `run_conversation()`: `run_conversation()` pulls in
session history, SOUL, Skills, the tool-dispatch loop, Foundry IQ retrieval,
trajectory/telemetry, and persistence side effects that Case Enrichment must
never inherit (Case Enrichment analyzes previously-observed evidence; it must
never re-query the knowledge source or blur the Evidence-vs-Inference
boundary). `call_llm()` is Hermes' existing, independent side-task inference
boundary (already used in this overlay's own gateway/run.py for auto-title
generation via the same `main_runtime` pattern used here) -- it takes a
plain, caller-controlled `messages` list with zero implicit injection, which
is exactly what an offline, independent analysis task needs.

`llm_call` is dependency-injected and defaults to a LAZY import of
`agent.auxiliary_client.call_llm` (resolved inside the function body, not at
module import time) because `agent.auxiliary_client` only exists inside a
built Hermes sandbox image (/opt/hermes/agent/...) -- this repo's own test
environment does not have it installed, and must never need it: every test
in test_case_enrichment_analyzer.py injects a fake `llm_call`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping

from tools.case_enrichment import (
    CaseEnrichmentInput,
    CaseEnrichmentResult,
    CaseRetrievalEvidence,
    CaseTurnEvidence,
    DIAGNOSIS_VALUES,
    EVIDENCE_TYPES,
    ISSUE_TYPE_VALUES,
    parse_case_enrichment_result,
)
from tools.feedback_store_v2 import PRODUCT_SOURCE_VALUES

logger = logging.getLogger(__name__)

# Prompt + analysis contract generation version -- bump only on a breaking
# change to prompt semantics or the CaseEnrichmentResult contract, never for
# routine prompt wording tweaks. Deliberately not a timestamp (see Stage D
# audit report §18): a version string must be able to identify "which
# contract/prompt generation produced this row", which a timestamp cannot do
# for prompt text that changes between analysis runs on the same day.
ANALYSIS_VERSION = "case-enrichment-v1"


class CaseEnrichmentAnalyzerError(RuntimeError):
    """Raised by analyze_case_with_llm() for any failure that prevents
    producing a valid CaseEnrichmentResult -- provider/network failure,
    empty response, invalid JSON, or contract validation failure. A single,
    minimal exception type on purpose (Stage D audit report §16): Stage C's
    run_case_enrichment._process_case() already isolates per-case failures
    with a broad `except Exception`, so callers do not need to distinguish
    failure kinds by exception subclass -- the message + chained __cause__
    (via `raise ... from exc`) carries that detail for logs/debugging
    instead of a multi-class hierarchy.
    """


# ---------------------------------------------------------------------------
# Prompt boundary -- system instructions
# ---------------------------------------------------------------------------
#
# Deliberately excludes: SOUL, memory, Skills, tools, Foundry IQ
# re-retrieval, whole session history, Telegram context, Case routing
# prompt. Case Enrichment is an independent offline analysis task over
# already-observed evidence, not a conversational turn (Stage D audit
# report §3/§8/§19) -- calling call_llm() directly with exactly these two
# messages (system + user, see _build_messages) is what structurally
# guarantees none of the excluded context leaks in; there is no separate
# "off switch" to maintain.
#
# Taxonomy value lists are interpolated from tools.case_enrichment's
# imports (themselves sourced from tools.feedback_store_v2, the taxonomy
# authority) rather than hard-coded here, so this prompt can never silently
# drift out of sync with the actual CaseEnrichmentResult contract.

_OUTPUT_SCHEMA_HINT = f"""{{
  "case_title": "<short string>",
  "issue_summary": "<short string>",
  "issue_type": "<one of: {' | '.join(ISSUE_TYPE_VALUES)}>",
  "issue_type_confidence": <number 0.0-1.0>,
  "diagnosis": "<one of: {' | '.join(DIAGNOSIS_VALUES)}>",
  "diagnosis_confidence": <number 0.0-1.0>,
  "product_model": "<string, or null>",
  "product_source": "<one of: {' | '.join(PRODUCT_SOURCE_VALUES)}, or null>",
  "product_confidence": <number 0.0-1.0, or null>,
  "evidence": [
    {{"type": "<one of: {' | '.join(EVIDENCE_TYPES)}>", "turn_id": "<string>", "fact": "<short string>"}}
  ]
}}"""

_SYSTEM_INSTRUCTIONS = f"""You are analyzing one technical-support Case for product support quality review. This is an independent offline analysis task, not a conversation -- you are given a fixed snapshot of evidence and must not assume any other context.

Rules:
1. You are analyzing one technical-support Case. Base your analysis ONLY on the evidence provided in the user message below -- do not use outside knowledge about the product beyond what is stated there.
2. Do not assume or invent retrieval document content. The "retrievals" evidence is telemetry only (whether/how a retrieval call executed) -- it is not the retrieved documents themselves, and no retrieved document text is provided to you or exists to imagine.
3. Distinguish observed facts from your own inference. Every entry you put in "evidence" must state what was observed (a message, a telemetry value, a feedback field) -- never a conclusion you drew from it.
4. Respond with a single JSON object and nothing else -- no markdown code fences, no explanation before or after the JSON.
5. "issue_type" must be exactly one of: {', '.join(ISSUE_TYPE_VALUES)}.
6. "diagnosis" must be exactly one of: {', '.join(DIAGNOSIS_VALUES)}. Weigh the user's turns, the assistant's answers, retrieval telemetry, and feedback together -- never infer diagnosis from feedback polarity alone (positive feedback does not force no_issue_detected; negative feedback does not force an issue diagnosis).
7. "issue_type_confidence" and "diagnosis_confidence" must each be a number from 0.0 to 1.0.
8. "product_model" is optional. Set "product_source" to "explicit_user_text" only when the product name literally appears in a user's question_text; set it to "inference" when you identified the product from context instead. If you cannot reasonably identify a product, set product_model, product_source, and product_confidence all to null.
9. "evidence" must contain at least one item. Each item has "type" (one of: {', '.join(EVIDENCE_TYPES)}), "turn_id", and "fact" -- "fact" must state an observed fact, never a conclusion.
10. Respond with a JSON object matching exactly this shape -- no extra fields, no missing fields:
{_OUTPUT_SCHEMA_HINT}"""


# ---------------------------------------------------------------------------
# Input serialization -- deterministic, evidence-only
# ---------------------------------------------------------------------------


def _serialize_turn(turn: CaseTurnEvidence) -> dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "question_text": turn.question_text,
        "answer_text": turn.answer_text,
        "created_at": turn.created_at,
        "retrieval_observation_status": turn.retrieval_observation_status,
        "case_assignment_method": turn.case_assignment_method,
        "case_assignment_confidence": turn.case_assignment_confidence,
    }


def _serialize_retrieval(retrieval: CaseRetrievalEvidence) -> dict[str, Any]:
    return {
        "turn_id": retrieval.turn_id,
        "execution_status": retrieval.execution_status,
        "foundry_iq_ok": retrieval.foundry_iq_ok,
        "result_count": retrieval.result_count,
        "reference_count": retrieval.reference_count,
        "observation_status": retrieval.observation_status,
        "observation_reason": retrieval.observation_reason,
    }


def _serialize_feedback(feedback: Any) -> dict[str, Any]:
    return {
        "turn_id": feedback.turn_id,
        "helpful": feedback.helpful,
        "reason_code": feedback.reason_code,
        "suggestion_text": feedback.suggestion_text,
    }


def _serialize_case_input(case_input: CaseEnrichmentInput) -> dict[str, Any]:
    """Deterministic, JSON-serializable projection of CaseEnrichmentInput.

    Carries exactly the fields CaseEnrichmentInput already has -- no whole
    session, no raw retrieved documents (CaseRetrievalEvidence has no such
    field to carry even if this wanted to), no fabricated KB content.
    Ordering is inherited unchanged from CaseEnrichmentInput (already
    deterministic -- see build_case_enrichment_input's docstring): turns
    chronological, each turn's retrievals in invocation order, feedback in
    (submitted_at, feedback_id) order. Only dict *key* order is additionally
    normalized (via json.dumps(..., sort_keys=True) at the call site) for a
    stable prompt string across runs with identical evidence.
    """
    return {
        "case_id": case_input.case_id,
        "session_id": case_input.session_id,
        "current_title": case_input.current_title,
        "current_product_model": case_input.current_product_model,
        "evidence_watermark": case_input.evidence_watermark,
        "turns": [_serialize_turn(t) for t in case_input.turns],
        "retrievals": [_serialize_retrieval(r) for r in case_input.retrievals],
        "feedback": [_serialize_feedback(f) for f in case_input.feedback],
    }


def _build_messages(case_input: CaseEnrichmentInput) -> list[dict[str, str]]:
    """Exactly two messages: system instructions + serialized evidence.
    Nothing else -- no session history, no SOUL, no tool definitions. This
    is the entire prompt boundary (Stage D audit report §8/§9)."""
    serialized = _serialize_case_input(case_input)
    evidence_json = json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Case evidence (JSON):\n{evidence_json}"},
    ]


# ---------------------------------------------------------------------------
# Response extraction -- minimal, OpenAI-compatible only
# ---------------------------------------------------------------------------


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of the OpenAI-compatible response object
    agent.auxiliary_client.call_llm() returns (`.choices[0].message.content`
    as a plain string -- confirmed against the real Hermes sandbox in the
    Stage D0 runtime smoke test). Deliberately not a generic multi-shape
    parser (Stage D audit report §14): any other shape, or a non-string
    content, returns "" so the caller's empty-response check fails closed
    rather than silently accepting something unexpected.
    """
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Analyzer entry point
# ---------------------------------------------------------------------------


def analyze_case_with_llm(
    case_input: CaseEnrichmentInput,
    *,
    main_runtime: Mapping[str, Any] | None = None,
    llm_call: Callable[..., Any] | None = None,
) -> CaseEnrichmentResult:
    """The real LLM analyzer. Matches the `Analyzer = Callable[[CaseEnrichmentInput],
    CaseEnrichmentResult]` boundary tools.run_case_enrichment already defines
    -- bind `main_runtime`/`llm_call` via functools.partial before passing
    this as a runner's `analyzer=` argument (Stage D audit report §12); this
    function's own signature never grows model/provider/timeout parameters.

    `main_runtime` should carry the SAME provider/model/base_url/api_key/
    api_mode the main Hermes agent is already using (Stage D audit report
    §6/§7/§17 risk 6) -- passed straight through to `llm_call`. Never
    resolved or hard-coded in this module.

    `llm_call` defaults to `agent.auxiliary_client.call_llm`, imported here
    (not at module level) so this module stays importable in this repo's
    test environment, which has no Hermes runtime installed.

    Raises CaseEnrichmentAnalyzerError -- never returns None -- for any
    failure: provider/network error, empty/unreadable response, invalid
    JSON, or CaseEnrichmentResult contract validation failure. This keeps
    the function's return type a real CaseEnrichmentResult always, matching
    the Analyzer contract (not Optional), and lets
    tools.run_case_enrichment._process_case()'s existing per-case
    try/except isolate a Stage D1 failure exactly like it already isolates
    a Stage C stub failure -- no changes needed there.
    """
    call = llm_call
    if call is None:
        from agent.auxiliary_client import call_llm as call

    messages = _build_messages(case_input)

    try:
        response = call(
            task=None,
            main_runtime=dict(main_runtime) if main_runtime is not None else None,
            messages=messages,
            tools=None,
            stream=False,
            temperature=0,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except Exception as exc:
        raise CaseEnrichmentAnalyzerError(
            f"call_llm failed for case {case_input.case_id!r}: {type(exc).__name__}: {exc}"
        ) from exc

    text = _extract_text(response)
    if not text.strip():
        raise CaseEnrichmentAnalyzerError(
            f"LLM response for case {case_input.case_id!r} contained no extractable text"
        )

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CaseEnrichmentAnalyzerError(
            f"LLM response for case {case_input.case_id!r} was not valid JSON: {exc}"
        ) from exc

    result = parse_case_enrichment_result(parsed)
    if result is None:
        raise CaseEnrichmentAnalyzerError(
            f"LLM response for case {case_input.case_id!r} failed CaseEnrichmentResult "
            "contract validation"
        )

    return result


__all__ = [
    "ANALYSIS_VERSION",
    "CaseEnrichmentAnalyzerError",
    "analyze_case_with_llm",
]
