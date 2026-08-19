"""The Reflector auxiliary LLM integration: structured-output parser and
deterministic validation, plus the call_llm() wiring that produces the
JSON this parser consumes.

    ReflectorPromptContext (tools.reflector_prompt_context)
     |
     v
    _build_messages()        (dedicated Reflector prompt -- no SOUL/memory/
     |                         Skills/tools/session history)
     v
    call_llm()                (agent.auxiliary_client -- Hermes' side-task
     |                         inference boundary, injected via `llm_call`
     |                         for testability)
     v
    _extract_text()           (minimal OpenAI-compatible response reader)
     v
    json.loads()
     v
    parse_reflector_output()  (strict shape check, candidate-bound /
     |                         case-id-bound deterministic validation,
     |                         fail closed)
     v
    ReflectionResult (tools.reflector_proposals)

This module calls call_llm() directly, not run_conversation(): the
Reflector analyzes already-structured Case Intelligence evidence and must
never re-query retrieval, see SOUL/memory/session history, or gain access
to conversational tools.

`llm_call` defaults to a lazy import of `agent.auxiliary_client.call_llm`
(resolved inside the function body) because that module only exists inside
a built Hermes sandbox image -- this repo's test environment does not have
it installed, and every test injects a fake `llm_call` instead.

A few behaviors worth knowing before editing:
- `parse_reflector_output()` fails the WHOLE parse closed on any single
  invalid finding -- there is no "skip the bad finding" mode.
- A finding's `title` must be non-blank when action=create_new and null
  when action=match_existing; either mismatch fails the parse.
- `supporting_case_ids` are checked against the Case ids actually present
  in the given ReflectorPromptContext -- a hallucinated case_id fails the
  parse.
- A create_new finding's ImprovementProposal.created_at is set to the same
  `observed_at` value as its founding ProposalObservation.observed_at.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Collection, Mapping

from tools.feedback_store_v2 import IMPROVEMENT_TARGET_VALUES, PROPOSAL_TREND_VALUES
from tools.proposal_matching import (
    PROPOSAL_RESOLUTION_ACTION_VALUES,
    ProposalCandidate,
    ProposalResolution,
    validate_proposal_resolution,
)
from tools.reflector_prompt_context import ReflectorPromptContext, serialize_reflector_prompt_context
from tools.reflector_proposals import (
    ImprovementProposal,
    ProposalObservation,
    ReflectionResult,
    build_supporting_case_ids,
)

# Prompt + reasoning-policy generation version -- bump only on a breaking
# change to _SYSTEM_INSTRUCTIONS' semantics or the ReflectionResult/
# ProposalResolution contract, never for routine prompt wording tweaks.
REFLECTOR_VERSION = "reflector-v1"

# ---------------------------------------------------------------------------
# Reasoning policy -- stable, model-facing instructions. Taxonomy value
# lists are interpolated from imported constants, never hard-coded, so this
# text can never silently drift out of sync with the actual
# ProposalResolution/ProposalObservation/ImprovementProposal contracts.
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA_HINT = f"""{{
  "run_summary": "<string>",
  "material_change_detected": <true | false>,
  "findings": [
    {{
      "resolution": {{
        "action": "<one of: {' | '.join(PROPOSAL_RESOLUTION_ACTION_VALUES)}>",
        "proposal_id": "<string, required when action=match_existing, otherwise null>",
        "improvement_target": "<one of: {' | '.join(IMPROVEMENT_TARGET_VALUES)}>"
      }},
      "title": "<non-blank string when action=create_new, otherwise null>",
      "trend": "<one of: {' | '.join(PROPOSAL_TREND_VALUES)}>",
      "pattern_summary": "<non-blank string>",
      "possible_cause": "<string, or null>",
      "recommended_improvement": "<non-blank string>",
      "expected_benefit": "<string, or null>",
      "limitations": "<string, or null>",
      "supporting_case_ids": ["<case_id from the cases given to you>", "..."],
      "confidence": <number 0.0-1.0>
    }}
  ]
}}"""

_SYSTEM_INSTRUCTIONS = f"""You are the Cross-case Reflector for a technical-support AI assistant. You are given a fixed snapshot of Case Intelligence (already-analyzed support Cases) and a fixed list of existing, still-pending Improvement Proposal candidates. This is an independent offline analysis task, not a conversation -- you must not assume any other context, and you must not re-query or re-imagine any knowledge-base content.

Program counts facts. You interpret patterns. Every count, aggregate, and candidate you are given is already computed and already trustworthy -- your job is semantic: decide whether a RECURRING PATTERN across multiple Cases represents a genuine IMPROVEMENT OPPORTUNITY, and if so, whether it is the same opportunity as one already known or a new one.

## Your role

Find recurring patterns across the given Cases that represent improvement opportunities -- not a general statistics report, not a per-Case summary, not an opinion on every Case you were given. A Case with no bearing on any recurring pattern simply contributes no finding.

## What counts as a recurring pattern

Do not treat two Cases as the same pattern just because their titles look similar, they mention the same product, or they touch a similar topic. Consider the underlying semantics: what was actually going wrong, and why. Two Cases about "SNMP" can be completely different improvement opportunities if one is a documentation gap and the other is an answer-quality problem, even though both mention SNMP.

## improvement_target

Classify each finding's improvement_target as exactly one of: {', '.join(IMPROVEMENT_TARGET_VALUES)}.

* "knowledge" -- the knowledge base is missing, incomplete, outdated, or hard to retrieve information about the topic.
* "agent_behavior" -- the knowledge/retrieval evidence already exists and was available, but the assistant's own answer, reasoning, or instruction-following did not make correct use of it.
* "retrieval" -- the problem is primarily in how information is queried, indexed, or surfaced: relevant knowledge may exist, but it was not found or was not found well.
* "workflow" -- the problem primarily needs a process, human handoff, or operational workflow change, not a knowledge/behavior/retrieval fix.
* "other" -- none of the above cleanly fits.

## Proposal identity: existing vs. new

The question is never "same title", "same product", or "same topic". It is: does this finding represent the SAME UNDERLYING IMPROVEMENT OPPORTUNITY as one of the candidates you were given? Weigh together: improvement_target, the problem/pattern semantics, and the recommended remediation direction. A knowledge-gap finding about SNMP and an agent_behavior finding about SNMP are different opportunities even though both mention SNMP, because what would actually need to change is different.

For every finding, set resolution.action to exactly one of: {', '.join(PROPOSAL_RESOLUTION_ACTION_VALUES)}.

* "match_existing" -- you MUST set resolution.proposal_id to one of the candidate proposal_id values you were given, exactly as given. Never invent a proposal_id. Never choose a candidate just because it is the closest-sounding one -- only choose it if you judge it to genuinely be the same underlying improvement opportunity.
* "create_new" -- resolution.proposal_id MUST be null. This finding is not the same underlying opportunity as any candidate you were given (including the case where no candidates were given at all).

## Evidence discipline

supporting_case_ids must be drawn only from the Case ids actually given to you in this context. Never invent a case_id. Never cite a Case you were not given.

## possible_cause is a hypothesis, not a proven root cause

State it as your best current explanation given the evidence, never as a confirmed fact. Do not overclaim certainty you do not have.

## Zero findings is a valid result

You are not required to produce a finding for every run. If nothing in the given Cases represents a recurring pattern worth proposing as an improvement, return material_change_detected=false and an empty findings list. Do not manufacture a finding to avoid an empty result.

## Human review boundary

An Improvement Proposal you produce is a recommendation for a human reviewer -- it is never permission to modify the knowledge base, SOUL, a Skill, or any production system directly, regardless of improvement_target.

## Language

Write run_summary, title, pattern_summary, possible_cause, recommended_improvement, expected_benefit, and limitations in Traditional Chinese (zh-TW). Preserve product names, technical terms, code, commands, and identifiers -- and standard English terminology where translating it would reduce precision -- in their original form within that Chinese text. JSON keys, enum values (action, improvement_target, trend), proposal_id, and case_id values must remain exactly as defined by the schema, in English, regardless of this rule.

## Output format

Respond with a single JSON object and nothing else -- no markdown code fences, no explanation before or after the JSON. Respond with a JSON object matching exactly this shape -- no extra fields, no missing fields:
{_OUTPUT_SCHEMA_HINT}"""


# ---------------------------------------------------------------------------
# Strict key-sets -- mirrors tools.case_enrichment.parse_case_enrichment_
# result's own "unknown top-level keys are REJECTED, not ignored" contract.
# ---------------------------------------------------------------------------

_RESULT_REQUIRED_KEYS = frozenset({"run_summary", "material_change_detected", "findings"})
_FINDING_REQUIRED_KEYS = frozenset({
    "resolution", "title", "trend", "pattern_summary", "possible_cause",
    "recommended_improvement", "expected_benefit", "limitations",
    "supporting_case_ids", "confidence",
})
_RESOLUTION_REQUIRED_KEYS = frozenset({"action", "proposal_id", "improvement_target"})


def _parse_finding(
    raw_finding: Any,
    *,
    candidates: Collection[ProposalCandidate],
    valid_case_ids: frozenset[str],
    reflection_run_id: str,
    observed_at: str,
    id_factory: Callable[[], str],
) -> tuple[ImprovementProposal | None, ProposalObservation] | None:
    """Parse and validate one raw finding, or return None on any failure
    (wrong shape, wrong key-set, invalid taxonomy, hallucinated proposal_id,
    target mismatch, hallucinated case_id, invalid confidence). A duplicate
    proposal_id inside `candidates` instead raises ValueError from
    validate_proposal_resolution(), caught by parse_reflector_output()'s
    own try/except.

    Returns (new_proposal_or_None, observation) on success: a create_new
    finding returns (ImprovementProposal, ProposalObservation); a
    match_existing finding returns (None, ProposalObservation) since it
    never creates or renames an ImprovementProposal.
    """
    if not isinstance(raw_finding, Mapping) or set(raw_finding.keys()) != _FINDING_REQUIRED_KEYS:
        return None

    raw_resolution = raw_finding.get("resolution")
    if not isinstance(raw_resolution, Mapping) or set(raw_resolution.keys()) != _RESOLUTION_REQUIRED_KEYS:
        return None

    resolution = ProposalResolution(
        action=raw_resolution.get("action"),
        proposal_id=raw_resolution.get("proposal_id"),
        improvement_target=raw_resolution.get("improvement_target"),
    )
    if not validate_proposal_resolution(resolution, candidates=candidates):
        return None

    title = raw_finding.get("title")
    if resolution.action == "create_new":
        if not isinstance(title, str) or not title.strip():
            return None
        proposal_id = id_factory()
    else:
        if title is not None:
            return None
        proposal_id = resolution.proposal_id

    raw_supporting_case_ids = raw_finding.get("supporting_case_ids")
    if not isinstance(raw_supporting_case_ids, (list, tuple)) or not all(
        isinstance(cid, str) for cid in raw_supporting_case_ids
    ):
        return None
    if not set(raw_supporting_case_ids) <= valid_case_ids:
        return None
    supporting_case_ids = build_supporting_case_ids(raw_supporting_case_ids)

    observation = ProposalObservation(
        observation_id=id_factory(),
        proposal_id=proposal_id,
        reflection_run_id=reflection_run_id,
        trend=raw_finding.get("trend"),
        pattern_summary=raw_finding.get("pattern_summary"),
        possible_cause=raw_finding.get("possible_cause"),
        recommended_improvement=raw_finding.get("recommended_improvement"),
        expected_benefit=raw_finding.get("expected_benefit"),
        limitations=raw_finding.get("limitations"),
        supporting_case_ids=supporting_case_ids,
        supporting_case_count=len(supporting_case_ids),
        confidence=raw_finding.get("confidence"),
        observed_at=observed_at,
    )

    new_proposal = None
    if resolution.action == "create_new":
        new_proposal = ImprovementProposal(
            proposal_id=proposal_id,
            improvement_target=resolution.improvement_target,
            title=title,
            review_status="pending",
            created_at=observed_at,
        )

    return new_proposal, observation


def parse_reflector_output(
    data: Any,
    *,
    candidates: Collection[ProposalCandidate],
    valid_case_ids: Collection[str],
    reflection_run_id: str,
    observed_at: str,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> ReflectionResult | None:
    """Validate an already-decoded Python mapping (e.g. json.loads() output)
    into a ReflectionResult, or return None if it is not a legal one.

    Only handles an already-decoded mapping (no markdown code-fence
    stripping, no streaming-chunk buffering). Never raises for a malformed/
    adversarial `data`; any structural problem anywhere returns None. Fail
    closed, never partial: a single invalid finding fails the WHOLE parse,
    not just that finding.

    `candidates` and `valid_case_ids` are the SAME two collections the
    model was actually shown for this run -- normally
    `ReflectorPromptContext.existing_proposals` and `{{c.case_id for c in
    ReflectorPromptContext.cases}}`. This function never queries a store
    for either.

    `reflection_run_id`/`observed_at` are plain caller-supplied strings,
    never generated here. `id_factory` is a minimal, keyword-only
    dependency injection point (default `lambda: uuid.uuid4().hex`) used
    for every id this function itself must mint (a create_new finding's
    new proposal_id, and every finding's observation_id).
    """
    if not isinstance(data, Mapping) or set(data.keys()) != _RESULT_REQUIRED_KEYS:
        return None

    raw_findings = data.get("findings")
    if not isinstance(raw_findings, (list, tuple)):
        return None

    valid_case_id_set = frozenset(valid_case_ids)
    candidates_tuple = tuple(candidates)

    new_proposals: list[ImprovementProposal] = []
    observations: list[ProposalObservation] = []

    try:
        for raw_finding in raw_findings:
            parsed = _parse_finding(
                raw_finding,
                candidates=candidates_tuple,
                valid_case_ids=valid_case_id_set,
                reflection_run_id=reflection_run_id,
                observed_at=observed_at,
                id_factory=id_factory,
            )
            if parsed is None:
                return None
            new_proposal, observation = parsed
            if new_proposal is not None:
                new_proposals.append(new_proposal)
            observations.append(observation)

        return ReflectionResult(
            reflection_run_id=reflection_run_id,
            run_summary=data.get("run_summary"),
            material_change_detected=data.get("material_change_detected"),
            new_proposals=tuple(new_proposals),
            proposal_observations=tuple(observations),
        )
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Slice 5B: the real LLM call -- mirrors tools.case_enrichment_analyzer's
# analyze_case_with_llm() shape exactly (see this module's own docstring).
# ---------------------------------------------------------------------------


class ReflectorAnalyzerError(RuntimeError):
    """Raised by analyze_reflection_with_llm() for any failure that
    prevents producing a valid ReflectionResult -- provider/network
    failure, empty response, invalid JSON, or parse_reflector_output()
    rejecting the decoded response. A single, minimal exception type: the
    message + chained __cause__ carries the failure detail instead of a
    multi-class hierarchy, since a caller never needs to distinguish a
    provider error from a rejected response -- both mean "this Reflection
    Run produced no usable result."
    """


def _build_messages(context: ReflectorPromptContext) -> list[dict[str, str]]:
    """Exactly two messages: system reasoning policy + serialized
    ReflectorPromptContext. Nothing else -- no session history, no SOUL, no
    tool definitions, no re-retrieval instructions. Business/reasoning
    policy lives entirely in `_SYSTEM_INSTRUCTIONS`; the user message is
    only ever a label plus the deterministic JSON payload.
    """
    serialized = serialize_reflector_prompt_context(context)
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Reflector context (JSON):\n{serialized}"},
    ]


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of the OpenAI-compatible response object
    agent.auxiliary_client.call_llm() returns (`.choices[0].message.content`
    as a plain string). Any other shape, or a non-string content, returns
    "" so the caller's empty-response check fails closed rather than
    silently accepting something unexpected.
    """
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def analyze_reflection_with_llm(
    context: ReflectorPromptContext,
    *,
    reflection_run_id: str,
    observed_at: str,
    main_runtime: Mapping[str, Any] | None = None,
    llm_call: Callable[..., Any] | None = None,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> ReflectionResult:
    """The real LLM analyzer: `context` -> one call_llm() invocation ->
    parse_reflector_output() -> ReflectionResult.

    `candidates`/`valid_case_ids` are derived from `context` itself
    (`context.existing_proposals` and `{{c.case_id for c in
    context.cases}}`) rather than taken as separate parameters, so a
    caller that already has a ReflectorPromptContext never needs to
    reconstruct either collection.

    `main_runtime` should carry the SAME provider/model/base_url/api_key/
    api_mode the main Hermes agent is already using -- passed straight
    through to `llm_call`, never resolved or hard-coded here.

    `llm_call` defaults to `agent.auxiliary_client.call_llm`, imported here
    (not at module level) so this module stays importable in this repo's
    test environment, which has no Hermes runtime installed -- calling
    without an injected `llm_call` surfaces the resulting ImportError
    undisguised (not wrapped into ReflectorAnalyzerError, since a missing
    package is an environment problem, not an LLM/parse failure).

    Raises ReflectorAnalyzerError -- never returns None -- for any failure
    downstream of a successful lazy import: provider/network error, empty/
    unreadable response, invalid JSON, or parse_reflector_output()
    rejecting the decoded response. No application-level retry anywhere in
    this function; no persistence either (returns a plain in-memory
    ReflectionResult, never calls FeedbackStoreV2 or creates a
    reflection_runs row).
    """
    call = llm_call
    if call is None:
        from agent.auxiliary_client import call_llm as call

    messages = _build_messages(context)

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
        raise ReflectorAnalyzerError(
            f"call_llm failed for reflection_run_id {reflection_run_id!r}: {type(exc).__name__}: {exc}"
        ) from exc

    text = _extract_text(response)
    if not text.strip():
        raise ReflectorAnalyzerError(
            f"LLM response for reflection_run_id {reflection_run_id!r} contained no extractable text"
        )

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReflectorAnalyzerError(
            f"LLM response for reflection_run_id {reflection_run_id!r} was not valid JSON: {exc}"
        ) from exc

    result = parse_reflector_output(
        parsed,
        candidates=context.existing_proposals,
        valid_case_ids=tuple(c.case_id for c in context.cases),
        reflection_run_id=reflection_run_id,
        observed_at=observed_at,
        id_factory=id_factory,
    )
    if result is None:
        raise ReflectorAnalyzerError(
            f"LLM response for reflection_run_id {reflection_run_id!r} failed ReflectionResult "
            "contract validation"
        )

    return result


__all__ = [
    "REFLECTOR_VERSION",
    "parse_reflector_output",
    "ReflectorAnalyzerError",
    "analyze_reflection_with_llm",
]
