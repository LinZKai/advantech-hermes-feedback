"""Phase 5 Slice 5A/5B: the Reflector auxiliary LLM integration --
structured-output parser + deterministic validation (5A), and the real
call_llm() wiring that produces the JSON this parser consumes (5B).

    ReflectorPromptContext
    (tools.reflector_prompt_context)
     |
     v
    _build_messages()                  (this module -- dedicated Reflector
     |                                   prompt, no SOUL/memory/Skills/
     |                                   tools/session history)
     v
    call_llm()                         (agent.auxiliary_client -- Hermes'
     |                                   existing side-task inference
     |                                   boundary, injected via `llm_call`
     |                                   for testability; same boundary
     |                                   tools.case_enrichment_analyzer
     |                                   already uses)
     v
    _extract_text()                    (this module -- minimal OpenAI-
     |                                   compatible response reader)
     v
    json.loads()
     |
     v
    parse_reflector_output()           (this module -- strict shape check,
                |                        candidate-bound / case-id-bound
                |                        deterministic validation, fail
                |                        closed)
                v
    ReflectionResult                   (tools.reflector_proposals -- reused
                                         unchanged; this module builds no
                                         new public type)

Slice 5A scope: the stable reasoning-policy prompt text
(`_SYSTEM_INSTRUCTIONS` / `_OUTPUT_SCHEMA_HINT`) and the deterministic
parser (`parse_reflector_output`) -- no LLM call.

Slice 5B scope (this addition): `analyze_reflection_with_llm()`, the real
call_llm() wiring, and `ReflectorAnalyzerError`, the single analyzer-layer
exception type. Still no persistence anywhere in this module
(FeedbackStoreV2 is never imported here; `reflection_runs`/
`improvement_proposals`/`proposal_observations` are never written) -- this
module's only output is an in-memory `ReflectionResult`, matching how
tools.case_enrichment_analyzer.analyze_case_with_llm() never calls
create_case_analysis() either (that stays a runner's job -- Slice 6 here,
Stage D2 there).

Why this module calls call_llm() directly and not run_conversation()
(mirrors tools.case_enrichment_analyzer's own reasoning, restated here for
the Reflector): Reflector analyzes previously-collected, already-
structured Case Intelligence evidence; it must never re-query Foundry IQ,
never see SOUL/memory/session history, and never gain access to
conversational tools -- `call_llm()` is Hermes' existing, independent
side-task inference boundary (already used by tools.case_enrichment_
analyzer for exactly this reason) that takes a plain, caller-controlled
`messages` list with zero implicit injection.

`llm_call` is dependency-injected and defaults to a LAZY import of
`agent.auxiliary_client.call_llm` (resolved inside the function body, not
at module import time) because `agent.auxiliary_client` only exists inside
a built Hermes sandbox image (/opt/hermes/agent/...) -- this repo's own
test environment does not have it installed, and must never need it: every
test in test_reflector_analyzer.py injects a fake `llm_call`.

Reuse, not reinvention (see the Phase 5 Slice 5A reconnaissance report for
the full audit this follows): this module builds no new dataclass. The
finding shape a future model response carries maps directly onto types
Slice 2/Slice 3 already built and already validate themselves:

    finding.resolution   -> tools.proposal_matching.ProposalResolution,
                             checked by that module's own
                             validate_proposal_resolution() against the
                             caller-supplied `candidates` -- never
                             re-implemented here.
    finding.{trend, pattern_summary, possible_cause,
             recommended_improvement, expected_benefit, limitations,
             supporting_case_ids, confidence}
                          -> tools.reflector_proposals.ProposalObservation
                             fields -- that dataclass's own __post_init__
                             already enforces every taxonomy/confidence/
                             cross-field rule (trend value, confidence
                             range, supporting_case_ids shape, the
                             trend-requires-supporting-cases rule); this
                             module never duplicates any of those checks.
    a create_new finding's title
                          -> tools.reflector_proposals.ImprovementProposal
                             (this module's own responsibility: only
                             `parse_reflector_output` decides WHEN a new
                             ImprovementProposal is warranted -- one per
                             create_new finding -- and generates its
                             identity; see "IDs and timestamps" below).

The one genuinely new deterministic guardrail this slice adds is the
Supporting Case allowlist (see `_parse_finding`'s docstring): no existing
Slice 1-4 contract checks a finding's supporting_case_ids against the set
of Case ids actually present in the ReflectorPromptContext the model was
given, because no earlier slice had both a finding-shaped model response
and a Case allowlist to check it against at the same time.

IDs and timestamps (Phase 5 Slice 5A task instruction, section 17): the
model never produces `reflection_run_id`, `observation_id`, a create_new
finding's `proposal_id`, `supporting_case_count`, `observed_at`, or
`created_at` -- exactly like tools.case_enrichment_analyzer.
analyze_case_with_llm() never asks the model for `analysis_id`/
`analyzed_at`. Unlike that analyzer, though, this module's own return type
(ReflectionResult) itself carries id/timestamp-bearing nested objects
(ImprovementProposal, ProposalObservation) -- Slice 2 built those as
identity-bearing records, not id-free like CaseEnrichmentResult -- so this
parser cannot defer id/timestamp generation to a later runner stage the
way tools.run_case_enrichment's `_process_case()` does for `analysis_id`/
`analyzed_at` (see that function's own `uuid.uuid4().hex` +
`datetime.now(timezone.utc).isoformat()` call site, the established
convention this module's defaults mirror). `parse_reflector_output()`
therefore accepts `observed_at` as a required, caller-supplied string
(matching how `create_reflection_run(started_at=...)` already takes the
run's own timing as a plain caller-supplied argument, never self-computed)
and `id_factory` as a keyword-only, minimal dependency injection point
(default `lambda: uuid.uuid4().hex`, the exact literal already used by
tools.run_case_enrichment) -- purely so a test can supply a deterministic,
predictable id sequence instead of a real random UUID. This is one small
callable, not a generic ID-factory framework: the same `id_factory` is
reused for both a create_new finding's new `proposal_id` and every
finding's `observation_id` (uuid4 collision between the two is not a
real-world concern, and two separate factory parameters would be
unjustified complexity for what is structurally the same "give me a fresh
opaque id string" operation).

A create_new finding's ImprovementProposal.created_at is set to the SAME
`observed_at` value as its founding ProposalObservation.observed_at --
never a separately generated timestamp -- because
tools.reflector_proposals.ImprovementProposal's own docstring already
states this explicitly: "within this slice's actual code paths, a Proposal
is only ever created once, at the exact moment its first Observation is
produced, so the two would always hold the identical value." This module
follows that documented invariant rather than inventing a second timestamp
with no semantic difference from the first.

Title contract (Phase 5 Slice 5A task instruction, section 18 -- resolved
here, not left ambiguous): `title` is a required key on every finding (part
of this module's own strict key-set check), but its legal VALUE depends on
`resolution.action`, mirroring the exact "no in-between state" strictness
tools.case_enrichment.CaseEnrichmentResult already uses for its own
product_model/product_source/product_confidence null-triad:

    action == "create_new"     -> title MUST be a non-blank string (a new
                                   ImprovementProposal cannot be created
                                   without one).
    action == "match_existing" -> title MUST be `null` (`None`). A
                                   match_existing finding never creates or
                                   renames an ImprovementProposal, so a
                                   model-supplied title here has no
                                   destination to write to; rather than
                                   silently accepting and discarding an
                                   arbitrary string (which would let a
                                   model's title opinion influence nothing
                                   while looking like it was consumed),
                                   this module fails the whole parse
                                   closed, exactly as strict as the
                                   product_model=None/product_source-must-
                                   also-be-None rule this mirrors.

Possible_cause / expected_benefit / limitations nullability (a resolved
conflict between this slice's task instruction and Slice 2's already-
established contract -- see this module's implementation report for the
full note): the task instruction's own illustrative JSON shows these three
fields as plain `"string"`, but tools.reflector_proposals.
ProposalObservation already declares all three OPTIONAL (`str | None`) on
purpose ("the AI may legitimately have nothing supportable to state for
one of them on a given Observation ... None must never be forced into a
fabricated one-line placeholder"). Per this slice's own instruction
(section 23: existing Slice 1-4 contracts are the source of truth, never
silently narrowed to fit a new prompt), this module's schema hint documents
all three as nullable and its parser accepts `null` for any of them --
ProposalObservation's own __post_init__ still enforces "a non-None value
must be a non-blank string" either way.
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

# ---------------------------------------------------------------------------
# Reasoning policy -- stable, model-facing instructions. Not runtime
# implementation, not a DB/storage instruction of any kind (mirrors tools.
# case_enrichment_analyzer._SYSTEM_INSTRUCTIONS's own "prompt boundary"
# framing). Taxonomy value lists are interpolated from imported constants,
# never hard-coded, so this text can never silently drift out of sync with
# the actual ProposalResolution/ProposalObservation/ImprovementProposal
# contracts (same convention tools.case_enrichment_analyzer already
# established for its own _SYSTEM_INSTRUCTIONS).
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
    """Parse and validate one raw finding, or return None on any failure.

    Any structural problem (wrong shape, wrong key-set, invalid taxonomy,
    hallucinated proposal_id, target mismatch, hallucinated case_id,
    invalid confidence) is reported as None to its own caller,
    parse_reflector_output() -- this function itself is not required to be
    exception-free in isolation (a duplicate proposal_id inside
    `candidates`, a caller-assembled-collection bug rather than a property
    of `raw_finding`, surfaces here as a ValueError from
    validate_proposal_resolution()), but parse_reflector_output() wraps
    every call to this function in one try/except that already treats any
    ValueError/TypeError the same way a malformed `raw_finding` is treated:
    fail the whole parse closed. See that function's own docstring.

    Returns (new_proposal_or_None, observation) on success: a create_new
    finding returns (ImprovementProposal, ProposalObservation) -- exactly
    one founding Observation for the new Proposal it also returns, matching
    ReflectionResult's own "every new_proposals entry requires a
    corresponding proposal_observations entry" invariant. A match_existing
    finding returns (None, ProposalObservation) -- it never creates or
    renames an ImprovementProposal (see this module's own docstring,
    "Title contract").

    The Supporting Case allowlist check below is the one genuinely new
    deterministic guardrail this slice adds (see this module's own
    docstring) -- no earlier slice checks a finding's supporting_case_ids
    against the Case ids actually present in the ReflectorPromptContext the
    model was given; every other check here already exists on
    ProposalResolution/ProposalObservation/ImprovementProposal and is only
    invoked, never re-implemented.
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

    Mirrors tools.case_enrichment.parse_case_enrichment_result's own
    contract shape: only handles an already-decoded mapping (no markdown
    code-fence stripping, no streaming-chunk buffering -- this slice is not
    wired to any model call; that is Slice 5B's job), strict key-set
    validation at every nesting level, never raises for a malformed/
    adversarial `data` -- any structural problem anywhere (wrong root type,
    missing/extra keys at the top level or inside any finding/resolution,
    an invalid taxonomy value, a hallucinated proposal_id, an
    improvement_target mismatch against the matched candidate, a
    hallucinated/unlisted case_id, an out-of-range or non-numeric
    confidence, or any ReflectionResult/ImprovementProposal/
    ProposalObservation `__post_init__` rejection) returns None. Fail
    closed, never partial: a single invalid finding fails the WHOLE parse,
    not just that finding -- there is no "skip the bad finding and keep the
    rest" mode, matching the Phase 5 Slice 5A task instruction's explicit
    "整個 parse fail closed" requirement.

    `candidates` and `valid_case_ids` are the SAME two collections the
    model was actually shown for this run -- normally
    `ReflectorPromptContext.existing_proposals` and `{{c.case_id for c in
    ReflectorPromptContext.cases}}` respectively (see tools.
    reflector_prompt_context). This function never queries a store for
    either; both are plain caller-supplied arguments, exactly like tools.
    case_routing.validate_case_routing's own `candidate_case_ids` parameter
    for the structurally identical Case Routing problem.

    `reflection_run_id`/`observed_at` are plain caller-supplied strings,
    never generated here -- a future Slice 5B/6 runner is expected to
    generate `reflection_run_id` the same way tools.run_case_enrichment
    already generates `analysis_id` (`uuid.uuid4().hex`) and `observed_at`
    the same way it already generates `analyzed_at`
    (`datetime.now(timezone.utc).isoformat()`), then call
    `create_reflection_run()` with that same id before calling this
    function -- none of that persistence/orchestration is built in this
    slice. `id_factory` is a minimal, keyword-only dependency injection
    point (default `lambda: uuid.uuid4().hex`) used for every id this
    function itself must mint (a create_new finding's new proposal_id,
    and every finding's observation_id) -- see this module's own docstring
    for why one factory is enough.
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
    rejecting the decoded response. A single, minimal exception type on
    purpose, matching tools.case_enrichment_analyzer.
    CaseEnrichmentAnalyzerError's own "one exception type, message carries
    the detail" convention -- the message + chained __cause__ (via
    `raise ... from exc`) carries the failure detail for logs/debugging
    instead of a multi-class hierarchy. A caller never needs to know
    whether a given failure was a provider error, a malformed response, or
    a rejected (e.g. hallucinated-id) response -- all three mean the same
    thing to it: "this Reflection Run produced no usable result."
    """


def _build_messages(context: ReflectorPromptContext) -> list[dict[str, str]]:
    """Exactly two messages: system reasoning policy + serialized
    ReflectorPromptContext. Nothing else -- no session history, no SOUL, no
    tool definitions, no re-retrieval instructions. This is the entire
    prompt boundary (mirrors tools.case_enrichment_analyzer._build_
    messages's own "no other context leaks in" guarantee).

    Reuses serialize_reflector_prompt_context() (Slice 4) unchanged -- this
    module builds no second, competing context serializer. Business/
    reasoning policy (recurring-pattern semantics, improvement_target
    meaning, match/new judgment, evidence discipline, etc.) lives entirely
    in `_SYSTEM_INSTRUCTIONS`, never repeated or diluted into this user
    message -- the user message is only ever a label plus the deterministic
    JSON payload, matching _build_messages's own precedent in tools.
    case_enrichment_analyzer (`f"Case evidence (JSON):\\n{evidence_json}"`).
    """
    serialized = serialize_reflector_prompt_context(context)
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Reflector context (JSON):\n{serialized}"},
    ]


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of the OpenAI-compatible response object
    agent.auxiliary_client.call_llm() returns (`.choices[0].message.content`
    as a plain string) -- identical shape to, and copied from, tools.
    case_enrichment_analyzer._extract_text (see that function's own
    docstring). Deliberately not a generic multi-shape parser: any other
    shape, or a non-string content, returns "" so the caller's empty-
    response check fails closed rather than silently accepting something
    unexpected.
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
    parse_reflector_output() -> ReflectionResult. The first place in this
    domain that actually performs semantic reasoning (recurring-pattern
    identification, improvement_target interpretation, match_existing vs.
    create_new judgment, trend/possible_cause/confidence) -- Slices 1-5A
    only assembled evidence and validated shape deterministically.

    `candidates`/`valid_case_ids` are deliberately NOT separate parameters
    here, unlike parse_reflector_output()'s own signature: `context.
    existing_proposals` already IS `tuple[ProposalCandidate, ...]` (see
    tools.reflector_prompt_context.ReflectorPromptContext's own field type
    -- confirmed by reading that module before writing this function, not
    assumed), the exact type validate_proposal_resolution() requires, so
    passing it straight through is real reuse, not a same-shaped-but-
    different-semantics conversion. `valid_case_ids` is derived the same
    way, as `{{c.case_id for c in context.cases}}`. A caller that already
    has a ReflectorPromptContext therefore never needs to reconstruct or
    re-pass either collection.

    `main_runtime` should carry the SAME provider/model/base_url/api_key/
    api_mode the main Hermes agent is already using -- passed straight
    through to `llm_call`, never resolved or hard-coded in this module
    (identical contract to tools.case_enrichment_analyzer.
    analyze_case_with_llm's own `main_runtime` parameter).

    `llm_call` defaults to `agent.auxiliary_client.call_llm`, imported here
    (not at module level) so this module stays importable in this repo's
    test environment, which has no Hermes runtime installed -- calling
    without an injected `llm_call` surfaces the resulting ImportError
    undisguised (not wrapped into ReflectorAnalyzerError, which means "the
    LLM call itself, or its output, failed" -- a missing package is an
    environment problem, not that), matching tools.case_enrichment_
    analyzer.analyze_case_with_llm's own lazy-import contract exactly.

    `id_factory` is forwarded unchanged to parse_reflector_output() -- see
    that function's own docstring for why one factory is enough.

    Raises ReflectorAnalyzerError -- never returns None -- for any failure
    downstream of a successful lazy import: provider/network error, empty/
    unreadable response, invalid JSON, or parse_reflector_output()
    rejecting the decoded response (a fail-closed rejection -- hallucinated
    proposal_id, hallucinated case_id, invalid taxonomy, invalid
    confidence, wrong key-set, or an invalid material-change combination
    -- is reported exactly the same way as any other analyzer failure;
    this function never retries, never asks the model to "fix" its JSON,
    and never guesses a closest match on the caller's behalf). No
    application-level retry anywhere in this function -- `agent.
    auxiliary_client.call_llm` already retries/falls back at the provider
    layer, matching tools.case_enrichment_analyzer.analyze_case_with_llm's
    own "no retry here" contract. No persistence anywhere in this
    function either: it returns a plain in-memory ReflectionResult, never
    calls FeedbackStoreV2, and never creates a reflection_runs row -- that
    stays a future runner's job (Slice 6), exactly like tools.
    case_enrichment_analyzer.analyze_case_with_llm never calls
    create_case_analysis().
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
    "parse_reflector_output",
    "ReflectorAnalyzerError",
    "analyze_reflection_with_llm",
]
