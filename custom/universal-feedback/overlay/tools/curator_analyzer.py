"""The Curator auxiliary LLM integration: structured-output parser and
deterministic validation, plus the call_llm() wiring that produces the
JSON this parser consumes -- Curator Slice 1.

    CuratorPromptContext (tools.curator_prompt_context)
     |
     v
    _build_messages()        (dedicated Curator prompt -- no SOUL/memory/
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
    parse_curator_output()    (strict shape check, fail closed --
     |                         cross-field validation delegated to
     |                         tools.curator_domain.CuratorChange's own
     |                         __post_init__)
     v
    CuratorChange (tools.curator_domain)

This module calls call_llm() directly, not run_conversation(): Curator
analyzes an already-structured, already-accepted Proposal and must never
re-query retrieval, see SOUL/memory/session history, or gain access to
conversational tools -- mirrors tools.reflector_analyzer's own reasoning
for the same choice exactly.

`llm_call` defaults to a lazy import of `agent.auxiliary_client.call_llm`
(resolved inside the function body) because that module only exists inside
a built Hermes sandbox image -- this repo's test environment does not have
it installed, and every test injects a fake `llm_call` instead.

A few behaviors worth knowing before editing:
- `parse_curator_output()` fails the WHOLE parse closed on any single
  invalid field -- there is no "best-effort" partial acceptance.
- target_file/change_type/confidence/proposed_content cross-field rules
  are NOT re-implemented here: constructing tools.curator_domain.
  CuratorChange does that validation, and any ValueError it raises is
  caught here and turned into a `None` return (same shape as tools.
  reflector_analyzer.parse_reflector_output's own try/except).
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Mapping

from tools.curator_domain import CURATOR_TARGET_FILE, CuratorChange
from tools.curator_prompt_context import CuratorPromptContext, serialize_curator_prompt_context
from tools.feedback_store_v2 import CHANGE_TYPE_VALUES

# ---------------------------------------------------------------------------
# Reasoning policy -- stable, model-facing instructions. The change_type
# value list is interpolated from an imported constant, never hard-coded,
# so this text can never silently drift out of sync with the actual
# CuratorChange contract.
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA_HINT = f"""{{
  "change_type": "<one of: {' | '.join(CHANGE_TYPE_VALUES)}>",
  "target_file": "<must be exactly {CURATOR_TARGET_FILE!r}>",
  "rationale": "<non-blank string>",
  "proposed_content": "<the COMPLETE replacement content of {CURATOR_TARGET_FILE}; required non-blank string when change_type is add_rule/modify_rule/remove_rule, otherwise null>",
  "expected_effect": "<string, or null>",
  "confidence": <number 0.0-1.0>
}}"""

_SYSTEM_INSTRUCTIONS = f"""You are the Improvement-Proposal Curator for a technical-support AI assistant. You are given ONE already human-accepted Improvement Proposal (improvement_target=agent_behavior -- you are never invoked for any other improvement_target; this has already been verified before you were invoked, you do not need to re-classify it), its latest supporting Observation, the Case evidence behind that Observation, and the CURRENT COMPLETE content of the one file you may propose a change to. This is an independent offline analysis task, not a conversation -- you must not assume any other context.

## Scope -- read carefully, this is a hard boundary

* The ONLY file you may propose a change to is {CURATOR_TARGET_FILE}. You are given its complete current content. You must never propose editing, mentioning, or referencing any other file -- not SOUL.md, not a Skill file (e.g. foundry-iq/SKILL.md), not any Python source, database schema, Dockerfile, credential, or any other path. Your "target_file" output must be exactly {CURATOR_TARGET_FILE!r}, always.
* Do not introduce product-specific facts, commands, values, or claims into {CURATOR_TARGET_FILE}. It is a GENERIC response-structure/behavior instruction file, never a place for Advantech product knowledge -- that lives in the knowledge base / retrieval Skill, which you cannot touch.
* Do not rewrite the assistant's whole personality, identity, or role. {CURATOR_TARGET_FILE} is deliberately narrow (response-structure instructions only) -- stay within that scope.
* Make the SMALLEST behavior change that plausibly addresses the recurring pattern described in the Proposal/Observation. Prefer a small, targeted edit over a broad rewrite.
* Preserve every existing rule in the current {CURATOR_TARGET_FILE} content that is not directly related to the pattern you are addressing -- an unrelated existing rule must survive into your proposed_content unchanged in substance, unless removing or changing it is the specific point of your change.
* If {CURATOR_TARGET_FILE} is genuinely not an appropriate place to address this Proposal (for example the recommended_improvement is not really about the assistant's general response structure/behavior, or the current content already covers it adequately), return change_type="no_change_recommended". This is a fully valid, expected outcome, not a failure -- do not force a change to avoid an empty-looking result.

## change_type

Set change_type to exactly one of: {', '.join(CHANGE_TYPE_VALUES)}.

* "add_rule" -- {CURATOR_TARGET_FILE} does not yet address this pattern; you are adding new instruction content.
* "modify_rule" -- {CURATOR_TARGET_FILE} already has a related rule, but it needs to change to address this pattern.
* "remove_rule" -- an existing rule in {CURATOR_TARGET_FILE} is itself the cause of the pattern and should be removed. Rare -- only when the rule itself is actively counterproductive.
* "no_change_recommended" -- see the Scope section above. proposed_content MUST be null for this value.

## proposed_content

For add_rule/modify_rule/remove_rule: the COMPLETE replacement content of {CURATOR_TARGET_FILE} -- not a diff, not a snippet, not just the new or changed lines. You are given the complete current content; return the complete new content, with your change applied and every unrelated existing rule preserved. This file is small by design, and a full-content replacement is the deliberate simplification for this slice -- there is no patch/diff mechanism here.

For no_change_recommended: proposed_content MUST be null.

## Evidence discipline

Ground your rationale in the Proposal's pattern_summary/recommended_improvement and the supporting Case evidence you were given. Do not invent a pattern that is not actually reflected in the given evidence.

## Language

proposed_content must be written in the same language and register as the current {CURATOR_TARGET_FILE} content you were given -- it becomes the literal runtime instruction file verbatim if a human reviewer later applies it. rationale and expected_effect are for a human reviewer and should be written in Traditional Chinese (zh-TW), preserving product names, technical terms, code, and identifiers in their original form, matching this system's existing convention for reviewer-facing narrative text.

## Human review boundary

A CuratorChange you produce is a recommendation for a human reviewer -- it is never applied automatically, regardless of confidence.

## Output format

Respond with a single JSON object and nothing else -- no markdown code fences, no explanation before or after the JSON. Respond with a JSON object matching exactly this shape -- no extra fields, no missing fields:
{_OUTPUT_SCHEMA_HINT}"""


# ---------------------------------------------------------------------------
# Strict key-set -- mirrors tools.reflector_analyzer.parse_reflector_
# output's own "unknown top-level keys are REJECTED, not ignored" contract.
# ---------------------------------------------------------------------------

_RESULT_REQUIRED_KEYS = frozenset({
    "change_type", "target_file", "rationale", "proposed_content", "expected_effect", "confidence",
})


def parse_curator_output(
    data: Any,
    *,
    proposal_id: str,
    before_content: str,
    created_at: str,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> CuratorChange | None:
    """Validate an already-decoded Python mapping (e.g. json.loads()
    output) into a CuratorChange, or return None if it is not a legal one.

    Only handles an already-decoded mapping (no markdown code-fence
    stripping, no streaming-chunk buffering). Never raises for a
    malformed/adversarial `data`; any structural problem returns None.
    Fails the WHOLE parse closed -- there is no partial acceptance.

    `proposal_id`/`before_content`/`created_at` are plain caller-supplied
    values, never read from `data` (before_content in particular must be
    the ACTUAL current file content the caller read, never whatever the
    model may have echoed back). `id_factory` is a minimal, keyword-only
    dependency injection point (default `lambda: uuid.uuid4().hex`) for
    this parse's own change_id.
    """
    if not isinstance(data, Mapping) or set(data.keys()) != _RESULT_REQUIRED_KEYS:
        return None

    try:
        return CuratorChange(
            change_id=id_factory(),
            proposal_id=proposal_id,
            target_file=data.get("target_file"),
            change_type=data.get("change_type"),
            rationale=data.get("rationale"),
            before_content=before_content,
            proposed_content=data.get("proposed_content"),
            expected_effect=data.get("expected_effect"),
            confidence=data.get("confidence"),
            status="proposed",
            created_at=created_at,
            reviewed_at=None,
            applied_at=None,
        )
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# The real LLM call -- mirrors tools.reflector_analyzer.
# analyze_reflection_with_llm's shape exactly.
# ---------------------------------------------------------------------------


class CuratorAnalyzerError(RuntimeError):
    """Raised by analyze_proposal_with_llm() for any failure that prevents
    producing a valid CuratorChange -- provider/network failure, empty
    response, invalid JSON, or parse_curator_output() rejecting the
    decoded response. A single, minimal exception type, matching tools.
    reflector_analyzer.ReflectorAnalyzerError's own reasoning: a caller
    never needs to distinguish a provider error from a rejected response,
    both mean "this Curator run produced no usable result."
    """


def _build_messages(context: CuratorPromptContext) -> list[dict[str, str]]:
    """Exactly two messages: system reasoning policy + serialized
    CuratorPromptContext. Nothing else -- no session history, no SOUL, no
    tool definitions."""
    serialized = serialize_curator_prompt_context(context)
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Curator context (JSON):\n{serialized}"},
    ]


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of the OpenAI-compatible response object
    agent.auxiliary_client.call_llm() returns (`.choices[0].message.content`
    as a plain string). Any other shape, or a non-string content, returns
    "" so the caller's empty-response check fails closed rather than
    silently accepting something unexpected. Identical to tools.
    reflector_analyzer._extract_text -- a small, intentional duplication
    (one four-line helper) rather than a cross-module import for something
    this trivial.
    """
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def analyze_proposal_with_llm(
    context: CuratorPromptContext,
    *,
    proposal_id: str,
    created_at: str,
    main_runtime: Mapping[str, Any] | None = None,
    llm_call: Callable[..., Any] | None = None,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> CuratorChange:
    """The real LLM analyzer: `context` -> one call_llm() invocation ->
    parse_curator_output() -> CuratorChange.

    `main_runtime` should carry the SAME provider/model/base_url/api_key/
    api_mode the main Hermes agent is already using -- passed straight
    through to `llm_call`, never resolved or hard-coded here.

    `llm_call` defaults to `agent.auxiliary_client.call_llm`, imported here
    (not at module level) so this module stays importable in this repo's
    test environment, which has no Hermes runtime installed -- calling
    without an injected `llm_call` surfaces the resulting ImportError
    undisguised (not wrapped into CuratorAnalyzerError), matching tools.
    reflector_analyzer.analyze_reflection_with_llm's own precedent exactly.

    Raises CuratorAnalyzerError -- never returns None -- for any failure
    downstream of a successful lazy import: provider/network error, empty/
    unreadable response, invalid JSON, or parse_curator_output() rejecting
    the decoded response. No application-level retry; no persistence
    either (returns a plain in-memory CuratorChange, never calls
    FeedbackStoreV2).
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
        raise CuratorAnalyzerError(
            f"call_llm failed for proposal_id {proposal_id!r}: {type(exc).__name__}: {exc}"
        ) from exc

    text = _extract_text(response)
    if not text.strip():
        raise CuratorAnalyzerError(
            f"LLM response for proposal_id {proposal_id!r} contained no extractable text"
        )

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CuratorAnalyzerError(
            f"LLM response for proposal_id {proposal_id!r} was not valid JSON: {exc}"
        ) from exc

    change = parse_curator_output(
        parsed,
        proposal_id=proposal_id,
        before_content=context.current_content,
        created_at=created_at,
        id_factory=id_factory,
    )
    if change is None:
        raise CuratorAnalyzerError(
            f"LLM response for proposal_id {proposal_id!r} failed CuratorChange contract validation"
        )

    return change


__all__ = [
    "parse_curator_output",
    "CuratorAnalyzerError",
    "analyze_proposal_with_llm",
]
