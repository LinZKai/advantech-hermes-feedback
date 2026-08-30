#!/usr/bin/env python3
"""Reflector Real Reasoning Smoke Test -- manual dev aid, NOT production code.

This script is not part of the Hermes overlay (it lives outside overlay/, so
it is never deployed alongside tools/reflector_analyzer.py), is not wired
into any production path, is not run by the test suite, and never touches a
database. It exists so a human can eyeball the real Reflector prompt +
parser against a controlled synthetic input; nothing depends on it.

Purpose: exercise the REAL production Reflector analyzer boundary --

    ReflectorPromptContext
      -> serialize_reflector_prompt_context()
      -> analyze_reflection_with_llm()   (tools.reflector_analyzer, Slice 5B)
      -> real agent.auxiliary_client.call_llm()
      -> parse_reflector_output()        (tools.reflector_analyzer, Slice 5A)
      -> ReflectionResult

against one small, hand-built, controlled synthetic ReflectorPromptContext,
so a human can read the model's raw JSON and the parsed ReflectionResult and
judge whether the Reflector reasoning policy (_SYSTEM_INSTRUCTIONS) produces
sensible groupings on cases whose intended relationships are already known
by construction.

This script deliberately calls analyze_reflection_with_llm() -- the exact
same function a future Slice 6 runner will call -- rather than hand-rolling
a provider request. It performs NO application-level retry, NO prompt
patching on failure, and does NOT bypass parse_reflector_output() if the
model's output is rejected; a rejection is printed and the script exits
non-zero.

--------------------------------------------------------------------------
Where to run this
--------------------------------------------------------------------------
`agent.auxiliary_client` only exists inside a built Hermes sandbox image
(this is the exact same lazy-import boundary tools.reflector_analyzer.
analyze_reflection_with_llm() and tools.case_enrichment_analyzer.
analyze_case_with_llm() both already rely on) -- it is NOT importable from
a plain local checkout of this repo. Run this script from wherever the
`agent` package and a configured provider are actually available (e.g. the
Hermes sandbox / VM this repo is normally deployed to), with this repo's
overlay/ on PYTHONPATH:

    cd /path/to/advantech-hermes-feedback/custom/universal-feedback
    python3 scripts/reflector_reasoning_smoke_test.py

Optional flags:
    --dry-run   Uses a scripted fake llm_call instead of the real one, so
                the SCRIPT'S OWN MECHANICS (context construction, serialization,
                the analyzer call, parsing) can be sanity-checked without a
                real Hermes runtime. This does NOT exercise real reasoning
                quality -- it only proves the script itself has no bugs.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import CaseAnalysisEvidence  # noqa: E402
from tools.proposal_matching import ProposalCandidate  # noqa: E402
from tools.reflector_analyzer import ReflectorAnalyzerError, analyze_reflection_with_llm  # noqa: E402
from tools.reflector_prompt_context import (  # noqa: E402
    AnalysisWindow,
    CaseIntelligenceProjection,
    ReflectorContextSummary,
    ReflectorPromptContext,
    serialize_reflector_prompt_context,
)


# ---------------------------------------------------------------------------
# Synthetic scenario: Group A (knowledge gap x3), Group B (agent_behavior x2),
# Group C (unrelated noise x1), Group D (ambiguous x1) -- 7 Cases total.
# ---------------------------------------------------------------------------


def _evidence(**overrides: Any) -> CaseAnalysisEvidence:
    base = dict(type="user_text", turn_id="turn-1", fact="")
    base.update(overrides)
    return CaseAnalysisEvidence(**base)


def build_synthetic_context() -> ReflectorPromptContext:
    cases = (
        # --- Group A: Knowledge Gap recurring pattern (3 independent Cases) ---
        CaseIntelligenceProjection(
            case_id="case-A1",
            case_title="ADAM-6266: how to disable SNMP",
            issue_summary="User asked how to disable SNMP on ADAM-6266; no explicit disable command was found.",
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.95,
            issue_type="product_usage_or_application", issue_type_confidence=0.85,
            diagnosis="knowledge_gap", diagnosis_confidence=0.8,
            evidence=(
                _evidence(type="user_text", turn_id="turn-a1-1", fact="User asked: 'ADAM-6266 要怎麼關閉 SNMP？'"),
                _evidence(type="retrieval", turn_id="turn-a1-1", fact="Retrieval executed, foundry_iq_ok=true, result_count=6, but no chunk contained an explicit SNMP disable command."),
                _evidence(type="assistant_text", turn_id="turn-a1-1", fact="Assistant answered with general guidance to check the network settings menu, without a specific command."),
            ),
        ),
        CaseIntelligenceProjection(
            case_id="case-A2",
            case_title="ADAM-6266 SNMP disable command",
            issue_summary="A different, independent session asked the same ADAM-6266 SNMP disable question.",
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.93,
            issue_type="product_usage_or_application", issue_type_confidence=0.82,
            diagnosis="knowledge_gap", diagnosis_confidence=0.78,
            evidence=(
                _evidence(type="user_text", turn_id="turn-a2-1", fact="User asked: 'How do I turn off SNMP on my ADAM-6266?'"),
                _evidence(type="retrieval", turn_id="turn-a2-1", fact="Retrieval executed, foundry_iq_ok=true, result_count=4, no explicit disable command located in returned chunks."),
                _evidence(type="feedback", turn_id="turn-a2-1", fact="Feedback: helpful=false, reason_code=incomplete."),
            ),
        ),
        CaseIntelligenceProjection(
            case_id="case-A3",
            case_title="ADAM-6266 SNMP configuration command needed",
            issue_summary="A third, independent Case again needs a specific SNMP configuration command for ADAM-6266.",
            product_model="ADAM-6266", product_source="inference", product_confidence=0.7,
            issue_type="product_usage_or_application", issue_type_confidence=0.8,
            diagnosis="knowledge_gap", diagnosis_confidence=0.75,
            evidence=(
                _evidence(type="user_text", turn_id="turn-a3-1", fact="User asked for the exact CLI/utility command to disable SNMP on their ADAM module."),
                _evidence(type="retrieval", turn_id="turn-a3-1", fact="Retrieval executed, foundry_iq_ok=true, result_count=5, general SNMP overview returned but no disable command."),
            ),
        ),
        # --- Group B: Agent Behavior (2 Cases) -- evidence already contains
        # the command, but the assistant's own answer omitted it. ---
        CaseIntelligenceProjection(
            case_id="case-B1",
            case_title="ADAM-6266 answer omitted SNMP disable command",
            issue_summary="Retrieval surfaced the SNMP disable command, but the assistant's final answer did not state it.",
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.9,
            issue_type="product_usage_or_application", issue_type_confidence=0.85,
            diagnosis="answer_quality_issue", diagnosis_confidence=0.75,
            evidence=(
                _evidence(type="user_text", turn_id="turn-b1-1", fact="User asked: 'ADAM-6266 SNMP disable 指令是什麼？'"),
                _evidence(type="retrieval", turn_id="turn-b1-1", fact="Retrieval executed, foundry_iq_ok=true, result_count=3, one returned chunk explicitly contains the SNMP disable command text."),
                _evidence(type="assistant_text", turn_id="turn-b1-1", fact="Assistant's final answer described navigating to the settings page but did not state the retrieved command."),
                _evidence(type="feedback", turn_id="turn-b1-1", fact="Feedback: helpful=false, reason_code=incomplete."),
            ),
        ),
        CaseIntelligenceProjection(
            case_id="case-B2",
            case_title="ADAM-6266 SNMP steps given without the command itself",
            issue_summary="Retrieval evidence contained the explicit command; the answer gave only conceptual steps.",
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.88,
            issue_type="product_usage_or_application", issue_type_confidence=0.8,
            diagnosis="answer_quality_issue", diagnosis_confidence=0.7,
            evidence=(
                _evidence(type="user_text", turn_id="turn-b2-1", fact="User asked how to disable SNMP and what the exact utility command is."),
                _evidence(type="retrieval", turn_id="turn-b2-1", fact="Retrieval executed, foundry_iq_ok=true, result_count=4, returned chunk includes the exact utility command."),
                _evidence(type="assistant_text", turn_id="turn-b2-1", fact="Assistant answered only with 'open the utility and disable SNMP in settings', never quoting the command from the retrieved chunk."),
            ),
        ),
        # --- Group C: Unrelated noise (1 Case, different product/topic) ---
        CaseIntelligenceProjection(
            case_id="case-C1",
            case_title="WISE-6610 power redundancy question",
            issue_summary="User asked whether WISE-6610 supports redundant power input; documentation confirmed it does.",
            product_model="WISE-6610", product_source="explicit_user_text", product_confidence=0.9,
            issue_type="product_capability_or_compatibility", issue_type_confidence=0.85,
            diagnosis="no_issue_detected", diagnosis_confidence=0.8,
            evidence=(
                _evidence(type="user_text", turn_id="turn-c1-1", fact="User asked: 'Does WISE-6610 support dual power input for redundancy?'"),
                _evidence(type="retrieval", turn_id="turn-c1-1", fact="Retrieval executed, foundry_iq_ok=true, result_count=2, spec sheet confirms dual power input support."),
                _evidence(type="feedback", turn_id="turn-c1-1", fact="Feedback: helpful=true."),
            ),
        ),
        # --- Group D: Ambiguous / thin evidence (1 Case, optional per task) ---
        CaseIntelligenceProjection(
            case_id="case-D1",
            case_title="ADAM-6266 SNMP-related question, unclear",
            issue_summary="User's SNMP-related question on ADAM-6266 was vague; evidence is too thin to attribute a clear cause.",
            product_model="ADAM-6266", product_source="inference", product_confidence=0.4,
            issue_type="other_or_unclear", issue_type_confidence=0.4,
            diagnosis="other_or_unclear", diagnosis_confidence=0.35,
            evidence=(
                _evidence(type="user_text", turn_id="turn-d1-1", fact="User asked a short, ambiguous SNMP-related question without enough detail to classify."),
            ),
        ),
    )

    existing_proposals = (
        ProposalCandidate(
            proposal_id="proposal-existing-knowledge",
            improvement_target="knowledge",
            title="ADAM-6266 SNMP Knowledge Improvement",
            review_status="pending",
            latest_pattern_summary="Multiple independent cases lack clear SNMP configuration command guidance.",
            latest_recommended_improvement="Add explicit validated SNMP command guidance to the knowledge source.",
            latest_trend="new",
            latest_supporting_case_count=2,
            latest_confidence=0.6,
        ),
    )

    by_product_model: dict[str, int] = {}
    by_issue_type: dict[str, int] = {}
    by_diagnosis: dict[str, int] = {}
    for case in cases:
        product_key = case.product_model if case.product_model is not None else "__unknown__"
        by_product_model[product_key] = by_product_model.get(product_key, 0) + 1
        by_issue_type[case.issue_type] = by_issue_type.get(case.issue_type, 0) + 1
        by_diagnosis[case.diagnosis] = by_diagnosis.get(case.diagnosis, 0) + 1

    return ReflectorPromptContext(
        analysis_window=AnalysisWindow(start="2026-07-01T00:00:00+00:00", end="2026-08-01T00:00:00+00:00"),
        summary=ReflectorContextSummary(
            analyzed_case_count=len(cases),
            by_product_model=by_product_model,
            by_issue_type=by_issue_type,
            by_diagnosis=by_diagnosis,
        ),
        cases=cases,
        existing_proposals=existing_proposals,
    )


# ---------------------------------------------------------------------------
# --dry-run only: a scripted fake, used ONLY to sanity-check this script's
# own mechanics without a real Hermes runtime. Never presented as evidence
# of real reasoning quality.
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def _dry_run_fake_llm_call(**kwargs: Any) -> _FakeResponse:
    data = {
        "run_summary": "[DRY RUN -- scripted fake, not a real model response] One knowledge-gap pattern and one agent-behavior pattern found; noise and ambiguous Cases excluded.",
        "material_change_detected": True,
        "findings": [
            {
                "resolution": {"action": "match_existing", "proposal_id": "proposal-existing-knowledge", "improvement_target": "knowledge"},
                "title": None,
                "trend": "growing",
                "pattern_summary": "[DRY RUN] Three independent Cases (A1-A3) ask for the ADAM-6266 SNMP disable command with no clear KB answer.",
                "possible_cause": "[DRY RUN] The KB may not document the exact SNMP disable command for this model.",
                "recommended_improvement": "[DRY RUN] Add the explicit SNMP disable command to the ADAM-6266 KB article.",
                "expected_benefit": "[DRY RUN] Fewer repeat questions on this topic.",
                "limitations": "[DRY RUN] Based on three Cases only.",
                "supporting_case_ids": ["case-A1", "case-A2", "case-A3"],
                "confidence": 0.7,
            },
            {
                "resolution": {"action": "create_new", "proposal_id": None, "improvement_target": "agent_behavior"},
                "title": "[DRY RUN] Hermes omits retrieved SNMP command from its answer",
                "trend": "new",
                "pattern_summary": "[DRY RUN] Two Cases (B1-B2) show retrieval surfaced the command but the answer omitted it.",
                "possible_cause": "[DRY RUN] The answer-generation step may not consistently surface a retrieved command.",
                "recommended_improvement": "[DRY RUN] Review answer generation for command-omission on evidence-usage.",
                "expected_benefit": "[DRY RUN] More complete answers when evidence already has the command.",
                "limitations": "[DRY RUN] Based on two Cases only.",
                "supporting_case_ids": ["case-B1", "case-B2"],
                "confidence": 0.6,
            },
        ],
    }
    return _FakeResponse(json.dumps(data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use a scripted fake llm_call to sanity-check this script's own mechanics only "
             "(does not exercise real reasoning quality).",
    )
    args = parser.parse_args()

    context = build_synthetic_context()
    reflection_run_id = f"smoke-test-{uuid.uuid4().hex[:8]}"
    observed_at = datetime.now(timezone.utc).isoformat()

    print("=" * 78)
    print("Serialized ReflectorPromptContext (this is the exact input the model sees)")
    print("=" * 78)
    print(serialize_reflector_prompt_context(context))
    print()

    llm_call = _dry_run_fake_llm_call if args.dry_run else None

    print("=" * 78)
    print(f"Calling analyze_reflection_with_llm() [{'DRY RUN -- fake' if args.dry_run else 'REAL auxiliary LLM'}]")
    print("=" * 78)

    try:
        result = analyze_reflection_with_llm(
            context,
            reflection_run_id=reflection_run_id,
            observed_at=observed_at,
            llm_call=llm_call,
        )
    except ReflectorAnalyzerError as exc:
        print("REJECTED: analyze_reflection_with_llm() raised ReflectorAnalyzerError")
        print(f"  {exc}")
        print("(This means either the LLM call itself failed, the response was not valid JSON,")
        print(" or parse_reflector_output() fail-closed rejected the decoded response. See the")
        print(" report for how to tell these apart.)")
        return 1
    except ImportError as exc:
        print("ImportError: agent.auxiliary_client is not importable in this environment.")
        print(f"  {exc}")
        print("Run this script where the Hermes `agent` package and a configured provider")
        print("are actually available (see this script's module docstring), or pass --dry-run")
        print("to sanity-check the script's own mechanics only.")
        return 2

    print()
    print("=" * 78)
    print("Parsed ReflectionResult")
    print("=" * 78)
    print(f"reflection_run_id: {result.reflection_run_id}")
    print(f"run_summary: {result.run_summary}")
    print(f"material_change_detected: {result.material_change_detected}")
    print()
    print(f"new_proposals ({len(result.new_proposals)}):")
    for proposal in result.new_proposals:
        print(f"  - proposal_id={proposal.proposal_id}")
        print(f"    improvement_target={proposal.improvement_target}")
        print(f"    title={proposal.title!r}")
        print(f"    review_status={proposal.review_status}")
    print()
    print(f"proposal_observations ({len(result.proposal_observations)}):")
    for observation in result.proposal_observations:
        matched_new = next((p for p in result.new_proposals if p.proposal_id == observation.proposal_id), None)
        action = "create_new" if matched_new is not None else "match_existing"
        print(f"  - proposal_id={observation.proposal_id} action={action}")
        print(f"    trend={observation.trend}")
        print(f"    confidence={observation.confidence}")
        print(f"    supporting_case_ids={observation.supporting_case_ids}")
        print(f"    pattern_summary={observation.pattern_summary!r}")
        print(f"    possible_cause={observation.possible_cause!r}")
        print(f"    recommended_improvement={observation.recommended_improvement!r}")
        print(f"    expected_benefit={observation.expected_benefit!r}")
        print(f"    limitations={observation.limitations!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
