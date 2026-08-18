"""Phase 5 Slice 5A/5B tests: the Reflector structured-output parser
(parse_reflector_output, Slice 5A) and the real LLM analyzer
(analyze_reflection_with_llm, Slice 5B) in tools.reflector_analyzer.

No DB, no network, no real Hermes provider anywhere in this file.

Slice 5A tests: parse_reflector_output() is a pure function over an
already-decoded Python mapping (the shape json.loads() would produce) plus
already-built in-memory objects (ProposalCandidate), so every fixture is
constructed directly, matching test_case_enrichment_analyzer.py's / test_
reflector_prompt_context.py's no-DB testing style for the equivalent Stage
D1 / Slice 4 contracts.

Slice 5B tests (AnalyzeReflectionWithLlmTests onward): every test injects a
fake `llm_call` -- no import of `agent.auxiliary_client` (that package only
exists inside a built Hermes sandbox image, not in this repo's test
environment). Only DefaultLlmCallLazyImportTests touches the lazy import at
all, exactly mirroring test_case_enrichment_analyzer.py's own
DefaultLlmCallLazyImportTests precedent.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.proposal_matching import ProposalCandidate  # noqa: E402
from tools.reflector_analyzer import (  # noqa: E402
    _OUTPUT_SCHEMA_HINT,
    _SYSTEM_INSTRUCTIONS,
    ReflectorAnalyzerError,
    analyze_reflection_with_llm,
    parse_reflector_output,
)
from tools.reflector_prompt_context import (  # noqa: E402
    AnalysisWindow,
    CaseIntelligenceProjection,
    ReflectorContextSummary,
    ReflectorPromptContext,
)
from tools.reflector_proposals import ImprovementProposal, ProposalObservation, ReflectionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate(**overrides: Any) -> ProposalCandidate:
    base = dict(
        proposal_id="proposal-1",
        improvement_target="knowledge",
        title="ADAM-6266 SNMP disable command undocumented",
        review_status="pending",
        latest_pattern_summary="Three Cases in the last 30 days ask how to disable SNMP on ADAM-6266.",
        latest_recommended_improvement="Add an explicit SNMP disable procedure to the ADAM-6266 KB article.",
        latest_trend="new",
        latest_supporting_case_count=3,
        latest_confidence=0.7,
    )
    base.update(overrides)
    return ProposalCandidate(**base)


def _resolution_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "action": "create_new",
        "proposal_id": None,
        "improvement_target": "knowledge",
    }
    base.update(overrides)
    return base


def _finding_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "resolution": _resolution_dict(),
        "title": "ADAM-6350 UAexpert cert trust step undocumented",
        "trend": "new",
        "pattern_summary": "Two Cases ask how to trust the UAexpert certificate on ADAM-6350.",
        "possible_cause": "The KB may not clearly document the certificate trust step.",
        "recommended_improvement": "Add the certificate trust step to the ADAM-6350 KB article.",
        "expected_benefit": "Fewer repeat questions on this specific topic.",
        "limitations": "Based on only two Cases; may not generalize.",
        "supporting_case_ids": ["case-1", "case-2"],
        "confidence": 0.7,
    }
    base.update(overrides)
    return base


def _result_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_summary": "Found one recurring knowledge gap across the analyzed Cases.",
        "material_change_detected": True,
        "findings": [_finding_dict()],
    }
    base.update(overrides)
    return base


class _ReflectorAnalyzerTestCase(unittest.TestCase):
    def _parse(self, data: Any, **overrides: Any) -> ReflectionResult | None:
        counter = iter(f"id-{i}" for i in range(1, 100))
        kwargs: dict[str, Any] = dict(
            candidates=(_candidate(),),
            valid_case_ids=("case-1", "case-2"),
            reflection_run_id="run-1",
            observed_at="2026-01-02T00:00:00+00:00",
            id_factory=lambda: next(counter),
        )
        kwargs.update(overrides)
        return parse_reflector_output(data, **kwargs)


# ---------------------------------------------------------------------------
# A. Valid create_new
# ---------------------------------------------------------------------------


class CreateNewTests(_ReflectorAnalyzerTestCase):
    def test_valid_create_new_builds_proposal_and_founding_observation(self):
        result = self._parse(_result_dict())
        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(result.reflection_run_id, "run-1")
        self.assertEqual(result.run_summary, "Found one recurring knowledge gap across the analyzed Cases.")
        self.assertTrue(result.material_change_detected)

        self.assertEqual(len(result.new_proposals), 1)
        new_proposal = result.new_proposals[0]
        self.assertIsInstance(new_proposal, ImprovementProposal)
        self.assertEqual(new_proposal.improvement_target, "knowledge")
        self.assertEqual(new_proposal.title, "ADAM-6350 UAexpert cert trust step undocumented")
        self.assertEqual(new_proposal.review_status, "pending")
        # created_at mirrors the founding Observation's observed_at exactly
        # (see ImprovementProposal's own docstring, quoted in reflector_
        # analyzer.py's module docstring).
        self.assertEqual(new_proposal.created_at, "2026-01-02T00:00:00+00:00")

        self.assertEqual(len(result.proposal_observations), 1)
        observation = result.proposal_observations[0]
        self.assertIsInstance(observation, ProposalObservation)
        self.assertEqual(observation.proposal_id, new_proposal.proposal_id)
        self.assertEqual(observation.reflection_run_id, "run-1")
        self.assertEqual(observation.supporting_case_ids, ("case-1", "case-2"))
        self.assertEqual(observation.supporting_case_count, 2)
        self.assertEqual(observation.observed_at, "2026-01-02T00:00:00+00:00")

    def test_create_new_requires_nonblank_title(self):
        for bad_title in (None, "", "   "):
            with self.subTest(title=bad_title):
                data = _result_dict(findings=[_finding_dict(title=bad_title)])
                self.assertIsNone(self._parse(data))

    def test_create_new_uses_id_factory_for_new_proposal_id(self):
        result = self._parse(_result_dict())
        self.assertEqual(result.new_proposals[0].proposal_id, "id-1")
        self.assertEqual(result.proposal_observations[0].observation_id, "id-2")


# ---------------------------------------------------------------------------
# B. Valid match_existing
# ---------------------------------------------------------------------------


class MatchExistingTests(_ReflectorAnalyzerTestCase):
    def test_valid_match_existing_does_not_create_duplicate_proposal(self):
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
            trend="growing",
        )
        result = self._parse(_result_dict(findings=[finding]))
        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(result.new_proposals, ())
        self.assertEqual(len(result.proposal_observations), 1)
        observation = result.proposal_observations[0]
        self.assertEqual(observation.proposal_id, "proposal-1")
        self.assertEqual(observation.trend, "growing")

    def test_match_existing_with_nonnull_title_fails_closed(self):
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title="A title the model should not have supplied here",
        )
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))


# ---------------------------------------------------------------------------
# C. Hallucinated proposal_id
# ---------------------------------------------------------------------------


class HallucinatedProposalIdTests(_ReflectorAnalyzerTestCase):
    def test_match_existing_unknown_proposal_id_fails_closed(self):
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="not-a-real-candidate", improvement_target="knowledge"),
            title=None,
        )
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_match_existing_with_empty_candidate_set_fails_closed(self):
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
        )
        self.assertIsNone(self._parse(_result_dict(findings=[finding]), candidates=()))


# ---------------------------------------------------------------------------
# D. Target mismatch
# ---------------------------------------------------------------------------


class TargetMismatchTests(_ReflectorAnalyzerTestCase):
    def test_match_existing_target_mismatch_fails_closed(self):
        # candidate's own improvement_target is "knowledge" (see _candidate())
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="agent_behavior"),
            title=None,
        )
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))


# ---------------------------------------------------------------------------
# E. Hallucinated case_id
# ---------------------------------------------------------------------------


class HallucinatedCaseIdTests(_ReflectorAnalyzerTestCase):
    def test_unknown_supporting_case_id_fails_closed(self):
        finding = _finding_dict(supporting_case_ids=["case-1", "case-Z"])
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_all_unknown_supporting_case_ids_fail_closed(self):
        finding = _finding_dict(supporting_case_ids=["case-Z"])
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))


# ---------------------------------------------------------------------------
# F. Invalid confidence
# ---------------------------------------------------------------------------


class InvalidConfidenceTests(_ReflectorAnalyzerTestCase):
    def test_invalid_confidence_values_fail_closed(self):
        for bad_confidence in (-0.1, 1.1, True, False, math.nan, math.inf, -math.inf, "0.7", None):
            with self.subTest(confidence=bad_confidence):
                finding = _finding_dict(confidence=bad_confidence)
                self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_boundary_confidence_values_accepted(self):
        for ok_confidence in (0.0, 1.0, 0.5):
            with self.subTest(confidence=ok_confidence):
                finding = _finding_dict(confidence=ok_confidence)
                result = self._parse(_result_dict(findings=[finding]))
                self.assertIsInstance(result, ReflectionResult)


# ---------------------------------------------------------------------------
# G. Invalid taxonomy
# ---------------------------------------------------------------------------


class InvalidTaxonomyTests(_ReflectorAnalyzerTestCase):
    def test_invalid_improvement_target_fails_closed(self):
        finding = _finding_dict(resolution=_resolution_dict(improvement_target="not_a_real_target"))
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_invalid_trend_fails_closed(self):
        finding = _finding_dict(trend="not_a_real_trend")
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_invalid_resolution_action_fails_closed(self):
        finding = _finding_dict(resolution=_resolution_dict(action="not_a_real_action"))
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))


# ---------------------------------------------------------------------------
# H. Missing / extra keys, strict key-set at every nesting level
# ---------------------------------------------------------------------------


class StrictKeySetTests(_ReflectorAnalyzerTestCase):
    def test_top_level_missing_key_fails_closed(self):
        data = _result_dict()
        del data["material_change_detected"]
        self.assertIsNone(self._parse(data))

    def test_top_level_extra_key_fails_closed(self):
        data = _result_dict()
        data["extra_field"] = "unexpected"
        self.assertIsNone(self._parse(data))

    def test_top_level_not_a_mapping_fails_closed(self):
        self.assertIsNone(self._parse(["not", "a", "mapping"]))
        self.assertIsNone(self._parse("also not a mapping"))
        self.assertIsNone(self._parse(None))

    def test_finding_missing_key_fails_closed(self):
        finding = _finding_dict()
        del finding["pattern_summary"]
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_finding_extra_key_fails_closed(self):
        finding = _finding_dict()
        finding["extra_field"] = "unexpected"
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_finding_not_a_mapping_fails_closed(self):
        self.assertIsNone(self._parse(_result_dict(findings=["not-a-mapping"])))

    def test_resolution_missing_key_fails_closed(self):
        finding = _finding_dict(resolution=_resolution_dict())
        del finding["resolution"]["proposal_id"]
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_resolution_extra_key_fails_closed(self):
        finding = _finding_dict(resolution=_resolution_dict())
        finding["resolution"]["extra_field"] = "unexpected"
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_resolution_not_a_mapping_fails_closed(self):
        finding = _finding_dict(resolution="not-a-mapping")
        self.assertIsNone(self._parse(_result_dict(findings=[finding])))

    def test_findings_not_a_list_fails_closed(self):
        self.assertIsNone(self._parse(_result_dict(findings="not-a-list")))
        self.assertIsNone(self._parse(_result_dict(findings={"a": "dict"})))


# ---------------------------------------------------------------------------
# I. Zero result
# ---------------------------------------------------------------------------


class ZeroResultTests(_ReflectorAnalyzerTestCase):
    def test_zero_findings_with_material_change_false_is_valid(self):
        data = _result_dict(
            run_summary="No recurring improvement opportunity found in this analysis horizon.",
            material_change_detected=False,
            findings=[],
        )
        result = self._parse(data)
        self.assertIsInstance(result, ReflectionResult)
        self.assertFalse(result.material_change_detected)
        self.assertEqual(result.new_proposals, ())
        self.assertEqual(result.proposal_observations, ())

    def test_material_change_true_with_zero_findings_fails_closed(self):
        # ReflectionResult.__post_init__ itself forbids this (Slice 2,
        # unchanged) -- confirms the parser does not loosen or bypass it.
        data = _result_dict(material_change_detected=True, findings=[])
        self.assertIsNone(self._parse(data))


# ---------------------------------------------------------------------------
# J. Existing stable observation with material_change_detected=false
#    (guards against accidentally adding a material_change == bool(findings)
#    invariant that Slice 2 deliberately does not have)
# ---------------------------------------------------------------------------


class MaterialChangeSemanticsTests(_ReflectorAnalyzerTestCase):
    def test_match_existing_stable_observation_with_material_change_false_is_valid(self):
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
            trend="stable",
        )
        data = _result_dict(
            run_summary="Routine re-observation only; no material change this run.",
            material_change_detected=False,
            findings=[finding],
        )
        result = self._parse(data)
        self.assertIsInstance(result, ReflectionResult)
        self.assertFalse(result.material_change_detected)
        self.assertEqual(len(result.proposal_observations), 1)
        self.assertEqual(result.proposal_observations[0].trend, "stable")


# ---------------------------------------------------------------------------
# K. Duplicate supporting Case IDs -- normalized via build_supporting_case_ids
# ---------------------------------------------------------------------------


class DuplicateSupportingCaseIdsTests(_ReflectorAnalyzerTestCase):
    def test_duplicate_supporting_case_ids_are_deduped_and_sorted(self):
        finding = _finding_dict(supporting_case_ids=["case-2", "case-1", "case-2", "case-1"])
        result = self._parse(_result_dict(findings=[finding]))
        self.assertIsInstance(result, ReflectionResult)
        observation = result.proposal_observations[0]
        self.assertEqual(observation.supporting_case_ids, ("case-1", "case-2"))
        self.assertEqual(observation.supporting_case_count, 2)


# ---------------------------------------------------------------------------
# Additional: multiple findings, no-candidates create_new, prompt sanity
# ---------------------------------------------------------------------------


class MultipleFindingsTests(_ReflectorAnalyzerTestCase):
    def test_one_create_new_and_one_match_existing_together(self):
        new_finding = _finding_dict(supporting_case_ids=["case-1"])
        existing_finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
            supporting_case_ids=["case-2"],
        )
        data = _result_dict(findings=[new_finding, existing_finding])
        result = self._parse(data)
        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(len(result.new_proposals), 1)
        self.assertEqual(len(result.proposal_observations), 2)

    def test_two_findings_matching_the_same_candidate_fails_closed(self):
        # ReflectionResult.__post_init__ forbids more than one Observation
        # per proposal_id in a single run (Slice 2, unchanged) -- confirms
        # the parser surfaces that failure as None rather than raising.
        finding_a = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
            supporting_case_ids=["case-1"],
        )
        finding_b = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
            supporting_case_ids=["case-2"],
        )
        data = _result_dict(findings=[finding_a, finding_b])
        self.assertIsNone(self._parse(data))

    def test_create_new_with_no_candidates_offered(self):
        result = self._parse(_result_dict(), candidates=())
        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(len(result.new_proposals), 1)


class PromptTextSanityTests(unittest.TestCase):
    def test_system_instructions_contains_schema_hint(self):
        self.assertIn(_OUTPUT_SCHEMA_HINT, _SYSTEM_INSTRUCTIONS)

    def test_system_instructions_mentions_all_improvement_targets(self):
        from tools.feedback_store_v2 import IMPROVEMENT_TARGET_VALUES

        for target in IMPROVEMENT_TARGET_VALUES:
            with self.subTest(target=target):
                self.assertIn(target, _SYSTEM_INSTRUCTIONS)

    def test_system_instructions_mentions_all_resolution_actions(self):
        from tools.proposal_matching import PROPOSAL_RESOLUTION_ACTION_VALUES

        for action in PROPOSAL_RESOLUTION_ACTION_VALUES:
            with self.subTest(action=action):
                self.assertIn(action, _SYSTEM_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Slice 5B fixtures -- fake call_llm, ReflectorPromptContext construction
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: Any) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: Any) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeLlmCall:
    """Records every call's kwargs and returns a scripted response (or
    raises a scripted exception) -- never touches the network. Identical
    shape to test_case_enrichment_analyzer.py's own _FakeLlmCall."""

    def __init__(self, *, text: str | None = None, exc: BaseException | None = None) -> None:
        self._text = text
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._text)


def _fake_returning(data: dict[str, Any]) -> _FakeLlmCall:
    return _FakeLlmCall(text=json.dumps(data))


def _id_factory() -> Any:
    counter = iter(f"id-{i}" for i in range(1, 1000))
    return lambda: next(counter)


def _case_projection(**overrides: Any) -> CaseIntelligenceProjection:
    base: dict[str, Any] = dict(
        case_id="case-1",
        case_title="Cannot disable SNMP",
        issue_summary="User asked how to disable SNMP on ADAM-6266.",
        product_model="ADAM-6266",
        product_source="explicit_user_text",
        product_confidence=0.9,
        issue_type="product_usage_or_application",
        issue_type_confidence=0.8,
        diagnosis="knowledge_gap",
        diagnosis_confidence=0.7,
        evidence=(),
    )
    base.update(overrides)
    return CaseIntelligenceProjection(**base)


def _context(
    cases: tuple[CaseIntelligenceProjection, ...] = (),
    existing_proposals: tuple[ProposalCandidate, ...] = (),
) -> ReflectorPromptContext:
    """Minimal, valid ReflectorPromptContext builder -- computes the same
    by_product_model/by_issue_type/by_diagnosis aggregates tools.
    reflector_prompt_context.build_reflector_prompt_context() would, for
    whatever `cases` this helper is given. Deliberately a small, local
    helper (not a fixture framework) -- matches test_reflector_prompt_
    context.py's own `_make_reflector_input` style."""
    by_product_model: dict[str, int] = {}
    by_issue_type: dict[str, int] = {}
    by_diagnosis: dict[str, int] = {}
    for case in cases:
        product_key = case.product_model if case.product_model is not None else "__unknown__"
        by_product_model[product_key] = by_product_model.get(product_key, 0) + 1
        by_issue_type[case.issue_type] = by_issue_type.get(case.issue_type, 0) + 1
        by_diagnosis[case.diagnosis] = by_diagnosis.get(case.diagnosis, 0) + 1

    return ReflectorPromptContext(
        analysis_window=AnalysisWindow(start="2026-01-01T00:00:00+00:00", end="2026-02-01T00:00:00+00:00"),
        summary=ReflectorContextSummary(
            analyzed_case_count=len(cases),
            by_product_model=by_product_model,
            by_issue_type=by_issue_type,
            by_diagnosis=by_diagnosis,
        ),
        cases=tuple(cases),
        existing_proposals=tuple(existing_proposals),
    )


# ---------------------------------------------------------------------------
# A/B/C. Valid create_new / match_existing / zero-result LLM responses
# ---------------------------------------------------------------------------


class AnalyzeReflectionWithLlmTests(unittest.TestCase):
    def test_valid_create_new_response_builds_proposal_and_observation(self):
        # _result_dict()'s default finding cites supporting_case_ids
        # ["case-1", "case-2"] (see the Slice 5A fixtures above) -- the
        # context must actually contain both, or the Supporting Case
        # allowlist correctly rejects the response as hallucinated.
        context = _context(cases=(_case_projection(), _case_projection(case_id="case-2")))
        fake = _fake_returning(_result_dict())
        result = analyze_reflection_with_llm(
            context,
            reflection_run_id="run-1",
            observed_at="2026-01-02T00:00:00+00:00",
            llm_call=fake,
            id_factory=_id_factory(),
        )
        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(len(result.new_proposals), 1)
        self.assertIsInstance(result.new_proposals[0], ImprovementProposal)
        self.assertEqual(len(result.proposal_observations), 1)
        self.assertIsInstance(result.proposal_observations[0], ProposalObservation)

    def test_valid_match_existing_response_creates_no_duplicate_proposal(self):
        candidate = _candidate()
        context = _context(
            cases=(_case_projection(), _case_projection(case_id="case-2")),
            existing_proposals=(candidate,),
        )
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="proposal-1", improvement_target="knowledge"),
            title=None,
            supporting_case_ids=["case-1", "case-2"],
        )
        fake = _fake_returning(_result_dict(findings=[finding]))
        result = analyze_reflection_with_llm(
            context,
            reflection_run_id="run-1",
            observed_at="2026-01-02T00:00:00+00:00",
            llm_call=fake,
            id_factory=_id_factory(),
        )
        self.assertEqual(result.new_proposals, ())
        self.assertEqual(len(result.proposal_observations), 1)
        self.assertEqual(result.proposal_observations[0].proposal_id, "proposal-1")

    def test_zero_result_is_a_valid_reflection_result(self):
        context = _context(cases=(_case_projection(),))
        data = {
            "run_summary": "No recurring improvement opportunity found in this analysis horizon.",
            "material_change_detected": False,
            "findings": [],
        }
        fake = _fake_returning(data)
        result = analyze_reflection_with_llm(
            context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        self.assertIsInstance(result, ReflectionResult)
        self.assertFalse(result.material_change_detected)
        self.assertEqual(result.new_proposals, ())
        self.assertEqual(result.proposal_observations, ())


# ---------------------------------------------------------------------------
# D/E/F/G. Failure semantics -- all surface as ReflectorAnalyzerError
# ---------------------------------------------------------------------------


class AnalyzerFailureTests(unittest.TestCase):
    def _context(self) -> ReflectorPromptContext:
        return _context(cases=(_case_projection(),))

    def test_invalid_json_raises_analyzer_error(self):
        fake = _FakeLlmCall(text="not-json")
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_json_list_instead_of_object_raises_analyzer_error(self):
        fake = _FakeLlmCall(text=json.dumps([1, 2, 3]))
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_parser_rejection_hallucinated_case_id_raises_analyzer_error(self):
        finding = _finding_dict(supporting_case_ids=["case-1", "case-DOES-NOT-EXIST"])
        fake = _fake_returning(_result_dict(findings=[finding]))
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_parser_rejection_hallucinated_proposal_id_raises_analyzer_error(self):
        finding = _finding_dict(
            resolution=_resolution_dict(action="match_existing", proposal_id="not-a-real-candidate", improvement_target="knowledge"),
            title=None,
        )
        fake = _fake_returning(_result_dict(findings=[finding]))
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_llm_call_exception_becomes_analyzer_error_with_cause_chained(self):
        fake = _FakeLlmCall(exc=ConnectionError("network down"))
        with self.assertRaises(ReflectorAnalyzerError) as ctx:
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )
        self.assertIsInstance(ctx.exception.__cause__, ConnectionError)

    def test_empty_response_raises_analyzer_error(self):
        fake = _FakeLlmCall(text="")
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_whitespace_only_response_raises_analyzer_error(self):
        fake = _FakeLlmCall(text="   \n  ")
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_none_content_response_raises_analyzer_error(self):
        fake = _FakeLlmCall(text=None)
        with self.assertRaises(ReflectorAnalyzerError):
            analyze_reflection_with_llm(
                self._context(), reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )


# ---------------------------------------------------------------------------
# H/I/18. call_llm() argument contract + dependency injection + context
# integrity (the serialized ReflectorPromptContext, not a second serializer)
# ---------------------------------------------------------------------------


class CallLlmArgumentContractTests(unittest.TestCase):
    def test_temperature_zero_and_json_object_response_format(self):
        context = _context(cases=(_case_projection(),))
        data = {"run_summary": "no findings", "material_change_detected": False, "findings": []}
        fake = _fake_returning(data)
        analyze_reflection_with_llm(
            context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        self.assertEqual(fake.calls[0]["temperature"], 0)
        self.assertEqual(fake.calls[0]["extra_body"], {"response_format": {"type": "json_object"}})
        self.assertEqual(fake.calls[0]["stream"], False)
        self.assertIsNone(fake.calls[0]["tools"])

    def test_messages_are_system_policy_plus_serialized_context(self):
        marker_case = _case_projection(case_id="case-marker-xyz-123")
        context = _context(cases=(marker_case,))
        data = {"run_summary": "no findings", "material_change_detected": False, "findings": []}
        fake = _fake_returning(data)
        analyze_reflection_with_llm(
            context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        messages = fake.calls[0]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], _SYSTEM_INSTRUCTIONS)
        self.assertEqual(messages[1]["role"], "user")
        # Context integrity: the serialized payload comes from
        # serialize_reflector_prompt_context(), not a second serializer --
        # a case_id present only in `context` must appear verbatim.
        self.assertIn("case-marker-xyz-123", messages[1]["content"])

    def test_context_with_existing_proposal_id_reaches_the_prompt(self):
        candidate = _candidate(proposal_id="proposal-marker-abc")
        context = _context(cases=(_case_projection(),), existing_proposals=(candidate,))
        data = {"run_summary": "no findings", "material_change_detected": False, "findings": []}
        fake = _fake_returning(data)
        analyze_reflection_with_llm(
            context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        self.assertIn("proposal-marker-abc", fake.calls[0]["messages"][1]["content"])

    def test_main_runtime_forwarded_unchanged(self):
        context = _context(cases=(_case_projection(),))
        data = {"run_summary": "no findings", "material_change_detected": False, "findings": []}
        fake = _fake_returning(data)
        runtime = {"provider": "custom", "model": "gpt-5.6-luna", "base_url": "https://inference.local/v1"}
        analyze_reflection_with_llm(
            context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00",
            main_runtime=runtime, llm_call=fake,
        )
        self.assertEqual(fake.calls[0]["main_runtime"], runtime)

    def test_prompt_excludes_whole_session_history_and_telegram_context(self):
        # Exactly system + user -- no session-history messages prepended,
        # no Telegram/platform-specific content anywhere. Note: the system
        # message legitimately MENTIONS the words "SOUL"/"Skill" as part of
        # the Human Review Boundary policy itself ("never permission to
        # modify ... SOUL, a Skill ...") -- that is this module's own
        # reasoning-policy text, not leaked conversational context, so this
        # test (unlike test_case_enrichment_analyzer.py's substring-based
        # version) checks message COUNT/shape rather than forbidden
        # substrings.
        context = _context(cases=(_case_projection(),))
        data = {"run_summary": "no findings", "material_change_detected": False, "findings": []}
        fake = _fake_returning(data)
        analyze_reflection_with_llm(
            context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        messages = fake.calls[0]["messages"]
        self.assertEqual(len(messages), 2)
        combined = messages[0]["content"] + "\n" + messages[1]["content"]
        for forbidden in ("Telegram", "tool_call"):
            self.assertNotIn(forbidden, combined)


class DefaultLlmCallLazyImportTests(unittest.TestCase):
    def test_default_llm_call_lazy_imports_agent_auxiliary_client(self):
        # This repo's test environment has no `agent` package installed
        # (agent.auxiliary_client only exists inside a built Hermes sandbox
        # image) -- calling without an injected llm_call must attempt the
        # lazy import and surface ImportError undisguised, distinct from a
        # ReflectorAnalyzerError (which means "the LLM call itself, or its
        # output, failed" -- an import failure is an environment problem,
        # not that). Mirrors test_case_enrichment_analyzer.py's own
        # DefaultLlmCallLazyImportTests precedent exactly.
        context = _context(cases=(_case_projection(),))
        with self.assertRaises(ImportError):
            analyze_reflection_with_llm(context, reflection_run_id="run-1", observed_at="2026-01-02T00:00:00+00:00")


# ---------------------------------------------------------------------------
# Synthetic Reflector Scenario -- end-to-end function-boundary smoke test.
#
# Not a unit test of one rule: this confirms the whole boundary
#   ReflectorPromptContext -> serialize -> fake LLM -> parse -> ReflectionResult
# runs end to end, AND that two findings sharing the same product but
# different improvement_target values are treated as two distinct
# underlying improvement opportunities, never merged into one Proposal
# just because they mention the same product ("SNMP") -- exactly the
# Proposal Identity Rule tools.proposal_matching's own module docstring
# states (improvement_target + pattern semantics + remediation direction,
# never bare topic similarity).
# ---------------------------------------------------------------------------


class SyntheticReflectorScenarioTests(unittest.TestCase):
    def test_knowledge_gap_and_agent_behavior_are_distinct_proposals(self):
        # Case A and Case B: two independent Cases, same underlying
        # knowledge gap (SNMP disable command missing from the KB).
        case_a = _case_projection(
            case_id="case-A",
            case_title="How to disable SNMP on ADAM-6266",
            issue_summary="User could not find the SNMP disable command for ADAM-6266.",
            diagnosis="knowledge_gap",
        )
        case_b = _case_projection(
            case_id="case-B",
            case_title="ADAM-6266 SNMP disable command",
            issue_summary="A different user asked the same SNMP disable question about ADAM-6266.",
            diagnosis="knowledge_gap",
        )
        # Case C: same product, same general topic (SNMP), but the KB
        # already had the command available -- the assistant's own answer
        # omitted it. A different underlying improvement opportunity.
        case_c = _case_projection(
            case_id="case-C",
            case_title="ADAM-6266 answer missing SNMP command despite retrieval",
            issue_summary="Retrieval succeeded and the command was present in evidence, but the answer omitted it.",
            diagnosis="answer_quality_issue",
        )
        context = _context(cases=(case_a, case_b, case_c))

        knowledge_finding = _finding_dict(
            resolution=_resolution_dict(action="create_new", proposal_id=None, improvement_target="knowledge"),
            title="ADAM-6266 SNMP disable command undocumented in KB",
            trend="new",
            pattern_summary="Two independent Cases ask how to disable SNMP on ADAM-6266; the KB does not document the command.",
            possible_cause="The KB article for ADAM-6266 may not document the SNMP disable command.",
            recommended_improvement="Add the SNMP disable command to the ADAM-6266 KB article.",
            expected_benefit="Fewer repeat questions on this specific topic.",
            limitations="Based on only two Cases; may not generalize.",
            supporting_case_ids=["case-A", "case-B"],
            confidence=0.75,
        )
        agent_behavior_finding = _finding_dict(
            resolution=_resolution_dict(action="create_new", proposal_id=None, improvement_target="agent_behavior"),
            title="Hermes omits the SNMP disable command despite having evidence",
            trend="new",
            pattern_summary="Retrieval surfaced the SNMP disable command, but the final answer omitted it.",
            possible_cause="The answer-generation step may not be using all retrieved evidence.",
            recommended_improvement="Review the answer-generation step for this evidence-usage pattern.",
            expected_benefit="More complete answers when evidence already contains the needed command.",
            limitations="Based on only one Case; may not generalize.",
            supporting_case_ids=["case-C"],
            confidence=0.6,
        )
        data = {
            "run_summary": (
                "Found two distinct improvement opportunities involving ADAM-6266 SNMP: "
                "a knowledge gap and a separate agent-behavior issue."
            ),
            "material_change_detected": True,
            "findings": [knowledge_finding, agent_behavior_finding],
        }
        fake = _fake_returning(data)

        result = analyze_reflection_with_llm(
            context,
            reflection_run_id="run-synthetic-1",
            observed_at="2026-01-02T00:00:00+00:00",
            llm_call=fake,
            id_factory=_id_factory(),
        )

        self.assertIsInstance(result, ReflectionResult)
        self.assertTrue(result.material_change_detected)
        self.assertEqual(len(result.new_proposals), 2)
        self.assertEqual(len(result.proposal_observations), 2)

        targets = {p.improvement_target for p in result.new_proposals}
        self.assertEqual(targets, {"knowledge", "agent_behavior"})

        proposal_ids = {p.proposal_id for p in result.new_proposals}
        self.assertEqual(len(proposal_ids), 2, "the two findings must not collapse into one Proposal")

        observations_by_target = {
            next(p.improvement_target for p in result.new_proposals if p.proposal_id == obs.proposal_id): obs
            for obs in result.proposal_observations
        }
        self.assertEqual(observations_by_target["knowledge"].supporting_case_ids, ("case-A", "case-B"))
        self.assertEqual(observations_by_target["agent_behavior"].supporting_case_ids, ("case-C",))


if __name__ == "__main__":
    unittest.main()
