"""Phase 4.5 Stage D1 tests: the real LLM Case Enrichment analyzer in
tools.case_enrichment_analyzer.

Every test in this file injects a fake `llm_call` -- no real network call,
no import of `agent.auxiliary_client` (that package only exists inside a
built Hermes sandbox image, not in this repo's test environment). This is
also what proves the lazy-import requirement actually holds: this whole
file imports tools.case_enrichment_analyzer successfully without Hermes
installed, and only the one dedicated test for the default `llm_call=None`
path touches the lazy import at all (see
DefaultLlmCallLazyImportTests below).

No persistence anywhere in this file -- create_case_analysis() is never
imported or called (Stage D2's job, not Stage D1's).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import (  # noqa: E402
    DIAGNOSIS_VALUES,
    EVIDENCE_TYPES,
    ISSUE_TYPE_VALUES,
    CaseEnrichmentInput,
    CaseEnrichmentResult,
    CaseFeedbackEvidence,
    CaseRetrievalEvidence,
    CaseTurnEvidence,
)
from tools.case_enrichment_analyzer import (  # noqa: E402
    ANALYSIS_VERSION,
    CaseEnrichmentAnalyzerError,
    _build_messages,
    _extract_text,
    _serialize_case_input,
    _SYSTEM_INSTRUCTIONS,
    analyze_case_with_llm,
)


# ---------------------------------------------------------------------------
# Fixtures -- pure in-memory dataclass construction, no DB (Stage D1's
# analyzer never touches storage; that stays Stage D2's job)
# ---------------------------------------------------------------------------


def _make_turn(**overrides: Any) -> CaseTurnEvidence:
    base = dict(
        turn_id="turn-1",
        question_text="ADAM-6266 SNMP 怎麼關閉？",
        answer_text="請至設定頁停用 SNMP。",
        created_at="2026-08-01T00:00:00+00:00",
        retrieval_observation_status="complete",
        case_assignment_method="explicit",
        case_assignment_confidence="high",
    )
    base.update(overrides)
    return CaseTurnEvidence(**base)


def _make_retrieval(**overrides: Any) -> CaseRetrievalEvidence:
    base = dict(
        turn_id="turn-1",
        execution_status="completed",
        foundry_iq_ok=True,
        result_count=18,
        reference_count=3,
        observation_status="complete",
        observation_reason=None,
    )
    base.update(overrides)
    return CaseRetrievalEvidence(**base)


def _make_feedback(**overrides: Any) -> CaseFeedbackEvidence:
    base = dict(turn_id="turn-1", helpful=False, reason_code="incomplete", suggestion_text=None)
    base.update(overrides)
    return CaseFeedbackEvidence(**base)


def _make_case_input(**overrides: Any) -> CaseEnrichmentInput:
    base = dict(
        case_id="case-1",
        session_id="sess-1",
        current_title=None,
        current_product_model=None,
        turns=(_make_turn(),),
        retrievals=(),
        feedback=(),
        evidence_watermark="2026-08-01T00:00:00+00:00",
    )
    base.update(overrides)
    return CaseEnrichmentInput(**base)


def _valid_evidence_item(**overrides: Any) -> dict[str, Any]:
    base = {
        "type": "user_text",
        "turn_id": "turn-1",
        "fact": "User asked about ADAM-6266 SNMP configuration in Turn turn-1.",
    }
    base.update(overrides)
    return base


def _valid_result_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "case_title": "ADAM-6266 SNMP configuration issue",
        "issue_summary": "User could not find how to disable SNMP on ADAM-6266.",
        "issue_type": "product_usage_or_application",
        "issue_type_confidence": 0.8,
        "diagnosis": "knowledge_gap",
        "diagnosis_confidence": 0.7,
        "product_model": "ADAM-6266",
        "product_source": "explicit_user_text",
        "product_confidence": 0.95,
        "evidence": [_valid_evidence_item()],
    }
    base.update(overrides)
    return base


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
    raises a scripted exception) -- never touches the network."""

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


# ---------------------------------------------------------------------------
# A. Valid path
# ---------------------------------------------------------------------------


class ValidResultTests(unittest.TestCase):
    def test_valid_json_returns_valid_case_enrichment_result(self):
        fake = _fake_returning(_valid_result_dict())
        result = analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertIsInstance(result, CaseEnrichmentResult)
        self.assertEqual(result.case_title, "ADAM-6266 SNMP configuration issue")
        self.assertEqual(result.product_model, "ADAM-6266")

    def test_all_issue_types_accepted_end_to_end(self):
        for value in ISSUE_TYPE_VALUES:
            with self.subTest(issue_type=value):
                fake = _fake_returning(_valid_result_dict(issue_type=value))
                result = analyze_case_with_llm(_make_case_input(), llm_call=fake)
                self.assertEqual(result.issue_type, value)

    def test_all_diagnosis_values_accepted_end_to_end(self):
        for value in DIAGNOSIS_VALUES:
            with self.subTest(diagnosis=value):
                fake = _fake_returning(_valid_result_dict(diagnosis=value))
                result = analyze_case_with_llm(_make_case_input(), llm_call=fake)
                self.assertEqual(result.diagnosis, value)

    def test_product_present(self):
        fake = _fake_returning(_valid_result_dict(
            product_model="WISE-6610", product_source="inference", product_confidence=0.4,
        ))
        result = analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertEqual(result.product_model, "WISE-6610")
        self.assertEqual(result.product_source, "inference")

    def test_product_absent(self):
        fake = _fake_returning(_valid_result_dict(
            product_model=None, product_source=None, product_confidence=None,
        ))
        result = analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertIsNone(result.product_model)
        self.assertIsNone(result.product_source)
        self.assertIsNone(result.product_confidence)

    def test_multiple_evidence_accepted(self):
        fake = _fake_returning(_valid_result_dict(evidence=[
            _valid_evidence_item(),
            _valid_evidence_item(type="retrieval", fact="Retrieval telemetry reports foundry_iq_ok=true, result_count=18 for Turn turn-1."),
            _valid_evidence_item(type="feedback", fact="Feedback for Turn turn-1 was negative with reason_code=incomplete."),
        ]))
        result = analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertEqual(len(result.evidence), 3)

    def test_llm_call_receives_no_tools(self):
        fake = _fake_returning(_valid_result_dict())
        analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertIsNone(fake.calls[0]["tools"])

    def test_llm_call_receives_no_stream_and_zero_temperature(self):
        fake = _fake_returning(_valid_result_dict())
        analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertEqual(fake.calls[0]["stream"], False)
        self.assertEqual(fake.calls[0]["temperature"], 0)

    def test_main_runtime_forwarded_unchanged(self):
        fake = _fake_returning(_valid_result_dict())
        runtime = {"provider": "custom", "model": "gpt-5.6-luna", "base_url": "https://inference.local/v1"}
        analyze_case_with_llm(_make_case_input(), main_runtime=runtime, llm_call=fake)
        self.assertEqual(fake.calls[0]["main_runtime"], runtime)

    def test_serialized_case_input_contains_required_observed_data(self):
        marker = "ADAM-6266-UNIQUE-MARKER-XYZ"
        case_input = _make_case_input(turns=(_make_turn(question_text=marker),))
        fake = _fake_returning(_valid_result_dict())
        analyze_case_with_llm(case_input, llm_call=fake)
        user_content = fake.calls[0]["messages"][1]["content"]
        self.assertIn(marker, user_content)
        self.assertIn("case-1", user_content)

    def test_prompt_excludes_whole_session_soul_skills_telegram(self):
        fake = _fake_returning(_valid_result_dict())
        analyze_case_with_llm(_make_case_input(), llm_call=fake)
        messages = fake.calls[0]["messages"]
        # Exactly system + user -- no session-history messages prepended.
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        combined = messages[0]["content"] + "\n" + messages[1]["content"]
        for forbidden in ("SOUL", "Skill", "Telegram", "tool_call"):
            self.assertNotIn(forbidden, combined)


# ---------------------------------------------------------------------------
# B. Failure semantics
# ---------------------------------------------------------------------------


class FailureTests(unittest.TestCase):
    def test_provider_exception_propagates_as_analyzer_error(self):
        fake = _FakeLlmCall(exc=ConnectionError("network down"))
        with self.assertRaises(CaseEnrichmentAnalyzerError) as ctx:
            analyze_case_with_llm(_make_case_input(), llm_call=fake)
        self.assertIsInstance(ctx.exception.__cause__, ConnectionError)

    def test_empty_response_fails(self):
        fake = _FakeLlmCall(text="")
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_whitespace_only_response_fails(self):
        fake = _FakeLlmCall(text="   \n  ")
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_none_content_response_fails(self):
        fake = _FakeLlmCall(text=None)
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_invalid_json_fails(self):
        fake = _FakeLlmCall(text="{not valid json")
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_json_list_instead_of_object_fails(self):
        fake = _FakeLlmCall(text=json.dumps([1, 2, 3]))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_unknown_field_fails(self):
        data = _valid_result_dict()
        data["severity"] = "high"
        fake = _fake_returning(data)
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_missing_required_field_fails(self):
        data = _valid_result_dict()
        del data["diagnosis"]
        fake = _fake_returning(data)
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_invalid_issue_type_fails(self):
        fake = _fake_returning(_valid_result_dict(issue_type="bogus_type"))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_invalid_diagnosis_fails(self):
        fake = _fake_returning(_valid_result_dict(diagnosis="bogus_diagnosis"))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_invalid_confidence_out_of_range_fails(self):
        fake = _fake_returning(_valid_result_dict(issue_type_confidence=1.5))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_zero_evidence_fails(self):
        fake = _fake_returning(_valid_result_dict(evidence=[]))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_invalid_evidence_type_fails(self):
        fake = _fake_returning(_valid_result_dict(evidence=[
            _valid_evidence_item(type="bogus_evidence_type"),
        ]))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_blank_turn_id_rejected(self):
        fake = _fake_returning(_valid_result_dict(evidence=[
            _valid_evidence_item(turn_id=""),
        ]))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)

    def test_null_turn_id_rejected(self):
        fake = _fake_returning(_valid_result_dict(evidence=[
            _valid_evidence_item(turn_id=None),
        ]))
        with self.assertRaises(CaseEnrichmentAnalyzerError):
            analyze_case_with_llm(_make_case_input(), llm_call=fake)


# ---------------------------------------------------------------------------
# C. Safety boundary
# ---------------------------------------------------------------------------


class SafetyBoundaryTests(unittest.TestCase):
    def test_retrieval_telemetry_serialized_with_no_invented_document_field(self):
        case_input = _make_case_input(retrievals=(_make_retrieval(),))
        serialized = _serialize_case_input(case_input)
        self.assertEqual(len(serialized["retrievals"]), 1)
        self.assertEqual(
            set(serialized["retrievals"][0].keys()),
            {
                "turn_id", "execution_status", "foundry_iq_ok", "result_count",
                "reference_count", "observation_status", "observation_reason",
            },
        )

    def test_feedback_serialized_verbatim(self):
        case_input = _make_case_input(feedback=(_make_feedback(),))
        serialized = _serialize_case_input(case_input)
        self.assertEqual(serialized["feedback"], [{
            "turn_id": "turn-1", "helpful": False,
            "reason_code": "incomplete", "suggestion_text": None,
        }])

    def test_explicit_user_text_semantics_documented_not_forged(self):
        # The rule is documented for the model to apply itself -- there is
        # no client-side regex/heuristic anywhere in this module that sets
        # product_source based on question_text content.
        self.assertIn("explicit_user_text", _SYSTEM_INSTRUCTIONS)
        self.assertIn("question_text", _SYSTEM_INSTRUCTIONS)

    def test_no_tools_passed_to_llm_call_regardless_of_case_content(self):
        case_input = _make_case_input(retrievals=(_make_retrieval(),), feedback=(_make_feedback(),))
        fake = _fake_returning(_valid_result_dict())
        analyze_case_with_llm(case_input, llm_call=fake)
        self.assertIn("tools", fake.calls[0])
        self.assertIsNone(fake.calls[0]["tools"])

    def test_all_evidence_type_values_accepted_by_taxonomy(self):
        # Sanity check that the analyzer's serialization/prompt boundary
        # agrees with the same EVIDENCE_TYPES authority parse_case_enrichment_result
        # already validates against -- not a taxonomy re-declaration.
        self.assertIn("user_text", EVIDENCE_TYPES)
        self.assertIn("retrieval", EVIDENCE_TYPES)
        for value in EVIDENCE_TYPES:
            self.assertIn(value, _SYSTEM_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# D. _extract_text -- direct unit coverage of response-shape edge cases
# ---------------------------------------------------------------------------


class ExtractTextTests(unittest.TestCase):
    def test_normal_openai_shaped_response(self):
        self.assertEqual(_extract_text(_FakeResponse('{"ok": true}')), '{"ok": true}')

    def test_missing_choices_attribute_returns_empty(self):
        class _NoChoices:
            pass
        self.assertEqual(_extract_text(_NoChoices()), "")

    def test_empty_choices_list_returns_empty(self):
        class _EmptyChoices:
            choices: list[Any] = []
        self.assertEqual(_extract_text(_EmptyChoices()), "")

    def test_none_content_returns_empty(self):
        self.assertEqual(_extract_text(_FakeResponse(None)), "")

    def test_non_string_content_returns_empty(self):
        self.assertEqual(_extract_text(_FakeResponse(["not", "a", "string"])), "")


# ---------------------------------------------------------------------------
# E. Default llm_call -- proves the lazy-import requirement
# ---------------------------------------------------------------------------


class DefaultLlmCallLazyImportTests(unittest.TestCase):
    def test_default_llm_call_lazy_imports_agent_auxiliary_client(self):
        # This repo's test environment has no `agent` package installed
        # (agent.auxiliary_client only exists inside a built Hermes sandbox
        # image) -- calling without an injected llm_call must attempt the
        # lazy import and surface ImportError undisguised, distinct from a
        # CaseEnrichmentAnalyzerError (which means "the LLM call itself, or
        # its output, failed" -- an import failure is an environment
        # problem, not that). The whole test file already imports
        # tools.case_enrichment_analyzer successfully above, which is the
        # other half of the proof: the import is not at module level.
        with self.assertRaises(ImportError):
            analyze_case_with_llm(_make_case_input())


# ---------------------------------------------------------------------------
# F. Misc contract
# ---------------------------------------------------------------------------


class MiscTests(unittest.TestCase):
    def test_analysis_version_is_a_fixed_non_timestamp_string(self):
        self.assertEqual(ANALYSIS_VERSION, "case-enrichment-v1")

    def test_analyzer_error_is_runtime_error(self):
        self.assertTrue(issubclass(CaseEnrichmentAnalyzerError, RuntimeError))

    def test_build_messages_is_deterministic_for_identical_input(self):
        case_input = _make_case_input()
        self.assertEqual(_build_messages(case_input), _build_messages(case_input))


if __name__ == "__main__":
    unittest.main()
