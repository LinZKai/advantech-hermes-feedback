"""Curator Slice 1 tests: tools.curator_analyzer -- parse_curator_output()
and analyze_proposal_with_llm().

No network anywhere in this file -- every LLM-calling test injects a fake
`llm_call` (matching test_reflector_analyzer.py's own _FakeLlmCall shape),
never `agent.auxiliary_client`. Deliberately does not assert on the exact
wording of _SYSTEM_INSTRUCTIONS (source-string tests are explicitly out of
scope for this slice) -- only on the deterministic CONTRACT: message
shape, call_llm() argument shape, and parse/validation outcomes.
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

from tools.curator_analyzer import (  # noqa: E402
    CuratorAnalyzerError,
    analyze_proposal_with_llm,
    parse_curator_output,
)
from tools.curator_domain import CURATOR_TARGET_FILE, CuratorChange  # noqa: E402
from tools.curator_prompt_context import CuratorPromptContext, build_curator_prompt_context  # noqa: E402


def _context(**overrides: Any) -> CuratorPromptContext:
    fields: dict[str, Any] = dict(
        proposal_id="proposal-1",
        title="技術支援回答應優先呈現直接結論、必要條件與可執行步驟",
        pattern_summary="多筆案例顯示回答過於冗長，重要結論被埋在說明後面。",
        possible_cause="AGENTS.md 目前未要求優先呈現結論。",
        recommended_improvement="調整回答結構，讓結論與必要條件更早出現。",
        expected_benefit="使用者能更快找到答案。",
        limitations="樣本數不多。",
        observation_confidence=0.75,
        observed_at="2026-01-02T00:00:00+00:00",
        supporting_case_records=(),
        unavailable_case_ids=(),
        current_content="# Advantech Technical Support Instructions\n\n## Response Structure\n\nPut the direct answer first.\n",
    )
    fields.update(overrides)
    return build_curator_prompt_context(**fields)


def _valid_output(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = dict(
        change_type="modify_rule",
        target_file=CURATOR_TARGET_FILE,
        rationale="多筆案例顯示回答過於冗長。",
        proposed_content="# Advantech Technical Support Instructions\n\nBe direct.\n",
        expected_effect="回答更精簡。",
        confidence=0.82,
    )
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# parse_curator_output -- fail-closed structural validation
# ---------------------------------------------------------------------------


class ParseCuratorOutputTests(unittest.TestCase):
    def test_valid_output_builds_curator_change(self):
        change = parse_curator_output(
            _valid_output(), proposal_id="proposal-1", before_content="before",
            created_at="2026-01-02T00:00:00+00:00",
        )
        self.assertIsInstance(change, CuratorChange)
        self.assertEqual(change.change_type, "modify_rule")
        self.assertEqual(change.status, "proposed")
        self.assertEqual(change.before_content, "before")

    def test_valid_no_change_recommended_output_parses(self):
        change = parse_curator_output(
            _valid_output(change_type="no_change_recommended", proposed_content=None, expected_effect=None),
            proposal_id="proposal-1", before_content="before",
            created_at="2026-01-02T00:00:00+00:00",
        )
        self.assertIsInstance(change, CuratorChange)
        self.assertEqual(change.change_type, "no_change_recommended")
        self.assertIsNone(change.proposed_content)

    def test_not_a_mapping_rejected(self):
        self.assertIsNone(
            parse_curator_output(["not", "a", "mapping"], proposal_id="p", before_content="b", created_at="t")
        )

    def test_missing_key_rejected(self):
        data = _valid_output()
        del data["confidence"]
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_extra_key_rejected(self):
        data = _valid_output()
        data["extra_field"] = "unexpected"
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_wrong_target_file_rejected(self):
        data = _valid_output(target_file="/sandbox/SOUL.md")
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_invalid_change_type_rejected(self):
        data = _valid_output(change_type="rewrite_everything")
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_missing_proposed_content_for_add_rule_rejected(self):
        data = _valid_output(change_type="add_rule", proposed_content=None)
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_non_null_proposed_content_for_no_change_recommended_rejected(self):
        data = _valid_output(change_type="no_change_recommended", proposed_content="not null")
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_confidence_out_of_range_rejected(self):
        data = _valid_output(confidence=1.5)
        self.assertIsNone(parse_curator_output(data, proposal_id="p", before_content="b", created_at="t"))

    def test_before_content_comes_from_caller_never_from_llm_output(self):
        # before_content is not even a key in the LLM's schema -- proving
        # it is always the caller-supplied actual file content.
        change = parse_curator_output(
            _valid_output(), proposal_id="proposal-1", before_content="ACTUAL CURRENT CONTENT",
            created_at="2026-01-02T00:00:00+00:00",
        )
        self.assertEqual(change.before_content, "ACTUAL CURRENT CONTENT")


# ---------------------------------------------------------------------------
# Fake call_llm -- mirrors test_reflector_analyzer.py's own _FakeLlmCall
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
# analyze_proposal_with_llm -- happy path, malformed output, empty/invalid
# response, provider error
# ---------------------------------------------------------------------------


class AnalyzeProposalWithLlmTests(unittest.TestCase):
    def test_valid_response_returns_curator_change(self):
        fake = _fake_returning(_valid_output())
        change = analyze_proposal_with_llm(
            _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        self.assertIsInstance(change, CuratorChange)
        self.assertEqual(change.change_type, "modify_rule")

    def test_malformed_json_raises_analyzer_error(self):
        fake = _FakeLlmCall(text="not valid json")
        with self.assertRaises(CuratorAnalyzerError):
            analyze_proposal_with_llm(
                _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_output_failing_contract_validation_raises_analyzer_error(self):
        fake = _fake_returning(_valid_output(change_type="rewrite_everything"))
        with self.assertRaises(CuratorAnalyzerError):
            analyze_proposal_with_llm(
                _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_empty_response_raises_analyzer_error(self):
        fake = _FakeLlmCall(text="")
        with self.assertRaises(CuratorAnalyzerError):
            analyze_proposal_with_llm(
                _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )

    def test_provider_exception_raises_analyzer_error(self):
        fake = _FakeLlmCall(exc=RuntimeError("provider unavailable"))
        with self.assertRaises(CuratorAnalyzerError):
            analyze_proposal_with_llm(
                _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
            )


class CallLlmArgumentContractTests(unittest.TestCase):
    def test_temperature_zero_and_json_object_response_format(self):
        fake = _fake_returning(_valid_output())
        analyze_proposal_with_llm(
            _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        self.assertEqual(fake.calls[0]["temperature"], 0)
        self.assertEqual(fake.calls[0]["extra_body"], {"response_format": {"type": "json_object"}})
        self.assertEqual(fake.calls[0]["stream"], False)
        self.assertIsNone(fake.calls[0]["tools"])

    def test_exactly_two_messages_system_then_user(self):
        fake = _fake_returning(_valid_output())
        analyze_proposal_with_llm(
            _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        messages = fake.calls[0]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_current_agents_content_reaches_the_prompt(self):
        marker_context = _context(current_content="MARKER-CONTENT-xyz-123")
        fake = _fake_returning(_valid_output())
        analyze_proposal_with_llm(
            marker_context, proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00", llm_call=fake,
        )
        self.assertIn("MARKER-CONTENT-xyz-123", fake.calls[0]["messages"][1]["content"])

    def test_main_runtime_forwarded_unchanged(self):
        fake = _fake_returning(_valid_output())
        runtime = {"provider": "custom", "model": "gpt-5.6-luna", "base_url": "https://inference.local/v1"}
        analyze_proposal_with_llm(
            _context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00",
            main_runtime=runtime, llm_call=fake,
        )
        self.assertEqual(fake.calls[0]["main_runtime"], runtime)


class DefaultLlmCallLazyImportTests(unittest.TestCase):
    def test_default_llm_call_lazy_imports_agent_auxiliary_client(self):
        # This repo's test environment has no `agent` package installed --
        # calling without an injected llm_call must attempt the lazy
        # import and surface ImportError undisguised, matching tools.
        # reflector_analyzer's own DefaultLlmCallLazyImportTests precedent.
        with self.assertRaises(ImportError):
            analyze_proposal_with_llm(_context(), proposal_id="proposal-1", created_at="2026-01-02T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
