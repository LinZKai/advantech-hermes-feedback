"""Phase 3A parser tests: tools.retrieval_observer.

Pure-function tests only -- no database, no gateway, no MessageEvent.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.retrieval_observer import (  # noqa: E402
    INNER_RESULT_UNPARSEABLE_REASON,
    MISSING_INNER_RESULT_REASON,
    OUTER_RESULT_UNPARSEABLE_REASON,
    TOOL_RESULT_MISSING_REASON,
    UNTRUSTED_TURN_BOUNDARY_REASON,
    FoundryRetrievalObservation,
    TurnRetrievalObservation,
    observation_to_safe_dict,
    parse_turn_retrieval_observations,
    safe_dict_to_observation,
)
from tools.universal_feedback import FOUNDARY_SCRIPT  # noqa: E402

CANARY_SECRET = "sk-CANARY-SECRET-DO-NOT-LEAK-1234567890"


def _assistant_terminal_call(call_id, command, extra_calls=None):
    tool_calls = [
        {"id": call_id, "function": {"name": "terminal", "arguments": json.dumps({"command": command})}}
    ]
    if extra_calls:
        tool_calls.extend(extra_calls)
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _assistant_non_terminal_call(call_id, name="web_search", arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments or {"q": "x"})}}
        ],
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _foundry_command(question="ADAM-6266 SNMP"):
    return f"python3 {FOUNDARY_SCRIPT} '{question}'"


def _outer(output=None, exit_code=0, error=None):
    return json.dumps({"output": output, "exit_code": exit_code, "error": error})


def _inner_success(documents=None, references=None, request_attempted=True):
    return json.dumps(
        {
            "schema_version": "foundry-iq-result-v2",
            "ok": True,
            "request_attempted": request_attempted,
            "error_code": None,
            "question": "q",
            "http_status": None,
            "documents": documents if documents is not None else [{"ref_id": 0, "content": "doc text"}],
            "references": references if references is not None else [{"id": 1, "type": "doc"}],
            "activity": [{"type": "modelQueryPlanning"}],
        }
    )


def _inner_failure(error_code, http_status=None, request_attempted=True, error="human message"):
    return json.dumps(
        {
            "schema_version": "foundry-iq-result-v2",
            "ok": False,
            "request_attempted": request_attempted,
            "error_code": error_code,
            "question": "q",
            "http_status": http_status,
            "error": error,
        }
    )


class BoundaryTrustTests(unittest.TestCase):
    def test_1_trusted_boundary_no_foundry_call(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(result.turn_observation_status, "complete")
        self.assertIsNone(result.turn_observation_reason)
        self.assertEqual(result.retrievals, ())

    def test_19_untrusted_boundary_produces_no_retrieval(self):
        messages = [
            _assistant_terminal_call("call-1", _foundry_command()),
            _tool_result("call-1", _outer(output=_inner_success())),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=False)
        self.assertEqual(result.turn_observation_status, "unavailable")
        self.assertEqual(result.turn_observation_reason, UNTRUSTED_TURN_BOUNDARY_REASON)
        self.assertEqual(result.retrievals, ())

    def test_offset_missing_is_untrusted(self):
        result = parse_turn_retrieval_observations([{"role": "user"}], history_offset=None, boundary_trusted=True)
        self.assertEqual(result.turn_observation_status, "unavailable")
        self.assertEqual(result.turn_observation_reason, UNTRUSTED_TURN_BOUNDARY_REASON)

    def test_offset_out_of_range_is_untrusted(self):
        result = parse_turn_retrieval_observations([{"role": "user"}], history_offset=99, boundary_trusted=True)
        self.assertEqual(result.turn_observation_status, "unavailable")

    def test_offset_negative_is_untrusted(self):
        result = parse_turn_retrieval_observations([{"role": "user"}], history_offset=-1, boundary_trusted=True)
        self.assertEqual(result.turn_observation_status, "unavailable")

    def test_18_old_history_foundry_call_excluded_by_offset(self):
        old_call = _assistant_terminal_call("old-call", _foundry_command())
        old_result = _tool_result("old-call", _outer(output=_inner_success()))
        messages = [
            old_call,
            old_result,
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "no tools needed this time"},
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=2, boundary_trusted=True)
        self.assertEqual(result.turn_observation_status, "complete")
        self.assertEqual(result.retrievals, ())


class DetectionTests(unittest.TestCase):
    def test_2_final_answer_mentions_foundry_but_no_tool_call(self):
        messages = [
            {"role": "user", "content": "what does the manual say?"},
            {"role": "assistant", "content": "Per Foundry IQ and ADAM-6266.pdf, [NEW_TOPIC] the answer is X. [IN_SCOPE]"},
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(result.turn_observation_status, "complete")
        self.assertEqual(result.retrievals, ())

    def test_15_non_foundry_terminal_call_ignored(self):
        messages = [
            _assistant_terminal_call("call-1", "ls -la /tmp"),
            _tool_result("call-1", _outer(output="total 0")),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(result.retrievals, ())

    def test_non_terminal_tool_ignored(self):
        messages = [
            _assistant_non_terminal_call("call-1"),
            _tool_result("call-1", json.dumps({"result": "ok"})),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(result.retrievals, ())

    def test_16_multiple_foundry_invocations_in_order(self):
        messages = [
            _assistant_terminal_call("call-1", _foundry_command("q1")),
            _tool_result("call-1", _outer(output=_inner_success())),
            _assistant_terminal_call("call-2", _foundry_command("q2")),
            _tool_result("call-2", _outer(output=_inner_failure("no_documents"))),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(len(result.retrievals), 2)
        self.assertEqual(result.retrievals[0].tool_call_id, "call-1")
        self.assertEqual(result.retrievals[0].execution_status, "completed")
        self.assertEqual(result.retrievals[1].tool_call_id, "call-2")
        self.assertEqual(result.retrievals[1].execution_status, "no_documents")

    def test_17_tool_results_out_of_order_paired_by_id(self):
        messages = [
            _assistant_terminal_call("call-1", _foundry_command("q1")),
            _assistant_terminal_call("call-2", _foundry_command("q2")),
            # Results arrive in reverse order relative to the calls.
            _tool_result("call-2", _outer(output=_inner_failure("http_error", http_status=500))),
            _tool_result("call-1", _outer(output=_inner_success())),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(len(result.retrievals), 2)
        self.assertEqual(result.retrievals[0].tool_call_id, "call-1")
        self.assertEqual(result.retrievals[0].execution_status, "completed")
        self.assertEqual(result.retrievals[1].tool_call_id, "call-2")
        self.assertEqual(result.retrievals[1].execution_status, "http_error")
        self.assertEqual(result.retrievals[1].http_status, 500)


class MappingTests(unittest.TestCase):
    def _single(self, messages):
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        self.assertEqual(len(result.retrievals), 1)
        return result.retrievals[0]

    def test_3_inner_ok_true(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_success(documents=[{"ref_id": 0, "content": "a"}, {"ref_id": 1, "content": "b"}], references=[{"id": 1}]))),
        ])
        self.assertEqual(obs.execution_status, "completed")
        self.assertEqual(obs.foundry_iq_ok, True)
        self.assertEqual(obs.observation_status, "complete")
        self.assertIsNone(obs.observation_reason)
        self.assertEqual(obs.result_count, 2)
        self.assertEqual(obs.reference_count, 1)
        self.assertEqual(obs.foundry_schema_version, "foundry-iq-result-v2")
        self.assertTrue(obs.request_attempted)

    def test_4_inner_ok_false(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_failure("invalid_input"))),
        ])
        self.assertEqual(obs.execution_status, "failed")
        self.assertEqual(obs.foundry_iq_ok, False)
        self.assertEqual(obs.observation_status, "complete")
        self.assertEqual(obs.error_code, "invalid_input")

    def test_5_outer_exit_nonzero_inner_ok_false_parseable(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_failure("missing_query_key"), exit_code=1, error="script exited 1")),
        ])
        self.assertEqual(obs.execution_status, "failed")
        self.assertEqual(obs.foundry_iq_ok, False)
        self.assertEqual(obs.observation_status, "complete")

    def test_6_http_error_status_captured(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                obs = self._single([
                    _assistant_terminal_call("c", _foundry_command()),
                    _tool_result("c", _outer(output=_inner_failure("http_error", http_status=status))),
                ])
                self.assertEqual(obs.execution_status, "http_error")
                self.assertEqual(obs.http_status, status)
                self.assertEqual(obs.foundry_iq_ok, False)

    def test_7_timeout(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_failure("request_timeout"))),
        ])
        self.assertEqual(obs.execution_status, "timed_out")
        self.assertEqual(obs.foundry_iq_ok, False)
        self.assertEqual(obs.observation_status, "complete")

    def test_7b_outer_only_timeout_text_fails_closed(self):
        # No verified, machine-readable Hermes terminal-tool contract for
        # an outer-only timeout signal exists anywhere in this repository
        # (see MISSING_INNER_RESULT_REASON docstring in
        # retrieval_observer.py) -- even a very timeout-looking outer JSON
        # (exit_code=124, error="timeout", matching the POSIX `timeout`(1)
        # convention) must NOT be classified as "timed_out". It must fail
        # closed to unparseable/partial instead, exactly like any other
        # outer-only result with no inner Foundry JSON.
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output="", exit_code=124, error="timeout")),
        ])
        self.assertEqual(obs.execution_status, "unparseable")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, MISSING_INNER_RESULT_REASON)

    def test_8_outer_only_blocked_text_fails_closed(self):
        # Same reasoning as above: an outer JSON that merely *looks*
        # policy-blocked (error="blocked") must not be classified as
        # "blocked" without a sourced contract -- fails closed instead.
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output="", exit_code=126, error="blocked")),
        ])
        self.assertEqual(obs.execution_status, "unparseable")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, MISSING_INNER_RESULT_REASON)

    def test_outer_only_no_output_at_all_fails_closed(self):
        # No error/exit_code text at all -- the baseline "no inner result"
        # case, must be handled identically to the text-plausible ones
        # above (the parser must not special-case any particular exit_code
        # or error string, since none of them are verified).
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=None, exit_code=1, error=None)),
        ])
        self.assertEqual(obs.execution_status, "unparseable")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, MISSING_INNER_RESULT_REASON)

    def test_inner_request_timeout_still_reliably_classified(self):
        # In contrast to the outer-only cases above, the INNER Foundry IQ
        # v2 JSON's request_timeout error_code IS backed by real source
        # (advantech-hermes-support-config/skills/foundry-iq/scripts/
        # query_foundry_iq.py, ERROR_CODE_REQUEST_TIMEOUT), so this must
        # still reliably classify as timed_out -- Blocker 3 only removes
        # the *outer-only* guess, not the sourced inner mapping.
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_failure("request_timeout"))),
        ])
        self.assertEqual(obs.execution_status, "timed_out")
        self.assertEqual(obs.foundry_iq_ok, False)
        self.assertEqual(obs.observation_status, "complete")

    def test_9_network_error(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_failure("network_error"))),
        ])
        self.assertEqual(obs.execution_status, "network_error")
        self.assertEqual(obs.foundry_iq_ok, False)

    def test_10_no_documents(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output=_inner_failure("no_documents"))),
        ])
        self.assertEqual(obs.execution_status, "no_documents")
        self.assertEqual(obs.foundry_iq_ok, False)

    def test_11_truncated_outer_json(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", '{"output": "{\\"ok\\": true, ' ),  # deliberately truncated
        ])
        self.assertEqual(obs.execution_status, "unparseable")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, OUTER_RESULT_UNPARSEABLE_REASON)

    def test_12_truncated_inner_json(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", _outer(output='{"schema_version": "foundry-iq-result-v2", "ok": true, "doc')),
        ])
        self.assertEqual(obs.execution_status, "unparseable")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, INNER_RESULT_UNPARSEABLE_REASON)

    def test_13_persisted_truncated_placeholder(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", "[Tool result persisted to /sandbox/.hermes/tool_results/ref-8f3a.json - truncated]"),
        ])
        self.assertEqual(obs.execution_status, "unparseable")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, OUTER_RESULT_UNPARSEABLE_REASON)

    def test_14_tool_call_with_missing_result(self):
        obs = self._single([
            _assistant_terminal_call("c", _foundry_command()),
        ])
        self.assertEqual(obs.execution_status, "unknown")
        self.assertIsNone(obs.foundry_iq_ok)
        self.assertEqual(obs.observation_status, "partial")
        self.assertEqual(obs.observation_reason, TOOL_RESULT_MISSING_REASON)


class SafetyTests(unittest.TestCase):
    def test_20_output_type_excludes_forbidden_fields(self):
        field_names = {f.name for f in dataclasses.fields(FoundryRetrievalObservation)}
        for forbidden in ("documents", "content", "output", "raw_output", "command", "headers", "response_body"):
            self.assertNotIn(forbidden, field_names)
        turn_field_names = {f.name for f in dataclasses.fields(TurnRetrievalObservation)}
        for forbidden in ("messages", "documents", "content"):
            self.assertNotIn(forbidden, turn_field_names)

    def test_20b_constructing_with_forbidden_kwarg_raises(self):
        with self.assertRaises(TypeError):
            FoundryRetrievalObservation(  # type: ignore[call-arg]
                tool_call_id="c",
                execution_status="completed",
                foundry_iq_ok=True,
                observation_status="complete",
                observation_reason=None,
                error_code=None,
                http_status=None,
                result_count=1,
                reference_count=1,
                foundry_schema_version="v2",
                request_attempted=True,
                documents=["leaked document text"],
            )

    def test_21_canary_secret_never_leaks_into_output_or_reason(self):
        messages = [
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result(
                "c",
                _outer(
                    output=json.dumps(
                        {
                            "schema_version": "foundry-iq-result-v2",
                            "ok": False,
                            "request_attempted": True,
                            "error_code": "internal_error",
                            "question": "q",
                            "http_status": None,
                            "error": f"leaked secret: {CANARY_SECRET}",
                        }
                    ),
                    error=f"outer leak {CANARY_SECRET}",
                ),
            ),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        dumped = json.dumps(observation_to_safe_dict(result))
        self.assertNotIn(CANARY_SECRET, dumped)
        for obs in result.retrievals:
            for value in dataclasses.astuple(obs):
                self.assertNotIn(CANARY_SECRET, str(value))

    def test_21b_canary_secret_in_truncated_placeholder_never_leaks(self):
        messages = [
            _assistant_terminal_call("c", _foundry_command()),
            _tool_result("c", f"[persisted ref with leaked {CANARY_SECRET} inline]"),
        ]
        result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        dumped = json.dumps(observation_to_safe_dict(result))
        self.assertNotIn(CANARY_SECRET, dumped)

    def test_parser_never_raises_on_malformed_messages(self):
        malformed_inputs = [
            None,
            [],
            [None],
            ["not a dict"],
            [{"role": "assistant", "tool_calls": "not a list"}],
            [{"role": "assistant", "tool_calls": [{"id": "c", "function": "not a dict"}]}],
            [{"role": "tool", "tool_call_id": None, "content": "x"}],
        ]
        for messages in malformed_inputs:
            with self.subTest(messages=messages):
                result = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
                self.assertIn(result.turn_observation_status, ("complete", "unavailable"))


class RoundTripTests(unittest.TestCase):
    def test_safe_dict_round_trip(self):
        messages = [
            _assistant_terminal_call("c1", _foundry_command("q1")),
            _tool_result("c1", _outer(output=_inner_success())),
        ]
        original = parse_turn_retrieval_observations(messages, history_offset=0, boundary_trusted=True)
        restored = safe_dict_to_observation(observation_to_safe_dict(original))
        self.assertEqual(original, restored)

    def test_safe_dict_round_trip_untrusted(self):
        original = parse_turn_retrieval_observations([{"role": "user"}], history_offset=None, boundary_trusted=False)
        restored = safe_dict_to_observation(observation_to_safe_dict(original))
        self.assertEqual(original.turn_observation_status, restored.turn_observation_status)
        self.assertEqual(original.turn_observation_reason, restored.turn_observation_reason)


if __name__ == "__main__":
    unittest.main()
