"""Phase 4A Stage C tests: gateway candidate-loading / streaming-filter /
non-streaming / persistence wiring.

overlay/gateway/run.py is a full-file upstream overlay that requires the
proprietary Hermes agent runtime (run_agent.AIAgent, the real Telegram
adapter stack, etc.) to actually import -- confirmed here the same way
test_phase3a_wiring.py already established: ``import gateway.run`` raises
``ModuleNotFoundError: No module named 'agent'`` in this repo's test
environment. Three different verification strategies are used below,
matched to what each piece of Stage C wiring actually needs:

1. Everything that lives in tools/retrieval_runtime.py (candidate loading,
   the prompt builder, the candidate_context_unavailable taxonomy, the
   build_turn_observation_context/persist_turn_observation_context
   case_routing round trip) is plain importable Python -- tested here with
   real execution against a temporary FeedbackStoreV2, exactly like
   test_case_assignment.py.

2. ``_StreamEnvelopeFilter`` lives in gateway/run.py but has ZERO
   dependency on anything gateway/run.py-specific (only a lazy import of
   tools.case_routing constants inside its own method body) -- its exact
   source text is extracted from the real, deployed run.py file (between
   fixed markers) and exec()'d in an isolated namespace. This tests the
   REAL class body actually shipped in run.py, not a hand-copied
   duplicate that could silently drift from it -- if the class body in
   run.py ever changes, this test picks up the change automatically.

3. Wiring that cannot be isolated this way (the post-run_conversation
   parse-and-clean block, the phase4_* keys threaded onto the return
   dicts, the persistence call sites) is verified structurally at the
   source-text level, matching test_phase3a_wiring.py's established
   convention exactly.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

_RUN_PY = _OVERLAY_ROOT / "gateway" / "run.py"
_BASE_PY = _OVERLAY_ROOT / "gateway" / "platforms" / "base.py"

from tools.case_routing import (  # noqa: E402
    CASE_ROUTING_CLOSE_DELIM,
    CASE_ROUTING_OPEN_DELIM,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_ENVELOPE_PREFIX_BYTES,
    ROUTING_VERSION,
    CaseRoutingResult,
)
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.retrieval_observer import TurnRetrievalObservation  # noqa: E402
from tools.retrieval_runtime import (  # noqa: E402
    CASE_ASSIGNMENT_METHOD_CANDIDATE_CONTEXT_UNAVAILABLE,
    CASE_ASSIGNMENT_METHOD_FIRST_TURN,
    CASE_ASSIGNMENT_METHOD_NEW,
    build_case_routing_prompt,
    build_turn_observation_context,
    load_candidate_cases,
    persist_turn_and_retrieval,
    persist_turn_observation_context,
)


def _empty_observation() -> TurnRetrievalObservation:
    return TurnRetrievalObservation(
        boundary_trusted=True,
        turn_observation_status="complete",
        turn_observation_reason=None,
        retrievals=(),
    )


# ---------------------------------------------------------------------------
# 1. Candidate loading + prompt builder (real execution)
# ---------------------------------------------------------------------------


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_case(self, case_id, session_id):
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        return case_id

    def _seed_turn(self, turn_id, case_id, question_text, *, platform_user_message_id=None):
        self.store.create_turn(
            turn_id,
            case_id,
            platform_user_id="user-1",
            platform_user_message_id=platform_user_message_id or f"msg-{turn_id}",
            question_text=question_text,
            answer_text="answer",
            feedback_eligible=True,
        )
        return turn_id


class _ExplodingStore:
    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated failure in {name}")
        return _boom


class _TurnContextExplodingStore:
    """Delegates the cases lookup to a real store but always raises on the
    turn-context lookup -- used to verify that a turn-context failure
    (candidate Cases found, but their routing context could not be
    fetched) is treated as unavailable, never silently downgraded to
    "confirmed zero candidates"."""

    def __init__(self, store):
        self._store = store

    def list_cases_for_session(self, session_id):
        return self._store.list_cases_for_session(session_id)

    def list_turn_questions_for_cases(self, case_ids):
        raise RuntimeError("simulated turn-context lookup failure")


class CandidateLoadingTests(_StoreTestCase):
    def test_candidate_query_returns_empty(self):
        self.store.create_or_update_session("sess-1", "telegram")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertEqual(candidates, [])
        self.assertFalse(unavailable)

    def test_single_turn_case_has_first_but_no_latest(self):
        self._seed_case("case-a", "sess-1")
        self._seed_turn("turn-1", "case-a", "ADAM-6266 要怎麼關閉 SNMP？")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertFalse(unavailable)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["case_id"], "case-a")
        self.assertEqual(candidates[0]["first_user_question"], "ADAM-6266 要怎麼關閉 SNMP？")
        self.assertNotIn("latest_user_question", candidates[0])

    def test_multi_turn_case_has_first_and_latest_only(self):
        self._seed_case("case-a", "sess-1")
        self._seed_turn("turn-1", "case-a", "ADAM-6717 Node-RED crash 後 Web GUI 打不開")
        self._seed_turn("turn-2", "case-a", "怎麼用 SSH 看 error？")
        self._seed_turn("turn-3", "case-a", "看到 EADDRINUSE:502，接下來怎麼辦？")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertFalse(unavailable)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["first_user_question"], "ADAM-6717 Node-RED crash 後 Web GUI 打不開")
        self.assertEqual(candidates[0]["latest_user_question"], "看到 EADDRINUSE:502，接下來怎麼辦？")
        # the middle turn's question must never leak into the candidate context
        for value in candidates[0].values():
            self.assertNotIn("SSH", value)

    def test_multiple_cases_do_not_cross_contaminate(self):
        self._seed_case("case-a", "sess-1")
        self._seed_case("case-b", "sess-1")
        self._seed_turn("turn-a1", "case-a", "兩台 BB-USR604 Serial Number 相同造成衝突，要怎麼處理？")
        self._seed_turn("turn-b1", "case-b", "ADAM-6266 要怎麼關閉 SNMP？")
        self._seed_turn("turn-b2", "case-b", "ADAM-6266 是否能限制 SNMP GET/SET 的來源 IP？")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertFalse(unavailable)
        by_id = {c["case_id"]: c for c in candidates}
        self.assertEqual(
            by_id["case-a"]["first_user_question"],
            "兩台 BB-USR604 Serial Number 相同造成衝突，要怎麼處理？",
        )
        self.assertNotIn("latest_user_question", by_id["case-a"])
        self.assertEqual(by_id["case-b"]["first_user_question"], "ADAM-6266 要怎麼關閉 SNMP？")
        self.assertEqual(
            by_id["case-b"]["latest_user_question"],
            "ADAM-6266 是否能限制 SNMP GET/SET 的來源 IP？",
        )

    def test_candidate_query_returns_multiple_cases(self):
        self._seed_case("case-a", "sess-1")
        self._seed_case("case-b", "sess-1")
        self._seed_case("case-c", "sess-1")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertFalse(unavailable)
        self.assertEqual({c["case_id"] for c in candidates}, {"case-a", "case-b", "case-c"})

    def test_candidate_query_exception_is_unavailable_not_raised(self):
        candidates, unavailable = load_candidate_cases("sess-1", store=_ExplodingStore())
        self.assertEqual(candidates, [])
        self.assertTrue(unavailable)

    def test_candidate_query_store_none_is_unavailable(self):
        from unittest import mock
        import tools.retrieval_runtime as rr

        with mock.patch.object(rr, "get_store", return_value=None):
            candidates, unavailable = load_candidate_cases("sess-1")
        self.assertEqual(candidates, [])
        self.assertTrue(unavailable)

    def test_turn_context_query_failure_is_unavailable_not_empty(self):
        self._seed_case("case-a", "sess-1")
        self._seed_turn("turn-1", "case-a", "ADAM-6266 要怎麼關閉 SNMP？")
        candidates, unavailable = load_candidate_cases(
            "sess-1", store=_TurnContextExplodingStore(self.store)
        )
        self.assertEqual(candidates, [])
        self.assertTrue(unavailable)


class CaseRoutingPromptTests(unittest.TestCase):
    def test_prompt_includes_latest_only_when_present(self):
        candidates = [
            {"case_id": "case-a", "first_user_question": "first only"},
            {
                "case_id": "case-b",
                "first_user_question": "first q",
                "latest_user_question": "latest q",
            },
        ]
        prompt = build_case_routing_prompt(candidates)
        self.assertIn("first only", prompt)
        self.assertNotIn("latest_user_question: first only", prompt)
        self.assertIn("latest_user_question: latest q", prompt)

    def test_prompt_uses_protocol_delimiters_and_version(self):
        prompt = build_case_routing_prompt([{"case_id": "case-a"}])
        self.assertIn(CASE_ROUTING_OPEN_DELIM, prompt)
        self.assertIn(CASE_ROUTING_CLOSE_DELIM, prompt)
        self.assertIn(ROUTING_VERSION, prompt)

    def test_prompt_final_response_only_and_no_tool_call_rule(self):
        prompt = build_case_routing_prompt([{"case_id": "case-a"}])
        self.assertIn("final user-facing assistant response only", prompt)
        self.assertIn("tool call", prompt)

    def test_prompt_has_same_product_topic_not_same_case_rule(self):
        prompt = build_case_routing_prompt([{"case_id": "case-a"}])
        self.assertIn("Same product/topic/category does not by itself mean the same Case", prompt)

    def test_prompt_does_not_leak_forbidden_content_categories(self):
        # Only case_id/first_user_question/latest_user_question may appear
        # -- never anything resembling assistant answers, feedback, or
        # retrieval telemetry.
        candidates = [
            {"case_id": "case-a", "first_user_question": "q1", "latest_user_question": "q2"}
        ]
        prompt = build_case_routing_prompt(candidates)
        for forbidden in ("feedback", "retrieval_run", "answer_text"):
            self.assertNotIn(forbidden, prompt.lower())

    def test_prompt_never_reduces_existing_case_to_bare_case_id(self):
        # Core regression: BB-USR604 (Case A) vs ADAM-6266 SNMP (Case B) --
        # the model must be able to tell them apart from the prompt text
        # alone, not just see two indistinguishable UUIDs.
        candidates = [
            {
                "case_id": "11111111-1111-1111-1111-111111111111",
                "first_user_question": "兩台 BB-USR604 接同一台電腦，Serial Number 相同造成辨識衝突，要怎麼處理？",
            },
            {
                "case_id": "22222222-2222-2222-2222-222222222222",
                "first_user_question": "ADAM-6266 要怎麼關閉 SNMP？",
            },
        ]
        prompt = build_case_routing_prompt(candidates)
        self.assertIn("BB-USR604", prompt)
        self.assertIn("Serial Number", prompt)
        self.assertIn("ADAM-6266", prompt)
        self.assertIn("SNMP", prompt)
        # each candidate's evidence must be scoped to its own case_id block,
        # not merged into one undifferentiated blob
        case_a_idx = prompt.index("11111111-1111-1111-1111-111111111111")
        case_b_idx = prompt.index("22222222-2222-2222-2222-222222222222")
        bb_usr604_idx = prompt.index("BB-USR604")
        snmp_idx = prompt.index("ADAM-6266 要怎麼關閉 SNMP")
        self.assertLess(case_a_idx, bb_usr604_idx)
        self.assertLess(bb_usr604_idx, case_b_idx)
        self.assertLess(case_b_idx, snmp_idx)


# ---------------------------------------------------------------------------
# 2. _StreamEnvelopeFilter -- real class body extracted from gateway/run.py
# ---------------------------------------------------------------------------


def _load_stream_envelope_filter_class():
    text = _RUN_PY.read_text(encoding="utf-8")
    start_marker = "class _StreamEnvelopeFilter:"
    end_marker = "\ndef _phase4_strip_final_assistant_message("
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    source = text[start:end]
    namespace: dict = {}
    exec(compile(source, str(_RUN_PY) + ":_StreamEnvelopeFilter", "exec"), namespace)
    return namespace["_StreamEnvelopeFilter"]


_StreamEnvelopeFilter = _load_stream_envelope_filter_class()


def _valid_envelope_text(action="new", case_id=None, confidence=0.9):
    import json

    body = json.dumps({
        "case_action": action, "case_id": case_id,
        "confidence": confidence, "routing_version": ROUTING_VERSION,
    })
    return f"{CASE_ROUTING_OPEN_DELIM}{body}{CASE_ROUTING_CLOSE_DELIM}"


# ---------------------------------------------------------------------------
# Shared malformed Case Routing envelope fixtures, reused below by BOTH
# StreamEnvelopeFilterTests (the live streaming display filter,
# _StreamEnvelopeFilter) and NonStreamingEnvelopeLeakSafetyTests (the
# non-streaming authoritative strip in run.py's _run_agent_inner). These are
# two genuinely different code paths -- one incrementally buffers chunks,
# the other parses one complete string -- so they are NOT collapsed into a
# single shared test; only the malformed-shape TEXT itself is defined once
# to avoid re-typing the same garbage bytes in both classes.
# ---------------------------------------------------------------------------

_MALFORMED_ENVELOPE_JSON_BODY = "{not valid json"
_OVERSIZED_ENVELOPE_PADDING = "x" * (MAX_ENVELOPE_PREFIX_BYTES + 500)


def _unsupported_routing_version_body() -> str:
    return json.dumps({
        "case_action": "new", "case_id": None,
        "confidence": 0.9, "routing_version": "case-router-v0",
    })


class StreamEnvelopeFilterTests(unittest.TestCase):
    def _run(self, chunks):
        out = []
        f = _StreamEnvelopeFilter(out.append)
        for chunk in chunks:
            f.feed(chunk)
        return out

    def test_valid_envelope_in_one_chunk(self):
        out = self._run([_valid_envelope_text() + "the answer"])
        self.assertEqual("".join(x for x in out if x), "the answer")
        self.assertNotIn(None, out)  # no stray None injected by the filter itself

    def test_opening_delimiter_split_across_chunks(self):
        env = _valid_envelope_text() + "the answer"
        # Split right in the middle of "<case-routing>"
        chunks = [env[:5], env[5:]]
        out = self._run(chunks)
        self.assertEqual("".join(out), "the answer")

    def test_json_split_across_chunks(self):
        env = _valid_envelope_text(action="existing", case_id="case-a", confidence=0.5)
        mid = len(CASE_ROUTING_OPEN_DELIM) + 10
        chunks = [env[:mid], env[mid:] + "the answer"]
        out = self._run(chunks)
        self.assertEqual("".join(out), "the answer")

    def test_closing_delimiter_split_across_chunks(self):
        env = _valid_envelope_text() + "the answer"
        close_pos = env.index(CASE_ROUTING_CLOSE_DELIM)
        split = close_pos + 3  # split inside the closing tag itself
        chunks = [env[:split], env[split:]]
        out = self._run(chunks)
        self.assertEqual("".join(out), "the answer")

    def test_close_and_answer_in_same_chunk(self):
        out = self._run([_valid_envelope_text() + "the answer, all in one go"])
        self.assertEqual("".join(out), "the answer, all in one go")

    def test_normal_text_diverges_immediately_no_delay(self):
        out = self._run(["H", "ello, ", "this is a normal answer"])
        self.assertEqual(out, ["H", "ello, ", "this is a normal answer"])

    def test_valid_envelope_never_visible_to_consumer(self):
        out = self._run([_valid_envelope_text(action="existing", case_id="case-a") + "answer"])
        joined = "".join(out)
        self.assertNotIn(CASE_ROUTING_OPEN_DELIM, joined)
        self.assertNotIn("case_action", joined)
        self.assertNotIn("case-a", joined)

    def test_invalid_envelope_never_visible_and_answer_still_delivered(self):
        # Malformed JSON between well-formed delimiters, fed as one chunk:
        # neither the garbage body nor the delimiter itself may reach the
        # consumer, and the real answer after the close delimiter must
        # still be delivered intact.
        malformed = (
            CASE_ROUTING_OPEN_DELIM + _MALFORMED_ENVELOPE_JSON_BODY + CASE_ROUTING_CLOSE_DELIM + "the real answer"
        )
        out = self._run([malformed])
        joined = "".join(out)
        self.assertEqual(joined, "the real answer")
        self.assertNotIn(CASE_ROUTING_OPEN_DELIM, joined)
        self.assertNotIn(_MALFORMED_ENVELOPE_JSON_BODY, joined)

    def test_oversized_prefix_does_not_leak_control_frame(self):
        # Split across chunks on purpose (unlike the malformed-JSON case
        # above): this proves the filter gives up mid-stream once the
        # buffered candidate exceeds MAX_ENVELOPE_PREFIX_BYTES, and that a
        # closing delimiter arriving in a LATER chunk (after give-up) is
        # then treated as plain text rather than re-triggering envelope
        # detection -- a chunk-boundary behavior the non-streaming leg has
        # no equivalent of, since it parses one already-complete string.
        chunks = [
            CASE_ROUTING_OPEN_DELIM + _OVERSIZED_ENVELOPE_PADDING,
            CASE_ROUTING_CLOSE_DELIM + "answer after giving up",
        ]
        out = self._run(chunks)
        joined = "".join(out)
        self.assertNotIn("x" * 100, joined)  # none of the padding ever leaked
        self.assertNotIn(CASE_ROUTING_OPEN_DELIM, joined)

    def test_none_sentinel_passed_through_unchanged(self):
        out = []
        f = _StreamEnvelopeFilter(out.append)
        f.feed("partial <cas")
        f.feed(None)
        self.assertIn(None, out)

    def test_none_sentinel_discards_buffered_candidate_before_final_completion(self):
        # Simulates: an intermediate tool-call completion streams some
        # envelope-shaped text, then the None boundary fires (tool
        # execution about to start, per conversation_loop.py evidence),
        # then the NEXT (final) completion streams a normal answer.
        # Whatever was buffered from the intermediate completion must be
        # discarded, not merged into the next completion's stream.
        out = []
        f = _StreamEnvelopeFilter(out.append)
        f.feed(CASE_ROUTING_OPEN_DELIM + '{"case_action":"existing"')  # intermediate, incomplete
        f.feed(None)  # tool-call boundary
        f.feed("the final answer")
        self.assertEqual([x for x in out if x is not None], ["the final answer"])

    def test_state_resets_fresh_for_next_completion_after_none(self):
        # First completion has NO envelope at all (goes straight to
        # NORMAL). After a None reset, the SECOND completion DOES start
        # with a real envelope -- it must still be caught and hidden, not
        # accidentally streamed because the filter stayed in NORMAL mode.
        out = []
        f = _StreamEnvelopeFilter(out.append)
        f.feed("first completion text, no envelope here")
        f.feed(None)
        f.feed(_valid_envelope_text() + "second completion answer")
        visible = [x for x in out if x is not None]
        self.assertEqual(visible, ["first completion text, no envelope here", "second completion answer"])


# ---------------------------------------------------------------------------
# 3. Structural wiring verification (source-text level, matching
#    test_phase3a_wiring.py's established convention for gateway/run.py and
#    gateway/platforms/base.py, neither of which is importable here).
# ---------------------------------------------------------------------------


class _SourceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_py_text = _RUN_PY.read_text(encoding="utf-8")
        cls.base_py_text = _BASE_PY.read_text(encoding="utf-8")


class StartupFeedbackStoreInitWiringTests(_SourceTestCase):
    """Fresh-onboard DB initialization must not require a manual
    `FeedbackStoreV2(...)` construction -- GatewayRunner.start() eagerly
    (and idempotently) initializes the store once at process startup."""

    def test_get_store_called_eagerly_in_start(self):
        self.assertIn(
            "from tools.retrieval_runtime import get_store as _get_feedback_store",
            self.run_py_text,
        )
        self.assertIn("_get_feedback_store()", self.run_py_text)

    def test_eager_init_happens_inside_start_after_gateway_state_starting(self):
        start_idx = self.run_py_text.index("async def start(self) -> bool:")
        starting_state_idx = self.run_py_text.index(
            'write_runtime_status(gateway_state="starting", exit_reason=None)', start_idx
        )
        init_idx = self.run_py_text.index("_get_feedback_store()", start_idx)
        self.assertGreater(init_idx, starting_state_idx)

    def test_eager_init_never_raises_out_of_start(self):
        init_idx = self.run_py_text.index("_get_feedback_store()")
        surrounding = self.run_py_text[max(0, init_idx - 400):init_idx]
        self.assertIn("try:", surrounding)


class StreamingCallbackWiringTests(_SourceTestCase):
    def test_stream_delta_cb_wraps_with_envelope_filter(self):
        self.assertIn("_phase4_stream_filter = _StreamEnvelopeFilter(_stream_consumer.on_delta)", self.run_py_text)
        self.assertIn("_phase4_stream_filter.feed(text)", self.run_py_text)
        # Must not call _stream_consumer.on_delta directly from
        # _stream_delta_cb anymore -- it must always go through the filter.
        cb_start = self.run_py_text.index("def _stream_delta_cb(text: str) -> None:")
        cb_body = self.run_py_text[cb_start:cb_start + 200]
        self.assertNotIn("_stream_consumer.on_delta(text)", cb_body)


class RoutingAcceptanceWiringTests(_SourceTestCase):
    def test_final_response_parsed_via_frozen_stage_a_parser(self):
        self.assertIn(
            "from tools.case_routing import parse_and_strip_prefix as _phase4_parse_and_strip_prefix",
            self.run_py_text,
        )
        self.assertIn(
            "final_response, case_routing = _phase4_parse_and_strip_prefix(final_response)",
            self.run_py_text,
        )

    def test_final_response_dict_key_is_overwritten_with_clean_value(self):
        self.assertIn('result["final_response"] = final_response', self.run_py_text)

    def test_messages_cleanup_helper_called_after_run_conversation(self):
        run_conv_idx = self.run_py_text.index("result = agent.run_conversation(_api_run_message")
        cleanup_idx = self.run_py_text.index("_phase4_strip_final_assistant_message(result.get(\"messages\"))")
        self.assertGreater(cleanup_idx, run_conv_idx)

    def test_parse_happens_before_maybe_auto_title(self):
        parse_idx = self.run_py_text.index("_phase4_parse_and_strip_prefix(final_response)")
        title_idx = self.run_py_text.index("maybe_auto_title(")
        self.assertLess(parse_idx, title_idx)

    def test_messages_cleanup_function_only_touches_last_assistant_message(self):
        func_start = self.run_py_text.index("def _phase4_strip_final_assistant_message(")
        func_end = self.run_py_text.index("\ndef _sanitize_gateway_final_response(", func_start)
        body = self.run_py_text[func_start:func_end]
        self.assertIn("messages[-1]", body)
        self.assertIn('last.get("role") != "assistant"', body)
        self.assertIn("strip_case_routing_prefix", body)


class PersistenceCallWiringTests(_SourceTestCase):
    def test_streaming_persist_call_passes_phase4_routing_context(self):
        call_start = self.run_py_text.index('response["phase3a_turn_persisted"] = observe_and_persist_turn(')
        call_end = self.run_py_text.index("\n                )", call_start)
        call_block = self.run_py_text[call_start:call_end]
        self.assertIn('case_routing=response.get("phase4_case_routing")', call_block)
        self.assertIn("phase4_candidate_case_ids", call_block)
        self.assertIn("candidate_context_unavailable=", call_block)

    def test_non_streaming_context_build_passes_phase4_routing_context(self):
        call_start = self.run_py_text.index("_phase3a_ctx = build_turn_observation_context(")
        call_end = self.run_py_text.index("\n                    )", call_start)
        call_block = self.run_py_text[call_start:call_end]
        self.assertIn("candidate_case_ids=", call_block)
        self.assertIn("candidate_context_unavailable=", call_block)
        self.assertIn('case_routing=agent_result.get("phase4_case_routing")', call_block)

    def test_base_py_persist_call_unchanged_relies_on_context_fallback(self):
        # base.py must NOT need its own case_routing= override -- it
        # relies entirely on persist_turn_observation_context's
        # context["case_routing"] fallback (plan section 9: single parse
        # point, base.py never re-parses the envelope).
        call_start = self.base_py_text.index("persist_turn_observation_context(")
        call_end = self.base_py_text.index("\n                            )", call_start)
        call_block = self.base_py_text[call_start:call_end]
        self.assertNotIn("case_routing=", call_block)
        self.assertNotIn("parse_and_strip_prefix", self.base_py_text)

    def test_candidate_loading_injected_before_ephemeral_prompt_finalized(self):
        ephemeral_idx = self.run_py_text.index("if self._ephemeral_system_prompt:")
        candidate_idx = self.run_py_text.index("load_candidate_cases(session_id)")
        # _current_max_iterations() is called at multiple unrelated sites in
        # this large file -- search for the one immediately downstream of
        # the candidate-loading block, not the first occurrence anywhere.
        max_iter_idx = self.run_py_text.index("max_iterations = _current_max_iterations()", candidate_idx)
        self.assertLess(ephemeral_idx, candidate_idx)
        self.assertLess(candidate_idx, max_iter_idx)
        self.assertLess(max_iter_idx - candidate_idx, 2000, "candidate loading should sit directly before this call")

    def test_prompt_only_injected_when_candidates_present_and_available(self):
        self.assertIn(
            "if _phase4_candidate_cases and not _phase4_candidate_context_unavailable:",
            self.run_py_text,
        )


# ---------------------------------------------------------------------------
# 4. Stage C additions to retrieval_runtime.py: candidate_context_unavailable
#    end-to-end, and the build/persist context case_routing round trip.
#    (First-turn/existing/new/uncertain/low-confidence/unknown-case_id are
#    already covered by Stage B's test_case_assignment.py -- not repeated
#    here, per the plan's "don't re-test parser internals" guidance.)
# ---------------------------------------------------------------------------


class CandidateContextUnavailableIntegrationTests(_StoreTestCase):
    def test_unavailable_creates_real_new_case_not_default_not_first_turn(self):
        self.store.create_or_update_session("sess-1", "telegram")
        ok = persist_turn_and_retrieval(
            self.store,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="turn-1-msg",
            platform_assistant_message_id=None, question_text="q", answer_text="a",
            feedback_eligible=True, model=None, provider=None, hermes_version=None,
            support_config_commit=None, feedback_code_commit=None,
            observation=_empty_observation(),
            candidate_case_ids=frozenset(),
            candidate_context_unavailable=True,
        )
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_CANDIDATE_CONTEXT_UNAVAILABLE)
        self.assertNotEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FIRST_TURN)
        self.assertIsNone(turn["case_assignment_confidence"])

    def test_unavailable_ignores_case_routing_content_entirely(self):
        self.store.create_or_update_session("sess-1", "telegram")
        routing = CaseRoutingResult(
            status="valid", case_action="existing", case_id="case-a",
            confidence=0.99, routing_version=ROUTING_VERSION,
        )
        persist_turn_and_retrieval(
            self.store,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="turn-1-msg",
            platform_assistant_message_id=None, question_text="q", answer_text="a",
            feedback_eligible=True, model=None, provider=None, hermes_version=None,
            support_config_commit=None, feedback_code_commit=None,
            observation=_empty_observation(),
            case_routing=routing,
            candidate_case_ids=frozenset(),
            candidate_context_unavailable=True,
        )
        turn = self.store.get_turn("turn-1")
        self.assertNotEqual(turn["case_id"], "case-a")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_CANDIDATE_CONTEXT_UNAVAILABLE)


class BuildPersistContextCaseRoutingRoundTripTests(_StoreTestCase):
    def test_case_routing_survives_context_round_trip(self):
        agent_result = {
            "final_response": "answer", "model": None, "messages": [], "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        routing = CaseRoutingResult(
            status="valid", case_action="new", case_id=None,
            confidence=0.77, routing_version=ROUTING_VERSION,
        )
        # Non-empty candidate_case_ids: this test verifies the case_routing
        # round trip mechanism itself, not the (separately-tested)
        # first-turn short-circuit that an empty set would trigger
        # regardless of case_routing content.
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
            candidate_case_ids=frozenset({"case-existing"}),
            case_routing=routing,
        )
        self.assertIsNotNone(ctx)
        import json
        json.dumps(ctx)  # still JSON-safe after stashing case_routing
        self.assertEqual(ctx["case_routing"]["case_action"], "new")
        self.assertEqual(ctx["case_routing"]["confidence"], 0.77)

        # persist_turn_observation_context does NOT receive an explicit
        # case_routing= override -- it must reconstruct it from context.
        ok = persist_turn_observation_context(
            ctx,
            platform_assistant_message_id="asst-1",
            question_text="q", answer_text="a", feedback_eligible=True,
            store=self.store,
        )
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_NEW)
        self.assertEqual(turn["case_assignment_confidence"], "0.77")

    def test_direct_case_routing_override_wins_over_context(self):
        agent_result = {
            "final_response": "answer", "model": None, "messages": [], "history_offset": 0,
            "phase3a_boundary_trusted": True,
        }
        stashed_routing = CaseRoutingResult(
            status="valid", case_action="new", case_id=None,
            confidence=0.5, routing_version=ROUTING_VERSION,
        )
        ctx = build_turn_observation_context(
            agent_result,
            session_id="sess-1", platform="telegram", platform_chat_id="chat-1",
            turn_id="turn-1", platform_user_id="user-1", platform_user_message_id="msg-1",
            candidate_case_ids=frozenset(),
            case_routing=stashed_routing,
        )
        override_routing = CaseRoutingResult(status="absent")
        ok = persist_turn_observation_context(
            ctx,
            platform_assistant_message_id="asst-1",
            question_text="q", answer_text="a", feedback_eligible=True,
            store=self.store, case_routing=override_routing,
        )
        self.assertTrue(ok)
        turn = self.store.get_turn("turn-1")
        # override_routing (absent) + candidate_case_ids=frozenset() ->
        # first_turn, NOT the stashed "new" from context.
        self.assertEqual(turn["case_assignment_method"], CASE_ASSIGNMENT_METHOD_FIRST_TURN)


# ---------------------------------------------------------------------------
# 5. Non-streaming delivery leak-safety -- complements StreamEnvelopeFilterTests
#    (section 2 above), which only proves the LIVE token-stream display
#    filter hides a malformed envelope. The non-streaming delivery leg never
#    goes through ``_StreamEnvelopeFilter`` at all: run.py's
#    ``_handle_message_with_agent`` reads ``agent_result.get("final_response")``
#    directly (~line 11114) and hands that text straight to base.py's
#    ``_send_with_retry`` with no further case-routing-aware processing --
#    base.py never imports ``tools.case_routing`` and never looks for the
#    delimiters (see ``test_base_py_never_re_strips_case_routing`` below). So
#    on this leg, the ONLY thing standing between a malformed model output
#    and the Telegram user is the "authoritative parse" block in
#    ``_run_agent_inner`` (run.py ~17910-17919) -- extracted here, verbatim,
#    from the real deployed file, exec()'d in an isolated namespace (same
#    technique ``_load_stream_envelope_filter_class`` uses above), and run
#    against protocol-malformed input to prove what actually reaches
#    delivery. This tests the REAL wiring, not a hand-copied paraphrase of
#    ``strip_case_routing_prefix`` -- per the Stage C regression-gap
#    request, a passing ``tools.case_routing`` unit test alone does not
#    prove the delivery/wiring contract holds.
# ---------------------------------------------------------------------------


def _load_phase4_authoritative_strip_snippet() -> str:
    text = _RUN_PY.read_text(encoding="utf-8")
    start_marker = '            case_routing = None\n            if final_response:'
    end_marker = '\n            _phase4_strip_final_assistant_message(result.get("messages"))'
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return textwrap.dedent(text[start:end])


_PHASE4_AUTHORITATIVE_STRIP_SNIPPET = _load_phase4_authoritative_strip_snippet()


def _apply_authoritative_strip(raw_final_response: str):
    """Run the REAL run.py wiring block (not a paraphrase) that produces the
    ``final_response`` text every non-streaming delivery is built from.
    Binds the same free variables (``result``, ``final_response``,
    ``logger``) the snippet references in its native context in run.py, so
    this executes unmodified production source, not a reimplementation of
    it. Returns ``(final_response, case_routing)`` exactly like the real
    block leaves them in its enclosing scope."""
    namespace: dict = {
        "result": {"final_response": raw_final_response},
        "final_response": raw_final_response,
        "logger": logging.getLogger("test_phase4_stage_c_wiring.authoritative_strip"),
    }
    exec(
        compile(
            _PHASE4_AUTHORITATIVE_STRIP_SNIPPET,
            str(_RUN_PY) + ":phase4_authoritative_strip",
            "exec",
        ),
        namespace,
    )
    return namespace["result"]["final_response"], namespace.get("case_routing")


def _non_streaming_delivered_text(final_response_after_strip: str) -> str:
    """The real extraction line from run.py's _handle_message_with_agent
    (~line 11114): ``response = agent_result.get("final_response") or ""``
    -- the exact text handed to base.py's ``_send_with_retry`` with no
    further case-routing-aware processing on this leg (see
    ``test_base_py_never_re_strips_case_routing``)."""
    agent_result = {"final_response": final_response_after_strip}
    return agent_result.get("final_response") or ""


class NonStreamingEnvelopeLeakSafetyTests(_SourceTestCase):
    def test_delivery_extraction_line_matches_real_source(self):
        # Pins the exact line _non_streaming_delivered_text mirrors, so a
        # future edit to this extraction in run.py invalidates the
        # simulation above instead of silently drifting from it.
        self.assertIn(
            'response = agent_result.get("final_response") or ""',
            self.run_py_text,
        )

    def test_base_py_never_re_strips_case_routing(self):
        # Confirms the authoritative strip really is the ONLY thing between
        # the model and the Telegram user on the non-streaming leg: the
        # send path in base.py never imports tools.case_routing and never
        # looks for the envelope delimiters.
        self.assertNotIn("case_routing", self.base_py_text)
        self.assertNotIn(CASE_ROUTING_OPEN_DELIM, self.base_py_text)

    def test_valid_envelope_stripped_before_non_streaming_delivery(self):
        raw = _valid_envelope_text(action="existing", case_id="case-a") + "the real answer"
        stripped, routing = _apply_authoritative_strip(raw)
        delivered = _non_streaming_delivered_text(stripped)
        self.assertEqual(delivered, "the real answer")
        self.assertNotIn(CASE_ROUTING_OPEN_DELIM, delivered)
        self.assertEqual(routing.status, "valid")

    def test_malformed_envelope_shapes_never_reach_non_streaming_delivery(self):
        # Four malformed shapes, all proven against the one real run.py
        # snippet extracted above -- this leg parses one already-complete
        # string (no chunk-boundary concerns, unlike the streaming filter),
        # so each shape's outcome can be checked uniformly in one loop
        # without changing what each case individually proves.
        cases = (
            (
                "malformed_json",
                CASE_ROUTING_OPEN_DELIM + _MALFORMED_ENVELOPE_JSON_BODY + CASE_ROUTING_CLOSE_DELIM,
                (_MALFORMED_ENVELOPE_JSON_BODY,),
            ),
            (
                # The model's turn ends mid-envelope -- no closing
                # delimiter ever arrives (e.g. truncated by a length limit
                # or upstream error).
                "unclosed",
                CASE_ROUTING_OPEN_DELIM + '{"case_action":"existing"',
                (),
            ),
            (
                "oversized",
                CASE_ROUTING_OPEN_DELIM + _OVERSIZED_ENVELOPE_PADDING + CASE_ROUTING_CLOSE_DELIM,
                ("x" * 100,),
            ),
            (
                "unsupported_routing_version",
                CASE_ROUTING_OPEN_DELIM + _unsupported_routing_version_body() + CASE_ROUTING_CLOSE_DELIM,
                ("case_action",),
            ),
        )
        for name, envelope_prefix, forbidden in cases:
            with self.subTest(shape=name):
                raw = envelope_prefix + "the real answer"
                stripped, routing = _apply_authoritative_strip(raw)
                delivered = _non_streaming_delivered_text(stripped)
                self.assertNotIn(CASE_ROUTING_OPEN_DELIM, delivered)
                for token in forbidden:
                    self.assertNotIn(token, delivered)


if __name__ == "__main__":
    unittest.main()
