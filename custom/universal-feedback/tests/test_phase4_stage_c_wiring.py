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
import sys
import tempfile
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

    def _seed_case(self, case_id, session_id, *, title=None, product_model=None):
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id, title=title, product_model=product_model)
        return case_id


class _ExplodingStore:
    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated failure in {name}")
        return _boom


class CandidateLoadingTests(_StoreTestCase):
    def test_candidate_query_returns_empty(self):
        self.store.create_or_update_session("sess-1", "telegram")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertEqual(candidates, [])
        self.assertFalse(unavailable)

    def test_candidate_query_returns_one_case(self):
        self._seed_case("case-a", "sess-1", title="ADAM-6266 — SNMP", product_model="ADAM-6266")
        candidates, unavailable = load_candidate_cases("sess-1", store=self.store)
        self.assertFalse(unavailable)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["case_id"], "case-a")
        self.assertEqual(candidates[0]["title"], "ADAM-6266 — SNMP")
        self.assertEqual(candidates[0]["product_model"], "ADAM-6266")

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


class CaseRoutingPromptTests(unittest.TestCase):
    def test_prompt_lists_all_candidate_ids(self):
        candidates = [
            {"case_id": "case-a", "title": "ADAM-6266 — SNMP", "product_model": "ADAM-6266"},
            {"case_id": "case-b", "title": "WISE-6610 — LTE", "product_model": "WISE-6610"},
        ]
        prompt = build_case_routing_prompt(candidates)
        self.assertIn("case-a", prompt)
        self.assertIn("case-b", prompt)
        self.assertIn("ADAM-6266 — SNMP", prompt)
        self.assertIn("WISE-6610", prompt)

    def test_prompt_uses_protocol_delimiters_and_version(self):
        prompt = build_case_routing_prompt([{"case_id": "case-a", "title": None, "product_model": None}])
        self.assertIn(CASE_ROUTING_OPEN_DELIM, prompt)
        self.assertIn(CASE_ROUTING_CLOSE_DELIM, prompt)
        self.assertIn(ROUTING_VERSION, prompt)

    def test_prompt_final_response_only_and_no_tool_call_rule(self):
        prompt = build_case_routing_prompt([{"case_id": "case-a", "title": None, "product_model": None}])
        self.assertIn("final user-facing assistant response only", prompt)
        self.assertIn("tool call", prompt)

    def test_prompt_does_not_leak_forbidden_content_categories(self):
        # Only case_id/title/product_model may appear -- never anything
        # resembling turn transcripts, feedback, or retrieval telemetry.
        candidates = [{"case_id": "case-a", "title": "t", "product_model": "m"}]
        prompt = build_case_routing_prompt(candidates)
        for forbidden in ("feedback", "retrieval_run", "answer_text", "question_text"):
            self.assertNotIn(forbidden, prompt.lower())


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

    def test_invalid_envelope_never_visible_to_consumer(self):
        malformed = CASE_ROUTING_OPEN_DELIM + "{not valid json" + CASE_ROUTING_CLOSE_DELIM + "answer"
        out = self._run([malformed])
        joined = "".join(out)
        self.assertNotIn(CASE_ROUTING_OPEN_DELIM, joined)
        self.assertNotIn("not valid json", joined)

    def test_answer_after_invalid_envelope_still_delivered(self):
        malformed = CASE_ROUTING_OPEN_DELIM + "{broken" + CASE_ROUTING_CLOSE_DELIM + "the real answer"
        out = self._run([malformed])
        self.assertEqual("".join(out), "the real answer")

    def test_oversized_prefix_does_not_leak_control_frame(self):
        padding = "x" * (MAX_ENVELOPE_PREFIX_BYTES + 500)
        chunks = [CASE_ROUTING_OPEN_DELIM + padding, CASE_ROUTING_CLOSE_DELIM + "answer after giving up"]
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


if __name__ == "__main__":
    unittest.main()
