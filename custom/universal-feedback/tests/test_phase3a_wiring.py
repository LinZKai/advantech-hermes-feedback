"""Phase 3A wiring tests.

overlay/gateway/run.py and overlay/gateway/platforms/base.py are full-file
upstream overlays that require the proprietary Hermes agent runtime
(run_agent.AIAgent, the real Telegram adapter stack, etc.) to actually
execute -- none of that is importable in this repo (see AGENTS.md; Phase 2's
own test suite only ever imports tools/*, never run.py/base.py directly).
These tests therefore verify the wiring *structurally*, at the source-text
level: that both delivery paths call into the one shared parser/persistence
module, that neither path uses a per-turn tool_complete_callback closure,
that every Phase 3A call site is guarded by its own try/except so a
parser/store failure can never reach the answer or the legacy feedback
flow, and -- the point of the blocker-2 revision -- that v2 persistence is
gated purely on delivery confirmation and never on
feedback_eligible()/claim_feedback_ownership()/universal_feedback_handled.
Behavioral proof that the two flows are actually independent at runtime
(legacy hook failing doesn't block v2, v2 failing doesn't block the legacy
hook, no double-write) is in test_phase3a_decoupling.py, using the real
tools.retrieval_runtime functions with a simulated call sequence that
mirrors what these source blocks actually do. Runtime behavior of the
shared parser/storage module itself is covered by test_retrieval_observer.py
and test_retrieval_runtime.py.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

_RUN_PY = _OVERLAY_ROOT / "gateway" / "run.py"
_BASE_PY = _OVERLAY_ROOT / "gateway" / "platforms" / "base.py"

# Block boundaries, shared across test classes below. The streaming and
# non-streaming Phase 3A blocks are each bounded by their own start marker
# and the marker of whatever comes next in the file (the legacy feedback
# hook, in both cases -- Phase 3A now runs BEFORE it in the source, see
# Blocker 2 revision) so a check can never accidentally match unrelated
# code elsewhere in these very large files.
_STREAMING_START = "# Phase 3A telemetry: independent of the legacy hook below"
_STREAMING_END = "# Universal message-level feedback hook: streaming final content has"
_NON_STREAMING_PARSE_START = "# Phase 3A telemetry, non-streaming leg: agent_result still has"
_NON_STREAMING_PARSE_END = 'response = agent_result.get("final_response") or ""'
_NON_STREAMING_PERSIST_START = "# Phase 3A telemetry, non-streaming leg: independent of"
_NON_STREAMING_PERSIST_END = "# Universal feedback is owned by this normal-send path"


class _SourceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_py_text = _RUN_PY.read_text(encoding="utf-8")
        cls.base_py_text = _BASE_PY.read_text(encoding="utf-8")

    @staticmethod
    def _code_only(block: str) -> str:
        """Strip full-line comments (leading '#' after whitespace) so
        assertions about what the *code* does or doesn't reference aren't
        tripped up by prose in the surrounding explanatory comment, which
        necessarily mentions the very names/flags it explains are NOT
        used."""
        return "\n".join(
            line for line in block.splitlines() if line.strip() and not line.strip().startswith("#")
        )

    def _streaming_block(self, *, code_only: bool = False) -> str:
        start = self.run_py_text.index(_STREAMING_START)
        end = self.run_py_text.index(_STREAMING_END)
        block = self.run_py_text[start:end]
        return self._code_only(block) if code_only else block

    def _non_streaming_parse_block(self, *, code_only: bool = False) -> str:
        start = self.run_py_text.index(_NON_STREAMING_PARSE_START)
        end = self.run_py_text.index(_NON_STREAMING_PARSE_END)
        block = self.run_py_text[start:end]
        return self._code_only(block) if code_only else block

    def _non_streaming_persist_block(self, *, code_only: bool = False) -> str:
        start = self.base_py_text.index(_NON_STREAMING_PERSIST_START)
        end = self.base_py_text.index(_NON_STREAMING_PERSIST_END)
        block = self.base_py_text[start:end]
        return self._code_only(block) if code_only else block


class NoToolCompleteCallbackTests(_SourceTestCase):
    def test_run_py_has_no_tool_complete_callback(self):
        self.assertNotIn("tool_complete_callback", self.run_py_text)

    def test_base_py_has_no_tool_complete_callback(self):
        self.assertNotIn("tool_complete_callback", self.base_py_text)


class SharedModuleUsageTests(_SourceTestCase):
    """Both delivery paths must call into tools.retrieval_runtime -- never
    a second, parallel parser/storage implementation."""

    def test_run_py_streaming_path_uses_observe_and_persist_turn(self):
        self.assertIn("from tools.retrieval_runtime import observe_and_persist_turn", self.run_py_text)
        self.assertEqual(self.run_py_text.count("observe_and_persist_turn("), 1)

    def test_run_py_non_streaming_path_uses_build_turn_observation_context(self):
        self.assertIn("from tools.retrieval_runtime import build_turn_observation_context", self.run_py_text)
        self.assertEqual(self.run_py_text.count("build_turn_observation_context("), 1)

    def test_base_py_uses_persist_turn_observation_context(self):
        self.assertIn("from tools.retrieval_runtime import persist_turn_observation_context", self.base_py_text)
        self.assertEqual(self.base_py_text.count("persist_turn_observation_context("), 1)

    def test_run_py_does_not_reimplement_foundry_detection(self):
        # foundry_marker/parsing logic must live only in retrieval_observer;
        # run.py's Phase 3A glue must not hand-roll its own query_foundry_iq
        # string search.
        self.assertNotIn("query_foundry_iq.py", self._streaming_block())
        self.assertNotIn("query_foundry_iq.py", self._non_streaming_parse_block())

    def test_base_py_does_not_reimplement_foundry_detection(self):
        self.assertNotIn("query_foundry_iq.py", self._non_streaming_persist_block())


class FailureIsolationTests(_SourceTestCase):
    """Every Phase 3A call site must be inside its own try/except Exception
    block, distinct from (and never replacing) the existing legacy feedback
    hook's own try/except."""

    def test_run_py_streaming_block_wrapped_in_try_except(self):
        block = self._streaming_block()
        self.assertIn("try:", block)
        self.assertIn("except Exception:", block)
        self.assertIn('logger.debug("Phase 3A retrieval observation failed", exc_info=True)', block)

    def test_run_py_non_streaming_parse_step_wrapped_in_try_except(self):
        block = self._non_streaming_parse_block()
        self.assertIn("try:", block)
        self.assertIn("except Exception:", block)
        self.assertIn(
            'logger.debug("Phase 3A retrieval observation (pre-send) failed", exc_info=True)', block
        )

    def test_base_py_persist_step_wrapped_in_try_except(self):
        block = self._non_streaming_persist_block()
        self.assertIn("try:", block)
        self.assertIn("except Exception:", block)
        self.assertIn(
            'logger.debug("[%s] Phase 3A retrieval persistence failed", self.name, exc_info=True)', block
        )


class IndependenceFromLegacyFeedbackTests(_SourceTestCase):
    """The blocker-2 fix itself: v2 persistence must never be gated on, or
    inferable from, the legacy universal-feedback ownership flag."""

    def test_run_py_streaming_gate_does_not_use_universal_feedback_handled(self):
        block = self._streaming_block(code_only=True)
        self.assertNotIn('response.get("universal_feedback_handled")', block)
        self.assertNotIn('response["universal_feedback_handled"]', block)

    def test_run_py_non_streaming_parse_gate_does_not_use_universal_feedback_handled(self):
        block = self._non_streaming_parse_block(code_only=True)
        self.assertNotIn('agent_result.get("universal_feedback_handled")', block)

    def test_base_py_persist_gate_does_not_use_universal_feedback_handled(self):
        block = self._non_streaming_persist_block(code_only=True)
        self.assertNotIn("universal_feedback_handled", block)

    def test_run_py_streaming_gate_uses_attempted_flag_not_persisted(self):
        # The GATE (whether to attempt at all) must check "attempted", not
        # "persisted" -- a failed attempt still counts as attempted, and
        # must not be retried.
        block = self._streaming_block()
        self.assertIn('response.get("phase3a_turn_persistence_attempted")', block)
        self.assertIn('response["phase3a_turn_persistence_attempted"] = True', block)

    def test_run_py_streaming_persisted_flag_set_from_real_return_value(self):
        # The OUTCOME flag must be assigned from observe_and_persist_turn's
        # actual boolean return value, never hardcoded to True regardless
        # of success -- that would conflate "attempted" with "succeeded".
        block = self._streaming_block(code_only=True)
        self.assertIn('response["phase3a_turn_persisted"] = observe_and_persist_turn(', block)
        self.assertNotIn('response["phase3a_turn_persisted"] = True', block)

    def test_run_py_non_streaming_parse_gate_uses_attempted_flag(self):
        block = self._non_streaming_parse_block()
        self.assertIn('not agent_result.get("phase3a_turn_persistence_attempted")', block)
        # Must not gate on the outcome flag -- that would incorrectly
        # allow a retry after a failed (but attempted) streaming write.
        self.assertNotIn('agent_result.get("phase3a_turn_persisted")', block)

    def test_run_py_streaming_gate_uses_delivery_confirmation_not_ownership_claim(self):
        # Gated on already_sent (a pure delivery-confirmation signal, never
        # set for a failed response) -- never on claim_feedback_ownership()
        # or feedback_eligible() being the deciding condition for whether
        # to attempt persistence at all.
        block = self._streaming_block(code_only=True)
        self.assertIn('response.get("already_sent")', block)
        self.assertNotIn("claim_feedback_ownership(", block)

    def test_base_py_persist_gate_uses_delivery_confirmation_not_ownership_claim(self):
        block = self._non_streaming_persist_block(code_only=True)
        self.assertIn("result.success", block)
        self.assertNotIn("claim_feedback_ownership(", block)

    def test_run_py_streaming_block_precedes_legacy_hook(self):
        # Order no longer matters for correctness (the two flows use
        # different flags entirely), but the source is intentionally
        # written with Phase 3A first and the legacy hook second so the
        # independence reads clearly -- this pins that layout.
        phase3a_idx = self.run_py_text.index(_STREAMING_START)
        legacy_idx = self.run_py_text.index(
            "# Universal message-level feedback hook: streaming final content has"
        )
        self.assertLess(phase3a_idx, legacy_idx)

    def test_base_py_persist_block_precedes_legacy_hook(self):
        phase3a_idx = self.base_py_text.index(_NON_STREAMING_PERSIST_START)
        legacy_idx = self.base_py_text.index("# Universal feedback is owned by this normal-send path")
        self.assertLess(phase3a_idx, legacy_idx)


class BoundaryFieldProducerConsumerTests(_SourceTestCase):
    """Item 1: phase3a_boundary_trusted must be set explicitly at every
    response-dict construction site in run.py's _run_agent_inner (the
    producers), and read only via resolve_boundary_trust (the consumer) --
    never reconstructed from compacted_in_place/session_id comparisons in
    the glue code itself."""

    def test_run_py_sets_boundary_trusted_at_all_three_producer_sites(self):
        # The two normal-completion return points (computed from the real
        # _session_was_split/_compacted_in_place local state) and the
        # inactivity-timeout synthetic dict (explicitly False -- the run
        # was interrupted mid-flight, so nothing is known for certain).
        self.assertEqual(
            self.run_py_text.count('"phase3a_boundary_trusted": not (_session_was_split or _compacted_in_place)'),
            2,
        )
        self.assertEqual(
            self.run_py_text.count('"phase3a_boundary_trusted": False,'),
            1,
        )

    def test_run_py_glue_does_not_reconstruct_trust_from_compacted_in_place_or_session_id(self):
        # The glue blocks themselves must not re-derive boundary trust by
        # inspecting compacted_in_place or comparing session ids -- that
        # reconstruction is exactly what caused the None != real-id false
        # positive this revision fixes. Trust must come from the one
        # explicit field only.
        for block in (self._streaming_block(code_only=True), self._non_streaming_parse_block(code_only=True)):
            self.assertNotIn("compacted_in_place", block)
            self.assertNotIn("response_session_id", block)
            self.assertNotIn("pre_run_session_id", block)


class MetadataConsistencyTests(_SourceTestCase):
    def test_run_py_stashes_context_on_event_metadata(self):
        self.assertIn('event.metadata["phase3a_turn_context"]', self.run_py_text)

    def test_base_py_reads_same_metadata_key(self):
        self.assertIn('event.metadata.get("phase3a_turn_context")', self.base_py_text)


class TurnIdReuseTests(_SourceTestCase):
    def test_run_py_reuses_turn_key_helper_not_a_second_id_scheme(self):
        self.assertIn("from tools.universal_feedback import turn_key as _phase3a_turn_key", self.run_py_text)
        # Reused twice: once in the streaming block, once in the
        # non-streaming parse step.
        self.assertEqual(self.run_py_text.count("_phase3a_turn_key("), 2)


class CoreOverlayFootprintTests(_SourceTestCase):
    """Sanity bound on how much Phase 3A actually added to the two core
    overlay files -- the real byte-accurate diff stat is reported
    separately via `git diff --stat`, this just guards against a future
    edit silently ballooning the glue code back into a large inline
    implementation."""

    def test_run_py_phase3a_blocks_are_short_glue(self):
        self.assertLess(self._streaming_block().count("\n"), 75)
        self.assertLess(self._non_streaming_parse_block().count("\n"), 45)

    def test_base_py_phase3a_block_is_short_glue(self):
        self.assertLess(self._non_streaming_persist_block().count("\n"), 55)


if __name__ == "__main__":
    unittest.main()
