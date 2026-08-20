"""Curator Slice 1 tests: the Curator runner in tools.run_curator.

No LLM, no network anywhere in this file -- every test injects a fake
`analyzer` callable (matching tools.run_curator.Analyzer's own reduced
signature: `(context, *, proposal_id, created_at) -> CuratorChange`, never
`agent.auxiliary_client`). Uses a real temp-file SQLite DB (matching
test_run_reflector.py's own `_StoreTestCase` style), because genuinely
exercising the deterministic guards and persistence outcomes needs a real
store -- not a mock.

Covers exactly the deterministic-guard contract this slice's task
instruction calls for: pending/rejected/wrong-target Proposals never reach
the analyzer; a missing AGENTS.md fails closed; a malformed/invalid-
target_file analyzer response fails closed without persisting; a valid
Proposal persists a proposed CuratorChange; no_change_recommended persists
and reports its own distinct status.
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.curator_analyzer import CuratorAnalyzerError  # noqa: E402
from tools.curator_domain import CURATOR_TARGET_FILE, CuratorChange  # noqa: E402
from tools.curator_prompt_context import CuratorPromptContext  # noqa: E402
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.run_curator import CuratorRunOutcome, main, run_curator  # noqa: E402

_DEFAULT_AGENTS_CONTENT = "# Advantech Technical Support Instructions\n\n## Response Structure\n\nPut the direct answer first.\n"


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)
        self.agents_file = Path(self._tmpdir.name) / "AGENTS.md"
        self.agents_file.write_text(_DEFAULT_AGENTS_CONTENT, encoding="utf-8")

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_proposal(
        self, proposal_id: str = "proposal-1", *,
        review_status: str = "accepted",
        improvement_target: str = "agent_behavior",
        title: str = "技術支援回答應優先呈現直接結論、必要條件與可執行步驟",
        with_observation: bool = True,
        supporting_case_ids: tuple[str, ...] = (),
    ) -> str:
        self.store.create_improvement_proposal(
            proposal_id, improvement_target=improvement_target, title=title,
            created_at="2026-01-01T00:00:00+00:00",
        )
        if review_status != "pending":
            ok = self.store.update_proposal_review_status(proposal_id, review_status)
            assert ok
        if with_observation:
            self.store.create_reflection_run(
                "run-seed", started_at="2025-12-01T00:00:00+00:00", analyzed_case_count=1,
                reflector_version="reflector-v0",
            )
            self.store.create_proposal_observation(
                f"obs-seed-{proposal_id}", proposal_id, "run-seed",
                trend="new", pattern_summary="多筆案例顯示回答過於冗長。", possible_cause=None,
                recommended_improvement="調整回答結構。", expected_benefit=None, limitations=None,
                supporting_case_ids_json=json.dumps(list(supporting_case_ids)),
                supporting_case_count=len(supporting_case_ids),
                confidence=0.7, observed_at="2025-12-01T00:00:00+00:00",
            )
        return proposal_id


# ---------------------------------------------------------------------------
# Fake analyzer -- records every call, returns a scripted CuratorChange (or
# raises a scripted exception). Never touches agent.auxiliary_client.
# ---------------------------------------------------------------------------


class _FakeAnalyzer:
    def __init__(
        self, *,
        exc: BaseException | None = None,
        change_type: str = "modify_rule",
        target_file: str = CURATOR_TARGET_FILE,
        proposed_content: str | None = "# Advantech Technical Support Instructions\n\nBe direct.\n",
        confidence: float = 0.82,
    ) -> None:
        self._exc = exc
        self._change_type = change_type
        self._target_file = target_file
        self._proposed_content = proposed_content
        self._confidence = confidence
        self.calls: list[dict[str, Any]] = []

    def __call__(self, context: CuratorPromptContext, *, proposal_id: str, created_at: str) -> CuratorChange:
        self.calls.append({"context": context, "proposal_id": proposal_id, "created_at": created_at})
        if self._exc is not None:
            raise self._exc
        return CuratorChange(
            change_id=uuid.uuid4().hex,
            proposal_id=proposal_id,
            target_file=self._target_file,
            change_type=self._change_type,
            rationale="多筆案例顯示回答過於冗長。",
            before_content=context.current_content,
            proposed_content=self._proposed_content,
            expected_effect="回答更精簡。" if self._change_type != "no_change_recommended" else None,
            confidence=self._confidence,
            status="proposed",
            created_at=created_at,
        )


# ---------------------------------------------------------------------------
# A. Deterministic guards -- BEFORE the analyzer
# ---------------------------------------------------------------------------


class ProposalNotFoundTests(_StoreTestCase):
    def test_unknown_proposal_id_fails_without_calling_analyzer(self):
        analyzer = _FakeAnalyzer()
        outcome = run_curator(
            self.store, proposal_id="does-not-exist", agents_file=self.agents_file, analyzer=analyzer,
        )
        self.assertEqual(outcome.status, "proposal_not_found")
        self.assertEqual(len(analyzer.calls), 0)


class NotAcceptedTests(_StoreTestCase):
    def test_pending_proposal_is_refused(self):
        proposal_id = self._seed_proposal(review_status="pending")
        analyzer = _FakeAnalyzer()
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "not_accepted")
        self.assertEqual(len(analyzer.calls), 0)

    def test_rejected_proposal_is_refused(self):
        proposal_id = self._seed_proposal(review_status="rejected")
        analyzer = _FakeAnalyzer()
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "not_accepted")
        self.assertEqual(len(analyzer.calls), 0)


class WrongTargetTests(_StoreTestCase):
    def test_knowledge_target_proposal_is_refused(self):
        proposal_id = self._seed_proposal(improvement_target="knowledge")
        analyzer = _FakeAnalyzer()
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "wrong_target")
        self.assertEqual(len(analyzer.calls), 0)

    def test_retrieval_target_proposal_is_refused(self):
        proposal_id = self._seed_proposal(improvement_target="retrieval")
        analyzer = _FakeAnalyzer()
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "wrong_target")
        self.assertEqual(len(analyzer.calls), 0)


class NoObservationTests(_StoreTestCase):
    def test_accepted_proposal_with_no_observation_is_refused(self):
        proposal_id = self._seed_proposal(with_observation=False)
        analyzer = _FakeAnalyzer()
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "no_observation")
        self.assertEqual(len(analyzer.calls), 0)


class AgentsFileUnreadableTests(_StoreTestCase):
    def test_missing_agents_file_fails_without_calling_analyzer(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer()
        missing_file = Path(self._tmpdir.name) / "does-not-exist" / "AGENTS.md"
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=missing_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "agents_file_unreadable")
        self.assertEqual(len(analyzer.calls), 0)


# ---------------------------------------------------------------------------
# B. Deterministic guards -- AFTER the analyzer, before persisting
# ---------------------------------------------------------------------------


class AnalyzerFailureTests(_StoreTestCase):
    def test_analyzer_error_fails_without_persistence(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer(exc=CuratorAnalyzerError("fake LLM failure"))
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)
        self.assertEqual(outcome.status, "analyzer_failed")
        self.assertIn("fake LLM failure", outcome.error)
        self.assertIsNone(self.store.get_curator_change(outcome.change_id or "no-such-id"))

    def test_wrong_target_file_from_analyzer_fails_without_persistence(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer(target_file="/sandbox/SOUL.md")
        # CuratorChange itself rejects a wrong target_file at construction
        # -- the fake analyzer must raise the same way a real one's
        # parse_curator_output() failure would surface, proving
        # run_curator()'s own guard is reachable even if CuratorChange's
        # own __post_init__ were ever bypassed. Simulate via a raw object
        # duck-typing CuratorChange's public attributes instead of the
        # real dataclass, since the real dataclass cannot be constructed
        # with an invalid target_file at all.
        class _BadChange:
            change_id = "bad-1"
            proposal_id = "proposal-1"
            target_file = "/sandbox/SOUL.md"
            change_type = "modify_rule"
            rationale = "r"
            before_content = "b"
            proposed_content = "p"
            expected_effect = None
            confidence = 0.5
            status = "proposed"
            created_at = "2026-01-01T00:00:00+00:00"

        class _BadAnalyzer:
            def __init__(self):
                self.calls = []

            def __call__(self, context, *, proposal_id, created_at):
                self.calls.append(1)
                return _BadChange()

        bad_analyzer = _BadAnalyzer()
        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=bad_analyzer)
        self.assertEqual(outcome.status, "analyzer_failed")
        self.assertIn("target_file", outcome.error)
        self.assertIsNone(self.store.get_curator_change("bad-1"))


# ---------------------------------------------------------------------------
# C. Happy path -- real change and no_change_recommended
# ---------------------------------------------------------------------------


class HappyPathTests(_StoreTestCase):
    def test_valid_accepted_proposal_persists_proposed_change(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer()

        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)

        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(analyzer.calls), 1)
        self.assertIsNotNone(outcome.change_id)

        row = self.store.get_curator_change(outcome.change_id)
        self.assertEqual(row["status"], "proposed")
        self.assertEqual(row["proposal_id"], proposal_id)
        self.assertEqual(row["target_file"], CURATOR_TARGET_FILE)
        self.assertEqual(row["before_content"], _DEFAULT_AGENTS_CONTENT)

    def test_no_change_recommended_persists_and_reports_distinct_status(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer(change_type="no_change_recommended", proposed_content=None)

        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)

        self.assertEqual(outcome.status, "no_change_recommended")
        self.assertNotEqual(outcome.status, "succeeded")
        row = self.store.get_curator_change(outcome.change_id)
        self.assertEqual(row["change_type"], "no_change_recommended")
        self.assertEqual(row["status"], "proposed")
        self.assertIsNone(row["proposed_content"])

    def test_supporting_case_evidence_reaches_the_analyzer_context(self):
        self.store.create_or_update_session("sess-1", "telegram")
        self.store.create_case("case-1", "sess-1")
        self.store.create_case_analysis(
            "analysis-1", "case-1",
            case_title="title", issue_summary="summary",
            issue_type="product_usage_or_application", issue_type_confidence=0.6,
            diagnosis="answer_quality_issue", diagnosis_confidence=0.6,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2025-12-01T00:00:00+00:00", source_evidence_watermark="2025-12-01T00:00:00+00:00",
        )
        proposal_id = self._seed_proposal(supporting_case_ids=("case-1",))
        analyzer = _FakeAnalyzer()

        outcome = run_curator(self.store, proposal_id=proposal_id, agents_file=self.agents_file, analyzer=analyzer)

        self.assertEqual(outcome.status, "succeeded")
        context = analyzer.calls[0]["context"]
        self.assertEqual([c.case_id for c in context.supporting_cases], ["case-1"])


# ---------------------------------------------------------------------------
# D. main() -- the thin CLI boundary
# ---------------------------------------------------------------------------


class MainCliTests(_StoreTestCase):
    def _run_main(self, *extra_argv: str, analyzer: Any = None) -> tuple[int, str, str]:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        argv = ["--db", str(self.db_path), "--agents-file", str(self.agents_file), *extra_argv]
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(argv, analyzer=analyzer)
        return exit_code, buf_out.getvalue(), buf_err.getvalue()

    def test_without_analyzer_override_and_without_hermes_config_fails_closed(self):
        proposal_id = self._seed_proposal()
        exit_code, _, err = self._run_main("--proposal-id", proposal_id)
        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR", err)
        self.assertEqual(self.store.list_improvement_proposals(review_status="accepted")[0]["proposal_id"], proposal_id)

    def test_cli_wiring_with_injected_analyzer_reaches_stdout_summary(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer()
        exit_code, out, _ = self._run_main("--proposal-id", proposal_id, analyzer=analyzer)
        self.assertEqual(exit_code, 0)
        self.assertIn("status=succeeded", out)
        self.assertIn(f"proposal_id={proposal_id}", out)
        self.assertIn("change_type=modify_rule", out)
        self.assertIn(f"target_file={CURATOR_TARGET_FILE}", out)

    def test_cli_exit_code_zero_on_no_change_recommended(self):
        proposal_id = self._seed_proposal()
        analyzer = _FakeAnalyzer(change_type="no_change_recommended", proposed_content=None)
        exit_code, out, _ = self._run_main("--proposal-id", proposal_id, analyzer=analyzer)
        self.assertEqual(exit_code, 0)
        self.assertIn("status=no_change_recommended", out)

    def test_cli_exit_code_nonzero_on_not_accepted(self):
        proposal_id = self._seed_proposal(review_status="pending")
        analyzer = _FakeAnalyzer()
        exit_code, out, _ = self._run_main("--proposal-id", proposal_id, analyzer=analyzer)
        self.assertEqual(exit_code, 1)
        self.assertIn("status=not_accepted", out)
        self.assertEqual(len(analyzer.calls), 0)


if __name__ == "__main__":
    unittest.main()
