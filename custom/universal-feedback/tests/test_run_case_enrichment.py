"""Phase 4.5 Stage C tests: the manual Case Enrichment dry-run runner in
tools.run_case_enrichment.

Covers three surfaces:
  1. analyze_case() -- the stub analyzer, pure function, no DB.
  2. _process_case() / run() -- the pipeline, real execution against a
     temporary FeedbackStoreV2 (matching test_case_analysis.py's fixture
     style).
  3. main() -- the CLI entry point, argv in / stdout out.

No LLM, no network (there is none to mock this stage), and no call to
FeedbackStoreV2.create_case_analysis() anywhere -- Stage C is dry-run only.
Every test that runs the pipeline against real Cases also asserts
case_analysis stays empty afterward.
"""
from __future__ import annotations

import contextlib
import gc
import io
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import CaseEnrichmentResult  # noqa: E402
from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput  # noqa: E402
from tools.run_case_enrichment import (  # noqa: E402
    _CaseOutcome,
    _format_summary,
    _process_case,
    analyze_case,
    main,
    run,
)


# ---------------------------------------------------------------------------
# Fixture base: real on-disk SQLite through the actual store, matching the
# _StoreTestCase convention already used across this test suite.
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

    def _seed_session_and_case(self, session_id="sess-1", case_id="case-1"):
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        return session_id, case_id

    def _add_turn(self, case_id, turn_id, *, question="SNMP 怎麼關閉？", answer="請至設定頁停用 SNMP。"):
        self.store.create_turn(
            turn_id, case_id,
            platform_user_id="user-1",
            platform_user_message_id=turn_id,
            question_text=question, answer_text=answer,
            feedback_eligible=True,
        )
        return turn_id

    def _add_retrieval_run(self, turn_id, *, result_count=3):
        ok = self.store.add_retrieval_runs(turn_id, [
            RetrievalRunInput(
                execution_status="completed", observation_status="complete",
                result_count=result_count, foundry_iq_ok=True,
            ),
        ])
        assert ok

    def _add_feedback(self, turn_id, *, helpful=True):
        feedback_id = uuid.uuid4().hex
        ok = self.store.submit_feedback(
            feedback_id, turn_id, helpful, feedback_policy_version="v1",
            reason_code=None if helpful else "incomplete",
        )
        assert ok

    def _case_analysis_row_count(self) -> int:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM case_analysis").fetchone()[0]


# ---------------------------------------------------------------------------
# 1. analyze_case() -- the stub analyzer
# ---------------------------------------------------------------------------


class AnalyzeCaseStubTests(_StoreTestCase):
    def test_returns_valid_case_enrichment_result(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        from tools.case_enrichment import build_case_enrichment_input
        case_input = build_case_enrichment_input(self.store, "case-1")
        result = analyze_case(case_input)
        self.assertIsInstance(result, CaseEnrichmentResult)

    def test_deterministic_conservative_fallback_values(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        from tools.case_enrichment import build_case_enrichment_input
        case_input = build_case_enrichment_input(self.store, "case-1")
        result = analyze_case(case_input)
        self.assertEqual(result.issue_type, "other_or_unclear")
        self.assertEqual(result.issue_type_confidence, 0.0)
        self.assertEqual(result.diagnosis, "no_issue_detected")
        self.assertEqual(result.diagnosis_confidence, 0.0)
        self.assertIsNone(result.product_model)
        self.assertIsNone(result.product_source)
        self.assertIsNone(result.product_confidence)

    def test_same_input_always_produces_same_classification(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        from tools.case_enrichment import build_case_enrichment_input
        case_input = build_case_enrichment_input(self.store, "case-1")
        r1 = analyze_case(case_input)
        r2 = analyze_case(case_input)
        self.assertEqual(r1.issue_type, r2.issue_type)
        self.assertEqual(r1.diagnosis, r2.diagnosis)

    def test_evidence_references_real_first_turn(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_turn("case-1", "turn-2")
        from tools.case_enrichment import build_case_enrichment_input
        case_input = build_case_enrichment_input(self.store, "case-1")
        result = analyze_case(case_input)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].type, "user_text")
        self.assertEqual(result.evidence[0].turn_id, "turn-1")  # the first turn, by created_at order

    def test_evidence_fact_does_not_contain_real_question_or_answer_text(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1", question="ADAM-6266 要怎麼設定 IP？", answer="請進入網頁設定介面。")
        from tools.case_enrichment import build_case_enrichment_input
        case_input = build_case_enrichment_input(self.store, "case-1")
        result = analyze_case(case_input)
        fact = result.evidence[0].fact
        self.assertNotIn("ADAM-6266", fact)
        self.assertNotIn("IP", fact)
        self.assertNotIn("網頁設定介面", fact)

    def test_works_with_multi_turn_case(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_turn("case-1", "turn-2")
        self._add_turn("case-1", "turn-3")
        from tools.case_enrichment import build_case_enrichment_input
        case_input = build_case_enrichment_input(self.store, "case-1")
        result = analyze_case(case_input)  # must not raise
        self.assertIsInstance(result, CaseEnrichmentResult)


# ---------------------------------------------------------------------------
# 2. _process_case() / run() -- the pipeline
# ---------------------------------------------------------------------------


def _always_fails_analyzer(case_input):
    raise RuntimeError("simulated analyzer failure")


class ProcessCaseTests(_StoreTestCase):
    def test_succeeds_for_valid_case(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._add_retrieval_run("turn-1")
        self._add_feedback("turn-1", helpful=False)

        outcome = _process_case(self.store, "case-1", "2026-01-01T00:00:00+00:00")
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(outcome.turns_count, 1)
        self.assertEqual(outcome.retrievals_count, 1)
        self.assertEqual(outcome.feedback_count, 1)
        self.assertEqual(outcome.issue_type, "other_or_unclear")
        self.assertEqual(outcome.diagnosis, "no_issue_detected")
        self.assertIsNone(outcome.error)

    def test_build_failed_for_unknown_case(self):
        outcome = _process_case(self.store, "does-not-exist", None)
        self.assertEqual(outcome.status, "build_failed")
        self.assertIsNotNone(outcome.error)
        self.assertIsNone(outcome.turns_count)

    def test_build_failed_for_zero_turn_case(self):
        self._seed_session_and_case()
        outcome = _process_case(self.store, "case-1", None)
        self.assertEqual(outcome.status, "build_failed")

    def test_analyze_failed_does_not_raise_and_still_reports_counts(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        outcome = _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_always_fails_analyzer,
        )
        self.assertEqual(outcome.status, "analyze_failed")
        self.assertEqual(outcome.turns_count, 1)  # input was built successfully before the analyzer failed
        self.assertIn("simulated analyzer failure", outcome.error)

    def test_never_creates_case_analysis_row(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        _process_case(self.store, "case-1", "2026-01-01T00:00:00+00:00")
        self.assertEqual(self._case_analysis_row_count(), 0)


class RunPipelineTests(_StoreTestCase):
    def test_no_cases_needing_analysis(self):
        outcomes = run(self.store)
        self.assertEqual(outcomes, [])

    def test_one_case_processed(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        outcomes = run(self.store)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].case_id, "case-1")
        self.assertEqual(outcomes[0].status, "succeeded")

    def test_multiple_cases_processed(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self.store.create_case("case-c", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        self._add_turn("case-c", "turn-c")
        outcomes = run(self.store)
        self.assertEqual({o.case_id for o in outcomes}, {"case-a", "case-b", "case-c"})
        self.assertTrue(all(o.status == "succeeded" for o in outcomes))

    def test_one_failing_case_does_not_stop_the_batch(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")

        def _analyzer(case_input):
            if case_input.case_id == "case-a":
                raise RuntimeError("boom")
            return analyze_case(case_input)

        outcomes = run(self.store, analyzer=_analyzer)
        by_id = {o.case_id: o for o in outcomes}
        self.assertEqual(by_id["case-a"].status, "analyze_failed")
        self.assertEqual(by_id["case-b"].status, "succeeded")

    def test_limit_respected(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        outcomes = run(self.store, limit=1)
        self.assertEqual(len(outcomes), 1)

    def test_session_id_filter_respected(self):
        self.store.create_or_update_session("sess-a", "telegram")
        self.store.create_or_update_session("sess-b", "telegram")
        self.store.create_case("case-a", "sess-a")
        self.store.create_case("case-b", "sess-b")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        outcomes = run(self.store, session_id="sess-a")
        self.assertEqual([o.case_id for o in outcomes], ["case-a"])

    def test_run_never_creates_case_analysis_rows(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        run(self.store)
        self.assertEqual(self._case_analysis_row_count(), 0)


# ---------------------------------------------------------------------------
# 3. main() -- CLI entry point
# ---------------------------------------------------------------------------


class MainCliTests(_StoreTestCase):
    def _run_main(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = main(["--db", str(self.db_path), *args])
        return exit_code, buf.getvalue()

    def test_exit_code_zero_on_success(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        exit_code, _ = self._run_main()
        self.assertEqual(exit_code, 0)

    def test_summary_counts_correct(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        _, output = self._run_main()
        self.assertIn("Cases needing analysis: 2", output)
        self.assertIn("processed=2 succeeded=2 failed=0", output)

    def test_dry_run_marker_present(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        _, output = self._run_main()
        self.assertIn("DRY RUN ONLY", output)
        self.assertIn("no case_analysis rows were created", output)

    def test_limit_flag(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        _, output = self._run_main("--limit", "1")
        self.assertIn("Cases needing analysis: 1", output)

    def test_session_id_flag(self):
        self.store.create_or_update_session("sess-a", "telegram")
        self.store.create_or_update_session("sess-b", "telegram")
        self.store.create_case("case-a", "sess-a")
        self.store.create_case("case-b", "sess-b")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        _, output = self._run_main("--session-id", "sess-a")
        self.assertIn("case_id=case-a", output)
        self.assertNotIn("case_id=case-b", output)

    def test_no_full_question_or_answer_text_printed(self):
        self._seed_session_and_case()
        self._add_turn(
            "case-1", "turn-1",
            question="ADAM-6266 要怎麼設定 IP 位址？機器一直重開機。",
            answer="請進入網頁管理介面設定 Static IP，並檢查電源供應是否穩定。",
        )
        _, output = self._run_main()
        for leaked in ("ADAM-6266", "IP 位址", "重開機", "網頁管理介面", "Static IP", "電源供應"):
            self.assertNotIn(leaked, output)

    def test_never_creates_case_analysis_row(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self._run_main()
        self.assertEqual(self._case_analysis_row_count(), 0)

    def test_no_cases_needing_analysis_output(self):
        exit_code, output = self._run_main()
        self.assertEqual(exit_code, 0)
        self.assertIn("Cases needing analysis: 0", output)
        self.assertIn("processed=0 succeeded=0 failed=0", output)


class FormatSummaryTests(unittest.TestCase):
    """_format_summary() in isolation -- no DB, pure formatting."""

    def test_failed_case_shows_status_and_error_not_counts(self):
        outcomes = [
            _CaseOutcome(case_id="case-x", evidence_watermark=None, status="build_failed", error="boom"),
        ]
        text = _format_summary(outcomes)
        self.assertIn("case_id=case-x", text)
        self.assertIn("status=build_failed", text)
        self.assertIn("error=boom", text)

    def test_mixed_success_and_failure_counts(self):
        outcomes = [
            _CaseOutcome(
                case_id="ok", evidence_watermark="t", status="succeeded",
                turns_count=1, retrievals_count=0, feedback_count=0,
                issue_type="other_or_unclear", diagnosis="no_issue_detected", product_model=None,
            ),
            _CaseOutcome(case_id="bad", evidence_watermark="t", status="analyze_failed", error="x"),
        ]
        text = _format_summary(outcomes)
        self.assertIn("processed=2 succeeded=1 failed=1", text)


if __name__ == "__main__":
    unittest.main()
