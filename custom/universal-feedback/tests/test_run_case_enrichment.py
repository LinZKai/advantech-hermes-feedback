"""Phase 4.5 Stage C/D2 tests: the Case Enrichment batch runner in
tools.run_case_enrichment.

Covers five surfaces:
  1. analyze_case() -- the Stage C stub analyzer, pure function, no DB.
  2. _process_case() / run() with persist=False (the library-level default)
     -- the pipeline, real execution against a temporary FeedbackStoreV2
     (matching test_case_analysis.py's fixture style), never touching
     case_analysis.
  3. _process_case() / run() with persist=True (Stage D2) -- real
     persistence, metadata mapping, failure isolation.
  4. Watermark lifecycle across two analysis runs (Stage D2's core value):
     first analysis -> Case no longer needs analysis -> new evidence ->
     Case needs analysis again -> second analysis appends a second row.
  5. main() -- the CLI entry point, argv in / stdout out, default-persists/
     --dry-run semantics.

No LLM, no network anywhere in this file -- every test injects a fake,
deterministic analyzer (Stage D1's real agent.auxiliary_client-backed
analyze_case_with_llm was already verified against a real Hermes sandbox in
Stage D1.5; this file never imports agent.auxiliary_client or
hermes_cli.config).
"""
from __future__ import annotations

import contextlib
import gc
import io
import sqlite3
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import (  # noqa: E402
    CaseAnalysisEvidence,
    CaseEnrichmentResult,
)
from tools.case_enrichment_analyzer import ANALYSIS_VERSION  # noqa: E402
from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput  # noqa: E402
from tools.run_case_enrichment import (  # noqa: E402
    _CaseOutcome,
    _format_summary,
    _process_case,
    analyze_case,
    main,
    run,
)


def _fake_llm_analyzer(case_input):
    """A deterministic stand-in for analyze_case_with_llm -- never touches
    the network, never reads case_input.turns[*].question_text/answer_text
    into its output (matching the real analyzer's own no-leak evidence
    contract), but returns a real, non-stub-looking, fully valid
    CaseEnrichmentResult so persistence/mapping tests exercise the actual
    field set create_case_analysis() expects."""
    first_turn = case_input.turns[0]
    return CaseEnrichmentResult(
        case_title=f"Case {case_input.case_id} needs SNMP guidance",
        issue_summary="User asked how to configure a product feature.",
        issue_type="product_usage_or_application",
        issue_type_confidence=0.75,
        diagnosis="knowledge_gap",
        diagnosis_confidence=0.6,
        product_model="ADAM-6266",
        product_source="explicit_user_text",
        product_confidence=0.9,
        evidence=(
            CaseAnalysisEvidence(
                type="user_text", turn_id=first_turn.turn_id,
                fact=f"User asked a configuration question in Turn {first_turn.turn_id}.",
            ),
        ),
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
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM case_analysis").fetchone()[0]

    def _case_analysis_rows(self, case_id: str) -> list[sqlite3.Row]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return list(conn.execute(
                "SELECT * FROM case_analysis WHERE case_id=? ORDER BY analyzed_at ASC",
                (case_id,),
            ).fetchall())


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
# 3. Persistence (Stage D2) -- _process_case()/run() with persist=True
# ---------------------------------------------------------------------------


class PersistCaseTests(_StoreTestCase):
    def test_successful_analyzer_creates_exactly_one_row(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        outcome = _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.persisted)
        self.assertEqual(self._case_analysis_row_count(), 1)

    def test_persisted_values_match_analyzer_result(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        row = self.store.get_latest_case_analysis("case-1")
        self.assertEqual(row["case_title"], "Case case-1 needs SNMP guidance")
        self.assertEqual(row["issue_summary"], "User asked how to configure a product feature.")
        self.assertEqual(row["issue_type"], "product_usage_or_application")
        self.assertEqual(row["issue_type_confidence"], 0.75)
        self.assertEqual(row["diagnosis"], "knowledge_gap")
        self.assertEqual(row["diagnosis_confidence"], 0.6)
        self.assertEqual(row["product_model"], "ADAM-6266")
        self.assertEqual(row["product_source"], "explicit_user_text")
        self.assertEqual(row["product_confidence"], 0.9)
        self.assertIn('"type": "user_text"', row["evidence_json"])
        self.assertIn('"turn_id": "turn-1"', row["evidence_json"])

    def test_analysis_version_matches_constant(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        row = self.store.get_latest_case_analysis("case-1")
        self.assertEqual(row["analysis_version"], ANALYSIS_VERSION)
        self.assertEqual(row["analysis_version"], "case-enrichment-v1")

    def test_source_watermark_uses_case_input_watermark_not_stale_outer_value(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        real_watermark = self.store.get_case_evidence_watermark("case-1")
        # Deliberately pass a WRONG/stale outer watermark (as if new evidence
        # arrived after list_cases_needing_analysis() read it but before this
        # Case was built) -- persistence must use the fresh value computed
        # inside build_case_enrichment_input(), never this stale parameter.
        _process_case(
            self.store, "case-1", "1999-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        row = self.store.get_latest_case_analysis("case-1")
        self.assertEqual(row["source_evidence_watermark"], real_watermark)
        self.assertNotEqual(row["source_evidence_watermark"], "1999-01-01T00:00:00+00:00")

    def test_analyzed_at_is_a_recent_utc_iso_timestamp(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        before = datetime.now(timezone.utc)
        _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        after = datetime.now(timezone.utc)
        row = self.store.get_latest_case_analysis("case-1")
        parsed = datetime.fromisoformat(row["analyzed_at"])
        self.assertLessEqual(before, parsed)
        self.assertLessEqual(parsed, after)

    def test_analysis_id_nonempty_and_unique_across_two_runs(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        outcome1 = _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        self._add_turn("case-1", "turn-2")  # new evidence so the second insert is not a duplicate analyzed_at pair by accident
        outcome2 = _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_fake_llm_analyzer, persist=True,
        )
        self.assertTrue(outcome1.analysis_id)
        self.assertTrue(outcome2.analysis_id)
        self.assertNotEqual(outcome1.analysis_id, outcome2.analysis_id)

    def test_build_failure_persists_nothing(self):
        outcome = _process_case(
            self.store, "does-not-exist", None,
            analyzer=_fake_llm_analyzer, persist=True,
        )
        self.assertEqual(outcome.status, "build_failed")
        self.assertEqual(self._case_analysis_row_count(), 0)

    def test_analyzer_failure_persists_nothing(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        outcome = _process_case(
            self.store, "case-1", "2026-01-01T00:00:00+00:00",
            analyzer=_always_fails_analyzer, persist=True,
        )
        self.assertEqual(outcome.status, "analyze_failed")
        self.assertEqual(self._case_analysis_row_count(), 0)

    def test_persist_failure_reported_as_persist_failed_without_raising(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        with mock.patch.object(self.store, "create_case_analysis", return_value=False):
            outcome = _process_case(
                self.store, "case-1", "2026-01-01T00:00:00+00:00",
                analyzer=_fake_llm_analyzer, persist=True,
            )
        self.assertEqual(outcome.status, "persist_failed")
        self.assertIsNotNone(outcome.error)
        self.assertEqual(self._case_analysis_row_count(), 0)

    def test_one_case_persist_failure_does_not_stop_the_batch(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")

        real_create = self.store.create_case_analysis

        def _flaky_create(analysis_id, case_id, **kwargs):
            if case_id == "case-a":
                return False
            return real_create(analysis_id, case_id, **kwargs)

        with mock.patch.object(self.store, "create_case_analysis", side_effect=_flaky_create):
            outcomes = run(self.store, analyzer=_fake_llm_analyzer, persist=True)

        by_id = {o.case_id: o for o in outcomes}
        self.assertEqual(by_id["case-a"].status, "persist_failed")
        self.assertEqual(by_id["case-b"].status, "succeeded")
        self.assertTrue(by_id["case-b"].persisted)
        self.assertEqual(self._case_analysis_row_count(), 1)

    def test_dry_run_analyzer_runs_but_persists_nothing(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        outcomes = run(self.store, analyzer=_fake_llm_analyzer, persist=False)
        self.assertEqual(outcomes[0].status, "succeeded")
        self.assertFalse(outcomes[0].persisted)
        self.assertEqual(self._case_analysis_row_count(), 0)

    def test_history_rows_append_only_never_updated(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        first_rows_before = self._case_analysis_rows("case-1")
        self.assertEqual(len(first_rows_before), 1)

        self._add_turn("case-1", "turn-2")
        run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        rows_after = self._case_analysis_rows("case-1")

        self.assertEqual(len(rows_after), 2)
        # The first row is byte-for-byte unchanged -- no UPDATE happened.
        self.assertEqual(dict(rows_after[0]), dict(first_rows_before[0]))
        self.assertNotEqual(rows_after[0]["analysis_id"], rows_after[1]["analysis_id"])
        self.assertNotEqual(rows_after[0]["analyzed_at"], rows_after[1]["analyzed_at"])


# ---------------------------------------------------------------------------
# 4. Watermark lifecycle (Stage D2's core value)
# ---------------------------------------------------------------------------


class WatermarkLifecycleTests(_StoreTestCase):
    def test_first_analysis_then_case_no_longer_needs_analysis(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        self.assertEqual(
            [r["case_id"] for r in self.store.list_cases_needing_analysis()], ["case-1"],
        )

        outcomes = run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        self.assertEqual(outcomes[0].status, "succeeded")
        self.assertEqual(self._case_analysis_row_count(), 1)

        self.assertEqual(self.store.list_cases_needing_analysis(), [])

    def test_new_turn_makes_case_stale_again_and_reanalysis_appends_second_row(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        self.assertEqual(self.store.list_cases_needing_analysis(), [])

        self._add_turn("case-1", "turn-2")  # real, legitimate new evidence -- no raw SQL
        self.assertEqual(
            [r["case_id"] for r in self.store.list_cases_needing_analysis()], ["case-1"],
        )

        run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        rows = self._case_analysis_rows("case-1")
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[1]["source_evidence_watermark"], rows[0]["source_evidence_watermark"])
        self.assertEqual(self.store.list_cases_needing_analysis(), [])

    def test_new_feedback_also_makes_case_stale_again(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        self.assertEqual(self.store.list_cases_needing_analysis(), [])

        self._add_feedback("turn-1", helpful=False)
        self.assertEqual(
            [r["case_id"] for r in self.store.list_cases_needing_analysis()], ["case-1"],
        )

        run(self.store, analyzer=_fake_llm_analyzer, persist=True)
        self.assertEqual(len(self._case_analysis_rows("case-1")), 2)
        self.assertEqual(self.store.list_cases_needing_analysis(), [])


# ---------------------------------------------------------------------------
# 5. main() -- CLI entry point
# ---------------------------------------------------------------------------


class MainCliTests(_StoreTestCase):
    def _run_main(self, *args, analyzer=_fake_llm_analyzer):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(["--db", str(self.db_path), *args], analyzer=analyzer)
        return exit_code, buf_out.getvalue(), buf_err.getvalue()

    def test_exit_code_zero_on_success(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        exit_code, _, _ = self._run_main("--dry-run")
        self.assertEqual(exit_code, 0)

    def test_default_persists_case_analysis_row(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        exit_code, output, _ = self._run_main()  # no --dry-run: default persists
        self.assertEqual(exit_code, 0)
        self.assertEqual(self._case_analysis_row_count(), 1)
        self.assertIn("persisted 1 case_analysis row(s).", output)
        self.assertNotIn("DRY RUN", output)

    def test_dry_run_flag_skips_persistence_and_marks_output(self):
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        _, output, _ = self._run_main("--dry-run")
        self.assertEqual(self._case_analysis_row_count(), 0)
        self.assertIn("DRY RUN ONLY", output)
        self.assertIn("no case_analysis rows were created", output)

    def test_summary_counts_correct(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        _, output, _ = self._run_main("--dry-run")
        self.assertIn("Cases needing analysis: 2", output)
        self.assertIn("processed=2 succeeded=2 failed=0", output)

    def test_limit_flag(self):
        self._seed_session_and_case(case_id="case-a")
        self.store.create_case("case-b", "sess-1")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        _, output, _ = self._run_main("--dry-run", "--limit", "1")
        self.assertIn("Cases needing analysis: 1", output)

    def test_session_id_flag(self):
        self.store.create_or_update_session("sess-a", "telegram")
        self.store.create_or_update_session("sess-b", "telegram")
        self.store.create_case("case-a", "sess-a")
        self.store.create_case("case-b", "sess-b")
        self._add_turn("case-a", "turn-a")
        self._add_turn("case-b", "turn-b")
        _, output, _ = self._run_main("--dry-run", "--session-id", "sess-a")
        self.assertIn("case_id=case-a", output)
        self.assertNotIn("case_id=case-b", output)

    def test_no_full_question_or_answer_text_printed(self):
        # Uses a product string distinct from _fake_llm_analyzer's own fixed
        # product_model ("ADAM-6266") so a legitimate, intentional
        # product_model=... summary line can never be mistaken for a leak of
        # this turn's actual question/answer text.
        self._seed_session_and_case()
        self._add_turn(
            "case-1", "turn-1",
            question="WISE-6610 要怎麼設定 IP 位址？機器一直重開機。",
            answer="請進入網頁管理介面設定 Static IP，並檢查電源供應是否穩定。",
        )
        _, output, _ = self._run_main()
        for leaked in ("WISE-6610", "IP 位址", "重開機", "網頁管理介面", "Static IP", "電源供應"):
            self.assertNotIn(leaked, output)

    def test_no_cases_needing_analysis_output(self):
        exit_code, output, _ = self._run_main("--dry-run")
        self.assertEqual(exit_code, 0)
        self.assertIn("Cases needing analysis: 0", output)
        self.assertIn("processed=0 succeeded=0 failed=0", output)

    def test_without_analyzer_override_and_without_hermes_config_fails_closed(self):
        # No analyzer= override, no --dry-run: main() must attempt to
        # resolve the real analyzer via _resolve_main_runtime(), which
        # cannot import hermes_cli.config in this repo's test environment
        # -- proving the fail-closed contract through the real CLI path
        # without needing a real Hermes sandbox to do it.
        self._seed_session_and_case()
        self._add_turn("case-1", "turn-1")
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(["--db", str(self.db_path)])
        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR", buf_err.getvalue())
        self.assertEqual(self._case_analysis_row_count(), 0)


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
