"""Phase 5 Slice 6B tests: the Reflector runner in tools.run_reflector.

No LLM, no network anywhere in this file -- every test injects a fake
`analyzer` callable (matching tools.run_reflector.Analyzer's own reduced
signature: `(context, *, reflection_run_id, observed_at) -> ReflectionResult`,
never `agent.auxiliary_client`). Uses a real temp-file SQLite DB (matching
test_reflector_persistence.py's / test_case_reflection_input.py's own
`_StoreTestCase` style), because genuinely exercising eligibility gating,
Proposal-candidate loading, and persistence outcomes needs a real store --
not a mock.
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
from pathlib import Path
from typing import Any
from unittest import mock

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402
from tools.reflector_analyzer import REFLECTOR_VERSION, ReflectorAnalyzerError  # noqa: E402
from tools.reflector_persistence import ReflectionPersistenceError  # noqa: E402
from tools.reflector_proposals import ImprovementProposal, ProposalObservation, ReflectionResult  # noqa: E402
from tools.run_reflector import ReflectorRunOutcome, main, run_reflector  # noqa: E402


class _StoreTestCase(unittest.TestCase):
    """Matches test_case_reflection_input.py's / test_reflector_
    persistence.py's own `_StoreTestCase` style (real temp-file DB,
    gc.collect() before cleanup for Windows file-lock safety)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _seed_case(self, case_id: str, session_id: str = "sess-1", *, created_at: str | None = None) -> str:
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        if created_at is not None:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("UPDATE cases SET created_at=? WHERE case_id=?", (created_at, case_id))
                conn.commit()
            finally:
                conn.close()
        return case_id

    def _create_analysis(
        self, case_id: str, *, analyzed_at: str,
        issue_type: str = "product_usage_or_application",
        diagnosis: str = "knowledge_gap",
        analysis_id: str | None = None,
    ) -> str:
        analysis_id = analysis_id or uuid.uuid4().hex
        ok = self.store.create_case_analysis(
            analysis_id, case_id,
            case_title="title", issue_summary="summary",
            issue_type=issue_type, issue_type_confidence=0.5,
            diagnosis=diagnosis, diagnosis_confidence=0.5,
            product_model=None, product_source=None, product_confidence=None,
            evidence_json=None, analysis_version="v1",
            analyzed_at=analyzed_at, source_evidence_watermark=analyzed_at,
        )
        assert ok
        return analysis_id

    def _seed_eligible_cases(self, n: int, *, created_at: str = "2026-01-01T00:00:00+00:00") -> list[str]:
        case_ids = [f"case-{i}" for i in range(n)]
        for case_id in case_ids:
            self._seed_case(case_id, created_at=created_at)
            self._create_analysis(case_id, analyzed_at=created_at)
        return case_ids

    def _seed_existing_proposal(
        self, proposal_id: str, *,
        improvement_target: str = "knowledge",
        title: str = "Existing Proposal",
    ) -> str:
        """A pending Proposal with a founding Observation -- the minimum
        shape build_proposal_candidates() requires (every Proposal in
        scope must have at least one latest Observation, or that function
        raises ProposalCandidateBuildError)."""
        self.store.create_reflection_run(
            "run-seed", started_at="2025-12-01T00:00:00+00:00", analyzed_case_count=1,
            reflector_version="reflector-v0",
        )
        self.store.create_improvement_proposal(
            proposal_id, improvement_target=improvement_target, title=title,
            created_at="2025-12-01T00:00:00+00:00",
        )
        self.store.create_proposal_observation(
            f"obs-seed-{proposal_id}", proposal_id, "run-seed",
            trend="new", pattern_summary="Seed pattern.", possible_cause=None,
            recommended_improvement="Seed recommendation.", expected_benefit=None, limitations=None,
            supporting_case_ids_json='["case-seed"]', supporting_case_count=1,
            confidence=0.5, observed_at="2025-12-01T00:00:00+00:00",
        )
        return proposal_id


# ---------------------------------------------------------------------------
# Fake analyzer -- records every call, returns a scripted ReflectionResult
# (or raises a scripted exception). Never touches agent.auxiliary_client.
# ---------------------------------------------------------------------------


class _FakeAnalyzer:
    def __init__(
        self, *,
        exc: BaseException | None = None,
        material_change_detected: bool = True,
        with_new_proposal: bool = True,
        with_match_existing: str | None = None,
    ) -> None:
        self._exc = exc
        self._material_change_detected = material_change_detected
        self._with_new_proposal = with_new_proposal
        self._with_match_existing = with_match_existing
        self.calls: list[dict[str, Any]] = []

    def __call__(self, context: Any, *, reflection_run_id: str, observed_at: str) -> ReflectionResult:
        self.calls.append({
            "context": context, "reflection_run_id": reflection_run_id, "observed_at": observed_at,
        })
        if self._exc is not None:
            raise self._exc

        case_ids = tuple(c.case_id for c in context.cases)
        new_proposals: tuple[ImprovementProposal, ...] = ()
        observations: tuple[ProposalObservation, ...] = ()

        if self._with_new_proposal:
            proposal = ImprovementProposal(
                proposal_id="proposal-fake-new", improvement_target="knowledge",
                title="Fake new proposal", review_status="pending", created_at=observed_at,
            )
            observation = ProposalObservation(
                observation_id=f"obs-{uuid.uuid4().hex}", proposal_id="proposal-fake-new",
                reflection_run_id=reflection_run_id, trend="new",
                pattern_summary="Fake pattern summary.", possible_cause=None,
                recommended_improvement="Fake recommendation.", expected_benefit=None, limitations=None,
                supporting_case_ids=case_ids, supporting_case_count=len(case_ids),
                confidence=0.7, observed_at=observed_at,
            )
            new_proposals = (proposal,)
            observations = observations + (observation,)

        if self._with_match_existing is not None:
            observation = ProposalObservation(
                observation_id=f"obs-{uuid.uuid4().hex}", proposal_id=self._with_match_existing,
                reflection_run_id=reflection_run_id, trend="stable",
                pattern_summary="Fake routine re-observation.", possible_cause=None,
                recommended_improvement="Fake recommendation.", expected_benefit=None, limitations=None,
                supporting_case_ids=case_ids, supporting_case_count=len(case_ids),
                confidence=0.6, observed_at=observed_at,
            )
            observations = observations + (observation,)

        return ReflectionResult(
            reflection_run_id=reflection_run_id,
            run_summary="Fake run summary.",
            material_change_detected=self._material_change_detected,
            new_proposals=new_proposals,
            proposal_observations=observations,
        )


# ---------------------------------------------------------------------------
# A. Happy path
# ---------------------------------------------------------------------------


class HappyPathTests(_StoreTestCase):
    def test_eligible_run_calls_analyzer_once_and_persists(self):
        self._seed_eligible_cases(5)
        analyzer = _FakeAnalyzer()

        outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertIsInstance(outcome, ReflectorRunOutcome)
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(analyzer.calls), 1)

        run_row = self.store.get_reflection_run(outcome.reflection_run_id)
        self.assertEqual(run_row["status"], "succeeded")
        self.assertIsNotNone(self.store.get_improvement_proposal("proposal-fake-new"))
        self.assertIsNotNone(self.store.get_latest_proposal_observation("proposal-fake-new"))


class ExistingProposalCandidateTests(_StoreTestCase):
    def test_context_passed_to_analyzer_includes_existing_candidate(self):
        self._seed_eligible_cases(5)
        self._seed_existing_proposal("proposal-existing", improvement_target="knowledge")
        analyzer = _FakeAnalyzer(with_new_proposal=False, with_match_existing="proposal-existing")

        outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertEqual(outcome.status, "succeeded")
        context = analyzer.calls[0]["context"]
        candidate_ids = {c.proposal_id for c in context.existing_proposals}
        self.assertIn("proposal-existing", candidate_ids)

        # match_existing must never create a duplicate Proposal.
        self.assertEqual(len(self.store.list_improvement_proposals()), 1)
        observation_row = self.store.get_latest_proposal_observation("proposal-existing")
        self.assertEqual(observation_row["trend"], "stable")


# ---------------------------------------------------------------------------
# B. Analyzer failure
# ---------------------------------------------------------------------------


class AnalyzerFailureTests(_StoreTestCase):
    def test_analyzer_error_fails_run_without_persistence(self):
        self._seed_eligible_cases(5)
        analyzer = _FakeAnalyzer(exc=ReflectorAnalyzerError("fake LLM failure"))

        outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertEqual(outcome.status, "analyzer_failed")
        self.assertIsNotNone(outcome.reflection_run_id)
        self.assertIn("fake LLM failure", outcome.error)
        # No reflection_runs row -- persist_reflection_result() (the only
        # code path that creates one) was never reached.
        self.assertIsNone(self.store.get_reflection_run(outcome.reflection_run_id))
        self.assertEqual(self.store.list_improvement_proposals(), [])


# ---------------------------------------------------------------------------
# C. Persistence failure -- two distinct categories
# ---------------------------------------------------------------------------


class PersistenceFailureTests(_StoreTestCase):
    def test_create_reflection_run_failure_reports_persistence_error(self):
        self._seed_eligible_cases(5)
        # Force a reflection_run_id collision: pre-create a reflection_runs
        # row with a known id, then patch uuid.uuid4() so run_reflector()
        # generates that exact id -- the only way to make persist_
        # reflection_result()'s own create_reflection_run() call fail
        # (run_reflector() itself exposes no id-injection point, matching
        # tools.run_case_enrichment's own precedent of not DI-ing id/clock
        # generation into its runner -- see this module's own report).
        self.store.create_reflection_run(
            "collision-id", started_at="2025-01-01T00:00:00+00:00", analyzed_case_count=1,
            reflector_version="reflector-v0",
        )
        analyzer = _FakeAnalyzer()

        class _FixedUUID:
            hex = "collision-id"

        with mock.patch("tools.run_reflector.uuid.uuid4", return_value=_FixedUUID()):
            outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertEqual(outcome.status, "persistence_error")
        self.assertEqual(outcome.reflection_run_id, "collision-id")
        self.assertIn("create_reflection_run failed", outcome.error)
        # The original, pre-existing row must be untouched -- still
        # 'running' (never overwritten or forced to a terminal state).
        self.assertEqual(self.store.get_reflection_run("collision-id")["status"], "running")

    def test_observation_referencing_unknown_proposal_reports_failed_with_a_row(self):
        self._seed_eligible_cases(5)
        analyzer = _FakeAnalyzer(with_new_proposal=False, with_match_existing="proposal-does-not-exist")

        outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertEqual(outcome.status, "failed")
        # Unlike persistence_error, THIS failure category DOES leave a row
        # -- create_reflection_run() succeeded; only the Proposal/
        # Observation transaction failed and was rolled back.
        run_row = self.store.get_reflection_run(outcome.reflection_run_id)
        self.assertIsNotNone(run_row)
        self.assertEqual(run_row["status"], "failed")
        self.assertEqual(self.store.list_improvement_proposals(), [])


# ---------------------------------------------------------------------------
# D. No material change
# ---------------------------------------------------------------------------


class NoMaterialChangeTests(_StoreTestCase):
    def test_material_change_false_still_persists_successfully(self):
        self._seed_eligible_cases(5)
        self._seed_existing_proposal("proposal-existing")
        analyzer = _FakeAnalyzer(
            material_change_detected=False, with_new_proposal=False, with_match_existing="proposal-existing",
        )

        outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertEqual(outcome.status, "succeeded")
        self.assertFalse(outcome.material_change_detected)
        self.assertEqual(outcome.observation_count, 1)
        run_row = self.store.get_reflection_run(outcome.reflection_run_id)
        self.assertEqual(bool(run_row["material_change_detected"]), False)
        self.assertIsNotNone(self.store.get_latest_proposal_observation("proposal-existing"))


# ---------------------------------------------------------------------------
# E. Zero eligible Cases
# ---------------------------------------------------------------------------


class ZeroEligibleCasesTests(_StoreTestCase):
    def test_no_eligible_cases_skips_analyzer_and_db_write(self):
        self._seed_eligible_cases(2)  # below DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION (5)
        analyzer = _FakeAnalyzer()

        outcome = run_reflector(self.store, analyzer=analyzer)

        self.assertEqual(outcome.status, "no_eligible_cases")
        self.assertEqual(outcome.analyzed_case_count, 2)
        self.assertEqual(len(analyzer.calls), 0)
        self.assertIsNone(outcome.reflection_run_id)

        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM reflection_runs").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)

    def test_zero_cases_at_all_is_also_no_eligible_cases(self):
        analyzer = _FakeAnalyzer()
        outcome = run_reflector(self.store, analyzer=analyzer)
        self.assertEqual(outcome.status, "no_eligible_cases")
        self.assertEqual(outcome.analyzed_case_count, 0)
        self.assertEqual(len(analyzer.calls), 0)

    def test_min_analyzed_cases_override_respected(self):
        self._seed_eligible_cases(3)
        analyzer = _FakeAnalyzer()
        outcome = run_reflector(self.store, analyzer=analyzer, min_analyzed_cases=3)
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(analyzer.calls), 1)


# ---------------------------------------------------------------------------
# F. Metadata propagation
# ---------------------------------------------------------------------------


class MetadataPropagationTests(_StoreTestCase):
    def test_reflector_version_analyzed_case_count_and_window_reach_the_db(self):
        self._seed_eligible_cases(5, created_at="2026-02-15T00:00:00+00:00")
        analyzer = _FakeAnalyzer()

        outcome = run_reflector(
            self.store, analyzer=analyzer,
            window_start="2026-02-01T00:00:00+00:00", window_end="2026-03-01T00:00:00+00:00",
        )

        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(outcome.analyzed_case_count, 5)
        self.assertEqual(outcome.window_start, "2026-02-01T00:00:00+00:00")
        self.assertEqual(outcome.window_end, "2026-03-01T00:00:00+00:00")

        run_row = self.store.get_reflection_run(outcome.reflection_run_id)
        self.assertEqual(run_row["reflector_version"], REFLECTOR_VERSION)
        self.assertEqual(run_row["analyzed_case_count"], 5)
        self.assertEqual(run_row["window_start"], "2026-02-01T00:00:00+00:00")
        self.assertEqual(run_row["window_end"], "2026-03-01T00:00:00+00:00")

    def test_observed_at_reaches_every_observation(self):
        self._seed_eligible_cases(5)
        analyzer = _FakeAnalyzer()
        outcome = run_reflector(self.store, analyzer=analyzer)
        observed_at = analyzer.calls[0]["observed_at"]
        observation_row = self.store.get_latest_proposal_observation("proposal-fake-new")
        self.assertEqual(observation_row["observed_at"], observed_at)


# ---------------------------------------------------------------------------
# G. main() -- the thin CLI boundary
# ---------------------------------------------------------------------------


class MainCliTests(_StoreTestCase):
    def _run_main(self, *extra_argv: str, analyzer: Any = None) -> tuple[int, str, str]:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        argv = ["--db", str(self.db_path), *extra_argv]
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exit_code = main(argv, analyzer=analyzer)
        return exit_code, buf_out.getvalue(), buf_err.getvalue()

    def test_without_analyzer_override_and_without_hermes_config_fails_closed(self):
        # No analyzer= override: main() must attempt to resolve the real
        # analyzer via _resolve_main_runtime(), which cannot import
        # hermes_cli.config in this repo's test environment -- proving the
        # fail-closed contract through the real CLI path without needing a
        # real Hermes sandbox, matching tools.run_case_enrichment's own
        # test_without_analyzer_override_and_without_hermes_config_fails_
        # closed precedent exactly.
        self._seed_eligible_cases(5)
        exit_code, _, err = self._run_main()
        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR", err)
        self.assertEqual(self.store.list_improvement_proposals(), [])

    def test_cli_wiring_with_injected_analyzer_reaches_stdout_summary(self):
        self._seed_eligible_cases(5)
        analyzer = _FakeAnalyzer()
        exit_code, out, _ = self._run_main(analyzer=analyzer)
        self.assertEqual(exit_code, 0)
        self.assertIn("Reflector 執行完成", out)
        self.assertIn("狀態：成功", out)
        self.assertEqual(len(analyzer.calls), 1)

    def test_cli_argv_window_flags_reach_run_reflector(self):
        self._seed_eligible_cases(5, created_at="2026-02-15T00:00:00+00:00")
        analyzer = _FakeAnalyzer()
        exit_code, out, _ = self._run_main(
            "--window-start", "2026-02-01T00:00:00+00:00",
            "--window-end", "2026-03-01T00:00:00+00:00",
            analyzer=analyzer,
        )
        self.assertEqual(exit_code, 0)
        context = analyzer.calls[0]["context"]
        self.assertEqual(context.analysis_window.start, "2026-02-01T00:00:00+00:00")
        self.assertEqual(context.analysis_window.end, "2026-03-01T00:00:00+00:00")

    def test_cli_exit_code_nonzero_on_analyzer_failure(self):
        self._seed_eligible_cases(5)
        analyzer = _FakeAnalyzer(exc=ReflectorAnalyzerError("fake failure"))
        exit_code, out, _ = self._run_main(analyzer=analyzer)
        self.assertEqual(exit_code, 1)
        self.assertIn("LLM 分析失敗", out)

    def test_cli_exit_code_zero_on_no_eligible_cases(self):
        # Zero eligible Cases is a legitimate, successful no-op -- not a
        # CLI failure (exit code 0, matching this module's own
        # main()'s "succeeded"/"no_eligible_cases" -> 0 convention).
        analyzer = _FakeAnalyzer()
        exit_code, out, _ = self._run_main(analyzer=analyzer)
        self.assertEqual(exit_code, 0)
        self.assertIn("可分析 Case 數量不足", out)
        self.assertEqual(len(analyzer.calls), 0)


if __name__ == "__main__":
    unittest.main()
