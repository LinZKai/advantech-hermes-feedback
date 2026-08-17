"""Phase 5 Slice 1 tests: the Cross-case Reflector input foundation.

Covers three surfaces:
  1. FeedbackStoreV2.list_case_ids_created_in_window -- the Case-occurrence
     window primitive (tools/feedback_store_v2.py).
  2. FeedbackStoreV2.list_latest_case_analysis -- the latest-analysis batch
     primitive (tools/feedback_store_v2.py).
  3. tools.case_reflection_input.build_reflector_input and its
     ReflectorInput/CaseIntelligenceRecord contracts.

Deliberately a separate file from test_case_analysis.py, matching the
one-file-per-concern convention this test suite already uses (see e.g.
test_run_case_enrichment.py / test_case_enrichment_analyzer.py each getting
their own file for their own module) -- this is a new, independently
reviewable Phase 5 slice, not an extension of the Phase 4.5 Stage A test
surface.

No LLM, no network, no scheduler anywhere in this file -- Slice 1 has none
of those.
"""
from __future__ import annotations

import gc
import json
import sqlite3
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import CaseAnalysisEvidence  # noqa: E402
from tools.case_reflection_input import (  # noqa: E402
    UNKNOWN_PRODUCT_MODEL_BUCKET,
    CaseIntelligenceRecord,
    ReflectorInput,
    build_reflector_input,
    is_reflection_eligible,
)
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture base: real on-disk SQLite through the actual store, matching the
# _StoreTestCase convention already used across this test suite (see
# test_case_analysis.py / test_run_case_enrichment.py).
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

    def _raw_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _seed_case(self, case_id: str, session_id: str = "sess-1", *, created_at: str | None = None) -> str:
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        if created_at is not None:
            with self._raw_connect() as conn:
                conn.execute("UPDATE cases SET created_at=? WHERE case_id=?", (created_at, case_id))
        return case_id

    def _create_analysis(
        self, case_id: str, *, analyzed_at: str,
        source_evidence_watermark: str | None = None,
        analysis_id: str | None = None,
        issue_type: str = "product_usage_or_application",
        issue_type_confidence: float = 0.5,
        diagnosis: str = "knowledge_gap",
        diagnosis_confidence: float = 0.5,
        product_model: str | None = None,
        product_source: str | None = None,
        product_confidence: float | None = None,
        evidence_json: str | None = None,
        case_title: str | None = "title",
        issue_summary: str | None = "summary",
        analysis_version: str = "v1",
    ) -> str:
        analysis_id = analysis_id or uuid.uuid4().hex
        ok = self.store.create_case_analysis(
            analysis_id, case_id,
            case_title=case_title, issue_summary=issue_summary,
            issue_type=issue_type, issue_type_confidence=issue_type_confidence,
            diagnosis=diagnosis, diagnosis_confidence=diagnosis_confidence,
            product_model=product_model, product_source=product_source,
            product_confidence=product_confidence,
            evidence_json=evidence_json,
            analysis_version=analysis_version,
            analyzed_at=analyzed_at,
            source_evidence_watermark=source_evidence_watermark or analyzed_at,
        )
        assert ok
        return analysis_id


# ---------------------------------------------------------------------------
# 1. Storage -- list_case_ids_created_in_window
# ---------------------------------------------------------------------------


class ListCaseIdsCreatedInWindowTests(_StoreTestCase):
    def test_lower_bound_inclusive(self):
        self._seed_case("case-a", created_at="2026-01-10T00:00:00+00:00")
        self._seed_case("case-b", created_at="2026-01-09T23:59:59+00:00")
        result = self.store.list_case_ids_created_in_window(created_from="2026-01-10T00:00:00+00:00")
        self.assertEqual(result, ["case-a"])

    def test_upper_bound_exclusive(self):
        self._seed_case("case-a", created_at="2026-01-10T00:00:00+00:00")
        self._seed_case("case-b", created_at="2026-01-09T23:59:59+00:00")
        result = self.store.list_case_ids_created_in_window(created_before="2026-01-10T00:00:00+00:00")
        self.assertEqual(result, ["case-b"])

    def test_both_bounds(self):
        self._seed_case("case-early", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-in", created_at="2026-01-15T00:00:00+00:00")
        self._seed_case("case-late", created_at="2026-02-01T00:00:00+00:00")
        result = self.store.list_case_ids_created_in_window(
            created_from="2026-01-01T00:00:00+00:00", created_before="2026-02-01T00:00:00+00:00",
        )
        self.assertEqual(result, ["case-early", "case-in"])

    def test_no_bounds_returns_all(self):
        self._seed_case("case-a", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-b", created_at="2026-02-01T00:00:00+00:00")
        result = self.store.list_case_ids_created_in_window()
        self.assertEqual(set(result), {"case-a", "case-b"})

    def test_empty_result(self):
        self._seed_case("case-a", created_at="2026-01-01T00:00:00+00:00")
        result = self.store.list_case_ids_created_in_window(created_from="2027-01-01T00:00:00+00:00")
        self.assertEqual(result, [])

    def test_deterministic_ordering(self):
        # Two Cases sharing the same created_at must still order
        # deterministically, by case_id as the tiebreaker.
        self._seed_case("case-z", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-a", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-m", created_at="2026-01-02T00:00:00+00:00")
        result = self.store.list_case_ids_created_in_window()
        self.assertEqual(result, ["case-a", "case-z", "case-m"])


# ---------------------------------------------------------------------------
# 2. Storage -- list_latest_case_analysis
# ---------------------------------------------------------------------------


class ListLatestCaseAnalysisTests(_StoreTestCase):
    def test_single_case_single_analysis(self):
        self._seed_case("case-1")
        analysis_id = self._create_analysis("case-1", analyzed_at="2026-01-01T00:00:00+00:00")
        rows = self.store.list_latest_case_analysis()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["analysis_id"], analysis_id)

    def test_same_case_multiple_historical_only_latest(self):
        self._seed_case("case-1")
        self._create_analysis("case-1", analyzed_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-1", analyzed_at="2026-01-02T00:00:00+00:00")
        latest_id = self._create_analysis("case-1", analyzed_at="2026-01-03T00:00:00+00:00")

        rows = self.store.list_latest_case_analysis()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["analysis_id"], latest_id)
        self.assertEqual(rows[0]["analyzed_at"], "2026-01-03T00:00:00+00:00")

    def test_multiple_cases_one_latest_each(self):
        self._seed_case("case-a")
        self._seed_case("case-b")
        self._create_analysis("case-a", analyzed_at="2026-01-01T00:00:00+00:00")
        latest_a = self._create_analysis("case-a", analyzed_at="2026-01-02T00:00:00+00:00")
        latest_b = self._create_analysis("case-b", analyzed_at="2026-01-01T00:00:00+00:00")

        rows = self.store.list_latest_case_analysis()
        by_case = {row["case_id"]: row["analysis_id"] for row in rows}
        self.assertEqual(by_case, {"case-a": latest_a, "case-b": latest_b})

    def test_case_ids_subset(self):
        self._seed_case("case-a")
        self._seed_case("case-b")
        self._create_analysis("case-a", analyzed_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-b", analyzed_at="2026-01-01T00:00:00+00:00")

        rows = self.store.list_latest_case_analysis(case_ids=["case-a"])
        self.assertEqual([row["case_id"] for row in rows], ["case-a"])

    def test_case_ids_empty_list_returns_empty(self):
        self._seed_case("case-a")
        self._create_analysis("case-a", analyzed_at="2026-01-01T00:00:00+00:00")
        rows = self.store.list_latest_case_analysis(case_ids=[])
        self.assertEqual(rows, [])

    def test_case_ids_none_returns_all_analyzed(self):
        self._seed_case("case-a")
        self._seed_case("case-b")  # never analyzed
        self._create_analysis("case-a", analyzed_at="2026-01-01T00:00:00+00:00")

        rows = self.store.list_latest_case_analysis(case_ids=None)
        self.assertEqual([row["case_id"] for row in rows], ["case-a"])

    def test_deterministic_ordering(self):
        self._seed_case("case-z")
        self._seed_case("case-a")
        self._create_analysis("case-z", analyzed_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-a", analyzed_at="2026-01-01T00:00:00+00:00")

        rows = self.store.list_latest_case_analysis()
        self.assertEqual([row["case_id"] for row in rows], ["case-a", "case-z"])


# ---------------------------------------------------------------------------
# 3. Reflection input -- build_reflector_input / ReflectorInput
# ---------------------------------------------------------------------------


def _evidence_json() -> str:
    return json.dumps([{"type": "user_text", "turn_id": "turn-1", "fact": "User asked a question."}])


class BuildReflectorInputTests(_StoreTestCase):
    def test_window_filters_cases_correctly(self):
        self._seed_case("case-in", created_at="2026-01-15T00:00:00+00:00")
        self._seed_case("case-out", created_at="2026-03-01T00:00:00+00:00")
        self._create_analysis("case-in", analyzed_at="2026-01-16T00:00:00+00:00")
        self._create_analysis("case-out", analyzed_at="2026-03-02T00:00:00+00:00")

        result = build_reflector_input(
            self.store, window_start="2026-01-01T00:00:00+00:00", window_end="2026-02-01T00:00:00+00:00",
        )
        self.assertEqual([c.case_id for c in result.cases], ["case-in"])

    def test_latest_snapshot_only(self):
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-1", analyzed_at="2026-01-02T00:00:00+00:00",
            diagnosis="retrieval_issue",
        )
        self._create_analysis(
            "case-1", analyzed_at="2026-01-03T00:00:00+00:00",
            diagnosis="knowledge_gap",
        )

        result = build_reflector_input(self.store)
        self.assertEqual(len(result.cases), 1)
        self.assertEqual(result.cases[0].diagnosis, "knowledge_gap")
        self.assertEqual(result.cases[0].analyzed_at, "2026-01-03T00:00:00+00:00")

    def test_no_historical_snapshot_double_count(self):
        # A Case re-analyzed three times must still contribute exactly one
        # count to analyzed_case_count and to every by_* bucket -- never
        # three.
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        for i in range(3):
            self._create_analysis(
                "case-1", analyzed_at=f"2026-01-0{i + 1}T00:00:00+00:00", diagnosis="knowledge_gap",
            )

        result = build_reflector_input(self.store)
        self.assertEqual(result.window_case_count, 1)
        self.assertEqual(result.analyzed_case_count, 1)
        self.assertEqual(dict(result.by_diagnosis), {"knowledge_gap": 1})

    def test_all_cases_analyzed_window_equals_analyzed(self):
        for i in range(3):
            case_id = f"case-{i}"
            self._seed_case(case_id, created_at="2026-01-01T00:00:00+00:00")
            self._create_analysis(case_id, analyzed_at="2026-01-02T00:00:00+00:00")

        result = build_reflector_input(self.store)
        self.assertEqual(result.window_case_count, 3)
        self.assertEqual(result.analyzed_case_count, 3)
        self.assertEqual(len(result.cases), 3)
        self.assertEqual(result.coverage_ratio, 1.0)

    def test_missing_analysis_window_greater_than_analyzed(self):
        self._seed_case("case-analyzed", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-unanalyzed", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-analyzed", analyzed_at="2026-01-02T00:00:00+00:00")

        result = build_reflector_input(self.store)
        self.assertGreater(result.window_case_count, result.analyzed_case_count)
        self.assertEqual(result.window_case_count, 2)
        self.assertEqual(result.analyzed_case_count, 1)

    def test_unparseable_counted_in_window_not_in_analyzed(self):
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-1", analyzed_at="2026-01-02T00:00:00+00:00", evidence_json="not valid json",
        )
        result = build_reflector_input(self.store)
        self.assertEqual(result.window_case_count, 1)
        self.assertEqual(result.analyzed_case_count, 0)
        self.assertEqual(result.cases_with_unparseable_analysis, ("case-1",))

    def test_window_equals_analyzed_plus_missing_plus_unparseable(self):
        self._seed_case("case-analyzed", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-missing", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-unparseable", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-analyzed", analyzed_at="2026-01-02T00:00:00+00:00")
        self._create_analysis(
            "case-unparseable", analyzed_at="2026-01-02T00:00:00+00:00", evidence_json="{bad",
        )

        result = build_reflector_input(self.store)
        self.assertEqual(
            result.window_case_count,
            result.analyzed_case_count
            + len(result.cases_missing_analysis)
            + len(result.cases_with_unparseable_analysis),
        )
        self.assertEqual(result.window_case_count, 3)
        self.assertEqual(result.analyzed_case_count, 1)
        self.assertEqual(len(result.cases_missing_analysis), 1)
        self.assertEqual(len(result.cases_with_unparseable_analysis), 1)

    def test_coverage_ratio_partial(self):
        for i in range(10):
            case_id = f"case-{i}"
            self._seed_case(case_id, created_at="2026-01-01T00:00:00+00:00")
            if i < 8:
                self._create_analysis(case_id, analyzed_at="2026-01-02T00:00:00+00:00")

        result = build_reflector_input(self.store)
        self.assertEqual(result.window_case_count, 10)
        self.assertEqual(result.analyzed_case_count, 8)
        self.assertEqual(result.coverage_ratio, 0.8)

    def test_by_product_model(self):
        self._seed_case("case-a", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-b", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-c", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-a", analyzed_at="2026-01-02T00:00:00+00:00",
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.9,
        )
        self._create_analysis(
            "case-b", analyzed_at="2026-01-02T00:00:00+00:00",
            product_model="ADAM-6266", product_source="inference", product_confidence=0.6,
        )
        self._create_analysis("case-c", analyzed_at="2026-01-02T00:00:00+00:00")  # no product

        result = build_reflector_input(self.store)
        self.assertEqual(
            dict(result.by_product_model), {"ADAM-6266": 2, UNKNOWN_PRODUCT_MODEL_BUCKET: 1},
        )

    def test_unknown_product_bucket(self):
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-1", analyzed_at="2026-01-02T00:00:00+00:00")
        result = build_reflector_input(self.store)
        self.assertIn(UNKNOWN_PRODUCT_MODEL_BUCKET, result.by_product_model)
        self.assertEqual(result.by_product_model[UNKNOWN_PRODUCT_MODEL_BUCKET], 1)

    def test_by_issue_type(self):
        self._seed_case("case-a", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-b", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-a", analyzed_at="2026-01-02T00:00:00+00:00", issue_type="product_issue",
        )
        self._create_analysis(
            "case-b", analyzed_at="2026-01-02T00:00:00+00:00", issue_type="product_issue",
        )
        result = build_reflector_input(self.store)
        self.assertEqual(dict(result.by_issue_type), {"product_issue": 2})

    def test_by_diagnosis(self):
        self._seed_case("case-a", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-b", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-a", analyzed_at="2026-01-02T00:00:00+00:00", diagnosis="answer_quality_issue",
        )
        self._create_analysis(
            "case-b", analyzed_at="2026-01-02T00:00:00+00:00", diagnosis="no_issue_detected",
        )
        result = build_reflector_input(self.store)
        self.assertEqual(dict(result.by_diagnosis), {"answer_quality_issue": 1, "no_issue_detected": 1})

    def test_evidence_json_parsing(self):
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-1", analyzed_at="2026-01-02T00:00:00+00:00", evidence_json=_evidence_json(),
        )
        result = build_reflector_input(self.store)
        self.assertEqual(len(result.cases[0].evidence), 1)
        self.assertIsInstance(result.cases[0].evidence[0], CaseAnalysisEvidence)
        self.assertEqual(result.cases[0].evidence[0].fact, "User asked a question.")

    def test_evidence_json_none_becomes_empty_tuple(self):
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-1", analyzed_at="2026-01-02T00:00:00+00:00", evidence_json=None)
        result = build_reflector_input(self.store)
        self.assertEqual(result.cases[0].evidence, ())

    def test_malformed_evidence_behavior(self):
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-1", analyzed_at="2026-01-02T00:00:00+00:00", evidence_json="not valid json",
        )
        result = build_reflector_input(self.store)
        # Never silently dropped, and never merged into `cases`.
        self.assertEqual(result.cases, ())
        self.assertEqual(result.cases_with_unparseable_analysis, ("case-1",))
        self.assertEqual(result.analyzed_case_count, 0)

    def test_missing_latest_analysis_behavior(self):
        self._seed_case("case-analyzed", created_at="2026-01-01T00:00:00+00:00")
        self._seed_case("case-unanalyzed", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis("case-analyzed", analyzed_at="2026-01-02T00:00:00+00:00")

        result = build_reflector_input(self.store)
        self.assertEqual([c.case_id for c in result.cases], ["case-analyzed"])
        self.assertEqual(result.cases_missing_analysis, ("case-unanalyzed",))
        # Never silently counted as analyzed.
        self.assertEqual(result.analyzed_case_count, 1)

    def test_empty_window_returns_empty_reflector_input(self):
        result = build_reflector_input(
            self.store, window_start="2099-01-01T00:00:00+00:00", window_end="2099-02-01T00:00:00+00:00",
        )
        self.assertEqual(result.cases, ())
        self.assertEqual(result.cases_missing_analysis, ())
        self.assertEqual(result.window_case_count, 0)
        self.assertEqual(result.analyzed_case_count, 0)
        self.assertEqual(result.coverage_ratio, 0.0)
        self.assertEqual(dict(result.by_product_model), {})


class ReflectorInputContractTests(unittest.TestCase):
    """Direct dataclass construction/validation -- no DB involved,
    matching test_case_enrichment.py's own OutputValidationInvalidTests /
    CaseAnalysisEvidenceDirectConstructionTests style for the write-side
    contract."""

    def _record(self, **overrides):
        defaults = dict(
            case_id="case-1", case_title="t", issue_summary="s",
            product_model=None, product_source=None, product_confidence=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            evidence=(), analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        defaults.update(overrides)
        return CaseIntelligenceRecord(**defaults)

    def test_frozen(self):
        record = self._record()
        with self.assertRaises(Exception):
            record.case_id = "other"  # type: ignore[misc]

    def test_invalid_issue_type_rejected(self):
        with self.assertRaises(ValueError):
            self._record(issue_type="not_a_real_issue_type")

    def test_invalid_diagnosis_rejected(self):
        with self.assertRaises(ValueError):
            self._record(diagnosis="not_a_real_diagnosis")

    def test_product_model_without_source_rejected(self):
        with self.assertRaises(ValueError):
            self._record(product_model="ADAM-6266", product_source=None, product_confidence=None)

    def test_empty_evidence_is_allowed(self):
        # Deliberately looser than CaseEnrichmentResult's write-time
        # contract -- see CaseIntelligenceRecord's own docstring.
        record = self._record(evidence=())
        self.assertEqual(record.evidence, ())


class ReflectorInputInvariantTests(unittest.TestCase):
    """Direct ReflectorInput construction/validation -- the
    window_case_count / analyzed_case_count / coverage_ratio invariants
    added in Slice 1.1, plus the pre-existing aggregation-sum invariant."""

    def _record(self, **overrides):
        defaults = dict(
            case_id="case-1", case_title="t", issue_summary="s",
            product_model=None, product_source=None, product_confidence=None,
            issue_type="product_usage_or_application", issue_type_confidence=0.5,
            diagnosis="knowledge_gap", diagnosis_confidence=0.5,
            evidence=(), analysis_version="v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            source_evidence_watermark="2026-01-01T00:00:00+00:00",
        )
        defaults.update(overrides)
        return CaseIntelligenceRecord(**defaults)

    def _reflector_input(self, **overrides):
        record = self._record()
        defaults = dict(
            window_start=None, window_end=None,
            cases=(record,),
            cases_missing_analysis=(),
            cases_with_unparseable_analysis=(),
            window_case_count=1,
            analyzed_case_count=1,
            coverage_ratio=1.0,
            by_product_model={UNKNOWN_PRODUCT_MODEL_BUCKET: 1},
            by_issue_type={"product_usage_or_application": 1},
            by_diagnosis={"knowledge_gap": 1},
        )
        defaults.update(overrides)
        return ReflectorInput(**defaults)

    def test_analyzed_case_count_mismatch_with_cases_rejected(self):
        with self.assertRaises(ValueError):
            self._reflector_input(analyzed_case_count=2)  # wrong -- len(cases) is 1

    def test_window_case_count_mismatch_rejected(self):
        # window_case_count must equal analyzed + missing + unparseable.
        with self.assertRaises(ValueError):
            self._reflector_input(window_case_count=5)

    def test_coverage_ratio_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self._reflector_input(coverage_ratio=0.5)  # should be 1.0 for 1/1

    def test_coverage_ratio_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            self._reflector_input(
                cases=(), cases_missing_analysis=(), cases_with_unparseable_analysis=(),
                window_case_count=0, analyzed_case_count=0, coverage_ratio=1.5,
                by_product_model={}, by_issue_type={}, by_diagnosis={},
            )

    def test_aggregation_sum_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self._reflector_input(
                by_product_model={UNKNOWN_PRODUCT_MODEL_BUCKET: 2},  # wrong -- must sum to 1
            )

    def test_valid_construction_with_gaps(self):
        # window_case_count = analyzed(1) + missing(1) + unparseable(1) = 3.
        result = self._reflector_input(
            cases_missing_analysis=("case-missing",),
            cases_with_unparseable_analysis=("case-bad",),
            window_case_count=3,
            analyzed_case_count=1,
            coverage_ratio=1 / 3,
        )
        self.assertEqual(result.window_case_count, 3)

    def test_frozen(self):
        result = self._reflector_input()
        with self.assertRaises(Exception):
            result.analyzed_case_count = 5  # type: ignore[misc]


class MappingImmutabilityTests(_StoreTestCase):
    """The by_* aggregation fields are types.MappingProxyType -- a
    genuinely immutable view, not just a type hint. See this module's
    docstring for why this is a serialization boundary (json.dumps()
    cannot consume a mappingproxy directly) rather than a JSON-ready
    shape -- no serializer is built in this slice."""

    def _result(self) -> ReflectorInput:
        self._seed_case("case-1", created_at="2026-01-01T00:00:00+00:00")
        self._create_analysis(
            "case-1", analyzed_at="2026-01-02T00:00:00+00:00",
            issue_type="product_issue", diagnosis="knowledge_gap",
        )
        return build_reflector_input(self.store)

    def test_by_product_model_rejects_item_assignment(self):
        result = self._result()
        with self.assertRaises(TypeError):
            result.by_product_model["ADAM-6266"] = 99  # type: ignore[index]

    def test_by_issue_type_rejects_item_assignment(self):
        result = self._result()
        with self.assertRaises(TypeError):
            result.by_issue_type["product_issue"] = 99  # type: ignore[index]

    def test_by_diagnosis_rejects_item_assignment(self):
        result = self._result()
        with self.assertRaises(TypeError):
            result.by_diagnosis["knowledge_gap"] = 99  # type: ignore[index]


class ReflectionEligibilityTests(_StoreTestCase):
    def _input_with_analyzed(
        self, count: int, *, window_count: int | None = None, prefix: str = "case",
    ) -> ReflectorInput:
        window_count = count if window_count is None else window_count
        for i in range(window_count):
            case_id = f"{prefix}-{i}"
            self._seed_case(case_id, created_at="2026-01-01T00:00:00+00:00")
            if i < count:
                self._create_analysis(case_id, analyzed_at="2026-01-02T00:00:00+00:00")
        return build_reflector_input(self.store)

    def test_exactly_at_threshold_is_eligible(self):
        reflector_input = self._input_with_analyzed(5)
        self.assertTrue(is_reflection_eligible(reflector_input, min_analyzed_cases=5))

    def test_below_threshold_is_not_eligible(self):
        reflector_input = self._input_with_analyzed(4)
        self.assertFalse(is_reflection_eligible(reflector_input, min_analyzed_cases=5))

    def test_above_threshold_is_eligible(self):
        reflector_input = self._input_with_analyzed(9)
        self.assertTrue(is_reflection_eligible(reflector_input, min_analyzed_cases=5))

    def test_invalid_threshold_zero_rejected(self):
        reflector_input = self._input_with_analyzed(5)
        with self.assertRaises(ValueError):
            is_reflection_eligible(reflector_input, min_analyzed_cases=0)

    def test_invalid_threshold_negative_rejected(self):
        reflector_input = self._input_with_analyzed(5)
        with self.assertRaises(ValueError):
            is_reflection_eligible(reflector_input, min_analyzed_cases=-1)

    def test_eligibility_ignores_window_case_count(self):
        # window=10, analyzed=5, threshold=5 -> True: eligibility is judged
        # purely on analyzed_case_count, never on the larger window count
        # that also includes Cases the Reflector has no evidence for.
        reflector_input = self._input_with_analyzed(5, window_count=10)
        self.assertEqual(reflector_input.window_case_count, 10)
        self.assertEqual(reflector_input.analyzed_case_count, 5)
        self.assertTrue(is_reflection_eligible(reflector_input, min_analyzed_cases=5))

    def test_default_threshold_accepts_five(self):
        reflector_input = self._input_with_analyzed(5)
        self.assertTrue(is_reflection_eligible(reflector_input))

    def test_default_threshold_rejects_four(self):
        reflector_input = self._input_with_analyzed(4)
        self.assertFalse(is_reflection_eligible(reflector_input))


if __name__ == "__main__":
    unittest.main()
