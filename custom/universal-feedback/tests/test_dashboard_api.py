"""Dashboard MVP API tests: dashboard_api.app (FastAPI thin adapter).

These are thin-adapter contract tests only -- verifying HTTP status/shape
mapping, not re-testing Slice 1/2 guard/state-transition edge cases
already covered by test_curator_storage.py / test_apply_curator_change.py /
test_review_curator_change.py / test_review_proposal.py. No LLM, no
network -- a real temp-file SQLite DB via app.dependency_overrides, and
FastAPI's own TestClient (no live server/socket).
"""
from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard_api.app import app, get_store  # noqa: E402
from tools.curator_domain import CURATOR_TARGET_FILE  # noqa: E402
from tools.feedback_store_v2 import FeedbackStoreV2, RetrievalRunInput  # noqa: E402
from tools.run_curator import CuratorRunOutcome  # noqa: E402
from tools.universal_feedback import POLICY_VERSION  # noqa: E402


class _ApiTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)
        app.dependency_overrides[get_store] = lambda: self.store
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    # -- seed helpers -----------------------------------------------------

    def _seed_case(
        self, case_id: str = "case-1", *, session_id: str = "sess-1",
        with_feedback: bool = True, helpful: bool = True,
    ) -> str:
        self.store.create_or_update_session(session_id, "telegram")
        self.store.create_case(case_id, session_id)
        turn_id = f"turn-{case_id}"
        self.store.create_turn(
            turn_id, case_id,
            platform_user_id="user-1", platform_user_message_id=f"msg-{case_id}",
            question_text="ADAM-6266 SNMP 怎麼關閉？", answer_text="...",
            feedback_eligible=True,
        )
        self.store.add_retrieval_runs(turn_id, [
            RetrievalRunInput(
                execution_status="completed", observation_status="complete",
                tool_call_id="call-1", request_attempted=True, foundry_iq_ok=True,
                result_count=3, reference_count=2,
            ),
        ])
        if with_feedback:
            self.store.submit_feedback(
                f"fb-{case_id}", turn_id, helpful,
                feedback_policy_version=POLICY_VERSION,
                reason_code=None if helpful else "incomplete",
                suggestion_text=None if helpful else "請補充步驟",
            )
        self.store.create_case_analysis(
            f"analysis-{case_id}", case_id,
            case_title="SNMP 無法關閉", issue_summary="使用者詢問如何關閉 SNMP",
            issue_type="product_usage_or_application", issue_type_confidence=0.8,
            diagnosis="answer_quality_issue", diagnosis_confidence=0.75,
            product_model="ADAM-6266", product_source="explicit_user_text", product_confidence=0.9,
            evidence_json=None, analysis_version="v1",
            analyzed_at="2026-01-02T00:00:00+00:00", source_evidence_watermark="2026-01-02T00:00:00+00:00",
        )
        return case_id

    def _seed_proposal_with_observation(
        self, proposal_id: str = "proposal-1", *,
        improvement_target: str = "agent_behavior",
        supporting_case_ids: tuple[str, ...] = ("case-1",),
    ) -> str:
        self.store.create_improvement_proposal(
            proposal_id, improvement_target=improvement_target,
            title="技術支援回答應優先呈現直接結論", created_at="2026-01-03T00:00:00+00:00",
        )
        self.store.create_reflection_run(
            "run-1", started_at="2026-01-03T00:00:00+00:00", analyzed_case_count=1,
            reflector_version="reflector-v1",
        )
        self.store.create_proposal_observation(
            "obs-1", proposal_id, "run-1",
            trend="new", pattern_summary="多筆案例顯示回答過於冗長。", possible_cause=None,
            recommended_improvement="調整回答結構。", expected_benefit=None, limitations=None,
            supporting_case_ids_json=json.dumps(list(supporting_case_ids)),
            supporting_case_count=len(supporting_case_ids),
            confidence=0.7, observed_at="2026-01-03T00:00:00+00:00",
        )
        return proposal_id

    def _seed_curator_change(
        self, proposal_id: str, *, status: str = "proposed",
        before_content: str = "# AGENTS.md\n", proposed_content: str | None = "# AGENTS.md\n\nBe direct.\n",
    ) -> str:
        change_id = uuid.uuid4().hex
        self.store.create_curator_change(
            change_id, proposal_id, CURATOR_TARGET_FILE,
            change_type="modify_rule", rationale="多筆案例顯示回答過於冗長。",
            before_content=before_content, proposed_content=proposed_content,
            expected_effect="回答更精簡。", confidence=0.9,
            created_at="2026-01-04T00:00:00+00:00",
        )
        if status != "proposed":
            self.store.review_curator_change(
                change_id, "approved" if status != "rejected" else "rejected",
                reviewed_at="2026-01-05T00:00:00+00:00",
            )
            if status == "applied":
                self.store.mark_curator_change_applied(change_id, applied_at="2026-01-06T00:00:00+00:00")
            elif status == "failed":
                self.store.mark_curator_change_failed(change_id)
        return change_id


# ---------------------------------------------------------------------------
# GET /overview
# ---------------------------------------------------------------------------


class OverviewTests(_ApiTestCase):
    def test_overview_success(self):
        self._seed_case("case-1", helpful=True)
        self._seed_case("case-2", helpful=False)
        proposal_id = self._seed_proposal_with_observation()
        self._seed_curator_change(proposal_id)

        response = self.client.get("/overview")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_cases"], 2)
        self.assertEqual(body["total_turns"], 2)
        self.assertEqual(body["total_feedback"], 2)
        self.assertEqual(body["helpful_count"], 1)
        self.assertEqual(body["negative_count"], 1)
        self.assertEqual(body["negative_reason_distribution"], {"incomplete": 1})
        self.assertEqual(body["case_diagnosis_distribution"], {"answer_quality_issue": 2})
        self.assertEqual(body["product_model_distribution"], {"ADAM-6266": 2})
        self.assertEqual(body["improvement_proposals"]["total"], 1)
        self.assertEqual(body["curator_changes"]["total"], 1)
        self.assertEqual(body["curator_changes"]["status_distribution"], {"proposed": 1})
        # Must never leak raw rows / raw JSON blobs.
        self.assertNotIn("evidence_json", json.dumps(body))

    def test_overview_empty_db_no_division_by_zero(self):
        response = self.client.get("/overview")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_cases"], 0)
        self.assertEqual(body["helpful_ratio"], 0.0)
        self.assertEqual(body["negative_ratio"], 0.0)


# ---------------------------------------------------------------------------
# GET /cases, GET /cases/{case_id}
# ---------------------------------------------------------------------------


class CasesTests(_ApiTestCase):
    def test_list_cases_success(self):
        self._seed_case("case-1")
        response = self.client.get("/cases")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        row = body[0]
        self.assertEqual(row["case_id"], "case-1")
        self.assertEqual(row["product_model"], "ADAM-6266")
        self.assertEqual(row["turn_count"], 1)
        self.assertEqual(row["feedback_summary"]["total"], 1)
        self.assertIn("latest_activity", row)
        self.assertIsNotNone(row["latest_feedback_submitted_at"])

    def test_list_cases_without_feedback_has_null_latest_feedback_submitted_at(self):
        # Cases are created from Turns, not from Feedback -- a Case can be
        # enriched (and therefore listed) with zero feedback rows.
        self._seed_case("case-1", with_feedback=False)
        response = self.client.get("/cases")
        self.assertEqual(response.status_code, 200)
        row = response.json()[0]
        self.assertEqual(row["feedback_summary"]["total"], 0)
        self.assertIsNone(row["latest_feedback_submitted_at"])

    def test_case_detail_success(self):
        self._seed_case("case-1")
        response = self.client.get("/cases/case-1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["case_id"], "case-1")
        self.assertEqual(body["analysis"]["product_model"], "ADAM-6266")
        self.assertEqual(len(body["turns"]), 1)
        self.assertEqual(body["turns"][0]["question"], "ADAM-6266 SNMP 怎麼關閉？")
        self.assertEqual(len(body["turns"][0]["retrieval_summary"]), 1)
        self.assertEqual(body["turns"][0]["retrieval_summary"][0]["result_count"], 3)
        self.assertEqual(len(body["feedback"]), 1)

    def test_case_detail_404(self):
        response = self.client.get("/cases/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["status"], "not_found")


# ---------------------------------------------------------------------------
# GET /improvements, GET /improvements/{proposal_id}
# ---------------------------------------------------------------------------


class ImprovementsTests(_ApiTestCase):
    def test_list_improvements_success(self):
        self._seed_case("case-1")
        proposal_id = self._seed_proposal_with_observation()
        self._seed_curator_change(proposal_id)

        response = self.client.get("/improvements")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["proposal_id"], proposal_id)
        self.assertEqual(body[0]["latest_observation"]["trend"], "new")
        self.assertEqual(body[0]["curator_change"]["change_type"], "modify_rule")
        # List page must NOT include full before/proposed content.
        self.assertNotIn("before_content", body[0])
        self.assertNotIn("before_content", body[0].get("curator_change") or {})

    def test_improvement_detail_success(self):
        self._seed_case("case-1")
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(proposal_id)

        response = self.client.get(f"/improvements/{proposal_id}")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["proposal"]["proposal_id"], proposal_id)
        self.assertEqual(body["observation"]["pattern_summary"], "多筆案例顯示回答過於冗長。")
        self.assertEqual(body["observation"]["supporting_case_ids"], ["case-1"])
        self.assertEqual(len(body["supporting_cases"]), 1)
        self.assertEqual(body["supporting_cases"][0]["case_id"], "case-1")
        self.assertEqual(len(body["curator_changes"]), 1)
        self.assertEqual(body["curator_changes"][0]["change_id"], change_id)
        # Detail page DOES include full content.
        self.assertIn("before_content", body["curator_changes"][0])
        self.assertIn("proposed_content", body["curator_changes"][0])

    def test_improvement_detail_404(self):
        response = self.client.get("/improvements/does-not-exist")
        self.assertEqual(response.status_code, 404)


# ---------------------------------------------------------------------------
# GET /curator-changes/{change_id}
# ---------------------------------------------------------------------------


class CuratorChangeDetailTests(_ApiTestCase):
    def test_curator_change_detail_success(self):
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(proposal_id)

        response = self.client.get(f"/curator-changes/{change_id}")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["change_id"], change_id)
        self.assertEqual(body["target_file"], CURATOR_TARGET_FILE)
        self.assertEqual(body["status"], "proposed")
        self.assertIn("proposed_content", body)

    def test_curator_change_detail_404(self):
        response = self.client.get("/curator-changes/does-not-exist")
        self.assertEqual(response.status_code, 404)


# ---------------------------------------------------------------------------
# POST /improvements/{proposal_id}/review
# ---------------------------------------------------------------------------


class ProposalReviewApiTests(_ApiTestCase):
    def test_review_success(self):
        proposal_id = self._seed_proposal_with_observation()
        response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "accepted"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "reviewed")
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")

    def test_review_invalid_transition_returns_409(self):
        proposal_id = self._seed_proposal_with_observation()
        self.store.update_proposal_review_status(proposal_id, "accepted")
        response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "rejected"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["status"], "not_pending")

    def test_review_unknown_proposal_returns_404(self):
        response = self.client.post("/improvements/does-not-exist/review", json={"status": "accepted"})
        self.assertEqual(response.status_code, 404)

    def test_review_invalid_body_returns_422(self):
        proposal_id = self._seed_proposal_with_observation()
        response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "pending"})
        self.assertEqual(response.status_code, 422)

    # -- Accept auto-invokes Curator ---------------------------------------

    def test_accept_invokes_curator_and_reports_success(self):
        from unittest import mock

        proposal_id = self._seed_proposal_with_observation()
        fake_outcome = CuratorRunOutcome(
            status="succeeded", proposal_id=proposal_id, change_id="change-x",
            change_type="modify_rule", target_file=CURATOR_TARGET_FILE, confidence=0.85,
        )
        with mock.patch("dashboard_api.app.resolve_default_analyzer", return_value=lambda *a, **k: None), \
             mock.patch("dashboard_api.app.run_curator", return_value=fake_outcome) as mock_run_curator:
            response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "accepted"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "reviewed")
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")
        self.assertEqual(body["curator"]["status"], "succeeded")
        self.assertEqual(body["curator"]["change_id"], "change-x")
        mock_run_curator.assert_called_once()
        self.assertEqual(mock_run_curator.call_args.kwargs["proposal_id"], proposal_id)

    def test_reject_never_invokes_curator(self):
        from unittest import mock

        proposal_id = self._seed_proposal_with_observation()
        with mock.patch("dashboard_api.app.run_curator") as mock_run_curator:
            response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "rejected"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "reviewed")
        self.assertNotIn("curator", body)
        mock_run_curator.assert_not_called()
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "rejected")
        self.assertEqual(self.store.list_curator_changes(proposal_id=proposal_id), [])

    def test_accept_curator_generation_failure_keeps_proposal_accepted(self):
        from unittest import mock

        proposal_id = self._seed_proposal_with_observation()
        fake_outcome = CuratorRunOutcome(status="analyzer_failed", proposal_id=proposal_id, error="LLM boom")
        with mock.patch("dashboard_api.app.resolve_default_analyzer", return_value=lambda *a, **k: None), \
             mock.patch("dashboard_api.app.run_curator", return_value=fake_outcome):
            response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "accepted"})

        # The Proposal Accept itself is never rolled back for a Curator failure.
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "reviewed")
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")
        self.assertEqual(body["curator"]["status"], "analyzer_failed")
        self.assertEqual(body["curator"]["error"], "LLM boom")
        self.assertEqual(self.store.list_curator_changes(proposal_id=proposal_id), [])

    def test_accept_curator_runtime_config_unavailable_is_reported_not_raised(self):
        # No mocking at all: this test's own environment has no hermes_cli
        # installed (see this module's own docstring), so resolve_default_
        # analyzer() genuinely raises RuntimeError here -- exactly what
        # happens if this endpoint is ever run outside a built Hermes
        # sandbox image. Proves the endpoint degrades gracefully rather
        # than 500ing.
        proposal_id = self._seed_proposal_with_observation()
        response = self.client.post(f"/improvements/{proposal_id}/review", json={"status": "accepted"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(self.store.get_improvement_proposal(proposal_id)["review_status"], "accepted")
        self.assertEqual(body["curator"]["status"], "analyzer_failed")
        self.assertIsNotNone(body["curator"]["error"])
        self.assertEqual(self.store.list_curator_changes(proposal_id=proposal_id), [])


# ---------------------------------------------------------------------------
# POST /curator-changes/{change_id}/review
# ---------------------------------------------------------------------------


class CuratorReviewApiTests(_ApiTestCase):
    def setUp(self):
        super().setUp()
        self._agents_tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.agents_path = Path(self._agents_tmpdir.name) / "AGENTS.md"

    def tearDown(self):
        self._agents_tmpdir.cleanup()
        super().tearDown()

    def test_approve_success_applies_immediately(self):
        from unittest import mock

        before_content = "# AGENTS.md\n"
        proposed_content = "# AGENTS.md\n\nBe direct.\n"
        self.agents_path.write_text(before_content, encoding="utf-8")
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(
            proposal_id, before_content=before_content, proposed_content=proposed_content,
        )

        with mock.patch("dashboard_api.app.CURATOR_TARGET_FILE", str(self.agents_path)):
            response = self.client.post(f"/curator-changes/{change_id}/review", json={"status": "approved"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "applied")
        self.assertIsNotNone(body["applied_at"])

        row = self.store.get_curator_change(change_id)
        self.assertEqual(row["status"], "applied")
        self.assertIsNotNone(row["applied_at"])
        self.assertEqual(self.agents_path.read_text(encoding="utf-8"), proposed_content)

    def test_reject_success_never_applies(self):
        from unittest import mock

        before_content = "# AGENTS.md\n"
        self.agents_path.write_text(before_content, encoding="utf-8")
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(proposal_id, before_content=before_content)

        with mock.patch("dashboard_api.app.CURATOR_TARGET_FILE", str(self.agents_path)):
            response = self.client.post(f"/curator-changes/{change_id}/review", json={"status": "rejected"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.get_curator_change(change_id)["status"], "rejected")
        self.assertEqual(self.agents_path.read_text(encoding="utf-8"), before_content)

    def test_invalid_transition_returns_409(self):
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(proposal_id, status="applied")
        response = self.client.post(f"/curator-changes/{change_id}/review", json={"status": "approved"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["status"], "not_proposed")

    def test_approve_apply_guard_failure_source_changed_leaves_status_approved(self):
        from unittest import mock

        before_content = "# AGENTS.md\n"
        proposed_content = "# AGENTS.md\n\nBe direct.\n"
        drifted_content = "# AGENTS.md\n\nSomeone else already changed this.\n"
        # AGENTS.md on "disk" has already drifted away from before_content --
        # simulates a change landing between Curator's proposal and this
        # Approve.
        self.agents_path.write_text(drifted_content, encoding="utf-8")
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(
            proposal_id, before_content=before_content, proposed_content=proposed_content,
        )

        with mock.patch("dashboard_api.app.CURATOR_TARGET_FILE", str(self.agents_path)):
            response = self.client.post(f"/curator-changes/{change_id}/review", json={"status": "approved"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["status"], "source_changed")

        row = self.store.get_curator_change(change_id)
        self.assertEqual(row["status"], "approved")
        self.assertIsNone(row["applied_at"])
        self.assertEqual(self.agents_path.read_text(encoding="utf-8"), drifted_content)


# ---------------------------------------------------------------------------
# POST /curator-changes/{change_id}/apply
#
# The endpoint always targets the real, fixed CURATOR_TARGET_FILE path
# (/sandbox/AGENTS.md) -- there is deliberately no caller-supplied path
# over HTTP (see dashboard_api.app's own docstring). "not_found" and
# "not_approved" are reachable without ever touching the filesystem (the
# guard short-circuits first), so those two are exercised end-to-end
# against the real apply_curator_change(). The write-succeeds path is
# ALREADY fully covered at the unit level by test_apply_curator_change.
# HappyPathTests -- re-proving file-write correctness here would just
# duplicate that, and would require writing to the real fixed
# /sandbox/AGENTS.md path on whatever machine runs this test, which this
# suite must not do. Instead, the "outcome -> HTTP status/shape" mapping
# itself (the actual thing this HTTP layer is responsible for) is tested
# by patching dashboard_api.app.apply_curator_change with a scripted fake
# -- exactly the "thin adapter contract, not the guard logic" boundary
# this module's own tests are scoped to.
# ---------------------------------------------------------------------------


class ApplyApiTests(_ApiTestCase):
    def test_apply_not_found_returns_404(self):
        response = self.client.post("/curator-changes/does-not-exist/apply")
        self.assertEqual(response.status_code, 404)

    def test_apply_not_approved_returns_409(self):
        proposal_id = self._seed_proposal_with_observation()
        change_id = self._seed_curator_change(proposal_id, status="proposed")
        response = self.client.post(f"/curator-changes/{change_id}/apply")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["status"], "not_approved")

    def test_apply_success_outcome_maps_to_200(self):
        from unittest import mock

        from tools.apply_curator_change import ApplyOutcome

        fake_outcome = ApplyOutcome(status="applied", change_id="change-x", applied_at="2026-01-07T00:00:00+00:00")
        with mock.patch("dashboard_api.app.apply_curator_change", return_value=fake_outcome):
            response = self.client.post("/curator-changes/change-x/apply")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "applied")
        self.assertEqual(body["applied_at"], "2026-01-07T00:00:00+00:00")

    def test_apply_source_changed_outcome_maps_to_409(self):
        from unittest import mock

        from tools.apply_curator_change import ApplyOutcome

        fake_outcome = ApplyOutcome(status="source_changed", change_id="change-x")
        with mock.patch("dashboard_api.app.apply_curator_change", return_value=fake_outcome):
            response = self.client.post("/curator-changes/change-x/apply")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["status"], "source_changed")

    def test_apply_write_failed_outcome_maps_to_500(self):
        from unittest import mock

        from tools.apply_curator_change import ApplyOutcome

        fake_outcome = ApplyOutcome(status="write_failed", change_id="change-x", error="disk full")
        with mock.patch("dashboard_api.app.apply_curator_change", return_value=fake_outcome):
            response = self.client.post("/curator-changes/change-x/apply")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["status"], "write_failed")
        self.assertEqual(response.json()["detail"]["message"], "disk full")


if __name__ == "__main__":
    unittest.main()
