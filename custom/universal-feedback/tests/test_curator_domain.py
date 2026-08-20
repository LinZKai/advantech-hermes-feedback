"""Curator Slice 1 tests: the CuratorChange dataclass contract in
tools.curator_domain.

No DB, no LLM, no network -- pure dataclass validation, matching test_
reflector_proposals.py's own style for ImprovementProposal/
ProposalObservation.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.curator_domain import CURATOR_TARGET_FILE, CuratorChange  # noqa: E402


def _make_change(**overrides):
    fields = dict(
        change_id="change-1",
        proposal_id="proposal-1",
        target_file=CURATOR_TARGET_FILE,
        change_type="modify_rule",
        rationale="多筆案例顯示回答過於冗長。",
        before_content="# Advantech Technical Support Instructions\n",
        proposed_content="# Advantech Technical Support Instructions\n\nBe direct.\n",
        expected_effect="回答更精簡。",
        confidence=0.8,
        status="proposed",
        created_at="2026-01-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return CuratorChange(**fields)


class ValidChangeTests(unittest.TestCase):
    def test_valid_real_change_constructs(self):
        change = _make_change()
        self.assertEqual(change.target_file, CURATOR_TARGET_FILE)
        self.assertIsNone(change.reviewed_at)
        self.assertIsNone(change.applied_at)

    def test_valid_no_change_recommended_constructs_with_null_proposed_content(self):
        change = _make_change(
            change_type="no_change_recommended", proposed_content=None, expected_effect=None,
        )
        self.assertEqual(change.change_type, "no_change_recommended")
        self.assertIsNone(change.proposed_content)


class TargetFileGuardTests(unittest.TestCase):
    def test_wrong_target_file_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(target_file="/sandbox/SOUL.md")

    def test_skill_file_target_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(target_file="/sandbox/hermes-support-config/skills/foundry-iq/SKILL.md")


class ChangeTypeGuardTests(unittest.TestCase):
    def test_invalid_change_type_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(change_type="rewrite_everything")


class ProposedContentCrossFieldTests(unittest.TestCase):
    def test_add_rule_requires_non_blank_proposed_content(self):
        with self.assertRaises(ValueError):
            _make_change(change_type="add_rule", proposed_content=None)

    def test_add_rule_rejects_blank_proposed_content(self):
        with self.assertRaises(ValueError):
            _make_change(change_type="add_rule", proposed_content="   ")

    def test_remove_rule_requires_non_blank_proposed_content(self):
        with self.assertRaises(ValueError):
            _make_change(change_type="remove_rule", proposed_content=None)

    def test_no_change_recommended_rejects_non_null_proposed_content(self):
        with self.assertRaises(ValueError):
            _make_change(change_type="no_change_recommended", proposed_content="not null")


class ConfidenceGuardTests(unittest.TestCase):
    def test_confidence_above_one_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(confidence=1.1)

    def test_confidence_below_zero_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(confidence=-0.1)

    def test_confidence_bool_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(confidence=True)


class StatusGuardTests(unittest.TestCase):
    def test_invalid_status_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(status="in_progress")


class BlankFieldGuardTests(unittest.TestCase):
    def test_blank_rationale_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(rationale="   ")

    def test_blank_before_content_rejected(self):
        with self.assertRaises(ValueError):
            _make_change(before_content="")

    def test_blank_expected_effect_rejected_but_none_allowed(self):
        with self.assertRaises(ValueError):
            _make_change(expected_effect="   ")
        _make_change(expected_effect=None)  # must not raise


if __name__ == "__main__":
    unittest.main()
