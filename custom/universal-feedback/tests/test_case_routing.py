"""Phase 4A Stage A parser tests: tools.case_routing.

Pure-function tests only -- no database, no gateway, no streaming, no
Telegram, no LLM. See tools/case_routing.py module docstring for the
protocol contract these tests lock in.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_routing import (  # noqa: E402
    CASE_ROUTING_CLOSE_DELIM,
    CASE_ROUTING_OPEN_DELIM,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_ENVELOPE_PREFIX_BYTES,
    REASON_EXISTING_MISSING_CASE_ID,
    REASON_EXISTING_UNKNOWN_CASE_ID,
    REASON_INVALID_CONFIDENCE,
    REASON_JSON_INVALID,
    REASON_MISSING_CLOSING_DELIMITER,
    REASON_NEW_WITH_CASE_ID,
    REASON_PREFIX_SIZE_EXCEEDED,
    REASON_SCHEMA_INVALID,
    REASON_UNCERTAIN_WITH_CASE_ID,
    REASON_UNKNOWN_CASE_ACTION,
    REASON_UNSUPPORTED_ROUTING_VERSION,
    ROUTING_VERSION,
    CaseRoutingResult,
    parse_and_strip_prefix,
    parse_case_routing_envelope,
    strip_case_routing_prefix,
    validate_case_routing,
)


def _envelope_json(
    *,
    case_action="new",
    case_id=None,
    confidence=0.9,
    routing_version=ROUTING_VERSION,
    extra_fields=None,
    omit_fields=None,
):
    """Build the JSON body of a valid-shaped envelope, with optional
    tampering (extra/omitted keys) for schema-violation tests."""
    body = {
        "case_action": case_action,
        "case_id": case_id,
        "confidence": confidence,
        "routing_version": routing_version,
    }
    if omit_fields:
        for key in omit_fields:
            body.pop(key, None)
    if extra_fields:
        body.update(extra_fields)
    return json.dumps(body)


def _envelope(answer="normal user-visible answer", **kwargs):
    """Build a full response: <case-routing>{...}</case-routing> + answer."""
    return f"{CASE_ROUTING_OPEN_DELIM}{_envelope_json(**kwargs)}{CASE_ROUTING_CLOSE_DELIM}{answer}"


class ValidEnvelopeTests(unittest.TestCase):
    def test_valid_existing(self):
        text = _envelope(case_action="existing", case_id="case-a", confidence=0.92)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.case_action, "existing")
        self.assertEqual(result.case_id, "case-a")
        self.assertEqual(result.confidence, 0.92)
        self.assertEqual(result.routing_version, ROUTING_VERSION)
        self.assertIsNone(result.invalid_reason)

    def test_valid_new(self):
        text = _envelope(case_action="new", case_id=None, confidence=0.81)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.case_action, "new")
        self.assertIsNone(result.case_id)

    def test_valid_uncertain(self):
        text = _envelope(case_action="uncertain", case_id=None, confidence=0.3)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.case_action, "uncertain")
        self.assertIsNone(result.case_id)

    def test_confidence_exactly_zero(self):
        text = _envelope(case_action="new", confidence=0.0)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.confidence, 0.0)

    def test_confidence_exactly_one(self):
        text = _envelope(case_action="existing", case_id="case-a", confidence=1.0)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.confidence, 1.0)

    def test_confidence_bare_integer_one_accepted(self):
        # JSON does not distinguish 1 from 1.0 -- a bare integer confidence
        # must still be accepted and normalized to float.
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action":"new","case_id":null,"confidence":1,"routing_version":"' + ROUTING_VERSION + '"}' + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.confidence, 1.0)
        self.assertIsInstance(result.confidence, float)


class EnvelopeStructureTests(unittest.TestCase):
    def test_no_envelope_is_absent(self):
        result = parse_case_routing_envelope("請先確認設備的 SNMP 設定...")
        self.assertEqual(result.status, "absent")
        self.assertIsNone(result.invalid_reason)

    def test_empty_string_is_absent(self):
        result = parse_case_routing_envelope("")
        self.assertEqual(result.status, "absent")

    def test_valid_prefix_plus_answer(self):
        text = _envelope(answer="這是給使用者看的正常回答。", case_action="new")
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")

    def test_envelope_only_no_trailing_answer(self):
        text = _envelope(answer="", case_action="new")
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "valid")

    def test_marker_later_in_natural_text_is_absent(self):
        text = "正常回答內容...\n" + CASE_ROUTING_OPEN_DELIM + _envelope_json(case_action="new") + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "absent")

    def test_leading_whitespace_before_marker_is_absent(self):
        text = "\n" + _envelope(case_action="new")
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "absent")

    def test_leading_space_before_marker_is_absent(self):
        text = " " + _envelope(case_action="new")
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "absent")

    def test_missing_close_delimiter(self):
        text = CASE_ROUTING_OPEN_DELIM + _envelope_json(case_action="new")
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_MISSING_CLOSING_DELIMITER)

    def test_oversized_envelope_with_eventual_close(self):
        # Closing delimiter exists, but the span between open/close exceeds
        # MAX_ENVELOPE_PREFIX_BYTES -- must be flagged for size, not treated
        # as merely malformed JSON.
        padding = "x" * (MAX_ENVELOPE_PREFIX_BYTES + 200)
        text = CASE_ROUTING_OPEN_DELIM + padding + CASE_ROUTING_CLOSE_DELIM + "answer"
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_PREFIX_SIZE_EXCEEDED)

    def test_oversized_with_no_close_at_all(self):
        padding = "x" * (MAX_ENVELOPE_PREFIX_BYTES + 200)
        text = CASE_ROUTING_OPEN_DELIM + padding
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_PREFIX_SIZE_EXCEEDED)

    def test_short_missing_close_is_not_oversized(self):
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action":"new"'
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_MISSING_CLOSING_DELIMITER)


class JsonSchemaTests(unittest.TestCase):
    def test_malformed_json(self):
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action": "new", oops' + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_JSON_INVALID)

    def test_envelope_body_not_a_json_object(self):
        text = CASE_ROUTING_OPEN_DELIM + "[1,2,3]" + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_SCHEMA_INVALID)

    def test_missing_required_field(self):
        text = _envelope(case_action="new", omit_fields=["confidence"])
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_SCHEMA_INVALID)

    def test_missing_case_id_key_entirely_is_schema_invalid(self):
        # case_id must be PRESENT (possibly null), not omitted.
        text = _envelope(case_action="new", omit_fields=["case_id"])
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_SCHEMA_INVALID)

    def test_unknown_extra_field_rejected(self):
        text = _envelope(case_action="new", extra_fields={"issue_category": "hardware"})
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_SCHEMA_INVALID)

    def test_wrong_routing_version(self):
        text = _envelope(case_action="new", routing_version="case-router-v2")
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_UNSUPPORTED_ROUTING_VERSION)

    def test_non_string_routing_version(self):
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action":"new","case_id":null,"confidence":0.9,"routing_version":1}' + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_UNSUPPORTED_ROUTING_VERSION)

    def test_unknown_case_action(self):
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action":"closed","case_id":null,"confidence":0.9,"routing_version":"' + ROUTING_VERSION + '"}' + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_UNKNOWN_CASE_ACTION)

    def test_non_string_case_id_is_schema_invalid(self):
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action":"existing","case_id":123,"confidence":0.9,"routing_version":"' + ROUTING_VERSION + '"}' + CASE_ROUTING_CLOSE_DELIM
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_SCHEMA_INVALID)


class ConfidenceTests(unittest.TestCase):
    def _confidence_text(self, confidence_json_literal):
        return (
            CASE_ROUTING_OPEN_DELIM
            + '{"case_action":"new","case_id":null,"confidence":'
            + confidence_json_literal
            + ',"routing_version":"'
            + ROUTING_VERSION
            + '"}'
            + CASE_ROUTING_CLOSE_DELIM
        )

    def test_negative_confidence(self):
        result = parse_case_routing_envelope(self._confidence_text("-0.1"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_above_one(self):
        result = parse_case_routing_envelope(self._confidence_text("1.1"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_as_string(self):
        result = parse_case_routing_envelope(self._confidence_text('"0.8"'))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_null(self):
        result = parse_case_routing_envelope(self._confidence_text("null"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_bool_true_rejected(self):
        # bool is an int subclass in Python -- must not be silently
        # accepted as confidence=1.
        result = parse_case_routing_envelope(self._confidence_text("true"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_bool_false_rejected(self):
        result = parse_case_routing_envelope(self._confidence_text("false"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_nan_rejected(self):
        # Python's json module accepts the bare NaN token by default.
        result = parse_case_routing_envelope(self._confidence_text("NaN"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_infinity_rejected(self):
        result = parse_case_routing_envelope(self._confidence_text("Infinity"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)

    def test_confidence_negative_infinity_rejected(self):
        result = parse_case_routing_envelope(self._confidence_text("-Infinity"))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_INVALID_CONFIDENCE)


class CaseSemanticsParseLevelTests(unittest.TestCase):
    """Protocol-level (no candidate_case_ids needed) case_action/case_id
    shape rules, enforced by parse_case_routing_envelope itself."""

    def test_existing_with_null_case_id_invalid(self):
        text = _envelope(case_action="existing", case_id=None, confidence=0.9)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_EXISTING_MISSING_CASE_ID)

    def test_new_with_case_id_invalid(self):
        text = _envelope(case_action="new", case_id="case-a", confidence=0.9)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_NEW_WITH_CASE_ID)

    def test_uncertain_with_case_id_invalid(self):
        text = _envelope(case_action="uncertain", case_id="case-a", confidence=0.5)
        result = parse_case_routing_envelope(text)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.invalid_reason, REASON_UNCERTAIN_WITH_CASE_ID)


class ValidateCaseRoutingTests(unittest.TestCase):
    """Session-scoped known-case validation layered on top of an already
    protocol-valid CaseRoutingResult."""

    def test_existing_known_case_stays_valid(self):
        parsed = parse_case_routing_envelope(
            _envelope(case_action="existing", case_id="case-a", confidence=0.92)
        )
        validated = validate_case_routing(
            parsed,
            candidate_case_ids=frozenset({"case-a", "case-b"}),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(validated.status, "valid")
        self.assertEqual(validated.case_id, "case-a")

    def test_existing_unknown_case_id_invalid(self):
        parsed = parse_case_routing_envelope(
            _envelope(case_action="existing", case_id="case-z", confidence=0.92)
        )
        validated = validate_case_routing(
            parsed,
            candidate_case_ids=frozenset({"case-a", "case-b"}),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(validated.status, "invalid")
        self.assertEqual(validated.invalid_reason, REASON_EXISTING_UNKNOWN_CASE_ID)

    def test_existing_unknown_case_id_never_fuzzy_matched(self):
        # "case-a-typo" is close to "case-a" but must never be corrected.
        parsed = parse_case_routing_envelope(
            _envelope(case_action="existing", case_id="case-a-typo", confidence=0.95)
        )
        validated = validate_case_routing(
            parsed,
            candidate_case_ids=frozenset({"case-a"}),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(validated.status, "invalid")
        self.assertEqual(validated.invalid_reason, REASON_EXISTING_UNKNOWN_CASE_ID)
        self.assertNotEqual(validated.case_id, "case-a")

    def test_new_with_empty_candidate_set_is_valid(self):
        parsed = parse_case_routing_envelope(_envelope(case_action="new", confidence=0.9))
        validated = validate_case_routing(
            parsed, candidate_case_ids=frozenset(), confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.assertEqual(validated.status, "valid")

    def test_uncertain_passes_through(self):
        parsed = parse_case_routing_envelope(_envelope(case_action="uncertain", confidence=0.4))
        validated = validate_case_routing(
            parsed,
            candidate_case_ids=frozenset({"case-a"}),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(validated.status, "valid")
        self.assertEqual(validated.case_action, "uncertain")

    def test_absent_passes_through_unchanged(self):
        absent = CaseRoutingResult(status="absent")
        validated = validate_case_routing(
            absent, candidate_case_ids=frozenset(), confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.assertEqual(validated.status, "absent")

    def test_invalid_passes_through_unchanged(self):
        invalid = CaseRoutingResult(status="invalid", invalid_reason=REASON_JSON_INVALID)
        validated = validate_case_routing(
            invalid, candidate_case_ids=frozenset(), confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.assertEqual(validated.status, "invalid")
        self.assertEqual(validated.invalid_reason, REASON_JSON_INVALID)

    def test_low_confidence_existing_stays_valid_stage_a_does_not_apply_threshold(self):
        # Stage A does not build a Case and does not apply the confidence
        # threshold as a validity gate -- that policy decision belongs to
        # Stage B. A below-threshold "existing" match is still a
        # protocol-valid result here.
        parsed = parse_case_routing_envelope(
            _envelope(case_action="existing", case_id="case-a", confidence=0.40)
        )
        validated = validate_case_routing(
            parsed,
            candidate_case_ids=frozenset({"case-a"}),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(validated.status, "valid")
        self.assertEqual(validated.confidence, 0.40)

    def test_invalid_confidence_threshold_raises(self):
        parsed = parse_case_routing_envelope(_envelope(case_action="new"))
        with self.assertRaises(ValueError):
            validate_case_routing(parsed, candidate_case_ids=frozenset(), confidence_threshold=1.5)

    def test_bool_confidence_threshold_raises(self):
        parsed = parse_case_routing_envelope(_envelope(case_action="new"))
        with self.assertRaises(ValueError):
            validate_case_routing(parsed, candidate_case_ids=frozenset(), confidence_threshold=True)


class SanitizationTests(unittest.TestCase):
    def test_strip_valid_prefix(self):
        text = _envelope(answer="這是正常答案", case_action="new")
        clean = strip_case_routing_prefix(text)
        self.assertEqual(clean, "這是正常答案")

    def test_preserve_normal_text(self):
        text = "沒有 envelope 的正常回答"
        self.assertEqual(strip_case_routing_prefix(text), text)

    def test_preserve_malformed_prefix(self):
        # Delimiters present, JSON inside is broken -- must not strip.
        text = CASE_ROUTING_OPEN_DELIM + "{not valid json" + CASE_ROUTING_CLOSE_DELIM + "answer"
        self.assertEqual(strip_case_routing_prefix(text), text)

    def test_preserve_incomplete_prefix(self):
        # Opening delimiter present, no closing delimiter at all.
        text = CASE_ROUTING_OPEN_DELIM + '{"case_action":"new"'
        self.assertEqual(strip_case_routing_prefix(text), text)

    def test_preserve_marker_appearing_later_in_answer(self):
        text = "正常回答內容...\n" + CASE_ROUTING_OPEN_DELIM + _envelope_json(case_action="new") + CASE_ROUTING_CLOSE_DELIM
        self.assertEqual(strip_case_routing_prefix(text), text)

    def test_parse_and_strip_prefix_returns_result_alongside_clean_text(self):
        text = _envelope(answer="乾淨的答案", case_action="existing", case_id="case-a", confidence=0.9)
        clean_text, result = parse_and_strip_prefix(text)
        self.assertEqual(clean_text, "乾淨的答案")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.case_id, "case-a")

    def test_parse_and_strip_prefix_non_destructive_on_invalid(self):
        text = CASE_ROUTING_OPEN_DELIM + "{broken" + CASE_ROUTING_CLOSE_DELIM + "answer"
        clean_text, result = parse_and_strip_prefix(text)
        self.assertEqual(clean_text, text)
        self.assertEqual(result.status, "invalid")

    def test_parse_and_strip_prefix_absent_returns_original(self):
        text = "正常回答，沒有 envelope"
        clean_text, result = parse_and_strip_prefix(text)
        self.assertEqual(clean_text, text)
        self.assertEqual(result.status, "absent")


class StructuralSafetyTests(unittest.TestCase):
    def test_bad_status_raises(self):
        with self.assertRaises(ValueError):
            CaseRoutingResult(status="pending")

    def test_valid_status_without_case_action_raises(self):
        with self.assertRaises(ValueError):
            CaseRoutingResult(status="valid", case_action=None)

    def test_valid_status_with_unknown_case_action_raises(self):
        with self.assertRaises(ValueError):
            CaseRoutingResult(status="valid", case_action="closed")

    def test_absent_and_invalid_do_not_require_case_action(self):
        # Must not raise.
        CaseRoutingResult(status="absent")
        CaseRoutingResult(status="invalid", invalid_reason=REASON_JSON_INVALID)

    def test_result_is_frozen(self):
        result = CaseRoutingResult(status="absent")
        with self.assertRaises(Exception):
            result.status = "valid"  # type: ignore[misc]


class ParserNeverRaisesOnMalformedInputTests(unittest.TestCase):
    """parse_case_routing_envelope must classify, never raise, on any
    malformed-but-string input."""

    def test_never_raises(self):
        malformed_inputs = [
            "",
            "not an envelope at all",
            CASE_ROUTING_OPEN_DELIM,
            CASE_ROUTING_OPEN_DELIM + CASE_ROUTING_CLOSE_DELIM,
            CASE_ROUTING_OPEN_DELIM + "{}" + CASE_ROUTING_CLOSE_DELIM,
            CASE_ROUTING_OPEN_DELIM + "null" + CASE_ROUTING_CLOSE_DELIM,
            CASE_ROUTING_OPEN_DELIM + "42" + CASE_ROUTING_CLOSE_DELIM,
            CASE_ROUTING_OPEN_DELIM + '"just a string"' + CASE_ROUTING_CLOSE_DELIM,
            CASE_ROUTING_OPEN_DELIM + "x" * 5000,
        ]
        for text in malformed_inputs:
            with self.subTest(text=text[:40]):
                result = parse_case_routing_envelope(text)
                self.assertIn(result.status, ("absent", "invalid"))


if __name__ == "__main__":
    unittest.main()
