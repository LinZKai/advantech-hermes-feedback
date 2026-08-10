"""Tests for tools.feedback_callbacks (structured fb: callback parsing).

Run locally with:
    python -m unittest discover -s custom/universal-feedback/tests -v
No production dependency is added; this uses only the standard library.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_callbacks import (  # noqa: E402
    REASON_CODES,
    FeedbackCallback,
    is_valid_reason_code,
    parse_feedback_callback,
)
from tools.feedback_storage import parse_callback_data  # noqa: E402


class ParseFeedbackCallbackTests(unittest.TestCase):
    def test_valid_helpful(self):
        result = parse_feedback_callback("fb:h:abc123")
        self.assertEqual(result, FeedbackCallback(action="h", run_id="abc123"))

    def test_valid_unhelpful(self):
        result = parse_feedback_callback("fb:u:abc123")
        self.assertEqual(result, FeedbackCallback(action="u", run_id="abc123"))

    def test_valid_reason_all_allowlist_codes(self):
        for code in REASON_CODES:
            with self.subTest(code=code):
                result = parse_feedback_callback(f"fb:r:abc123:{code}")
                self.assertEqual(
                    result,
                    FeedbackCallback(action="r", run_id="abc123", reason_code=code),
                )

    def test_unknown_action_rejected(self):
        self.assertIsNone(parse_feedback_callback("fb:x:abc123"))

    def test_invalid_reason_code_rejected(self):
        self.assertIsNone(parse_feedback_callback("fb:r:abc123:bogus"))

    def test_empty_run_id_rejected(self):
        self.assertIsNone(parse_feedback_callback("fb:h:"))
        self.assertIsNone(parse_feedback_callback("fb:r::incorrect"))

    def test_whitespace_only_run_id_rejected(self):
        self.assertIsNone(parse_feedback_callback("fb:h:   "))
        self.assertIsNone(parse_feedback_callback("fb:r:   :incorrect"))

    def test_missing_segment_rejected(self):
        self.assertIsNone(parse_feedback_callback("fb:h"))
        self.assertIsNone(parse_feedback_callback("fb:u"))
        self.assertIsNone(parse_feedback_callback("fb:r:abc123"))
        self.assertIsNone(parse_feedback_callback("fb"))

    def test_extra_segment_rejected(self):
        self.assertIsNone(parse_feedback_callback("fb:h:abc123:extra"))
        self.assertIsNone(parse_feedback_callback("fb:r:abc123:incorrect:extra"))

    def test_ascii_payload_at_65_bytes_is_rejected(self):
        # One byte over the limit: "fb:h:" (5 bytes) + 60 'a' bytes == 65
        # bytes. Pins the exact > MAX_CALLBACK_DATA_BYTES boundary, distinct
        # from the exactly-64-bytes-accepted case below.
        run_id = "a" * 60
        payload = f"fb:h:{run_id}"
        self.assertEqual(len(payload.encode("utf-8")), 65)
        self.assertIsNone(parse_feedback_callback(payload))

    def test_multibyte_utf8_payload_over_64_bytes_rejected(self):
        # 25 chars total, but each CJK char is 3 UTF-8 bytes: 5 + 20*3 = 65
        # bytes. A char-length check alone would wrongly accept this.
        run_id = "強" * 20
        payload = f"fb:h:{run_id}"
        self.assertLessEqual(len(payload), 64)
        self.assertGreater(len(payload.encode("utf-8")), 64)
        self.assertIsNone(parse_feedback_callback(payload))

    def test_payload_at_exactly_64_bytes_is_accepted(self):
        # "fb:h:" (5 bytes) + 59 'a' bytes == 64 bytes exactly.
        run_id = "a" * 59
        payload = f"fb:h:{run_id}"
        self.assertEqual(len(payload.encode("utf-8")), 64)
        self.assertEqual(
            parse_feedback_callback(payload),
            FeedbackCallback(action="h", run_id=run_id),
        )

    def test_malformed_non_string_input_fails_closed(self):
        for value in (None, 123, 12.5, b"fb:h:abc123", ["fb", "h", "abc123"], {}):
            with self.subTest(value=value):
                self.assertIsNone(parse_feedback_callback(value))

    def test_is_valid_reason_code_rejects_non_string_input(self):
        # Must fail closed (return False) rather than raise, even for
        # unhashable types that would otherwise blow up a naive
        # `x in frozenset(...)` membership test.
        for value in (None, 123, ["incorrect"], {"incorrect"}, {"code": "incorrect"}):
            with self.subTest(value=value):
                self.assertFalse(is_valid_reason_code(value))

    def test_is_valid_reason_code_accepts_allowlisted_strings(self):
        for code in REASON_CODES:
            with self.subTest(code=code):
                self.assertTrue(is_valid_reason_code(code))

    def test_legacy_y_n_prefixes_not_recognized_by_structured_parser(self):
        # fb:y / fb:n stay the exclusive responsibility of the legacy
        # parse_callback_data() contract below; the structured parser only
        # understands h/u/r.
        self.assertIsNone(parse_feedback_callback("fb:y:abc123"))
        self.assertIsNone(parse_feedback_callback("fb:n:abc123"))


class LegacyParseCallbackDataContractTests(unittest.TestCase):
    """Confirms this change did not touch parse_callback_data()'s contract."""

    def test_fb_h_is_helpful_true(self):
        self.assertEqual(parse_callback_data("fb:h:abc123"), ("abc123", True))

    def test_fb_u_is_helpful_false(self):
        self.assertEqual(parse_callback_data("fb:u:abc123"), ("abc123", False))

    def test_fb_y_is_helpful_true(self):
        self.assertEqual(parse_callback_data("fb:y:abc123"), ("abc123", True))

    def test_fb_n_is_helpful_false(self):
        self.assertEqual(parse_callback_data("fb:n:abc123"), ("abc123", False))

    def test_invalid_action_rejected(self):
        self.assertIsNone(parse_callback_data("fb:r:abc123"))
        self.assertIsNone(parse_callback_data("fb:x:abc123"))

    def test_missing_run_id_rejected(self):
        self.assertIsNone(parse_callback_data("fb:h:"))


if __name__ == "__main__":
    unittest.main()
