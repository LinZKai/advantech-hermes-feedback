"""Phase 3B wiring tests.

overlay/plugins/platforms/telegram/adapter.py is a full-file upstream
overlay requiring the proprietary python-telegram-bot/Hermes runtime to
actually execute (Update/CallbackQuery/Message objects, a live Bot, etc.)
-- not importable in this test environment, matching run.py/base.py's own
situation documented in test_phase3a_wiring.py. These tests therefore
verify the wiring *structurally*, at the source-text level: that the v2
mirror calls exist exactly once each, are placed strictly after the
legacy success actions (never before, never replacing them), are each
independently failure-isolated, and are bound to turn_key resolved off
the already-authorized feedback_runs row -- never a heuristic. Runtime
behavior of tools.feedback_mirror itself is covered by
test_feedback_mirror.py, using the real storage layer.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

_ADAPTER_PY = _OVERLAY_ROOT / "plugins" / "platforms" / "telegram" / "adapter.py"


class _SourceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = _ADAPTER_PY.read_text(encoding="utf-8")

    def _window_around(self, marker: str, *, before: int = 400, after: int = 400) -> str:
        idx = self.text.index(marker)
        return self.text[max(0, idx - before): idx + len(marker) + after]


class ImportTests(_SourceTestCase):
    def test_imports_all_three_mirror_functions(self):
        self.assertIn(
            "from tools.feedback_mirror import mirror_negative_feedback, mirror_positive_feedback, mirror_suggestion",
            self.text,
        )

    def test_does_not_import_feedback_store_v2_directly(self):
        # adapter.py must go through tools.feedback_mirror, never construct
        # or call FeedbackStoreV2 itself -- no duplicated storage logic in
        # the core adapter file.
        self.assertNotIn("FeedbackStoreV2", self.text)


class CallSiteCountTests(_SourceTestCase):
    def test_each_mirror_function_called_exactly_once(self):
        self.assertEqual(self.text.count("mirror_positive_feedback("), 1)
        self.assertEqual(self.text.count("mirror_negative_feedback("), 1)
        self.assertEqual(self.text.count("mirror_suggestion("), 1)


class TurnLinkageNoHeuristicTests(_SourceTestCase):
    """The blocker-level requirement: v2 mirror must bind to
    row["turn_key"] (resolved from the already-authorized feedback_runs
    row), never to question/answer text, timestamps, or "the latest
    Turn"."""

    def test_positive_mirror_uses_row_turn_key(self):
        window = self._window_around("mirror_positive_feedback(", before=200, after=50)
        self.assertIn('mirror_positive_feedback(row["turn_key"])', window)

    def test_negative_mirror_uses_row_turn_key(self):
        window = self._window_around("mirror_negative_feedback(", before=200, after=50)
        self.assertIn('mirror_negative_feedback(row["turn_key"], reason_code)', window)

    def test_suggestion_mirror_uses_row_turn_key(self):
        window = self._window_around("mirror_suggestion(", before=200, after=50)
        self.assertIn('mirror_suggestion(row["turn_key"], stripped)', window)

    def test_no_mirror_call_uses_question_or_answer_text_as_linkage(self):
        for marker in ("mirror_positive_feedback(", "mirror_negative_feedback(", "mirror_suggestion("):
            window = self._window_around(marker, before=300, after=100)
            call_line = [ln for ln in window.splitlines() if marker in ln][0]
            self.assertNotIn("question_text", call_line)
            self.assertNotIn("answer_text", call_line)
            self.assertNotIn("latest", call_line.lower())

    def test_no_new_per_chat_pending_state_dict_introduced(self):
        # Phase 3B must resolve turn_key from the DB row it already fetched
        # by run_id/feedback_message_id -- not a new chat_id-keyed dict
        # (that shape is exactly what _model_picker_state already shows is
        # unsafe for anything spanning more than one in-flight prompt).
        self.assertNotIn("_phase3b_pending", self.text)
        self.assertNotIn("_v2_feedback_pending", self.text)


class OrderingAndIsolationTests(_SourceTestCase):
    """"Legacy success" is the legacy STORAGE mutation, not the Telegram
    UX that follows it -- so each mirror call must sit strictly AFTER the
    storage call that decides `accepted`, and strictly BEFORE the
    Telegram UX (query.answer/edit_message_text/reply_text) that could
    independently fail without that failure ever being allowed to
    suppress an already-legal mirror attempt. Each mirror call is also
    independently try/except-isolated so its own failure can never affect
    the Telegram UX that follows it."""

    def test_positive_mirror_between_legacy_storage_and_telegram_ux(self):
        storage_idx = self.text.index("accepted = store.submit_helpful(run_id, True)")
        mirror_idx = self.text.index("mirror_positive_feedback(")
        ux_idx = self.text.index('await query.answer(text="已收到，謝謝您的回饋！")', storage_idx)
        self.assertLess(storage_idx, mirror_idx)
        self.assertLess(mirror_idx, ux_idx)
        self.assertIn(
            'logger.debug("[%s] v2 feedback mirror (positive) failed", self.name, exc_info=True)', self.text
        )

    def test_negative_mirror_between_legacy_storage_and_telegram_ux(self):
        storage_idx = self.text.index("accepted = store.submit_negative(run_id, reason_code)")
        mirror_idx = self.text.index("mirror_negative_feedback(")
        ux_idx = self.text.index('await query.answer(text="已收到，謝謝您的回饋！")', mirror_idx)
        self.assertLess(storage_idx, mirror_idx)
        self.assertLess(mirror_idx, ux_idx)
        self.assertIn(
            'logger.debug("[%s] v2 feedback mirror (negative) failed", self.name, exc_info=True)', self.text
        )

    def test_suggestion_mirror_between_legacy_storage_and_telegram_ux(self):
        storage_idx = self.text.index("accepted = store.submit_suggestion(")
        mirror_idx = self.text.index("mirror_suggestion(")
        ux_idx = self.text.index('await msg.reply_text("感謝您的建議！")')
        self.assertLess(storage_idx, mirror_idx)
        self.assertLess(mirror_idx, ux_idx)
        self.assertIn(
            'logger.debug("[%s] v2 feedback mirror (suggestion) failed", self.name, exc_info=True)', self.text
        )

    def test_suggestion_mirror_gated_on_legacy_accepted(self):
        # Must not attempt the v2 mirror when the legacy submit_suggestion
        # call itself lost the first-write-wins race.
        window = self._window_around("mirror_suggestion(", before=120, after=20)
        self.assertIn("if accepted:", window)

    def test_mirror_calls_are_not_nested_inside_telegram_ux_try_blocks(self):
        # Each mirror call must have its own try/except, never share one
        # with the Telegram UX call -- otherwise a UX exception could skip
        # the mirror attempt (or vice versa) depending on ordering inside
        # a single shared block.
        for marker in ("mirror_positive_feedback(", "mirror_negative_feedback(", "mirror_suggestion("):
            idx = self.text.index(marker)
            # The nearest preceding `try:` must belong to the mirror call
            # itself: the line immediately above it (ignoring blank/comment
            # lines) is `try:`.
            preceding = self.text[:idx]
            lines = [ln for ln in preceding.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            self.assertEqual(lines[-1].strip(), "try:")


class AuthorizationUnchangedTests(_SourceTestCase):
    """Phase 3B must not weaken or bypass any existing authorization
    check -- these markers must all still be present and untouched."""

    def test_owner_authorization_check_present(self):
        self.assertIn('await query.answer(text="⛔ 只有原使用者可以提交這筆回饋。")', self.text)

    def test_generic_callback_user_authorization_still_called(self):
        self.assertIn("self._is_callback_user_authorized(", self.text)

    def test_universal_feedback_authorization_helper_untouched_signature(self):
        self.assertIn("async def _authorize_universal_feedback_run(", self.text)
        self.assertIn('stored_message_id = row["feedback_message_id"]', self.text)

    def test_suggestion_reply_binding_checks_present(self):
        self.assertIn("async def _try_handle_feedback_suggestion_reply(self, msg: Message) -> bool:", self.text)
        self.assertIn(
            "Feedback suggestion reply binding mismatch for run %s", self.text
        )


class LegacyUxUnchangedTests(_SourceTestCase):
    def test_legacy_button_labels_unchanged(self):
        self.assertIn('InlineKeyboardButton("👍 有幫助", callback_data=f"fb:h:{run_id}")', self.text)
        self.assertIn('InlineKeyboardButton("👎 需改善", callback_data=f"fb:u:{run_id}")', self.text)

    def test_legacy_reason_prompt_unchanged(self):
        self.assertIn("請選擇這則回答需要改善的原因：", self.text)

    def test_legacy_suggestion_prompt_unchanged(self):
        self.assertIn("可直接回覆此訊息補充建議（選填）。", self.text)


if __name__ == "__main__":
    unittest.main()
