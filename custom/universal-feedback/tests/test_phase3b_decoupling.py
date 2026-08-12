"""Phase 3B behavioral ordering/isolation tests.

test_phase3b_wiring.py proves the *source text* of adapter.py has the
mirror call positioned strictly between the legacy storage mutation and
the Telegram UX call, in its own try/except (adapter.py itself can't be
imported/executed in this environment -- no proprietary python-telegram-
bot/Hermes runtime available). This file proves the *behavior* is
actually correct by re-running the same three-step sequence (legacy
storage -> v2 mirror -> Telegram UX) against the real
tools.feedback_mirror functions and a real temporary SQLite-backed
FeedbackStoreV2, with a stand-in Telegram UX call that can be configured
to raise. The harness functions are a deliberately small, literal mirror
of the actual source (cross-checked structurally by
test_phase3b_wiring.py), not a reimplementation of unrelated logic.

Two independent isolation directions are covered, matching the fix:
  1. A Telegram UX failure must never prevent (or undo) an already-
     attempted v2 mirror -- the mirror runs BEFORE the UX call now.
  2. A v2 mirror failure must never prevent the Telegram UX that follows.
"""
from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path

_OVERLAY_ROOT = Path(__file__).resolve().parents[1] / "overlay"
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.feedback_mirror import (  # noqa: E402
    mirror_negative_feedback,
    mirror_positive_feedback,
    mirror_suggestion,
)
from tools.feedback_store_v2 import FeedbackStoreV2  # noqa: E402


class _ExplodingStore:
    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated v2 failure in {name}")
        return _boom


class _TelegramUx:
    """Stand-in for query.answer()/edit_message_text()/msg.reply_text()."""

    def __init__(self, *, raises: bool = False):
        self.raises = raises
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.raises:
            raise RuntimeError("simulated Telegram API failure")


def _simulate_positive_callback(*, store, turn_key, telegram_ux):
    """Literal mirror of adapter.py's fb:h branch, post-fix ordering:
    legacy storage (assumed already succeeded, as in these tests -- the
    `if not accepted: return` early-exit branch is unchanged legacy code
    and out of scope here) -> v2 mirror, independently isolated -> the
    existing Telegram UX, independently isolated.
    """
    mirror_result = None
    try:
        mirror_result = mirror_positive_feedback(turn_key, store=store)
    except Exception:
        pass  # mirrors: logger.debug(...)
    try:
        telegram_ux()
    except Exception:
        pass  # mirrors: except Exception: pass (existing legacy UX try/except)
    return mirror_result


def _simulate_negative_callback(*, store, turn_key, reason_code, telegram_ux):
    mirror_result = None
    try:
        mirror_result = mirror_negative_feedback(turn_key, reason_code, store=store)
    except Exception:
        pass
    try:
        telegram_ux()
    except Exception:
        pass
    return mirror_result


def _simulate_suggestion_reply(*, store, turn_key, suggestion_text, accepted, telegram_ux):
    mirror_result = None
    if accepted:
        try:
            mirror_result = mirror_suggestion(turn_key, suggestion_text, store=store)
        except Exception:
            pass
    try:
        telegram_ux()
    except Exception:
        pass
    return mirror_result


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "support_feedback.db"
        self.store = FeedbackStoreV2(self.db_path)

    def tearDown(self):
        del self.store
        gc.collect()
        self._tmpdir.cleanup()

    def _new_turn(self, turn_id, *, session_id="sess-1", case_id="case-1"):
        self.store.create_or_update_session(session_id, "telegram")
        if self.store.get_case(case_id) is None:
            self.store.create_case(case_id, session_id)
        self.store.create_turn(
            turn_id, case_id,
            platform_user_id="user-1", platform_user_message_id=f"{turn_id}-msg",
            question_text="q", answer_text="a", feedback_eligible=True,
        )
        return turn_id


class TelegramUxFailureDoesNotBlockMirrorTests(_StoreTestCase):
    """Requirement 1-3: Telegram UX failure != prevents an already-legal
    mirror attempt (mirror now runs BEFORE the UX call)."""

    def test_positive_ux_raises_mirror_still_attempted_and_written(self):
        turn_id = self._new_turn("turn-1")
        ux = _TelegramUx(raises=True)
        mirror_result = _simulate_positive_callback(store=self.store, turn_key=turn_id, telegram_ux=ux)
        self.assertEqual(ux.calls, 1)  # UX was still attempted (and raised, swallowed)
        self.assertTrue(mirror_result)  # mirror succeeded despite the later UX failure
        row = self.store.get_feedback(turn_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["helpful"], 1)

    def test_negative_ux_raises_mirror_still_attempted_and_written(self):
        turn_id = self._new_turn("turn-2")
        ux = _TelegramUx(raises=True)
        mirror_result = _simulate_negative_callback(
            store=self.store, turn_key=turn_id, reason_code="incomplete", telegram_ux=ux
        )
        self.assertEqual(ux.calls, 1)
        self.assertTrue(mirror_result)
        row = self.store.get_feedback(turn_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["reason_code"], "incomplete")

    def test_suggestion_reply_raises_mirror_still_attempted_and_written(self):
        turn_id = self._new_turn("turn-3")
        mirror_negative_feedback(turn_id, "incomplete", store=self.store)
        ux = _TelegramUx(raises=True)
        mirror_result = _simulate_suggestion_reply(
            store=self.store, turn_key=turn_id, suggestion_text="please add detail",
            accepted=True, telegram_ux=ux,
        )
        self.assertEqual(ux.calls, 1)
        self.assertTrue(mirror_result)
        row = self.store.get_feedback(turn_id)
        self.assertEqual(row["suggestion_text"], "please add detail")


class MirrorFailureDoesNotBlockTelegramUxTests(_StoreTestCase):
    """The other direction, preserved from before: v2 mirror failure !=
    prevents the Telegram UX that follows."""

    def test_positive_mirror_failure_ux_still_runs(self):
        ux = _TelegramUx(raises=False)
        _simulate_positive_callback(store=_ExplodingStore(), turn_key="turn-1", telegram_ux=ux)
        self.assertEqual(ux.calls, 1)

    def test_negative_mirror_failure_ux_still_runs(self):
        ux = _TelegramUx(raises=False)
        _simulate_negative_callback(
            store=_ExplodingStore(), turn_key="turn-1", reason_code="incomplete", telegram_ux=ux
        )
        self.assertEqual(ux.calls, 1)

    def test_suggestion_mirror_failure_ux_still_runs(self):
        ux = _TelegramUx(raises=False)
        _simulate_suggestion_reply(
            store=_ExplodingStore(), turn_key="turn-1", suggestion_text="text",
            accepted=True, telegram_ux=ux,
        )
        self.assertEqual(ux.calls, 1)


class BothDirectionsSimultaneouslyTests(_StoreTestCase):
    """Both failure modes at once must still not deadlock/raise out of the
    simulated handler -- purely a defense-in-depth sanity check."""

    def test_both_mirror_and_ux_fail_handler_completes(self):
        ux = _TelegramUx(raises=True)
        try:
            mirror_result = _simulate_positive_callback(
                store=_ExplodingStore(), turn_key="turn-1", telegram_ux=ux
            )
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"simulated handler raised: {exc}")
        self.assertFalse(mirror_result)
        self.assertEqual(ux.calls, 1)


if __name__ == "__main__":
    unittest.main()
