"""SQLite storage for universal and legacy Telegram feedback."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.feedback_callbacks import REASON_CODES, is_valid_reason_code
from tools.universal_feedback import safe_feedback_text

DEFAULT_PATH = Path("/sandbox/.hermes/data/support_feedback.db")
MAX_SUGGESTION_TEXT_LENGTH = 1000
_REASON_CODE_PLACEHOLDERS = ",".join("?" for _ in REASON_CODES)
BASE_COLUMNS = {
    "run_id": "TEXT PRIMARY KEY", "chat_id": "TEXT NOT NULL", "resolved": "INTEGER",
    "created_at": "TEXT NOT NULL", "submitted_at": "TEXT",
    "turn_key": "TEXT", "telegram_user_id": "TEXT", "session_id": "TEXT",
    "user_message_id": "TEXT", "assistant_message_id": "TEXT", "feedback_message_id": "TEXT",
    "question_text": "TEXT", "answer_text": "TEXT", "helpful": "INTEGER",
    "reason_code": "TEXT", "suggestion_text": "TEXT", "feedback_ui_mode": "TEXT",
    "feedback_send_status": "TEXT", "feedback_trigger_reason": "TEXT",
    "foundry_iq_attempted": "INTEGER", "foundry_iq_ok": "INTEGER",
    "foundry_iq_metadata_json": "TEXT", "feedback_policy_version": "TEXT",
    "feedback_schema_version": "TEXT",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_callback_data(data: str):
    parts = str(data or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "fb" or parts[1] not in {"h", "u", "y", "n"} or not parts[2]:
        return None
    return parts[2], parts[1] in {"h", "y"}


class FeedbackStore:
    def __init__(self, path: Path | str = DEFAULT_PATH, *, migrate: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if migrate:
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Idempotently migrate feedback_runs without converting resolved."""
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS feedback_runs (run_id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, resolved INTEGER, created_at TEXT NOT NULL, submitted_at TEXT)")
            existing = {row[1] for row in db.execute("PRAGMA table_info(feedback_runs)")}
            for name, spec in BASE_COLUMNS.items():
                if name not in existing and name != "run_id":
                    db.execute(f"ALTER TABLE feedback_runs ADD COLUMN {name} {spec}")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_runs_turn_key ON feedback_runs(turn_key) WHERE turn_key IS NOT NULL")

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def create_run(self, run_id: str, chat_id: str, **fields: Any) -> bool:
        names = ["run_id", "chat_id", "created_at"]
        values = [str(run_id), str(chat_id), _now()]
        for name in BASE_COLUMNS:
            if name not in names and name in fields:
                value = fields[name]
                if name in {"question_text", "answer_text"}:
                    value = safe_feedback_text(value)
                names.append(name); values.append(value)
        placeholders = ",".join("?" for _ in names)
        with self._connect() as db:
            try:
                cur = db.execute(f"INSERT INTO feedback_runs ({','.join(names)}) VALUES ({placeholders})", values)
                return cur.rowcount == 1
            except sqlite3.IntegrityError:
                return False

    def get(self, run_id: str):
        with self._connect() as db:
            return db.execute("SELECT * FROM feedback_runs WHERE run_id = ?", (run_id,)).fetchone()

    def get_by_turn_key(self, key: str):
        with self._connect() as db:
            return db.execute("SELECT * FROM feedback_runs WHERE turn_key = ?", (key,)).fetchone()

    def get_by_feedback_message_id(self, chat_id: str, feedback_message_id: str):
        """Look up the row for a (chat_id, feedback_message_id) pair.

        Telegram message_id values are unique only within a single chat, so
        feedback_message_id alone is not a safe lookup key — both fields are
        required together. No DB constraint currently enforces uniqueness on
        this pair, so if more than one row somehow matches, fail closed and
        return None rather than arbitrarily picking one: an ambiguous
        binding must never be treated as a valid authorization target.
        """
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM feedback_runs WHERE chat_id = ? AND feedback_message_id = ?",
                (str(chat_id), str(feedback_message_id)),
            ).fetchall()
        if len(rows) != 1:
            return None
        return rows[0]

    def mark_send(self, run_id: str, *, status: str, feedback_message_id: str | None = None, assistant_message_id: str | None = None, ui_mode: str | None = None) -> bool:
        with self._connect() as db:
            cur = db.execute("UPDATE feedback_runs SET feedback_send_status=?, feedback_message_id=?, assistant_message_id=?, feedback_ui_mode=? WHERE run_id=?", (status, feedback_message_id, assistant_message_id, ui_mode, run_id))
            return cur.rowcount == 1

    def submit_helpful(self, run_id: str, helpful: bool) -> bool:
        with self._connect() as db:
            cur = db.execute("UPDATE feedback_runs SET helpful=?, submitted_at=? WHERE run_id=? AND submitted_at IS NULL", (int(bool(helpful)), _now(), run_id))
            return cur.rowcount == 1

    def submit(self, run_id: str, resolved: bool) -> bool:
        """Legacy /feedback_test compatibility; does not write helpful."""
        with self._connect() as db:
            cur = db.execute("UPDATE feedback_runs SET resolved=?, submitted_at=? WHERE run_id=? AND submitted_at IS NULL", (int(resolved), _now(), run_id))
            return cur.rowcount == 1

    def submit_negative(self, run_id: str, reason_code: str) -> bool:
        """Atomically finalize a negative-feedback row with its reason.

        Re-validates reason_code itself rather than trusting an
        already-parsed callback, and requires helpful/reason_code/
        submitted_at to all still be NULL so a prior positive or negative
        submission for this run_id can never be overwritten.
        """
        if not is_valid_reason_code(reason_code):
            return False
        with self._connect() as db:
            cur = db.execute(
                "UPDATE feedback_runs SET helpful=0, reason_code=?, submitted_at=? "
                "WHERE run_id=? AND helpful IS NULL AND reason_code IS NULL AND submitted_at IS NULL",
                (reason_code, _now(), run_id),
            )
            return cur.rowcount == 1

    def submit_suggestion(
        self,
        run_id: str,
        chat_id: str,
        telegram_user_id: str,
        feedback_message_id: str,
        suggestion_text: str,
    ) -> bool:
        """Atomically attach an optional suggestion to an already-finalized
        negative-feedback row.

        Re-validates the full chat/owner/message binding and eligibility
        (helpful=0, legal reason_code, already submitted) inside the same
        atomic UPDATE rather than trusting the caller's own authorization
        pass, and requires suggestion_text to still be NULL so a prior
        suggestion can never be overwritten. Rejects non-string, blank, or
        over-length input outright — it is never silently truncated.
        """
        if not isinstance(suggestion_text, str):
            return False
        text = suggestion_text.strip()
        if not text or len(text) > MAX_SUGGESTION_TEXT_LENGTH:
            return False
        text = safe_feedback_text(text)
        with self._connect() as db:
            cur = db.execute(
                "UPDATE feedback_runs SET suggestion_text=? "
                "WHERE run_id=? AND chat_id=? AND telegram_user_id=? AND feedback_message_id=? "
                f"AND helpful=0 AND reason_code IN ({_REASON_CODE_PLACEHOLDERS}) "
                "AND submitted_at IS NOT NULL AND suggestion_text IS NULL",
                (
                    text,
                    str(run_id),
                    str(chat_id),
                    str(telegram_user_id),
                    str(feedback_message_id),
                    *REASON_CODES,
                ),
            )
            return cur.rowcount == 1
