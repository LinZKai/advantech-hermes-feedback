"""SQLite storage for universal and legacy Telegram feedback."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.universal_feedback import safe_feedback_text

DEFAULT_PATH = Path("/sandbox/.hermes/data/support_feedback.db")
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
