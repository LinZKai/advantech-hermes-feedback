"""Pure helpers for staged universal Telegram message-level feedback."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

POLICY_VERSION = "universal-message-v1"
SCHEMA_VERSION = "v1"
FOUNDARY_SCRIPT = "/sandbox/hermes-support-config/skills/foundry-iq/scripts/query_foundry_iq.py"
MAX_FEEDBACK_TEXT_LENGTH = 20000


def turn_key(chat_id: Any, user_message_id: Any) -> str:
    return f"telegram:{chat_id}:{user_message_id}"


def is_excluded_input(text: str, message_type: Any = None) -> bool:
    value = str(text or "").strip().lower()
    if value in {"/start", "/help", "/feedback_test"}:
        return True
    name = getattr(message_type, "name", str(message_type or "")).upper()
    return name in {"COMMAND", "SYSTEM", "INTERNAL"}


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(query\s*key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(token\s*=\s*)[^\s,;]+"),
    re.compile(r"(?i)(api_key\s*=\s*)[^\s,;]+"),
    re.compile(r"(?i)(foundry_iq_query_key\s*=\s*)[^\s,;]+"),
    re.compile(r"(?i)(telegram_bot_token\s*=\s*)[^\s,;]+"),
)


def safe_feedback_text(value: Any) -> str:
    """Redact common credentials, then bound persisted user-visible text."""
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text[:MAX_FEEDBACK_TEXT_LENGTH]


def feedback_eligible(*, platform: Any, text: str, message_type: Any,
                      response: Mapping[str, Any], stream_task_ok: bool,
                      final_content_delivered: bool) -> bool:
    platform_name = str(getattr(platform, "value", platform) or "").lower()
    if platform_name != "telegram" or is_excluded_input(text, message_type):
        return False
    if not response or response.get("failed") or response.get("partial"):
        return False
    if response.get("interrupted") or response.get("error"):
        return False
    if not str(response.get("final_response") or "").strip():
        return False
    return bool(stream_task_ok and final_content_delivered)


def claim_feedback_ownership(response: Mapping[str, Any], event: Any = None) -> bool:
    """Claim one turn across response/event objects without shared global state."""
    already = bool(response.get("universal_feedback_handled"))
    if event is not None:
        already = already or bool(getattr(event, "universal_feedback_handled", False))
    if already:
        return False
    if isinstance(response, dict):
        response["universal_feedback_handled"] = True
    if event is not None:
        try:
            setattr(event, "universal_feedback_handled", True)
        except Exception:
            return False
    return True


def foundry_marker(tool_name: Any, arguments: Any) -> bool:
    if str(tool_name or "") != "terminal":
        return False
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(args, dict) and FOUNDARY_SCRIPT in str(args.get("command") or "")
