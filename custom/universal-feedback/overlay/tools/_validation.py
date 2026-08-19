"""Small, stateless validation primitives shared across the feedback/case/
reflector domain modules -- previously duplicated verbatim in each module
that needed them (case_enrichment, case_reflection_input, case_routing,
feedback_store_v2, proposal_matching, reflector_prompt_context,
reflector_proposals). Keep this module free of domain knowledge: only
primitive-value checks belong here.
"""
from __future__ import annotations

import math
from typing import Any


def is_valid_confidence(value: Any) -> bool:
    """True only for a real, finite number in [0.0, 1.0].

    ``bool`` is a subclass of ``int`` in Python, so it is rejected before
    the int check -- otherwise ``True`` would silently pass as
    confidence=1. The non-standard JSON tokens NaN/Infinity/-Infinity
    (which Python's ``json`` module accepts by default) are rejected
    explicitly too, since a NaN would otherwise silently fail the
    ``0.0 <= value <= 1.0`` range check for the wrong reason.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return 0.0 <= value <= 1.0


def require_nonblank_str(value: Any, field_name: str) -> None:
    """Raise ValueError unless value is a str with non-whitespace content."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string, got {value!r}")
