"""The Curator (Improvement-Proposal -> AGENTS.md) proposed-change domain
contract -- Slice 1.

    accepted, agent_behavior ImprovementProposal
     + its latest ProposalObservation
     + supporting Case evidence
     + current /sandbox/AGENTS.md content
     |
     v
    CuratorPromptContext        (tools.curator_prompt_context)
     |
     v
    [ Curator LLM call ]        (tools.curator_analyzer)
     |
     v
    CuratorChange                (this module)
     |
     v
    structural validation        (this dataclass's own __post_init__)
     |
     v
    curator_changes               (tools.feedback_store_v2)

Naming note -- read this before wiring anything up: this is NOT the same
"Curator" as `agent.curator.maybe_run_curator()`, a pre-existing, unrelated
weekly Skill-maintenance background job that already exists inside the
built Hermes sandbox image (wired from custom/universal-feedback/overlay/
gateway/run.py). This module and its siblings (tools.curator_analyzer,
tools.curator_prompt_context, tools.run_curator, tools.review_proposal)
implement a different thing entirely: turning one human-accepted
improvement_proposals row into one reviewable, NOT-YET-APPLIED, proposed
change to /sandbox/AGENTS.md. Nothing here imports or calls agent.curator,
and nothing there calls back into any module in this family.

Self-improvement boundary -- the same statement tools.reflector_proposals
makes for ImprovementProposal, restated here because CuratorChange is the
step in this whole domain closest to an actual file edit: producing a
CuratorChange is NEVER itself an edit. No function in this module, or in
tools.curator_analyzer/tools.curator_prompt_context/tools.run_curator/
tools.review_proposal, writes to /sandbox/AGENTS.md or any other file. A
CuratorChange is a recommendation persisted with status='proposed' --
reviewing it (-> approved/rejected) and applying it (-> applied/failed)
are both explicitly OUT OF SCOPE for this slice and are not implemented
anywhere in this codebase yet.
"""
from __future__ import annotations

from dataclasses import dataclass

from tools._validation import is_valid_confidence as _is_valid_confidence
from tools._validation import require_nonblank_str as _require_nonblank_str
from tools.feedback_store_v2 import CHANGE_TYPE_VALUES, CURATOR_CHANGE_STATUS_VALUES

_CHANGE_TYPE_VALUE_SET = frozenset(CHANGE_TYPE_VALUES)
_CURATOR_CHANGE_STATUS_VALUE_SET = frozenset(CURATOR_CHANGE_STATUS_VALUES)

# The ONLY file Curator v1 may target -- never derived, never LLM-chosen.
# A CuratorChange whose target_file does not equal this constant fails
# __post_init__ below; tools.run_curator's deterministic guards re-check
# this again defensively before persisting (see that module's own
# docstring for why -- belt-and-suspenders, matching this codebase's
# existing convention of never trusting an upstream layer alone).
CURATOR_TARGET_FILE = "/sandbox/AGENTS.md"

# change_type values that structurally REQUIRE non-blank proposed_content
# -- every value except 'no_change_recommended' (a real change always
# needs the actual replacement content; 'no_change_recommended' is the one
# change_type that legitimately means "AGENTS.md is not the right lever
# for this Proposal").
_CHANGE_TYPES_REQUIRING_PROPOSED_CONTENT = frozenset(CHANGE_TYPE_VALUES) - {"no_change_recommended"}


@dataclass(frozen=True)
class CuratorChange:
    """One Curator run's complete, already-validated proposed change.

    Unlike the Reflector's three-dataclass domain (ImprovementProposal +
    ProposalObservation + ReflectionResult), Curator v1 produces exactly
    ONE change per run against exactly ONE target_file, so a single flat
    dataclass mirrors both the LLM's structured output contract AND the
    curator_changes row it becomes -- there is no separate "batch result"
    shape to model.

    proposed_content is the COMPLETE replacement content of target_file,
    never a diff/patch -- target_file (AGENTS.md) is small by design for
    this slice, so a full-content replacement is the deliberate
    simplification; a future "show a diff" step, if ever built, derives
    one deterministically from before_content/proposed_content in Python,
    never stored separately here. proposed_content is required (non-blank)
    for every change_type except 'no_change_recommended', where it MUST be
    None -- never the unchanged current content re-typed back, so a
    reviewer can distinguish "Curator explicitly recommends no change"
    from "Curator produced a change that happens to equal the original"
    (the latter would be a real add_rule/modify_rule/remove_rule; two
    different outcomes must never look identical in storage).

    before_content is always recorded, even for no_change_recommended --
    a reviewer must be able to see exactly what content the run was given,
    without cross-referencing anything else.

    status is always 'proposed' at construction -- see this module's own
    docstring, "Self-improvement boundary": producing a CuratorChange is
    never itself an edit, and review/apply are out of scope for this
    slice. Structurally, this dataclass allows constructing any of the
    five status values (e.g. to represent an already-reviewed row freshly
    read back from storage), but nothing in tools.curator_domain/
    tools.curator_analyzer/tools.run_curator ever sets one other than
    'proposed'.
    """

    change_id: str
    proposal_id: str
    target_file: str
    change_type: str
    rationale: str
    before_content: str
    proposed_content: str | None
    expected_effect: str | None
    confidence: float
    status: str
    created_at: str
    reviewed_at: str | None = None
    applied_at: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank_str(self.change_id, "change_id")
        _require_nonblank_str(self.proposal_id, "proposal_id")

        if self.target_file != CURATOR_TARGET_FILE:
            raise ValueError(
                f"invalid target_file: {self.target_file!r}; Curator v1 may only "
                f"target {CURATOR_TARGET_FILE!r}"
            )

        if self.change_type not in _CHANGE_TYPE_VALUE_SET:
            raise ValueError(f"invalid change_type: {self.change_type!r}")

        _require_nonblank_str(self.rationale, "rationale")
        _require_nonblank_str(self.before_content, "before_content")

        if self.change_type in _CHANGE_TYPES_REQUIRING_PROPOSED_CONTENT:
            if not isinstance(self.proposed_content, str) or not self.proposed_content.strip():
                raise ValueError(
                    f"proposed_content is required (non-blank) when change_type={self.change_type!r}"
                )
        elif self.proposed_content is not None:
            raise ValueError("proposed_content must be null when change_type='no_change_recommended'")

        if self.expected_effect is not None and (
            not isinstance(self.expected_effect, str) or not self.expected_effect.strip()
        ):
            raise ValueError("expected_effect must be a non-blank string or null")

        if not _is_valid_confidence(self.confidence):
            raise ValueError(f"invalid confidence: {self.confidence!r}")

        if self.status not in _CURATOR_CHANGE_STATUS_VALUE_SET:
            raise ValueError(f"invalid status: {self.status!r}")

        _require_nonblank_str(self.created_at, "created_at")

        if self.reviewed_at is not None and not isinstance(self.reviewed_at, str):
            raise ValueError("reviewed_at must be a string or None")
        if self.applied_at is not None and not isinstance(self.applied_at, str):
            raise ValueError("applied_at must be a string or None")


__all__ = [
    "CURATOR_TARGET_FILE",
    "CuratorChange",
]
