"""Curator Slice 1 runner: accepted Improvement Proposal -> Curator ->
one persisted, reviewable curator_changes row (status='proposed').

    FeedbackStoreV2.get_improvement_proposal(proposal_id)
     |
     v
    deterministic guards               (proposal exists, review_status==
     |                                   'accepted', improvement_target==
     |                                   'agent_behavior', target_file
     |                                   readable -- see this module's own
     |                                   docstring, "Deterministic guards")
     v
    FeedbackStoreV2.get_latest_proposal_observation(proposal_id)
     |
     v
    build_case_intelligence_for_ids()  (tools.case_reflection_input,
     |                                   unchanged)
     v
    build_curator_prompt_context()     (tools.curator_prompt_context,
     |                                   unchanged)
     v
    analyzer(context, ...)              (THE analyzer boundary -- see
     |                                    below; defaults to tools.
     |                                    curator_analyzer.
     |                                    analyze_proposal_with_llm)
     v
    CuratorChange (tools.curator_domain)
     |
     v
    post-LLM guards                     (target_file re-checked; see
     |                                    "Deterministic guards" below)
     v
    FeedbackStoreV2.create_curator_change()
     |
     v
    CuratorRunOutcome (human-readable summary on stdout)
     |
     v
    exit

Usage:
    python3 tools/run_curator.py --proposal-id <proposal_id> [--db PATH]
        [--agents-file PATH]

Not a scheduler: manual execution only, one run against one already-
accepted Proposal, then exit. This slice never applies a change -- see
this module's own "Deterministic guards" section and tools.curator_domain's
module docstring, "Self-improvement boundary".

Analyzer boundary: a run's CuratorChange is produced by any `Analyzer`
callable, injected via the `analyzer` parameter on run_curator()/main(),
matching tools.run_reflector's own Analyzer-injection shape exactly.
Unlike run_reflector(), `analyzer` here has no default and must always be
supplied explicitly by a caller other than main() -- main() is the only
place that binds the real analyzer (tools.curator_analyzer.
analyze_proposal_with_llm, via functools.partial with a resolved
main_runtime) when no analyzer override is given.

Deterministic guards -- checked BEFORE the analyzer is ever called, so an
ineligible Proposal produces no LLM call and no DB write of any kind:
  * the proposal_id exists (get_improvement_proposal)
  * review_status == 'accepted' (never 'pending'/'rejected' -- Curator
    NEVER accepts a Proposal itself; see tools.feedback_store_v2's own
    "review_status is a HUMAN lifecycle field" boundary statement, and
    tools.review_proposal for the human-facing accept/reject helper)
  * improvement_target == 'agent_behavior' (Curator v1 handles no other
    target -- a 'knowledge'/'retrieval'/'workflow'/'other' Proposal is not
    something AGENTS.md can address)
  * a latest ProposalObservation exists for this proposal_id (Curator
    needs the actual pattern_summary/recommended_improvement narrative;
    an accepted Proposal with no Observation is a structurally-impossible
    state this module still defends against rather than assuming)
  * `agents_file` exists and is readable

Checked AFTER the analyzer returns, before persisting -- see run_curator()
below for exactly what tools.curator_analyzer.parse_curator_output() (via
tools.curator_domain.CuratorChange.__post_init__) already enforces, and
what this module re-checks defensively on top of that:
  * change.target_file exactly equals tools.curator_domain.
    CURATOR_TARGET_FILE (re-checked here too -- belt-and-suspenders,
    matching tools.feedback_store_v2's own "never trust an upstream layer
    alone" convention)
  * change_type is one of the taxonomy's allowed values (enforced by
    CuratorChange.__post_init__, which a malformed LLM response cannot
    bypass -- parse_curator_output() returns None instead)
  * confidence is a valid 0.0-1.0 number (same enforcement path)
  * proposed_content is non-empty exactly when change_type requires it,
    and exactly None for no_change_recommended (same enforcement path)

Any guard failure -- before or after the analyzer -- fails closed: no
curator_changes row is ever written for a run that did not fully pass
every guard. no_change_recommended is NOT a guard failure: it is a valid,
fully-persisted CuratorChange with change_type='no_change_recommended',
reported via a status distinct from a real change (see CuratorRunOutcome's
own docstring) so a human reviewer, or this module's own stdout summary,
never confuses "Curator ran and found nothing to change" with "Curator
failed to run."
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_reflection_input import build_case_intelligence_for_ids  # noqa: E402
from tools.curator_analyzer import (  # noqa: E402
    CuratorAnalyzerError,
    analyze_proposal_with_llm,
)
from tools.curator_domain import CURATOR_TARGET_FILE, CuratorChange  # noqa: E402
from tools.curator_prompt_context import CuratorPromptContext, build_curator_prompt_context  # noqa: E402
from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402

# (context, *, proposal_id, created_at) -> CuratorChange, raises
# CuratorAnalyzerError. Matches tools.curator_analyzer.analyze_proposal_
# with_llm's own signature shape (minus main_runtime/llm_call/id_factory,
# which are the REAL analyzer's own concerns -- a fake analyzer injected
# for a test needs none of them, only this reduced shape).
Analyzer = Callable[..., CuratorChange]


# ---------------------------------------------------------------------------
# Runtime config resolution -- the ONE place config.yaml is read
# ---------------------------------------------------------------------------

_REQUIRED_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key")


def _resolve_main_runtime() -> dict[str, Any]:
    """Read provider/model/base_url/api_key from Hermes' own config.yaml
    -- deliberately a separate copy from tools.run_reflector's own
    equivalent, not a cross-module import (this domain never imports
    another module's private helper across a file boundary). Curator must
    analyze with the SAME provider/model the main Hermes agent is already
    configured with -- never a hard-coded model, never a silent fallback
    to a different one.

    Lazy-imports hermes_cli.config so this module stays importable in
    this repo's test environment.

    Fails closed: raises RuntimeError if hermes_cli.config cannot be
    imported, or if config.yaml has no 'model' section, or if any of
    provider/model/base_url/api_key is missing or blank.
    """
    try:
        from hermes_cli.config import load_config
    except ImportError as exc:
        raise RuntimeError(
            "hermes_cli.config is not importable -- run_curator.py must run "
            "inside a Hermes sandbox to resolve the real LLM runtime configuration"
        ) from exc

    cfg = load_config() or {}
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise RuntimeError("config.yaml has no 'model' section; cannot resolve main_runtime")

    main_runtime: dict[str, Any] = {
        "provider": model_cfg.get("provider"),
        "model": model_cfg.get("default"),
        "base_url": model_cfg.get("base_url"),
        "api_key": model_cfg.get("api_key"),
    }
    missing = [field for field in _REQUIRED_MAIN_RUNTIME_FIELDS if not main_runtime.get(field)]
    if missing:
        raise RuntimeError(
            "config.yaml model section is missing required field(s): "
            f"{', '.join(missing)}; refusing to fall back to a different provider/model"
        )
    return main_runtime


def resolve_default_analyzer() -> Analyzer:
    """Resolve the real LLM-backed analyzer -- `_resolve_main_runtime()`
    bound via functools.partial to tools.curator_analyzer.analyze_
    proposal_with_llm. This is the exact same resolution main() has always
    performed for the CLI entrypoint, factored out so dashboard_api.app can
    reuse it verbatim (to auto-invoke Curator immediately after a Proposal
    Accept) without duplicating this domain's own runtime-config logic in
    the FastAPI layer.

    Raises RuntimeError (propagated from _resolve_main_runtime()) if
    hermes_cli.config is not importable, or config.yaml's model section is
    missing/incomplete -- never falls back to a different provider/model.
    Never raises for any other reason; callers needing a non-raising
    outcome (e.g. dashboard_api.app, which must fold this into a
    CuratorRunOutcome-shaped response rather than a 500) catch RuntimeError
    themselves.
    """
    main_runtime = _resolve_main_runtime()
    return functools.partial(analyze_proposal_with_llm, main_runtime=main_runtime)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CuratorRunOutcome:
    """One run_curator() call's outcome.

    `status` is exactly one of:
      * "proposal_not_found"    -- no improvement_proposals row for the
        given proposal_id; no LLM call, no DB write.
      * "not_accepted"          -- review_status is not 'accepted' (still
        'pending', or already 'rejected'); no LLM call, no DB write.
      * "wrong_target"          -- improvement_target is not
        'agent_behavior'; no LLM call, no DB write.
      * "no_observation"        -- the Proposal has never been observed
        (no proposal_observations row); no LLM call, no DB write.
      * "agents_file_unreadable" -- `agents_file` does not exist or could
        not be read; no LLM call, no DB write.
      * "input_failed"          -- an exception while resolving supporting
        Case evidence or building CuratorPromptContext; no LLM call, no
        DB write.
      * "analyzer_failed"        -- the analyzer raised
        CuratorAnalyzerError, OR returned a CuratorChange whose
        target_file failed this module's own post-LLM guard; no DB write.
      * "persistence_failed"     -- FeedbackStoreV2.create_curator_change()
        returned False for an already-validated CuratorChange (e.g. a
        change_id collision); should not occur in practice with the
        default uuid4 id_factory.
      * "succeeded"               -- a real change (add_rule/modify_rule/
        remove_rule) was persisted with status='proposed'.
      * "no_change_recommended"   -- Curator judged CURATOR_TARGET_FILE
        (/sandbox/AGENTS.md) not to be an appropriate lever for this
        Proposal; ALSO persisted
        with status='proposed' (change_type='no_change_recommended',
        proposed_content=None) -- this is a fully successful run, not a
        failure, reported as its own status so it is never confused with
        "succeeded" (a real change) or with any guard-failure status
        above.

    Deliberately a plain dataclass with a `status` string, not a runner-
    specific exception hierarchy -- run_curator() itself never raises for
    any of the above; a caller branches on `.status`.
    """

    status: str
    proposal_id: str | None = None
    change_id: str | None = None
    change_type: str | None = None
    target_file: str | None = None
    confidence: float | None = None
    error: str | None = None


def run_curator(
    store: FeedbackStoreV2,
    *,
    proposal_id: str,
    agents_file: Path,
    analyzer: Analyzer,
) -> CuratorRunOutcome:
    """The reusable pipeline body for one Curator run, separated from CLI/
    argv handling so main() (or a test) can call it directly with a
    different analyzer. See this module's own docstring for the full
    guard sequence.
    """
    proposal = store.get_improvement_proposal(proposal_id)
    if proposal is None:
        return CuratorRunOutcome(status="proposal_not_found", proposal_id=proposal_id)

    if proposal["review_status"] != "accepted":
        return CuratorRunOutcome(
            status="not_accepted", proposal_id=proposal_id,
            error=f"review_status={proposal['review_status']!r}, expected 'accepted'",
        )

    if proposal["improvement_target"] != "agent_behavior":
        return CuratorRunOutcome(
            status="wrong_target", proposal_id=proposal_id,
            error=f"improvement_target={proposal['improvement_target']!r}, expected 'agent_behavior'",
        )

    observation = store.get_latest_proposal_observation(proposal_id)
    if observation is None:
        return CuratorRunOutcome(status="no_observation", proposal_id=proposal_id)

    try:
        current_content = agents_file.read_text(encoding="utf-8")
    except OSError as exc:
        return CuratorRunOutcome(
            status="agents_file_unreadable", proposal_id=proposal_id,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        supporting_case_ids = json.loads(observation["supporting_case_ids_json"])
        records, unavailable_case_ids = build_case_intelligence_for_ids(store, supporting_case_ids)
        context: CuratorPromptContext = build_curator_prompt_context(
            proposal_id=proposal_id,
            title=proposal["title"],
            pattern_summary=observation["pattern_summary"],
            possible_cause=observation["possible_cause"],
            recommended_improvement=observation["recommended_improvement"],
            expected_benefit=observation["expected_benefit"],
            limitations=observation["limitations"],
            observation_confidence=observation["confidence"],
            observed_at=observation["observed_at"],
            supporting_case_records=records,
            unavailable_case_ids=unavailable_case_ids,
            current_content=current_content,
        )
    except Exception as exc:
        return CuratorRunOutcome(
            status="input_failed", proposal_id=proposal_id,
            error=f"{type(exc).__name__}: {exc}",
        )

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        change = analyzer(context, proposal_id=proposal_id, created_at=created_at)
    except CuratorAnalyzerError as exc:
        return CuratorRunOutcome(
            status="analyzer_failed", proposal_id=proposal_id,
            error=f"{type(exc).__name__}: {exc}",
        )

    if change.target_file != CURATOR_TARGET_FILE:
        return CuratorRunOutcome(
            status="analyzer_failed", proposal_id=proposal_id,
            error=f"analyzer returned target_file={change.target_file!r}, expected {CURATOR_TARGET_FILE!r}",
        )

    persisted = store.create_curator_change(
        change.change_id, change.proposal_id, change.target_file,
        change_type=change.change_type,
        rationale=change.rationale,
        before_content=change.before_content,
        proposed_content=change.proposed_content,
        expected_effect=change.expected_effect,
        confidence=change.confidence,
        created_at=change.created_at,
    )
    if not persisted:
        return CuratorRunOutcome(
            status="persistence_failed", proposal_id=proposal_id, change_type=change.change_type,
        )

    status = "no_change_recommended" if change.change_type == "no_change_recommended" else "succeeded"
    return CuratorRunOutcome(
        status=status,
        proposal_id=proposal_id,
        change_id=change.change_id,
        change_type=change.change_type,
        target_file=change.target_file,
        confidence=change.confidence,
    )


# ---------------------------------------------------------------------------
# Human-readable summary -- Traditional Chinese narrative, English
# machine-facing fields.
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    "proposal_not_found": "找不到指定的 Proposal",
    "not_accepted": "Proposal 尚未被人工 accept",
    "wrong_target": "Proposal 的 improvement_target 不是 agent_behavior",
    "no_observation": "找不到此 Proposal 的 Observation",
    "agents_file_unreadable": "無法讀取 AGENTS.md",
    "input_failed": "輸入資料組裝失敗",
    "analyzer_failed": "LLM 分析失敗",
    "persistence_failed": "寫入 curator_changes 失敗",
    "succeeded": "成功，已產生 proposed 變更",
    "no_change_recommended": "Curator 判斷不需要變更 AGENTS.md（no_change_recommended）",
}

_SUCCESSFUL_STATUSES = frozenset({"succeeded", "no_change_recommended"})


def _format_summary(outcome: CuratorRunOutcome) -> str:
    lines: list[str] = [
        "Curator 執行完成",
        f"status={outcome.status}",
        f"（{_STATUS_LABELS.get(outcome.status, outcome.status)}）",
        f"proposal_id={outcome.proposal_id}",
    ]
    if outcome.status in _SUCCESSFUL_STATUSES:
        lines.append(f"change_id={outcome.change_id}")
        lines.append(f"change_type={outcome.change_type}")
        lines.append(f"target_file={outcome.target_file}")
        lines.append(f"confidence={outcome.confidence}")
    if outcome.error:
        lines.append(f"error={outcome.error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, analyzer: Analyzer | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Curator Slice 1 runner: turn one accepted, agent_behavior "
            "Improvement Proposal into one proposed AGENTS.md change."
        ),
    )
    parser.add_argument(
        "--proposal-id", type=str, required=True,
        help="The accepted improvement_proposals.proposal_id to curate",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=f"SQLite DB path (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--agents-file", type=Path, default=Path(CURATOR_TARGET_FILE),
        help=f"Runtime AGENTS.md path (default: {CURATOR_TARGET_FILE})",
    )
    args = parser.parse_args(argv)

    store = FeedbackStoreV2(args.db) if args.db is not None else FeedbackStoreV2()

    real_analyzer = analyzer
    if real_analyzer is None:
        try:
            real_analyzer = resolve_default_analyzer()
        except RuntimeError as exc:
            print(f"ERROR: cannot resolve LLM runtime configuration: {exc}", file=sys.stderr)
            return 1

    outcome = run_curator(
        store, proposal_id=args.proposal_id, agents_file=args.agents_file, analyzer=real_analyzer,
    )
    print(_format_summary(outcome))
    return 0 if outcome.status in _SUCCESSFUL_STATUSES else 1


if __name__ == "__main__":
    sys.exit(main())
