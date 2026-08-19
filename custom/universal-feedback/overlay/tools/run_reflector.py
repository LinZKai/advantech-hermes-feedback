"""Reflector runner: real analysis and persistence.

    FeedbackStoreV2
     |
     v
    build_reflector_input()            (tools.case_reflection_input, unchanged)
     |
     v
    is_reflection_eligible()           (tools.case_reflection_input, unchanged)
     |                                   -- not eligible -> stop here, no LLM
     |                                      call, no DB write (see "Zero
     |                                      eligible Cases" below)
     v
    build_proposal_candidates()        (tools.proposal_matching, unchanged)
     |
     v
    build_reflector_prompt_context()   (tools.reflector_prompt_context, unchanged)
     |
     v
    analyzer(context, ...)              (THE analyzer boundary -- see below;
     |                                    defaults to tools.reflector_analyzer.
     |                                    analyze_reflection_with_llm)
     v
    ReflectionResult
     |
     v
    persist_reflection_result()        (tools.reflector_persistence, unchanged)
     |
     v
    ReflectorRunOutcome (human-readable summary on stdout)
     |
     v
    exit

Usage:
    python3 tools/run_reflector.py [--db PATH] [--window-start ISO] [--window-end ISO]
        [--min-analyzed-cases N]

Not a scheduler: manual execution only, one run of one Reflection window,
then exit. No cron, no timer, no background worker, no retry/queue
infrastructure -- deciding WHEN to run this is deliberately deferred to a
later trigger-policy decision.

Analyzer boundary: a run's ReflectionResult is produced by any `Analyzer`
callable, injected via the `analyzer` parameter on run_reflector()/main().
Unlike tools.run_case_enrichment.run(), `analyzer` here has NO default and
must always be supplied explicitly -- there is no existing Reflector stub
analyzer to reuse. main() is the only place that binds the real analyzer
(tools.reflector_analyzer.analyze_reflection_with_llm, via functools.partial
with a resolved main_runtime) when no analyzer override is given -- this
keeps the CLI's real-runtime wiring in exactly one place and keeps
run_reflector() fully testable with an injected fake.

ReflectionRun ownership: persist_reflection_result() (tools.
reflector_persistence) owns the ENTIRE reflection_runs lifecycle --
create_reflection_run(status='running') through complete_reflection_run
(status='succeeded'/'failed'). This module NEVER calls either of those
itself. Consequence: an analyzer-stage failure produces NO reflection_runs
row at all (persist_reflection_result() is never reached), whereas a
persistence-stage failure DOES leave one, with status='failed' (because
persist_reflection_result() got far enough to create it before its own
Proposal/Observation transaction failed). Both are still fully visible in
this run's own ReflectorRunOutcome/stdout summary/process exit code.

Zero eligible Cases: is_reflection_eligible()=False stops this run BEFORE
building a ReflectorPromptContext, BEFORE calling the analyzer, and BEFORE
any DB write -- reported as ReflectorRunOutcome(status="no_eligible_cases"),
never as a reflection_runs row. A reflection_runs row's meaning stays
uniform: "the analyzer was actually invoked for this attempt."

Input-stage failure: build_reflector_input()/build_proposal_candidates()/
build_reflector_prompt_context() are not expected to raise for ordinary
data gaps (a missing/unparseable Case analysis is a first-class
ReflectorInput field, never an exception) -- but a genuine DB/connectivity
failure, or a structurally-impossible pending-Proposal-with-no-Observation
state, can still raise. This module wraps that whole sequence in one
try/except and reports it as ReflectorRunOutcome(status="input_failed"),
distinct from an analyzer or persistence failure, and never reaches the
analyzer or persist_reflection_result() at all.
"""
from __future__ import annotations

import argparse
import functools
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_reflection_input import (  # noqa: E402
    DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION,
    build_reflector_input,
    is_reflection_eligible,
)
from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402
from tools.proposal_matching import build_proposal_candidates  # noqa: E402
from tools.reflector_analyzer import (  # noqa: E402
    REFLECTOR_VERSION,
    ReflectorAnalyzerError,
    analyze_reflection_with_llm,
)
from tools.reflector_persistence import ReflectionPersistenceError, persist_reflection_result  # noqa: E402
from tools.reflector_prompt_context import ReflectorPromptContext, build_reflector_prompt_context  # noqa: E402
from tools.reflector_proposals import ReflectionResult  # noqa: E402

# (context, *, reflection_run_id, observed_at) -> ReflectionResult, raises
# ReflectorAnalyzerError. Matches tools.reflector_analyzer.analyze_
# reflection_with_llm's own signature shape (minus main_runtime/llm_call/
# id_factory, which are the REAL analyzer's own concerns -- a fake analyzer
# injected for a test needs none of them, only this reduced shape).
Analyzer = Callable[..., ReflectionResult]

_TERMINAL_STATUSES_MEANING_A_ROW_WAS_CREATED = frozenset({"succeeded", "failed"})


# ---------------------------------------------------------------------------
# Runtime config resolution -- the ONE place config.yaml is read
# ---------------------------------------------------------------------------

_REQUIRED_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key")


def _resolve_main_runtime() -> dict[str, Any]:
    """Read provider/model/base_url/api_key from Hermes' own config.yaml
    -- deliberately a separate copy from tools.run_case_enrichment's own
    equivalent, not a cross-module import (this domain never imports
    another module's private helper across a file boundary). Reflector
    must analyze with the SAME provider/model the main Hermes agent is
    already configured with -- never a hard-coded model, never a silent
    fallback to a different one.

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
            "hermes_cli.config is not importable -- run_reflector.py must run "
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectorRunOutcome:
    """One run_reflector() call's outcome -- counts and classification
    only, never Case/Proposal narrative content (see _format_summary).

    `status` is exactly one of:
      * "no_eligible_cases" -- is_reflection_eligible() said no; no LLM
        call, no DB write of any kind.
      * "input_failed"       -- an exception while building ReflectorInput/
        candidates/context; no LLM call, no DB write.
      * "analyzer_failed"    -- the analyzer raised ReflectorAnalyzerError;
        no DB write (see this module's own docstring, "ReflectionRun
        ownership").
      * "persistence_error"  -- persist_reflection_result() raised
        ReflectionPersistenceError (create_reflection_run() itself
        failed, e.g. a duplicate reflection_run_id); no reflection_runs
        row exists for this attempt.
      * "succeeded"           -- persist_reflection_result() returned
        status="succeeded"; a reflection_runs row exists with that status.
      * "failed"              -- persist_reflection_result() returned
        status="failed" (the Proposal/Observation transaction itself
        failed); a reflection_runs row STILL exists, with status='failed'
        -- this is the one failure category that IS durable in the DB.

    Deliberately a plain dataclass with a `status` string, not a new
    runner-specific exception hierarchy -- run_reflector() itself never
    raises for any of the above; a caller branches on `.status`.
    """

    status: str
    reflection_run_id: str | None = None
    analyzed_case_count: int = 0
    window_start: str | None = None
    window_end: str | None = None
    material_change_detected: bool | None = None
    new_proposal_count: int | None = None
    observation_count: int | None = None
    run_summary: str | None = None
    error: str | None = None


def run_reflector(
    store: FeedbackStoreV2,
    *,
    window_start: str | None = None,
    window_end: str | None = None,
    min_analyzed_cases: int = DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION,
    analyzer: Analyzer,
) -> ReflectorRunOutcome:
    """The reusable pipeline body for one Reflection Run, separated from
    CLI/argv handling so main() (or a test) can call it directly with a
    different analyzer.

    `window_start`/`window_end` default to None/None -- build_reflector_
    input()'s own default (the whole Case history, no window restriction).
    `analyzer` is required, no default -- see this module's own docstring,
    "Analyzer boundary", for why.

    reflection_run_id is generated once, via uuid.uuid4().hex. `started_at`
    (passed to persist_reflection_result as the reflection_runs row's own
    started_at) and `observed_at` (passed to the analyzer, and from there
    into every Observation's own observed_at) are deliberately the SAME
    timestamp, generated once, immediately before the analyzer call -- not
    two independent datetime.now() calls a few lines apart that would
    differ by an accidental few milliseconds for no semantic reason.
    `completed_at` IS a second, later datetime.now() call, taken after the
    analyzer returns -- that one genuinely is a different moment (the
    analyzer call takes real wall-clock time), so a fresh timestamp there
    is correct, not redundant.
    """
    try:
        reflector_input = build_reflector_input(store, window_start=window_start, window_end=window_end)
        eligible = is_reflection_eligible(reflector_input, min_analyzed_cases=min_analyzed_cases)
    except Exception as exc:
        return ReflectorRunOutcome(
            status="input_failed",
            window_start=window_start,
            window_end=window_end,
            error=f"{type(exc).__name__}: {exc}",
        )

    if not eligible:
        return ReflectorRunOutcome(
            status="no_eligible_cases",
            analyzed_case_count=reflector_input.analyzed_case_count,
            window_start=reflector_input.window_start,
            window_end=reflector_input.window_end,
        )

    try:
        candidates = build_proposal_candidates(store)
        context: ReflectorPromptContext = build_reflector_prompt_context(reflector_input, candidates)
    except Exception as exc:
        return ReflectorRunOutcome(
            status="input_failed",
            analyzed_case_count=reflector_input.analyzed_case_count,
            window_start=reflector_input.window_start,
            window_end=reflector_input.window_end,
            error=f"{type(exc).__name__}: {exc}",
        )

    reflection_run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        result = analyzer(context, reflection_run_id=reflection_run_id, observed_at=started_at)
    except ReflectorAnalyzerError as exc:
        return ReflectorRunOutcome(
            status="analyzer_failed",
            reflection_run_id=reflection_run_id,
            analyzed_case_count=reflector_input.analyzed_case_count,
            window_start=reflector_input.window_start,
            window_end=reflector_input.window_end,
            error=f"{type(exc).__name__}: {exc}",
        )

    completed_at = datetime.now(timezone.utc).isoformat()

    try:
        persistence_outcome = persist_reflection_result(
            store, result,
            started_at=started_at,
            completed_at=completed_at,
            analyzed_case_count=reflector_input.analyzed_case_count,
            reflector_version=REFLECTOR_VERSION,
            window_start=reflector_input.window_start,
            window_end=reflector_input.window_end,
        )
    except ReflectionPersistenceError as exc:
        return ReflectorRunOutcome(
            status="persistence_error",
            reflection_run_id=reflection_run_id,
            analyzed_case_count=reflector_input.analyzed_case_count,
            window_start=reflector_input.window_start,
            window_end=reflector_input.window_end,
            material_change_detected=result.material_change_detected,
            new_proposal_count=len(result.new_proposals),
            observation_count=len(result.proposal_observations),
            run_summary=result.run_summary,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ReflectorRunOutcome(
        status=persistence_outcome.status,
        reflection_run_id=persistence_outcome.reflection_run_id,
        analyzed_case_count=reflector_input.analyzed_case_count,
        window_start=reflector_input.window_start,
        window_end=reflector_input.window_end,
        material_change_detected=result.material_change_detected,
        new_proposal_count=len(result.new_proposals),
        observation_count=len(result.proposal_observations),
        run_summary=result.run_summary,
    )


# ---------------------------------------------------------------------------
# Human-readable summary -- Traditional Chinese narrative, English
# machine-facing fields. This function does no translation of its own and
# never touches result.run_summary's content; it only labels that
# already-zh-TW text with a handful of fixed Chinese labels around it.
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    "no_eligible_cases": "可分析 Case 數量不足，本次未執行分析",
    "input_failed": "輸入資料組裝失敗",
    "analyzer_failed": "LLM 分析失敗",
    "persistence_error": "ReflectionRun 建立失敗",
    "succeeded": "成功",
    "failed": "失敗（已記錄於 reflection_runs）",
}


def _format_summary(outcome: ReflectorRunOutcome) -> str:
    lines: list[str] = [
        "Reflector 執行完成",
        f"狀態：{_STATUS_LABELS.get(outcome.status, outcome.status)}",
        f"reflection_run_id={outcome.reflection_run_id}",
        f"分析 Case：{outcome.analyzed_case_count}",
        f"window_start={outcome.window_start} window_end={outcome.window_end}",
    ]
    if outcome.status in _TERMINAL_STATUSES_MEANING_A_ROW_WAS_CREATED:
        material_change = "是" if outcome.material_change_detected else "否"
        lines.append(f"新增 Proposal：{outcome.new_proposal_count}")
        lines.append(f"Observation：{outcome.observation_count}")
        lines.append(f"Material change：{material_change}")
        lines.append(f"run_summary={outcome.run_summary}")
    if outcome.error:
        lines.append(f"error={outcome.error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, analyzer: Analyzer | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 Slice 6B: Reflector runner. Manual, one-shot execution of a "
            "single Reflection Run over the given (or unbounded) Case window."
        ),
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=f"SQLite DB path (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--window-start", type=str, default=None,
        help="Analysis window start, ISO-8601, inclusive (default: unbounded)",
    )
    parser.add_argument(
        "--window-end", type=str, default=None,
        help="Analysis window end, ISO-8601, exclusive (default: unbounded)",
    )
    parser.add_argument(
        "--min-analyzed-cases", type=int, default=DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION,
        help=f"Minimum analyzed Cases required to run (default: {DEFAULT_MIN_ANALYZED_CASES_FOR_REFLECTION})",
    )
    args = parser.parse_args(argv)

    store = FeedbackStoreV2(args.db) if args.db is not None else FeedbackStoreV2()

    real_analyzer = analyzer
    if real_analyzer is None:
        try:
            main_runtime = _resolve_main_runtime()
        except RuntimeError as exc:
            print(f"ERROR: cannot resolve LLM runtime configuration: {exc}", file=sys.stderr)
            return 1
        real_analyzer = functools.partial(analyze_reflection_with_llm, main_runtime=main_runtime)

    outcome = run_reflector(
        store,
        window_start=args.window_start,
        window_end=args.window_end,
        min_analyzed_cases=args.min_analyzed_cases,
        analyzer=real_analyzer,
    )
    print(_format_summary(outcome))
    return 0 if outcome.status in ("succeeded", "no_eligible_cases") else 1


if __name__ == "__main__":
    sys.exit(main())
