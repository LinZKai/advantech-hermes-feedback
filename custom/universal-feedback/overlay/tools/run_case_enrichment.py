"""Phase 4.5 Stage D2: Case Enrichment batch runner, with real persistence.

    FeedbackStoreV2
     |
     v
    list_cases_needing_analysis()
     |
     v (per Case)
    build_case_enrichment_input()      (tools.case_enrichment, unchanged)
     |
     v
    analyzer(case_input)                (THE analyzer boundary -- see below)
     |
     v
    [ if persist: create_case_analysis() ]   (only on a valid result)
     |
     v
    human-readable summary (stdout)
     |
     v
    exit

Usage:
    python tools/run_case_enrichment.py [--db PATH] [--limit N] [--session-id ID] [--dry-run]

Default (no --dry-run): runs the real LLM analyzer
(tools.case_enrichment_analyzer.analyze_case_with_llm) and, for every Case
whose result passes the full Stage B contract, INSERTs exactly one row into
case_analysis. `--dry-run`: runs the same real analyzer and prints the same
summary, but never calls create_case_analysis() -- useful for previewing
prompt/model output before committing it to history.

Persistence is INSERT-only and only ever happens after a Case's result has
already passed CaseEnrichmentResult's own __post_init__ validation (the
analyzer's return type is CaseEnrichmentResult, never Optional -- an invalid
result cannot exist to persist in the first place). No partial/failed
analysis ever reaches case_analysis: build failure, analyzer failure, and
storage-insert failure all short-circuit before create_case_analysis() is
even called, and are reported as build_failed/analyze_failed/persist_failed
respectively -- see _process_case()'s docstring.

Not a scheduler: manual execution only, one run, then exit. No retry/queue
infrastructure, and no application-level retry around the analyzer call --
tools.case_enrichment_analyzer.analyze_case_with_llm's own
agent.auxiliary_client.call_llm already retries/falls back at the provider
layer (Stage D Hermes LLM Integration Audit).

Analyzer boundary (unchanged since Stage C): a Case's analysis is produced
by any Analyzer callable, injected via the `analyzer` parameter on
_process_case()/run()/main() -- run()/_process_case()'s own signatures never
grow model/provider/timeout parameters. main() is the only place that binds
the real analyzer (analyze_case_with_llm, via functools.partial with a
resolved main_runtime) when no analyzer override is given -- this keeps the
CLI's real-runtime wiring in exactly one place and keeps run()/_process_case()
fully testable with an injected fake, no Hermes runtime required.

Does not update cases.title / cases.product_model: case_analysis is
append-only intelligence history, not the Case's own identity fields. Stage
B.5's create_case_analysis() docstring already states this explicitly;
Stage D2 does not change it (see that module for the deferred-to-later-stage
Human Override policy this would require).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import (  # noqa: E402
    CaseAnalysisEvidence,
    CaseEnrichmentInput,
    CaseEnrichmentResult,
    build_case_enrichment_input,
)
from tools.case_enrichment_analyzer import ANALYSIS_VERSION, analyze_case_with_llm  # noqa: E402
from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402

Analyzer = Callable[[CaseEnrichmentInput], CaseEnrichmentResult]


# ---------------------------------------------------------------------------
# Stub analyzer -- kept from Stage C. Still useful as a network-free,
# deterministic Analyzer for tests and manual plumbing checks; no longer the
# CLI default (see main()), but not removed -- nothing about its contract
# changed.
# ---------------------------------------------------------------------------


def analyze_case(case_input: CaseEnrichmentInput) -> CaseEnrichmentResult:
    """STUB analyzer: deterministic, conservative, no reasoning of any kind.

    Exists only to prove the CaseEnrichmentInput -> analyzer ->
    CaseEnrichmentResult seam actually connects end to end -- not to
    simulate real classification. Always returns the same conservative,
    maximally-uncertain result shape (both taxonomies' "no signal" fallback
    values, 0.0 confidence, no product identified) regardless of the
    Case's actual content, and never reads case_input.turns[*].
    question_text/answer_text to do so.

    The single required evidence entry states a fact that is ALWAYS true
    by construction -- CaseEnrichmentInput.__post_init__ already guarantees
    at least one turn exists, and turns.question_text is NOT NULL at the
    schema level -- so this never fabricates or infers anything; it is not
    a claim about what the user said, only that a first turn with a
    question exists. Every field is fixed/computed from data already on
    case_input; nothing here ever reads or repeats the user's or
    assistant's actual words, matching the CLI output's own policy of never
    printing full turn text (see _format_summary below).
    """
    first_turn = case_input.turns[0]
    evidence = (
        CaseAnalysisEvidence(
            type="user_text",
            turn_id=first_turn.turn_id,
            fact=(
                f"[STUB] Turn {first_turn.turn_id} has a recorded user "
                "question (dry-run stub does not read its content)."
            ),
        ),
    )
    return CaseEnrichmentResult(
        case_title=f"[STUB] Case {case_input.case_id} (dry-run placeholder)",
        issue_summary="[STUB] dry-run placeholder -- no real analysis was performed.",
        issue_type="other_or_unclear",
        issue_type_confidence=0.0,
        diagnosis="no_issue_detected",
        diagnosis_confidence=0.0,
        product_model=None,
        product_source=None,
        product_confidence=None,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Runtime config resolution -- the ONE place config.yaml is read
# ---------------------------------------------------------------------------

_REQUIRED_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key")


def _resolve_main_runtime() -> dict[str, Any]:
    """Read provider/model/base_url/api_key from Hermes' own config.yaml
    (model.provider / model.default / model.base_url / model.api_key) so
    Case Enrichment always analyzes with the SAME provider/model the main
    Hermes agent is already configured with -- never a hard-coded model or
    provider, never a fallback to a different one. Key paths and presence
    were confirmed against the real ae-mail-foundry-hermes sandbox in the
    Stage D0 runtime smoke test (provider=custom, model=gpt-5.6-luna,
    base_url_present=True, api_key_present=True).

    Deliberately the single call site for hermes_cli.config.load_config()
    in this file (Stage D2 audit instruction: do not scatter load_config()
    across _process_case()/run()/main()) -- called exactly once, from
    main(), before any Case is processed.

    Lazy-imports hermes_cli.config (only available inside a built Hermes
    sandbox image) so this module stays importable in this repo's test
    environment, matching tools.case_enrichment_analyzer's own lazy-import
    of agent.auxiliary_client.

    Fails closed: raises RuntimeError if hermes_cli.config cannot be
    imported, or if config.yaml has no 'model' section, or if any of
    provider/model/base_url/api_key is missing or blank. Never falls back
    to "auto" or to a hard-coded default -- a missing/incomplete config is
    a setup problem for the whole run, not a per-Case outcome, so this is
    checked once before any Case is processed rather than inside
    _process_case().

    api_mode/auth_mode are deliberately NOT read here: no code evidence in
    the Stage D Hermes LLM Integration Audit confirmed a config.yaml key
    path for either (api_mode is resolved by auto-detection inside Hermes
    itself when not given -- see that audit's §7). Adding either without
    evidence would be exactly the kind of guessed key path this project has
    repeatedly required avoiding.
    """
    try:
        from hermes_cli.config import load_config
    except ImportError as exc:
        raise RuntimeError(
            "hermes_cli.config is not importable -- run_case_enrichment.py must run "
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


def _evidence_to_json(evidence: tuple[CaseAnalysisEvidence, ...]) -> str:
    """Serialize a CaseEnrichmentResult's evidence tuple to the plain JSON
    string FeedbackStoreV2.create_case_analysis()'s evidence_json parameter
    requires (str | None -- the storage layer does no encoding of its own;
    confirmed by reading its exact signature, not assumed). One JSON array
    of {"type", "turn_id", "fact"} objects, matching CaseAnalysisEvidence's
    own field names exactly -- not a new shape invented here."""
    return json.dumps(
        [{"type": e.type, "turn_id": e.turn_id, "fact": e.fact} for e in evidence],
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CaseOutcome:
    """One Case's outcome -- counts and classification only, never
    question/answer/evidence text (see _format_summary)."""

    case_id: str
    evidence_watermark: str | None
    status: str  # "succeeded" | "build_failed" | "analyze_failed" | "persist_failed"
    turns_count: int | None = None
    retrievals_count: int | None = None
    feedback_count: int | None = None
    issue_type: str | None = None
    diagnosis: str | None = None
    product_model: str | None = None
    error: str | None = None
    analysis_id: str | None = None
    persisted: bool = False


def _process_case(
    store: FeedbackStoreV2,
    case_id: str,
    evidence_watermark: str | None,
    *,
    analyzer: Analyzer = analyze_case,
    persist: bool = False,
) -> _CaseOutcome:
    """Build + analyze (+ optionally persist) one Case, never raising -- a
    failure here must never take down the rest of the batch.

    build_case_enrichment_input() is already documented as fail-closed
    (returns None, never raises) on its own, so the try/except around it is
    defensive belt-and-suspenders. The analyzer call is a real risk this
    isolates (network/parsing failure against a real LLM as of Stage D1).
    Persistence is a THIRD, independent risk this isolates: a storage-layer
    failure (returns False, or -- defensively -- raises) never propagates
    past this function either, and never partially writes a row.

    `persist=False` (the default) never calls create_case_analysis() --
    matching this function's original Stage C dry-run behavior exactly, so
    every existing Stage C caller/test that does not pass `persist=` keeps
    working unchanged. `persist=True` calls create_case_analysis() only
    after `analyzer` has already returned a real CaseEnrichmentResult
    (impossible to be structurally invalid -- CaseEnrichmentResult's own
    __post_init__ already enforced that at construction time inside the
    analyzer), using `case_input.evidence_watermark` (the watermark computed
    fresh when this Case's input was actually built) as
    source_evidence_watermark -- deliberately NOT the `evidence_watermark`
    parameter passed into this function (that value was read earlier, by
    list_cases_needing_analysis(), and could be stale if new evidence
    arrived between listing and building).
    """
    try:
        case_input = build_case_enrichment_input(store, case_id)
    except Exception as exc:
        return _CaseOutcome(
            case_id, evidence_watermark, "build_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    if case_input is None:
        return _CaseOutcome(
            case_id, evidence_watermark, "build_failed",
            error="build_case_enrichment_input returned None",
        )

    try:
        result = analyzer(case_input)
    except Exception as exc:
        return _CaseOutcome(
            case_id, evidence_watermark, "analyze_failed",
            turns_count=len(case_input.turns),
            retrievals_count=len(case_input.retrievals),
            feedback_count=len(case_input.feedback),
            error=f"{type(exc).__name__}: {exc}",
        )

    counts = dict(
        turns_count=len(case_input.turns),
        retrievals_count=len(case_input.retrievals),
        feedback_count=len(case_input.feedback),
    )
    classification = dict(
        issue_type=result.issue_type,
        diagnosis=result.diagnosis,
        product_model=result.product_model,
    )

    if not persist:
        return _CaseOutcome(
            case_id, evidence_watermark, "succeeded",
            persisted=False, **counts, **classification,
        )

    analysis_id = uuid.uuid4().hex
    analyzed_at = datetime.now(timezone.utc).isoformat()
    try:
        ok = store.create_case_analysis(
            analysis_id, case_id,
            case_title=result.case_title,
            issue_summary=result.issue_summary,
            issue_type=result.issue_type,
            issue_type_confidence=result.issue_type_confidence,
            diagnosis=result.diagnosis,
            diagnosis_confidence=result.diagnosis_confidence,
            product_model=result.product_model,
            product_source=result.product_source,
            product_confidence=result.product_confidence,
            evidence_json=_evidence_to_json(result.evidence),
            analysis_version=ANALYSIS_VERSION,
            analyzed_at=analyzed_at,
            source_evidence_watermark=case_input.evidence_watermark,
        )
    except Exception as exc:
        return _CaseOutcome(
            case_id, evidence_watermark, "persist_failed",
            error=f"{type(exc).__name__}: {exc}", **counts, **classification,
        )

    if not ok:
        return _CaseOutcome(
            case_id, evidence_watermark, "persist_failed",
            error="create_case_analysis returned False", **counts, **classification,
        )

    return _CaseOutcome(
        case_id, evidence_watermark, "succeeded",
        analysis_id=analysis_id, persisted=True, **counts, **classification,
    )


def _format_summary(outcomes: list[_CaseOutcome], *, dry_run: bool = True) -> str:
    """Human-readable summary. Deliberately never includes
    question_text/answer_text/evidence.fact/case_title/issue_summary --
    only identifiers, counts, and classification fields -- so a run's log
    can never unboundedly leak Case content, dry-run or not."""
    lines: list[str] = [f"Cases needing analysis: {len(outcomes)}", ""]
    for outcome in outcomes:
        lines.append(f"case_id={outcome.case_id}")
        lines.append(f"  evidence_watermark={outcome.evidence_watermark}")
        if outcome.status == "succeeded":
            lines.append(
                f"  turns={outcome.turns_count} "
                f"retrievals={outcome.retrievals_count} "
                f"feedback={outcome.feedback_count}"
            )
            lines.append(
                f"  issue_type={outcome.issue_type} "
                f"diagnosis={outcome.diagnosis} "
                f"product_model={outcome.product_model}"
            )
            if outcome.persisted:
                lines.append(f"  persisted=True analysis_id={outcome.analysis_id}")
            else:
                lines.append("  persisted=False")
        else:
            lines.append(f"  status={outcome.status}")
            lines.append(f"  error={outcome.error}")
        lines.append("")

    succeeded = sum(1 for o in outcomes if o.status == "succeeded")
    failed = len(outcomes) - succeeded
    lines.append(f"processed={len(outcomes)} succeeded={succeeded} failed={failed}")
    lines.append("")
    if dry_run:
        lines.append("DRY RUN ONLY -- no case_analysis rows were created.")
    else:
        persisted_count = sum(1 for o in outcomes if o.persisted)
        lines.append(f"persisted {persisted_count} case_analysis row(s).")
    return "\n".join(lines)


def run(
    store: FeedbackStoreV2,
    *,
    session_id: str | None = None,
    limit: int | None = None,
    analyzer: Analyzer = analyze_case,
    persist: bool = False,
) -> list[_CaseOutcome]:
    """The reusable pipeline body, separated from CLI/argv handling so
    main() (or a test) can call it directly with a different analyzer
    and/or persist setting. `persist` defaults to False here (library-level
    safe default -- an explicit opt-in is required to write to the
    database), independent of the CLI's own default (see main(), which
    computes `persist=not args.dry_run` explicitly)."""
    candidates = store.list_cases_needing_analysis(session_id=session_id, limit=limit)
    return [
        _process_case(
            store, row["case_id"], row["evidence_watermark"],
            analyzer=analyzer, persist=persist,
        )
        for row in candidates
    ]


def main(argv: list[str] | None = None, *, analyzer: Analyzer | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4.5 Stage D2: Case Enrichment batch runner. Default: real "
            "LLM analysis, persisted to case_analysis. --dry-run: real LLM "
            "analysis, printed, never persisted."
        ),
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=f"SQLite DB path (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of Cases to process this run",
    )
    parser.add_argument(
        "--session-id", type=str, default=None,
        help="Only process Cases belonging to this session",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the real analyzer and print results, but never call create_case_analysis()",
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
        real_analyzer = functools.partial(analyze_case_with_llm, main_runtime=main_runtime)

    outcomes = run(
        store, session_id=args.session_id, limit=args.limit,
        analyzer=real_analyzer, persist=not args.dry_run,
    )
    print(_format_summary(outcomes, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
