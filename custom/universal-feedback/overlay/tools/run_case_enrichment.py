"""Phase 4.5 Stage C: manual Case Enrichment batch runner (dry-run only).

    FeedbackStoreV2
     |
     v
    list_cases_needing_analysis()
     |
     v (per Case)
    build_case_enrichment_input()      (tools.case_enrichment, unchanged)
     |
     v
    analyze_case()                     (THE analyzer boundary -- see below)
     |
     v
    human-readable dry-run summary (stdout)
     |
     v
    exit

Usage:
    python tools/run_case_enrichment.py [--db PATH] [--limit N] [--session-id ID]

Dry-run only: this file NEVER calls FeedbackStoreV2.create_case_analysis()
and never writes to case_analysis. Stage C's job is to prove the plumbing
(list -> build input -> analyze -> report) works end to end; Stage D adds a
real LLM analyzer, output validation against the analyzer's raw output, and
persistence -- none of that exists here yet.

Not a scheduler: manual execution only, one run, then exit. No retry/queue
infrastructure -- see _process_case()'s failure-isolation docstring for
exactly how far "POC-friendly" failure handling goes here.

Analyzer boundary (Stage D's replacement point): analyze_case(case_input)
-> CaseEnrichmentResult is the ONLY function in this file a real LLM
analyzer needs to replace. Everything else (listing candidates, building
input, the summary/failure-isolation loop) is reusable unchanged -- pass a
different callable via the `analyzer` parameter on _process_case()/main()
rather than editing this file's control flow. Deliberately not a class
hierarchy or plugin framework: a plain function is enough for this POC.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_OVERLAY_ROOT = Path(__file__).resolve().parents[1]
if str(_OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_ROOT))

from tools.case_enrichment import (  # noqa: E402
    CaseAnalysisEvidence,
    CaseEnrichmentInput,
    CaseEnrichmentResult,
    build_case_enrichment_input,
)
from tools.feedback_store_v2 import DEFAULT_PATH, FeedbackStoreV2  # noqa: E402

Analyzer = Callable[[CaseEnrichmentInput], CaseEnrichmentResult]


# ---------------------------------------------------------------------------
# Analyzer boundary -- Stage D swaps this out for a real LLM call. Nothing
# else in this file needs to change to do that.
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
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CaseOutcome:
    """One Case's dry-run outcome -- counts and classification only, never
    question/answer/evidence text (see _format_summary)."""

    case_id: str
    evidence_watermark: str | None
    status: str  # "succeeded" | "build_failed" | "analyze_failed"
    turns_count: int | None = None
    retrievals_count: int | None = None
    feedback_count: int | None = None
    issue_type: str | None = None
    diagnosis: str | None = None
    product_model: str | None = None
    error: str | None = None


def _process_case(
    store: FeedbackStoreV2,
    case_id: str,
    evidence_watermark: str | None,
    *,
    analyzer: Analyzer = analyze_case,
) -> _CaseOutcome:
    """Build + analyze one Case, never raising -- a failure here must never
    take down the rest of the batch (Stage C failure-isolation
    requirement). build_case_enrichment_input() is already documented as
    fail-closed (returns None, never raises) on its own, so the try/except
    around it is defensive belt-and-suspenders; the analyzer call is the
    real risk this isolates -- a stub bug today, a real LLM call's
    network/parsing failure in Stage D.
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

    return _CaseOutcome(
        case_id, evidence_watermark, "succeeded",
        turns_count=len(case_input.turns),
        retrievals_count=len(case_input.retrievals),
        feedback_count=len(case_input.feedback),
        issue_type=result.issue_type,
        diagnosis=result.diagnosis,
        product_model=result.product_model,
    )


def _format_summary(outcomes: list[_CaseOutcome]) -> str:
    """Human-readable dry-run summary. Deliberately never includes
    question_text/answer_text/evidence.fact/case_title/issue_summary --
    only identifiers, counts, and the stub's fixed classification fields --
    so a dry-run log can never unboundedly leak Case content."""
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
                "  [STUB RESULT] "
                f"issue_type={outcome.issue_type} "
                f"diagnosis={outcome.diagnosis} "
                f"product_model={outcome.product_model}"
            )
        else:
            lines.append(f"  status={outcome.status}")
            lines.append(f"  error={outcome.error}")
        lines.append("")

    succeeded = sum(1 for o in outcomes if o.status == "succeeded")
    failed = len(outcomes) - succeeded
    lines.append(f"processed={len(outcomes)} succeeded={succeeded} failed={failed}")
    lines.append("")
    lines.append("DRY RUN ONLY -- no case_analysis rows were created.")
    return "\n".join(lines)


def run(
    store: FeedbackStoreV2,
    *,
    session_id: str | None = None,
    limit: int | None = None,
    analyzer: Analyzer = analyze_case,
) -> list[_CaseOutcome]:
    """The reusable pipeline body, separated from CLI/argv handling so
    Stage D (or a test) can call it directly with a different analyzer."""
    candidates = store.list_cases_needing_analysis(session_id=session_id, limit=limit)
    return [
        _process_case(store, row["case_id"], row["evidence_watermark"], analyzer=analyzer)
        for row in candidates
    ]


def main(argv: list[str] | None = None, *, analyzer: Analyzer = analyze_case) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4.5 Stage C: manual Case Enrichment dry-run. "
            "Stub analyzer only, no LLM, no persistence."
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
    args = parser.parse_args(argv)

    store = FeedbackStoreV2(args.db) if args.db is not None else FeedbackStoreV2()
    outcomes = run(store, session_id=args.session_id, limit=args.limit, analyzer=analyzer)
    print(_format_summary(outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
