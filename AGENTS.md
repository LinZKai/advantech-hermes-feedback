# AGENTS.md

## Project purpose

This repository contains the Advantech Hermes universal-feedback customization
and its future feedback knowledge-loop components.

Upstream baseline:

- NemoClaw: v0.0.88
- Hermes Agent: v0.18.0
- Initial channel: Telegram
- Initial storage: SQLite

## Engineering constraints

1. Preserve the currently working Telegram response and feedback flow.
2. Do not store secrets, credentials, runtime databases, logs, or user data in Git.
3. Avoid broad changes to Hermes core runtime files.
4. Prefer isolated feedback modules and small integration patches.
5. If a core overlay must change, keep the change minimal and document why.
6. Do not replace whole upstream files merely to make a small behavior change.
7. Database schema changes must use explicit, versioned migrations.
8. Add or update tests for storage, callback parsing, and feedback state transitions.
9. Do not deploy to the formal sandbox without explicit approval.
10. Keep the current test environment separate from formal deployment.

## Feedback scope

The POC records turn-level answer quality:

- helpful
- not_helpful
- negative reason
- optional user suggestion

Resolution confirmation and unresolved-rate metrics are currently out of scope.

Recommended negative reason codes:

- incorrect
- incomplete
- not_relevant
- unclear
- other

## Current implementation note

The baseline contains full-file overlays of several Hermes files. Treat these as
a migration baseline, not the desired final architecture. Future work should
gradually move feedback logic into isolated modules while keeping integration
changes small and reviewable.
