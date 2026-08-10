# Advantech Hermes Feedback

A proof-of-concept feedback and support-intelligence extension for Hermes Agent
running under NemoClaw.

## Current status

The current Telegram flow can:

- send a feedback prompt after an eligible assistant response;
- collect helpful or not-helpful feedback;
- verify the originating chat and user;
- persist turn and feedback information in SQLite.

Negative-reason selection and optional suggestions are not implemented yet.

## Repository role

This repository is currently a customization source for a pinned NemoClaw and
Hermes build workspace. It is not yet a standalone build of NemoClaw.

Current upstream baseline:

- NemoClaw v0.0.88
- Hermes Agent v0.18.0

The files under `custom/universal-feedback/overlay/` are applied during the
custom image build.

## POC direction

The project will provide:

1. A structured feedback collection flow.
2. Data that can support a later knowledge-improvement loop.
3. Product, version, and issue-type aggregation.
4. Negative-feedback volume anomaly detection.
5. Reports and proactive AE notifications.

User-confirmed resolution is intentionally excluded from the current POC.

## Data safety

The following must never be committed:

- API keys and tokens
- environment files
- SQLite runtime databases
- Telegram user data
- logs
- sandbox runtime state
- backup copies of upstream files

## Development workflow

1. Make changes on a feature branch.
2. Run automated tests.
3. Review the diff against the baseline.
4. Build and test in the feedback test sandbox.
5. Tag an approved version before formal deployment.
