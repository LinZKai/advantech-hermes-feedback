# Current Feedback Baseline

## Working behavior

- Telegram assistant responses support post-delivery feedback prompts.
- Universal feedback uses `fb:h:<run_id>` and `fb:u:<run_id>`.
- Callback submission verifies the chat and original Telegram user.
- The first accepted response is stored in SQLite.
- The feedback message is updated after submission.

## Current limitations

- Negative feedback ends immediately after the thumbs-down click.
- There is no negative-reason selection UI.
- There is no optional suggestion input flow.
- The runtime table still contains legacy resolution-related fields.
- Migration behavior is partly implemented at runtime.
- Some large Hermes core files are copied as complete overlays.
- Custom automated tests have not yet been added.

## Next implementation order

1. Freeze this working version as the Git baseline.
2. Add tests for callback parsing and storage behavior.
3. Implement negative-reason selection.
4. Implement optional suggestion capture.
5. Refactor storage schema and versioned migrations.
6. Reduce the size and responsibility of core-file overlays.
