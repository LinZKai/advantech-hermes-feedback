-- Staged migration documentation. Runtime migration is applied idempotently by
-- tools.feedback_storage.FeedbackStore.__init__ using PRAGMA table_info + ALTER TABLE.
-- Existing rows remain intact; legacy rows retain resolved and NULL turn_key.

CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_runs_turn_key
ON feedback_runs(turn_key) WHERE turn_key IS NOT NULL;
