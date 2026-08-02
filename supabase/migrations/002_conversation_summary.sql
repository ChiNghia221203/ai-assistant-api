-- ---------------------------------------------------------------------------
-- Rolling conversation summary
-- Prompt = summary (older turns) + last N messages + RAG context + question
-- ---------------------------------------------------------------------------
alter table conversations
  add column if not exists summary text not null default '';

-- created_at of the newest message already folded into `summary`
alter table conversations
  add column if not exists summarized_through timestamptz;
