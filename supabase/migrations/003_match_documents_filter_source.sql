-- Per-source RAG: allow match_documents to filter by documents.source
-- (tripadvisor / chudu24 / google) so VI queries do not drown EN corpora.

create or replace function match_documents(
  query_embedding vector(1536),
  match_count int default 8,
  filter_place_id uuid default null,
  filter_source text default null
)
returns table (
  id uuid,
  place_id uuid,
  review_id uuid,
  source text,
  content text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    d.id,
    d.place_id,
    d.review_id,
    d.source,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  where d.embedding is not null
    and (filter_place_id is null or d.place_id = filter_place_id)
    and (filter_source is null or d.source = filter_source)
  order by d.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;
