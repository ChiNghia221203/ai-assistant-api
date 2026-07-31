-- Hotel multi-source review schema (Sài Gòn)
create extension if not exists vector;
create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- places
-- ---------------------------------------------------------------------------
create table if not exists places (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  name text not null,
  city text not null default 'Ho Chi Minh',
  address text,
  lat double precision,
  lng double precision,
  google_place_id text,
  chudu24_url text,
  tripadvisor_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists places_google_place_id_idx on places (google_place_id);
create index if not exists places_city_idx on places (city);

-- ---------------------------------------------------------------------------
-- source_snapshots (latest methodology + scores per place/source)
-- ---------------------------------------------------------------------------
create table if not exists source_snapshots (
  id uuid primary key default gen_random_uuid(),
  place_id uuid not null references places(id) on delete cascade,
  source text not null check (source in ('chudu24', 'google', 'tripadvisor')),
  source_url text not null default '',
  captured_at timestamptz not null default now(),
  sample_policy text not null,
  sample_size int not null default 0,
  date_min date,
  date_max date,
  site_overall numeric,
  site_overall_scale int not null default 5,
  site_n_total int,
  sample_mean numeric,
  sample_mean_scale int not null default 5,
  distribution jsonb not null default '{}'::jsonb,
  reviews_available boolean not null default false,
  unique (place_id, source)
);

create index if not exists source_snapshots_place_id_idx on source_snapshots (place_id);

-- ---------------------------------------------------------------------------
-- reviews
-- ---------------------------------------------------------------------------
create table if not exists reviews (
  id uuid primary key default gen_random_uuid(),
  place_id uuid not null references places(id) on delete cascade,
  snapshot_id uuid not null references source_snapshots(id) on delete cascade,
  source text not null check (source in ('chudu24', 'google', 'tripadvisor')),
  external_review_id text,
  review_date date,
  score numeric,
  score_scale int not null default 5,
  title text,
  body text not null,
  review_url text,
  author text,
  created_at timestamptz not null default now()
);

create index if not exists reviews_place_source_date_idx
  on reviews (place_id, source, review_date desc);
create index if not exists reviews_snapshot_id_idx on reviews (snapshot_id);

-- ---------------------------------------------------------------------------
-- documents (pgvector)
-- ---------------------------------------------------------------------------
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  place_id uuid not null references places(id) on delete cascade,
  review_id uuid references reviews(id) on delete cascade,
  source text not null check (source in ('chudu24', 'google', 'tripadvisor')),
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(1536),
  created_at timestamptz not null default now()
);

create index if not exists documents_place_id_idx on documents (place_id);
create index if not exists documents_review_id_idx on documents (review_id);

-- Optional IVFFlat index (run after you have data):
-- create index documents_embedding_idx on documents
--   using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- ---------------------------------------------------------------------------
-- conversations / messages (Supabase Auth user id)
-- ---------------------------------------------------------------------------
create table if not exists conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  title text not null default 'New conversation',
  place_ids uuid[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists conversations_user_id_idx on conversations (user_id);

create table if not exists messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  sources jsonb not null default '[]'::jsonb,
  evidence jsonb,
  created_at timestamptz not null default now()
);

create index if not exists messages_conversation_id_idx on messages (conversation_id);

-- ---------------------------------------------------------------------------
-- match_documents RPC
-- ---------------------------------------------------------------------------
create or replace function match_documents(
  query_embedding vector(1536),
  match_count int default 8,
  filter_place_id uuid default null
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
  order by d.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

-- ---------------------------------------------------------------------------
-- RLS (API uses service role; enable for future direct FE access)
-- ---------------------------------------------------------------------------
alter table places enable row level security;
alter table source_snapshots enable row level security;
alter table reviews enable row level security;
alter table documents enable row level security;
alter table conversations enable row level security;
alter table messages enable row level security;

drop policy if exists places_public_read on places;
create policy places_public_read on places for select using (true);

drop policy if exists snapshots_public_read on source_snapshots;
create policy snapshots_public_read on source_snapshots for select using (true);

drop policy if exists reviews_public_read on reviews;
create policy reviews_public_read on reviews for select using (true);

drop policy if exists documents_public_read on documents;
create policy documents_public_read on documents for select using (true);

drop policy if exists conversations_owner on conversations;
create policy conversations_owner on conversations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists messages_owner on messages;
create policy messages_owner on messages
  for all using (
    exists (
      select 1 from conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );
