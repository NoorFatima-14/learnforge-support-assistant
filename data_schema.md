# Data / Chunk Schema

## Chunking approach

One chunk per source entry: one FAQ, one policy article, one ticket
transcript (see `src/ingest.py`). These documents already are short,
single-topic units (a few hundred words), and several deliberately place a
"current vs. superseded" correction inside the same entry (e.g. POLICY-01:
"An older version of this article stated X, that is outdated, the current
rule is Y"). Splitting below the entry level would risk separating the
correction from the fact it corrects — a real hallucination hazard for this
specific dataset. See `README.md` → Trade-offs for when I'd chunk finer.

## Prototype record (what's actually implemented)

The prototype stores chunks as plain JSON (`index/chunks.json`) plus the
chunk embeddings as a numpy array (`index/embeddings.npy`, one 384-dim
vector per chunk from `all-MiniLM-L6-v2`, L2-normalized so cosine
similarity is a plain dot product). One record looks like:

```json
{
  "id": "policy-01",
  "source_type": "policy",
  "title": "Subscription Plans and Billing",
  "text": "POLICY: Subscription Plans and Billing\n\nLearnForge subscriptions provide access to a changing catalog ...",
  "status": null,
  "last_reviewed": "February 2026",
  "contains_superseded_note": true,
  "tags": ["policy"]
}
```

Field notes:
- `id` — stable, human-readable, derived from the source doc's own
  numbering (`FAQ-02` → `faq-02`). Doubles as the citation key the LLM is
  asked to return in `sources`.
- `source_type` — `faq` | `policy` | `ticket`. Used for filtering/boosting
  and shown to the LLM so it can weigh a resolved past ticket differently
  from an authoritative policy.
- `text` — the full entry, prefixed with a type label (`"FAQ: ..."`,
  `"POLICY: ..."`, `"PAST SUPPORT TICKET: ..."`) so the LLM sees the
  document type inline even outside the structured field.
- `status` — tickets only (`Resolved`, `Escalated`, `Pending monitoring`,
  etc.), parsed from the `STATUS:` line. Signals whether a past ticket's
  resolution should be treated as a settled precedent or a flag that this
  topic tends to need a human.
- `last_reviewed` — policies only, parsed from `Last reviewed:` / `Effective
  date:` / similar lines when present. This is the actual mechanism for
  "staleness" in this prototype: a policy chunk older than N months could be
  demoted or flagged (not yet wired up as a retrieval-time penalty — see
  Trade-offs).
- `contains_superseded_note` — heuristic boolean (regex over the text for
  "outdated", "no longer", "archived", etc.) surfaced to the LLM prompt as
  an instruction to trust the current statement over the superseded one.

## Production mapping (vector DB table)

At real scale this becomes a table in a vector-capable store (Postgres +
pgvector, for concreteness — the same shape maps to Pinecone/Weaviate/Chroma
with minor renames):

```sql
CREATE TABLE kb_chunks (
    id                  TEXT PRIMARY KEY,       -- e.g. "policy-01#chunk-2" if split further
    source_doc_id       TEXT NOT NULL,          -- groups chunks back to one source article
    source_type         TEXT NOT NULL,          -- 'faq' | 'policy' | 'ticket' | ...
    title               TEXT NOT NULL,
    text                TEXT NOT NULL,
    embedding           VECTOR(768),             -- pgvector column, real embedding model
    status               TEXT,                    -- tickets: resolution status
    last_reviewed_at    DATE,                    -- policies: for staleness scoring/decay
    superseded_by        TEXT REFERENCES kb_chunks(id),  -- explicit link, not just a text note
    source_url           TEXT,                    -- link back to the live help-center article
    ingested_at          TIMESTAMPTZ NOT NULL,
    content_hash         TEXT NOT NULL,           -- detect when a re-crawl actually changed content
    tags                 TEXT[]
);

CREATE INDEX ON kb_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON kb_chunks (source_type);
CREATE INDEX ON kb_chunks (last_reviewed_at);
```

What changes vs. the prototype, and why:
- `embedding` — same idea as the prototype (a real embedding vector, not
  TF-IDF), but backed by an `ivfflat` index for approximate nearest-neighbor
  search instead of brute-force comparison against every row — necessary
  once the table is too large to scan on every query. The prototype's
  ~40-chunk brute-force search doesn't need this yet.
- `superseded_by` — an explicit foreign key instead of a regex-detected note
  in free text. This is the single highest-leverage change for the
  staleness problem: content ingestion should be the place contradictions
  get resolved (by whoever owns the help center), not something inferred at
  query time.
- `content_hash` / `ingested_at` — lets a re-ingestion job detect drift and
  flag chunks that haven't been reviewed in N days, independent of whether
  the text happens to mention "outdated."
- `source_url` — every answer can link back to the live doc, so a user (or
  a human agent picking up an escalation) can verify currency themselves.
