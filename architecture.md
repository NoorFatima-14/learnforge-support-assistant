# System Design

## Query flow

```mermaid
flowchart TD
    U[User message] --> RW[Rewrite into standalone query<br/>using conversation history]
    RW --> RET[Retriever: sentence-transformer embeddings<br/>+ cosine similarity over 40-chunk index]
    RET --> TOPK[Top-k chunks + scores]
    TOPK --> GEN[Gemini call<br/>structured JSON output:<br/>answer, confidence, grounded, sources, escalate]
    GEN --> CHECK{Escalate?<br/>LLM said so, OR<br/>retrieval score below threshold}
    CHECK -- yes --> ESC[Flag for human agent<br/>+ reason shown to user]
    CHECK -- no --> OK[Return answer + cited sources]
    ESC --> RESP[Response to user]
    OK --> RESP
    RESP --> HIST[Append turn to history]
```

## Components

| Component | File | Responsibility |
|---|---|---|
| Ingestion | `src/ingest.py` | Parse markdown -> `Chunk` records |
| Retriever | `src/retriever.py` | Build/query the embedding index (sentence-transformers + cosine similarity) |
| LLM client | `src/llm.py` | Query rewriting + grounded generation (Gemini) |
| Pipeline | `src/pipeline.py` | Retrieve -> generate -> escalation decision |
| CLI | `src/chat.py` | Multi-turn conversation loop |
| Eval | `eval/run_eval.py` | Retrieval hit-rate + escalation accuracy checks |

## Out of scope for this prototype

- Persistent conversation storage (history is in-memory for the session)
- Real vector DB at production scale (brute-force cosine similarity over
  ~40 chunks instead — see `docs/data_schema.md` for the FAISS/pgvector
  mapping once the corpus is much larger)
- Human-agent handoff integration (escalation is a flag in the CLI output, not routed anywhere)
