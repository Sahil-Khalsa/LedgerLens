# LedgerLens — Roadmap

All phases derive from `DESIGN.md` (architecture rationale) and `SPEC.md` (feature spec). SPEC.md overrides DESIGN.md on conflicts.

---

## Phase 1 — Text Baseline & Eval Harness (Weeks 1–2)

**Goal:** Print a numeric exact-match score for the text-only pipeline. This is the comparison number the visual pipeline must beat.

**Scope:** NVDA + MSFT, 3–4 filings each (10-K, 10-Q).

### Deliverables

| Deliverable | Done criteria |
|---|---|
| EDGAR polite client | Rate-limited, cached, `User-Agent` set; all SEC fetches go through `ingest/edgar.py` |
| XBRL ingestion | `companyfacts` normalized into `xbrl_facts` table; scale detection applied; flow vs. stock tagged |
| Filing download + render | HTML → PDF → PNG pages; `pages` table populated; `filing.page_count` correct |
| Text baseline index | HTML → chunked text → `all-MiniLM-L6-v2` embeddings → `chunks` table with HNSW index |
| Eval harness | Gold set loaded from `eval/gold_set.jsonl`; `eval/run.py` produces exact-match + tolerance metrics |
| Eval gate | `eval/gate.py` enforces thresholds in CI; fails if not met (never lower thresholds) |
| CI green | ruff + mypy strict + pytest unit tests all pass |

### Key files

```
ingest/   edgar.py filings.py render.py xbrl.py xbrl_concepts.py
index/    text_baseline.py store.py
eval/     gold.py metrics.py run.py gate.py gold_set.jsonl
api/      main.py (baseline SSE endpoint)
```

### Status

**Scaffolded. CI green. End-to-end thin thread not yet wired.**

- All source files exist and pass lint/type checks.
- `ingest/pipeline.py` (the orchestrator that chains edgar → render → xbrl → text-index for a given filing) is not yet written.
- The eval gold set (`gold_set.jsonl`) needs real NVDA/MSFT numeric entries.
- Phase 1 is complete when `eval/gate.py` passes on the baseline pipeline.

---

## Phase 2 — ColPali Visual Index (Weeks 3–4)

**Goal:** Replace text retrieval with page-image retrieval. Beat the Phase 1 baseline score on the same gold set.

### Deliverables

| Deliverable | Done criteria |
|---|---|
| ColPali indexing | Per-page patch vectors stored (1030 patches × dim); mean-pooled vector in `pages.pooled_embedding` |
| Two-stage retrieval | Stage 1: pgvector cosine over mean-pooled vectors → top-N candidates; Stage 2: MaxSim rerank → top-k pages |
| VLM extraction | Top-k pages sent to VLM (GPT-4o); structured numeric facts extracted with `filing_accn:page_idx` citations |
| Eval delta | Re-run gold set; document exact-match improvement over Phase 1 |

### Key files

```
index/    visual.py (ColPali indexing + retrieval)
agents/   nodes.py::visual_retriever, nodes.py::extractor
```

### Constraints

- ColPali query encoding: always use the model's own `encode_query_meanpool()` — never a separate sentence-transformer for the stage-1 vector.
- The VLM only receives the top-k reranked pages, never an entire filing.
- HNSW index (`ix_pages_pooled_hnsw`) is already defined in schema; alembic migration must be applied.

### Status

**Not started.** `index/visual.py` skeleton exists with `load_patch_vecs()` stub.

---

## Phase 3 — Full LangGraph Agent Graph (Weeks 5–6)

**Goal:** Wire the full reasoning graph with XBRL verification, faithfulness critique, and bounded reflection.

### Agent graph

```
planner → [fast path: verifier] OR [visual_retriever → extractor → verifier] → synthesizer → critic
```

- **Planner** (gpt-4o-mini): classifies route (`fast` or `document`), decomposes multi-hop questions into sub-queries.
- **Fast path**: Single-fact lookups answered directly from `xbrl_facts` — no retrieval, no VLM call.
- **Numerical verifier**: Cross-checks every extracted figure against `xbrl_facts` ground truth. `mismatch` → synthesizer drops it. `unverifiable` → low-confidence flag only. Never emits an unverified number as fact.
- **Synthesizer** (gpt-4o): Assembles answer from verified facts only. Every numeric claim carries `filing_accn:page_idx` citation.
- **Critic** (gpt-4o-mini): Faithfulness check. If ungrounded and `retries < 2`, loops back to retriever with `critique.missing_evidence` as refined queries. At retry cap, degrades to flagged `low_confidence` rather than hallucinating.

### Deliverables

| Deliverable | Done criteria |
|---|---|
| All 5 nodes implemented | planner, visual_retriever, extractor, numerical_verifier, synthesizer, critic |
| Conditional routing | planner → fast/document; critic → retry/end |
| Reflection loop | Critic loops with refined queries, bounded at 2 retries |
| PostgresSaver checkpointing | Multi-turn conversation state persisted; graceful timeout recovery |
| Cost + latency logging | Langfuse spans per node; cost_usd + latency_ms in query log |

### Key files

```
agents/   state.py nodes.py graph.py
index/    store.py::QueryLog
```

### Status

**Scaffolded.** `agents/graph.py` and `agents/nodes.py` exist with stubs. Full node implementations pending.

---

## Phase 4 — Eval Hardening & Observability (Weeks 7–8)

**Goal:** Full three-tier eval harness in CI; production-grade tracing.

### Deliverables

| Deliverable | Done criteria |
|---|---|
| Three-tier gold set | Positive (answerable), negative (unanswerable), adversarial (tempt hallucination) cases |
| CI eval gate | Full gold set runs on every PR; gate enforces exact-match ≥ threshold, hallucination rate ≤ threshold |
| Amendment handling | Verifier prefers amendment facts (10-K/A, 10-Q/A) over original filing for same period |
| Langfuse tracing | All LLM calls instrumented; cost dashboard operational |
| Scale normalization audit | All verifier false-mismatches diagnosed; concept alias map in `xbrl_concepts.py` complete |

### Key files

```
eval/     gold.py metrics.py run.py gate.py gold_set.jsonl
ingest/   xbrl_concepts.py
```

### Status

**Not started.**

---

## Phase 5 — Frontend, ASR, and Forecasting (Weeks 9–10)

**Goal:** End-to-end user-facing product: SSE-streaming chat UI, cited-region viewer, optional earnings-call ASR.

### Deliverables

| Deliverable | Done criteria |
|---|---|
| Next.js frontend | SSE stream renders tokens live; citation chips link to page image; filing selector |
| Cited-region viewer | Page PNG served via `/pages/{accn}/{page_idx}`; bounding box overlay for extracted region |
| Earnings-call ASR | Whisper transcription of quarterly call audio; facts extracted and cross-checked against XBRL |
| Forecasting module | Trend extrapolation from XBRL time series; explicitly labeled as model output, not filing fact |
| Rate limiting | `slowapi` per-IP rate limit enforced on all POST routes |
| Auth hardening | X-API-Key stored as SHA-256 hash only; no plaintext in logs or error responses |

### Key files

```
app/      (Next.js — see SPEC.md §7)
api/      main.py schemas.py auth.py validation.py
```

### Status

**Not started.** API scaffolding (auth, rate limiting, SSE endpoint, `/pages` route) complete.

---

## Non-negotiables (apply to all phases)

1. Never emit an unverified number as fact.
2. Every numeric claim carries a `filing_accn:page_idx` citation.
3. XBRL `companyfacts` is ground truth — normalize before comparing.
4. All EDGAR access goes through `ingest/edgar.py` only.
5. Eval gate must pass before merge — never lower thresholds.
6. Keep the text-only baseline — it is the comparison number that justifies the project.
7. VLM only receives top-k reranked pages, never a whole filing.
8. All API routes require `X-API-Key` authentication.
9. Input validation on every user-facing boundary; user content wrapped in `<user_question>` delimiters.
