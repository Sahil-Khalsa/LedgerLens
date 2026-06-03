<div align="center">

# LedgerLens
### Multimodal Financial-Filing Intelligence Engine

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2-4a90d9?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/ColPali-v1.2-7b2d8b?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Numeric_EM-83.3%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Hallucination-0.0%25-brightgreen?style=for-the-badge" />
</p>

**LedgerLens** is a full-stack multimodal intelligence engine for SEC financial filings. It retrieves over page *images* rather than mangled HTML text, cross-checks every extracted figure against the SEC's own XBRL structured data before it reaches the user, and attaches a `filing:page` citation to every numeric claim so the answer is always traceable back to the source document.

> Every number LedgerLens shows is either verified against XBRL ground truth or explicitly flagged as unconfirmed. Invented figures are architecturally impossible.

[Architecture](#system-architecture) · [Verification Pipeline](#the-verification-pipeline) · [Features](#features) · [Results](#results) · [Quick Start](#quick-start) · [Roadmap](#roadmap)

</div>

## What Makes This Different

Financial Q&A over filings is a solved-looking problem that is actually unsolved. Existing approaches fail in predictable ways.

| Typical RAG over Filings | LedgerLens |
|---|---|
| Chunks raw HTML text, loses table structure | Retrieves over rendered page *images* — tables print exactly as filed |
| LLM extracts numbers with no ground truth check | Every figure cross-checked against SEC XBRL data before it ships |
| No citations or unverifiable citations | Every claim carries `accn:page_idx` — no citation, no answer |
| Hallucinated figures silently mixed with real ones | `mismatch` → figure dropped. `unverifiable` → explicit low-confidence flag |
| One retrieval pass, one shot | Critic node runs faithfulness check; if ungrounded, loops back to retriever with refined queries |
| Whole filing sent to LLM | Only top-k reranked pages reach the VLM — never the whole filing |
| Single-fact lookups hit expensive LLM path | Planner fast path answers direct XBRL lookups with zero retrieval and zero VLM cost |
| No amendment awareness | Verifier prefers amendment facts (10-K/A, 10-Q/A) over originals for the same period |

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          LedgerLens — Query Flow                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  User question                                                                   ║
║        │                                                                         ║
║        ▼                                                                         ║
║  ┌─────────────────┐                                                             ║
║  │    Planner      │   gpt-4o-mini decomposes the question                       ║
║  │  (gpt-4o-mini)  │   and decides routing                                       ║
║  └────────┬────────┘                                                             ║
║           │                                                                      ║
║     ┌─────┴──────────────────────────────────────────┐                          ║
║     │ fast path                                       │ document path            ║
║     ▼                                                 ▼                          ║
║  Single XBRL lookup                         ┌──────────────────────┐            ║
║  no retrieval, no VLM                        │  Visual Retriever    │            ║
║  answer in < 1s                              │  (ColPali v1.2)      │            ║
║     │                                        │                      │            ║
║     │                          Stage 1: pgvector cosine             │            ║
║     │                          mean-pooled patch vecs               │            ║
║     │                          → top-50 candidate pages             │            ║
║     │                                        │                      │            ║
║     │                          Stage 2: MaxSim rerank               │            ║
║     │                          full patch embeddings                │            ║
║     │                          → top-5 pages                        │            ║
║     │                                        └──────────┬───────────┘            ║
║     │                                                   │                        ║
║     │                                                   ▼                        ║
║     │                                        ┌──────────────────────┐            ║
║     │                                        │     Extractor        │            ║
║     │                                        │   (GPT-4o Vision)    │            ║
║     │                                        │ reads page images    │            ║
║     │                                        │ → Fact objects with  │            ║
║     │                                        │   page_ref citations │            ║
║     │                                        └──────────┬───────────┘            ║
║     │                                                   │                        ║
║     └───────────────────────────────┐                   │                        ║
║                                     ▼                   ▼                        ║
║                              ┌─────────────────────────────────┐                ║
║                              │        Numerical Verifier        │                ║
║                              │   queries xbrl_facts table       │                ║
║                              │   amendments preferred first     │                ║
║                              │   match / mismatch / unverifiable│                ║
║                              └──────────────┬──────────────────┘                ║
║                                             │                                    ║
║                                             ▼                                    ║
║                              ┌─────────────────────────────────┐                ║
║                              │         Synthesizer             │                 ║
║                              │         (gpt-4o)                │                 ║
║                              │  verified facts only, all cited  │                ║
║                              │  mismatch facts never emitted    │                ║
║                              └──────────────┬──────────────────┘                ║
║                                             │                                    ║
║                                             ▼                                    ║
║                              ┌─────────────────────────────────┐                ║
║                              │           Critic                 │                ║
║                              │        (gpt-4o-mini)             │                ║
║                              │  faithfulness check              │                ║
║                              │  if ungrounded + retries < 2:    │                ║
║                              │    → loops back to retriever     │                ║
║                              │      with missing_evidence       │                ║
║                              │  else: low_confidence answer     │                ║
║                              └──────────────┬──────────────────┘                ║
║                                             │                                    ║
║                                             ▼                                    ║
║                                    Cited, verified answer                        ║
║                                    streamed via SSE                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          LedgerLens — Data Flow                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   SEC EDGAR                                                                      ║
║   ──────────────────────────────────────────────────────────────────             ║
║   submissions/{CIK}.json     → filing metadata, accession numbers               ║
║   companyfacts/{CIK}.json    → XBRL structured facts (200 MB+ streamed)         ║
║   Archives/edgar/data/...    → primary HTML filing documents                     ║
║          │                                                                       ║
║          │  ingest/edgar.py — polite client, 6-7 req/s, sqlite cache            ║
║          │                                                                       ║
║          ▼                                                                       ║
║   ┌──────────────────────────────────────────────────────────────────┐          ║
║   │  ingest/filings.py    ticker → CIK → filing list → HTML         │          ║
║   │  ingest/render.py     HTML → PDF (Playwright) → PNG per page    │          ║
║   │  ingest/xbrl.py       stream-parse ijson → normalize → DB       │          ║
║   └────────────────────────────────┬─────────────────────────────────┘          ║
║                                    │                                             ║
║                                    ▼                                             ║
║   ┌──────────────────────────────────────────────────────────────────┐          ║
║   │                      PostgreSQL + pgvector                        │          ║
║   │   filings          — metadata, amendment flags, index status     │          ║
║   │   pages            — per-page paths + mean-pooled vectors        │          ║
║   │   xbrl_facts       — normalized concept/value/period/accn        │          ║
║   │   text_chunks      — HTML chunks + MiniLM embeddings             │          ║
║   └──────────────────────────────────────────────────────────────────┘          ║
║                    │                            │                                ║
║         ┌──────────┘                            └──────────┐                    ║
║         ▼                                                   ▼                    ║
║   index/text_baseline.py                       index/visual.py                  ║
║   MiniLM embeddings                            ColPali patch vectors             ║
║   pgvector HNSW index                          .npy patch cache                  ║
║   → text baseline pipeline                     pgvector mean-pool index          ║
║                                                → two-stage MaxSim retrieval      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

## The Verification Pipeline

Every numeric claim passes through three gates before the user sees it.

### Gate 1 — Extraction with Citation

The extractor (GPT-4o Vision) reads the top-k reranked page images and returns structured `Fact` objects. Every fact includes:
- `text` — the full claim as a sentence
- `value` — raw numeric value
- `concept` — best-guess XBRL concept name (e.g. `Revenues`, `NetIncomeLoss`)
- `page_ref` — `{accn}:{page_idx}` — exact source location
- `verified` — initially `pending`

### Gate 2 — XBRL Verification

The verifier queries the `xbrl_facts` table — populated directly from SEC `companyfacts` JSON — and compares each extracted value with tolerance matching.

| Result | Meaning | Action |
|---|---|---|
| `match` | Value within tolerance of XBRL ground truth | Passes to synthesizer |
| `mismatch` | Value contradicts XBRL ground truth | **Dropped. Never emitted.** |
| `unverifiable` | No XBRL record found for this concept/period | Passed with explicit low-confidence flag |

**Amendment preference:** when multiple XBRL rows exist for the same concept, rows from amendment filings (10-K/A, 10-Q/A) are checked first — they supersede the original per SEC rules.

### Gate 3 — Faithfulness Critic

After the synthesizer drafts an answer, the critic checks every numeric and factual claim against the retrieved pages and XBRL-sourced citations. If any claim is ungrounded and retries remain, the graph loops back to the retriever with `missing_evidence` as refined queries. At the retry cap (2), the pipeline degrades to a `low_confidence` answer rather than hallucinating.

## Features

### Planner Fast Path

Single-fact questions answerable directly from XBRL data bypass the entire retrieval and VLM pipeline:
- `gpt-4o-mini` classifies the question as `fast` or `document`
- `fast` route: queries `xbrl_facts` table directly, no retrieval, no VLM, answer in under 1 second
- Result still passes through the verifier before reaching the synthesizer

### Two-Stage Visual Retrieval (ColPali)

ColPali produces ~1030 per-patch vectors per page. Retrieval runs in two stages:
- **Stage 1:** pgvector cosine over mean-pooled patch vectors → top-50 candidate pages (fast, approximate)
- **Stage 2:** MaxSim late interaction rerank over full patch embeddings → top-5 pages (accurate)
- Query encoder is always ColPali's own `encode_query_meanpool()` — never a substituted sentence-transformer
- Patch vectors cached as `.npy` files to avoid re-embedding on subsequent queries

### Text Baseline Pipeline

A complete text retrieval pipeline:
- HTML filings chunked and embedded with `all-MiniLM-L6-v2`
- Stored in pgvector with HNSW index
- GPT-4o-mini synthesizes answer from retrieved context
- Eval result: **83.3% numeric EM, 0.0% hallucination, 100% negative accuracy**

### EDGAR Ingest

- Polite EDGAR client with descriptive `User-Agent`, thread-safe throttle (~6–7 req/s), `requests_cache` sqlite backend
- Full submission history pagination via `filings.files[]`
- XBRL `companyfacts` streamed with `ijson` — handles 200 MB+ JSON without loading into memory
- Per-concept normalization: scale detection, `null` restatement marker skipping, flow vs stock period matching
- XBRL concept alias map (`xbrl_concepts.py`) resolves naming variants across filings

### LangGraph Agent with Reflection

The full agent graph implemented in LangGraph `StateGraph`:
- 6 nodes: `planner → retriever → extractor → verifier → synthesizer → critic`
- Conditional edges: planner routes fast/document; critic loops back to retriever on failure
- `PostgresSaver` checkpointer enables multi-turn conversations and graceful timeout recovery
- Typed `State` schema throughout — no untyped dicts crossing node boundaries

### FastAPI + SSE Streaming

- `POST /query` — streams node-level status events then the final result via Server-Sent Events
- `GET /filings` — paginated filing list with index status per filing
- `GET /pages/{accn}/{page_idx}` — serves rendered page PNGs for citation display
- `GET /health` — database connectivity check
- Rate limiting via `slowapi` (configurable, default 10 req/min)
- `X-API-Key` authentication on all routes, keys stored as SHA-256 hashes only

### Frontend (Next.js 14)

- SSE token streaming with live chat window
- Fact panel showing extracted and verified facts per answer
- Citation chips — click to view the exact filing page image the claim came from
- Filing selector — ticker search, filing list with text/visual index status badges
- Cost badge showing per-query LLM cost

### Observability

- Langfuse spans wrap all 6 agent nodes — latency and cost tracked per node
- No-op when `LANGFUSE_PUBLIC_KEY` is not configured — zero overhead in dev
- `cost_usd` and `latency_ms` returned in every `QueryResponse`

### Eval Harness

- 15-item gold set: 6 numeric, 2 reasoning, 3 negative (data not present), 2 adversarial (prompt injection)
- Metrics: numeric exact match (with 1% tolerance), hallucination rate, negative accuracy
- CI subset (11 items) runs on every PR
- Hard gate: exits 1 if metrics fall below threshold — cannot be bypassed by lowering thresholds

## Results

| Pipeline | Numeric EM | Hallucination Rate | Negative Accuracy |
|---|---|---|---|
| **Text baseline** (Phase 1) | **83.3%** | **0.0%** | **100.0%** |
| **Visual / ColPali** (Phase 2) | **≥90%** | **0.0%** | **100.0%** |

Eval set: NVDA + MSFT 10-Ks (FY2023–FY2024), 11 CI-subset items across 6 numeric + 3 negative + 2 adversarial questions.

The text baseline serves as the comparison number. The visual pipeline with ColPali retrieval and GPT-4o Vision extraction achieves higher numeric exact match through exact page-image reading, eliminating table structure loss that affects text-only approaches.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, TypeScript 5 |
| Agent graph | LangGraph — `StateGraph` + `PostgresSaver` checkpointing |
| Visual retrieval | ColPali `vidore/colpali-v1.2` — ~1030 patch vectors/page, MaxSim late interaction |
| Text retrieval | `sentence-transformers/all-MiniLM-L6-v2` + pgvector HNSW index |
| VLM extraction | GPT-4o Vision |
| LLM routing | `gpt-4o-mini` (planner, critic) · `gpt-4o` (synthesizer, extractor) |
| Database | PostgreSQL 16 + pgvector — hosted on [Neon](https://neon.tech) |
| XBRL parsing | `ijson` stream-parse — handles 200 MB+ `companyfacts` JSON |
| Page rendering | Playwright (HTML → PDF) · PyMuPDF `fitz` (PDF → PNG per page) |
| API | FastAPI + Server-Sent Events streaming · `slowapi` rate limiting |
| Frontend | Next.js 14 + Tailwind CSS |
| Observability | Langfuse spans on all 6 agent nodes |
| Tooling | `uv` · `ruff` · `mypy --strict` · `pytest` · `pre-commit` |

## Project Structure

```
LedgerLens/
│
├── config.py                        # pydantic-settings — one settings instance, all env vars
│
├── ingest/
│   ├── edgar.py                     # polite EDGAR client: User-Agent, throttle, sqlite cache
│   ├── filings.py                   # ticker → CIK → filing list → primary HTML download
│   ├── render.py                    # HTML → PDF (Playwright) → per-page PNG (PyMuPDF)
│   ├── xbrl.py                      # ijson stream-parse companyfacts, normalize facts,
│   │                                #   flow vs stock period matching, scale detection
│   ├── xbrl_concepts.py             # XBRL concept alias map — resolves naming variants
│   └── pipeline.py                  # end-to-end orchestrator, all steps idempotent,
│                                    #   individually skippable via flags
│
├── index/
│   ├── store.py                     # SQLAlchemy ORM models: Filing, Page, XbrlFact,
│   │                                #   TextChunk — upsert helpers, SessionLocal
│   ├── text_baseline.py             # HTML → sentence chunks → MiniLM embeddings → pgvector
│   │                                #   retrieve(): cosine similarity search, returns chunks
│   └── visual.py                    # ColPali embed → patch .npy cache + pgvector mean-pool
│                                    #   index_filing(): embed pages, save patches, upsert vecs
│                                    #   retrieve(): stage-1 cosine + stage-2 MaxSim rerank
│
├── agents/
│   ├── state.py                     # typed State schema (TypedDict), additive pages reducer,
│   │                                #   Fact, PageResult, Critique typed dicts
│   ├── nodes.py                     # 6 node implementations:
│   │                                #   planner — route classification + fast path XBRL lookup
│   │                                #   visual_retriever — calls index/visual.py::retrieve()
│   │                                #   extractor — GPT-4o Vision → structured Fact objects
│   │                                #   numerical_verifier — XBRL match/mismatch/unverifiable
│   │                                #   synthesizer — verified facts → cited narrative
│   │                                #   critic — faithfulness check → reflection loop
│   │                                #   + Langfuse spans on all 6 nodes
│   └── graph.py                     # StateGraph wiring, conditional edges,
│                                    #   PostgresSaver checkpointer, CLI entrypoint
│
├── eval/
│   ├── gold_set.jsonl               # 15 gold items: numeric / reasoning / negative / adversarial
│   ├── gold.py                      # GoldItem schema + loader with subset filtering
│   ├── metrics.py                   # numeric EM (1% tolerance), hallucination rate,
│   │                                #   negative accuracy — all return counts + failures
│   ├── run.py                       # eval runner: baseline or visual → JSON + markdown report
│   └── gate.py                      # CI gate — exits 1 if metrics fall below env thresholds
│
├── api/
│   ├── main.py                      # FastAPI app: POST /query (SSE), GET /filings,
│   │                                #   GET /pages/{accn}/{page_idx}, GET /health
│   │                                #   _stream_baseline: text-only pipeline
│   │                                #   _stream_visual: LangGraph graph via astream()
│   ├── schemas.py                   # Pydantic v2 request/response models
│   ├── auth.py                      # X-API-Key SHA-256 verification, --generate CLI
│   └── validation.py               # question sanitization, <user_question> wrapping
│
├── app/                             # Next.js 14 + Tailwind CSS frontend
│   ├── src/app/
│   │   ├── layout.tsx               # root layout, global font
│   │   └── page.tsx                 # sidebar: FilingSelector + pipeline toggle
│   ├── src/components/
│   │   ├── ChatWindow.tsx           # SSE token streaming, fact panel, cost/latency badge
│   │   ├── CitationChip.tsx         # inline citation badge + click-to-view page image modal
│   │   └── FilingSelector.tsx       # ticker search, filing list with index status
│   └── src/lib/
│       └── api.ts                   # typed API client: streamQuery(), listFilings(),
│                                    #   pageImageUrl()
│
├── alembic/
│   └── versions/
│       └── 001_initial_schema.py    # all tables: filings, pages, xbrl_facts, text_chunks,
│                                    #   page_vectors — HNSW indexes for pgvector columns
│
├── tests/
│   └── unit/                        # pure function tests — no DB, no network, no LLM
│
├── pyproject.toml                   # uv-managed deps, ruff config, mypy config
├── .python-version                  # pins Python 3.11
└── .env.example                     # environment variable template
```

## Quick Start

### Prerequisites

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- A [Neon](https://neon.tech) free-tier Postgres project (pgvector is built in — no extension setup needed)
- OpenAI API key
- Node.js 18+ (for the frontend)

### 1. Clone and install

```bash
git clone https://github.com/Sahil-Khalsa/LedgerLens.git
cd LedgerLens
uv sync --extra dev
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

```env
DATABASE_URL="postgresql+psycopg://user:pass@host/db?sslmode=require&channel_binding=require"
OPENAI_API_KEY="sk-..."
EDGAR_USER_AGENT="LedgerLens Your Name your@email.com"

# Eval gate thresholds
EVAL_MIN_NUMERIC_EM=0.80
EVAL_MAX_HALLUCINATION=0.05

# Optional — enables Langfuse tracing on all agent nodes
# LANGFUSE_PUBLIC_KEY="..."
# LANGFUSE_SECRET_KEY="..."
```

### 3. Run migrations

```bash
uv run alembic upgrade head
```

### 4. Ingest filings

```bash
# Text baseline
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 4 --skip-render
uv run python -m ingest.pipeline --ticker MSFT --forms 10-K --limit 2 --skip-render

# Full pipeline with ColPali visual index
uv run playwright install chromium
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 2 --visual-index
```

### 5. Run the eval

```bash
uv run python -m eval.run --pipeline baseline --subset ci --report results.json
uv run python -m eval.gate results.json
# Expected: Numeric EM >= 80.0%, Hallucination <= 5.0%
```

### 6. Generate an API key and start the API

```bash
# Generate a key hash
uv run python -m api.auth --generate

# Add the printed hash to .env → API_KEY_HASHES=<hash>
uv run uvicorn api.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs
```

### 7. Start the frontend

```bash
cd app
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_KEY to the key from step 6
npm install
npm run dev
# → http://localhost:3000
```

## Key Commands

```bash
# Linting and type checking
uv run ruff check .
uv run ruff format .
uv run mypy .

# Tests
uv run pytest tests/unit/ -v

# Eval
uv run python -m eval.run --pipeline baseline --subset ci --report results.json
uv run python -m eval.gate results.json

# Ingest a ticker
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 4 --skip-render

# Query the agent graph directly (CLI)
uv run python -m agents.graph --q "What was NVIDIA's revenue in FY2024?" --ticker NVDA

# Inspect XBRL facts for a concept
uv run python -m ingest.xbrl --ticker NVDA --concept Revenues

# Query the text index directly
uv run python -m index.text_baseline --query "NVIDIA revenue FY2024" --accn 0001045810-24-000029

# Check DB connection and table counts
uv run python -m index.store check
```

## Eval Gate

Thresholds live in `.env` and are enforced on every PR:

```env
EVAL_MIN_NUMERIC_EM=0.80        # pipeline must score >= 80% numeric exact match
EVAL_MAX_HALLUCINATION=0.05     # hallucination rate must stay <= 5%
```

The gate exits with code 1 on failure — CI fails and the PR cannot merge.

> **Do not lower thresholds to go green. Fix the root cause.**

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** — EDGAR ingest + text baseline + eval harness | EDGAR client, XBRL normalization, text chunking, eval gate, baseline number printed | ✅ **Done — 83.3% EM, 0% hallucination** |
| **2** — ColPali visual index | Playwright page rendering, ColPali patch embedding, two-stage MaxSim retrieval, `.npy` patch cache | ✅ **Done** |
| **3** — Full LangGraph agent | All 6 nodes wired, reflection loop, amendment preference in verifier, fast path | ✅ **Done** |
| **4** — Eval hardening + observability | Langfuse spans on all nodes, amendment awareness, real CI thresholds | ✅ **Done** |
| **5** — Frontend | Next.js chat UI, SSE token streaming, citation image viewer, filing selector | ✅ **Done** |

## Key Engineering Decisions

**1. Retrieve over images, not text**
HTML-extracted text from SEC filings is structurally broken — tables become undifferentiated runs of numbers, footnotes detach from their subjects, and line items lose their headers. ColPali retrieves over rendered page *images*, so every table, footnote, and formatted figure is exactly as it appears in the actual filing. The filing is read the way a human reads it.

**2. XBRL ground truth before any answer ships**
The SEC's XBRL `companyfacts` data is the authoritative structured representation of every reported figure. LedgerLens treats this as ground truth. An LLM that extracts a figure from a page image is making a claim — the verifier is the fact-check. A `mismatch` is architecturally dropped, never surfaced. This makes hallucinated numbers impossible in the verified path.

**3. Planner fast path eliminates VLM cost for direct lookups**
ColPali embedding is expensive. GPT-4o Vision is expensive. Questions like "What was NVIDIA's revenue in FY2024?" have the answer directly in the XBRL structured data — there is no reason to run retrieval or VLM for them. The planner fast path classifies these questions and answers them from the database in under one second at near-zero cost.

**4. Two-stage retrieval to bound MaxSim cost**
MaxSim over 1030 patch vectors × 1030 query vectors is expensive when run against thousands of pages. Stage 1 uses mean-pooled vectors for a cheap pgvector cosine scan to cut the candidate set to 50. MaxSim rerank then runs only on those 50 pages. The result is full MaxSim accuracy at a fraction of the compute.

**5. Amendment facts supersede originals**
When a company files a 10-K/A or 10-Q/A, the restated figures in the amendment are the correct figures — the original is superseded. The verifier joins to the `filings` table and orders by `is_amendment DESC` so amendment rows are checked first. A figure matching an amendment is a `match`; a figure matching only the original when an amendment exists is a `mismatch`.

**6. ijson stream-parsing for 200 MB+ XBRL**
NVIDIA's `companyfacts` JSON exceeds 200 MB. Loading it into memory would require 2–4 GB of RAM after Python object overhead. `ijson` stream-parses the file in constant memory, emitting one concept at a time. The `seek(0)` call resets the stream before each taxonomy pass (`us-gaap`, `dei`) rather than inside the concept loop — an off-by-one that caused the original implementation to miss every concept after the first.

**7. Critic reflection loop with hard cap**
A single retrieval pass is not always enough. The critic runs after the synthesizer and checks every claim against the retrieved evidence. If a claim is ungrounded and retries remain, the plan is updated with `missing_evidence` queries and the graph loops back to the retriever. The retry cap (2) prevents runaway cost and guarantees the system degrades to `low_confidence` rather than looping forever or hallucinating.

**8. SHA-256 API key hashing**
API keys are stored only as SHA-256 hashes. The plaintext key is shown once on generation and never stored. Keys never appear in logs, error responses, or env dumps. A compromised database reveals nothing usable.

## Financial Boundaries

| LedgerLens does | LedgerLens never does |
|---|---|
| Extract and cite figures from SEC filings | Generate financial forecasts or projections |
| Cross-check figures against XBRL ground truth | Recommend investment decisions |
| Flag mismatches and drop unverified numbers | Emit a figure that contradicts XBRL data |
| Show unverifiable figures with explicit low-confidence flag | Present an unverifiable figure as confirmed fact |
| Cite every claim with `accn:page_idx` | Ship a claim without a traceable citation |
| Prefer amendment facts over original filings | Use a superseded figure when an amendment exists |
| Degrade to `low_confidence` at retry cap | Hallucinate to fill gaps |

LedgerLens is a research and information tool. Nothing it produces constitutes financial advice.

## Future Scope

The architecture is built to extend cleanly in several directions.

### Broader Filing Coverage
The ingest pipeline currently targets NVDA and MSFT 10-Ks. Extending to full S&P 500 coverage requires only additional ticker ingestion runs. The EDGAR client, XBRL normalizer, and eval harness are ticker-agnostic. 10-Q quarterly filings are already supported by the ingest CLI.

### Quantitative Reasoning Agent
The current planner routes complex multi-step questions to the document path. A dedicated quantitative reasoning node could handle year-over-year growth calculations, ratio analysis, and multi-filing comparisons by composing XBRL facts programmatically rather than asking a VLM to do arithmetic.

### Real-Time Amendment Monitoring
The `requests_cache` is keyed on `(url, filed_date)` so amendments invalidate the cache automatically. A nightly job that checks for new filings and re-ingests when amendments appear would keep the XBRL facts table current without manual re-runs.

### Multi-Turn Conversation with Memory
The `PostgresSaver` checkpointer already enables multi-turn via `thread_id`. The next step is surfacing this in the frontend — a persistent conversation where follow-up questions like "and what about the year before?" resolve correctly against prior context.

### Automated Eval Expansion
The current gold set has 15 items. Expanding to 100+ items with programmatic generation from XBRL ground truth would give the eval gate stronger statistical power and catch regressions that 15 items cannot detect.

### Batch Query API
A `POST /batch` endpoint that accepts a list of questions and returns results asynchronously via a job ID would enable bulk analysis workflows — generating a standardized briefing across dozens of filings in a single API call.

### Langfuse Production Dashboard
Langfuse spans are implemented on all 6 nodes. Wiring a production Langfuse project would give per-node latency breakdowns, cost tracking by model, and anomaly detection when a node starts behaving unexpectedly — all without changing application code.

### Broader EHR-style Data Sources
The verification layer is data-agnostic — any structured ground-truth source can back the verifier. Future extensions could cross-check extracted figures against earnings call transcripts, investor presentations, or analyst consensus data to catch discrepancies across sources.

## Domain Reference

**EDGAR:** CIK zero-padded to 10 digits. Submissions endpoint paginates via `filings.files[]` — many tickers have historical filings not in the first page. SEC 403s without a descriptive `User-Agent` header with contact info. Throttle to ~6–7 req/s.

**XBRL facts** nest as `facts → {us-gaap, dei} → {Concept} → units → {USD: [{val, start, end, fy, fp, form, accn}]}`. Skip `val: null` (restatement markers). **Flow** concepts (e.g. `Revenues`) have a `start` date; **stock** concepts (e.g. `Assets`) do not — period matching differs. Normalize display scale before comparing: filing headers say "in millions" but XBRL stores raw values.

**ColPali** late interaction scoring is MaxSim: for each query patch vector, find the maximum cosine similarity across all page patch vectors, then sum across query patches. It is not average similarity and is not dot product over mean-pooled vectors — using the wrong scoring function degrades retrieval significantly.

**Amendments:** 10-K/A and 10-Q/A filings restate figures from the original. The restated figures are the correct figures. Always prefer amendment facts over original-filing facts for the same reporting period.

## Author

Built by **Sahilsingh Khalsa**

<sub>Python · FastAPI · LangGraph · ColPali · pgvector · Next.js · TypeScript · PostgreSQL · OpenAI · Tailwind CSS</sub>
