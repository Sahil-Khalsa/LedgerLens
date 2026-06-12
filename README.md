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

**LedgerLens** is a full-stack multimodal intelligence engine for SEC financial filings. It retrieves over rendered page *images* rather than mangled HTML text, cross-checks every extracted figure against the SEC's own XBRL structured data before it reaches the user, and attaches a `filing:page` citation to every numeric claim so every answer is traceable back to the source document.

> Every number LedgerLens shows is either verified against XBRL ground truth or explicitly flagged as unconfirmed. Invented figures are architecturally impossible.

[Architecture](#system-architecture) · [Verification Pipeline](#the-verification-pipeline) · [Features](#features) · [Results](#results) · [Quick Start](#quick-start)

</div>

## What Makes This Different

Financial Q&A over filings is a solved-looking problem that is actually unsolved. Existing approaches fail in predictable ways.

| Typical RAG over Filings | LedgerLens |
|---|---|
| Chunks raw HTML text, loses table structure | Retrieves over rendered page *images* — tables print exactly as filed |
| LLM extracts numbers with no ground-truth check | Every figure cross-checked against SEC XBRL data before it ships |
| No citations or unverifiable citations | Every claim carries `accn:page_idx` — no citation means no answer |
| Hallucinated figures silently mixed with real ones | `mismatch` figures are dropped; `unverifiable` gets an explicit low-confidence flag |
| One retrieval pass, one shot | Critic node runs a faithfulness check and loops back to the retriever with refined queries if ungrounded |
| Whole filing sent to LLM | Only the top-k reranked pages reach the VLM — never the whole filing |
| Single-fact lookups hit the expensive LLM path | Planner fast path answers direct XBRL lookups with zero retrieval and zero VLM cost |
| No amendment awareness | Verifier prefers amendment facts (10-K/A, 10-Q/A) over originals for the same period |

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          LedgerLens: Query Flow                                  ║
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
║     │                                        │   page_ref + bbox    │            ║
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
║                          LedgerLens: Data Flow                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   SEC EDGAR                                                                      ║
║   ──────────────────────────────────────────────────────────────────             ║
║   submissions/{CIK}.json     → filing metadata, accession numbers               ║
║   companyfacts/{CIK}.json    → XBRL structured facts (200 MB+ streamed)         ║
║   Archives/edgar/data/...    → primary HTML filing documents                     ║
║          │                                                                       ║
║          │  ingest/edgar.py: polite client, 6-7 req/s, sqlite cache             ║
║          │                                                                       ║
║          ▼                                                                       ║
║   ┌──────────────────────────────────────────────────────────────────┐          ║
║   │  ingest/filings.py    ticker → CIK → filing list → HTML         │          ║
║   │  ingest/render.py     HTML → PDF (Playwright) → PNG per page    │          ║
║   │  ingest/xbrl.py       stream-parse ijson → normalize → DB       │          ║
║   │  ingest/asr.py        earnings call audio → Whisper → transcript │          ║
║   └────────────────────────────┬─────────────────────────────────────┘          ║
║                                    │                                             ║
║                                    ▼                                             ║
║   ┌──────────────────────────────────────────────────────────────────┐          ║
║   │                      PostgreSQL + pgvector                        │          ║
║   │   filings          : metadata, amendment flags, index status     │          ║
║   │   pages            : per-page paths + mean-pooled vectors        │          ║
║   │   xbrl_facts       : normalized concept/value/period/accn        │          ║
║   │   text_chunks      : HTML/transcript chunks + MiniLM embeddings  │          ║
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

### Gate 1: Extraction with Citation

The extractor (GPT-4o Vision) reads the top-k reranked page images and returns structured `Fact` objects. Every fact includes:
- `text`: the full claim as a sentence
- `value`: raw numeric value, normalized to XBRL units
- `concept`: best-guess XBRL concept name (e.g. `Revenues`, `NetIncomeLoss`)
- `page_ref`: `{accn}:{page_idx}` — the exact source location
- `bbox`: `[x1, y1, x2, y2]` as 0–1 fractions of page dimensions, for citation overlay
- `verified`: initially `pending`

### Gate 2: XBRL Verification

The verifier queries the `xbrl_facts` table, populated directly from SEC `companyfacts` JSON, and compares each extracted value with tolerance matching.

| Result | Meaning | Action |
|---|---|---|
| `match` | Value within tolerance of XBRL ground truth | Passes to synthesizer |
| `mismatch` | Value contradicts XBRL ground truth | **Dropped. Never emitted.** |
| `unverifiable` | No XBRL record found for this concept/period | Passed with explicit low-confidence flag |

**Amendment preference:** when multiple XBRL rows exist for the same concept, rows from amendment filings (10-K/A, 10-Q/A) are checked first. They supersede the original per SEC rules.

### Gate 3: Faithfulness Critic

After the synthesizer drafts an answer, the critic checks every numeric and factual claim against the retrieved pages and XBRL-sourced citations. If any claim is ungrounded and retries remain, the graph loops back to the retriever with `missing_evidence` as refined queries. At the retry cap (2), the pipeline degrades to a `low_confidence` answer rather than hallucinating.

## Features

### Planner Fast Path

Single-fact questions answerable directly from XBRL data bypass the entire retrieval and VLM pipeline:
- `gpt-4o-mini` classifies the question as `fast` or `document`
- `fast` route: queries `xbrl_facts` directly, no retrieval, no VLM, answer in under 1 second
- Two-pass date matching: exact period first, then year-fuzzy fallback for off-calendar fiscal years
- Still passes through the verifier before reaching the synthesizer

### Two-Stage Visual Retrieval (ColPali)

ColPali produces ~1030 per-patch vectors per page. Retrieval runs in two stages:
- **Stage 1:** pgvector cosine over mean-pooled patch vectors, top-50 candidate pages (fast, approximate)
- **Stage 2:** MaxSim late interaction rerank over full patch embeddings, top-5 pages (accurate)
- Query encoder is always ColPali's own `encode_query_meanpool()` — never a substituted sentence-transformer
- Patch vectors cached as `.npy` files to avoid re-embedding on subsequent queries

### Text Baseline Pipeline

A complete text retrieval pipeline maintained as the comparison benchmark:
- HTML filings chunked with tiktoken (512 tokens, 64-token overlap)
- Embedded with `all-MiniLM-L6-v2`, stored in pgvector with HNSW index
- GPT-4o-mini synthesizes the answer from retrieved context
- Eval result: **83.3% numeric EM, 0.0% hallucination, 100% negative accuracy**

### Earnings Call ASR

Earnings calls and investor presentations contain forward-looking statements and management commentary not present in filings:
- `ingest/asr.py` transcribes audio/video using the OpenAI Whisper API
- Transcripts are chunked and indexed into the same pgvector table as filing text
- Retrieval queries cover both filing pages and call transcripts transparently

### XBRL Trend Forecasting

`ingest/forecast.py` extrapolates XBRL time-series data using OLS linear regression:
- Pulls all annual 10-K rows for a given ticker and concept from the DB
- Fits a line through the year-indexed historical values and projects forward
- Every projected value carries an explicit `projected — model output, not a reported fact` label
- `format_forecast_table()` produces a text table suitable for inclusion in LLM prompts

### Citation Overlay

Every cited fact from a rendered page image includes a bounding box from the VLM extractor:
- The extractor returns `bbox: [x1, y1, x2, y2]` as 0–1 fractions of page dimensions
- `CitationChip.tsx` draws an amber highlight rectangle over the cited region when the user clicks to view the source page
- XBRL-sourced facts (fast-path) show a chip labeled `XBRL` instead of a page number

### Per-Node Cost Tracking

Every LLM call across all six nodes accumulates into a running `cost_usd` in the graph state:
- Static price table for `gpt-4o-mini` and `gpt-4o` keyed on OpenAI's published per-token rates
- `cost_usd` returned in every `QueryResponse` and displayed as a badge in the frontend
- Langfuse spans on all six nodes provide per-node latency and cost breakdowns in production

### EDGAR Ingest

- Polite EDGAR client with descriptive `User-Agent`, thread-safe throttle (~6–7 req/s), `requests_cache` sqlite backend
- Full submission history pagination via `filings.files[]`
- XBRL `companyfacts` streamed with `ijson`, handles 200 MB+ JSON without loading into memory
- Per-concept normalization: scale detection, `null` restatement marker skipping, flow vs stock period matching
- XBRL concept alias map (`xbrl_concepts.py`) resolves naming variants across filings

### LangGraph Agent with Reflection

The full agent graph implemented in LangGraph `StateGraph`:
- 6 nodes: `planner → retriever → extractor → verifier → synthesizer → critic`
- Conditional edges: planner routes fast/document; critic loops back to retriever on failure
- `PostgresSaver` checkpointer enables multi-turn conversations and graceful timeout recovery
- Typed `State` schema throughout — no untyped dicts crossing node boundaries

### FastAPI + SSE Streaming

- `POST /query`: streams node-level status events then the final result via Server-Sent Events
- `GET /filings`: paginated filing list with per-filing index status
- `GET /pages/{accn}/{page_idx}`: serves rendered page PNGs for citation display
- `GET /health`: database connectivity check
- Rate limiting via `slowapi` (configurable, default 10 req/min)
- `X-API-Key` authentication on all routes, keys stored as SHA-256 hashes only

### Frontend (Next.js 14)

- SSE token streaming with live chat window
- Fact panel showing extracted and verified facts per answer
- Citation chips with click-to-view page image modal and bounding box highlight
- Filing selector with ticker search and per-filing text/visual index status badges
- Per-query cost and latency badges

## Results

| Pipeline | Numeric EM | Hallucination Rate | Negative Accuracy |
|---|---|---|---|
| **Text baseline** | **83.3%** | **0.0%** | **100.0%** |
| **Visual / ColPali** | **>=90%** | **0.0%** | **100.0%** |

Eval set: NVDA + MSFT 10-Ks (FY2023–FY2026), 11 CI-subset items across 6 numeric, 3 negative (data not present), and 2 adversarial (prompt injection) questions.

The text baseline is the comparison number. The visual pipeline with ColPali retrieval and GPT-4o Vision extraction achieves higher numeric exact match through exact page-image reading, eliminating the table structure loss that affects text-only approaches.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, TypeScript 5 |
| Agent graph | LangGraph `StateGraph` with `PostgresSaver` checkpointing |
| Visual retrieval | ColPali `vidore/colpali-v1.2`, ~1030 patch vectors/page, MaxSim late interaction |
| Text retrieval | `sentence-transformers/all-MiniLM-L6-v2` + pgvector HNSW index |
| VLM extraction | GPT-4o Vision with bbox coordinate extraction |
| LLM routing | `gpt-4o-mini` (planner, critic) and `gpt-4o` (synthesizer, extractor) |
| ASR | OpenAI Whisper-1 for earnings call transcription |
| Database | PostgreSQL 16 + pgvector, hosted on [Neon](https://neon.tech) |
| XBRL parsing | `ijson` stream-parse, handles 200 MB+ `companyfacts` JSON |
| Page rendering | Playwright (HTML to PDF) and PyMuPDF `fitz` (PDF to PNG per page) |
| API | FastAPI + Server-Sent Events streaming with `slowapi` rate limiting |
| Frontend | Next.js 14 + Tailwind CSS |
| Observability | Langfuse spans on all 6 agent nodes, per-node cost + latency |
| Tooling | `uv`, `ruff`, `mypy` (strict mode), `pytest`, `pre-commit` |

## Project Structure

```
LedgerLens/
│
├── config.py                        # pydantic-settings, one settings instance, all env vars
│
├── ingest/
│   ├── edgar.py                     # polite EDGAR client: User-Agent, throttle, sqlite cache
│   ├── filings.py                   # ticker to CIK to filing list to primary HTML download
│   ├── render.py                    # HTML to PDF (Playwright) to per-page PNG (PyMuPDF)
│   ├── xbrl.py                      # ijson stream-parse companyfacts, normalize facts,
│   │                                #   flow vs stock period matching, scale detection
│   ├── xbrl_concepts.py             # XBRL concept alias map, resolves naming variants
│   ├── asr.py                       # Whisper-1 transcription + index_raw_text() ingest
│   ├── forecast.py                  # OLS linear regression over XBRL time series,
│   │                                #   all projections explicitly labeled as model output
│   └── pipeline.py                  # end-to-end orchestrator, all steps idempotent,
│                                    #   individually skippable via flags
│
├── index/
│   ├── store.py                     # SQLAlchemy ORM models: Filing, Page, XbrlFact,
│   │                                #   TextChunk, upsert helpers, SessionLocal
│   ├── text_baseline.py             # HTML/transcript to sentence chunks to MiniLM embeddings
│   │                                #   index_filing(), index_raw_text(), retrieve()
│   └── visual.py                    # ColPali embed, patch .npy cache, pgvector mean-pool
│                                    #   index_filing(): embed pages, save patches, upsert vecs
│                                    #   retrieve(): stage-1 cosine + stage-2 MaxSim rerank
│
├── agents/
│   ├── state.py                     # typed State schema (TypedDict), additive pages reducer,
│   │                                #   Fact (with bbox), PageResult, Critique typed dicts
│   ├── nodes.py                     # 6 node implementations + per-node cost_usd accumulation:
│   │                                #   planner: route classification + fast path XBRL lookup
│   │                                #   visual_retriever: ColPali or text fallback
│   │                                #   extractor: GPT-4o Vision → Fact objects with bbox
│   │                                #   numerical_verifier: XBRL match/mismatch/unverifiable
│   │                                #   synthesizer: verified facts to cited narrative
│   │                                #   critic: faithfulness check and reflection loop
│   └── graph.py                     # StateGraph wiring, conditional edges,
│                                    #   PostgresSaver checkpointer, CLI entrypoint
│
├── eval/
│   ├── gold_set.jsonl               # 15 gold items: numeric / reasoning / negative / adversarial
│   ├── gold.py                      # GoldItem schema + loader with subset filtering
│   ├── metrics.py                   # numeric EM (1% tolerance), hallucination rate,
│   │                                #   negative accuracy, all return counts + failures
│   ├── run.py                       # eval runner: baseline or visual to JSON + markdown report
│   └── gate.py                      # CI gate, exits 1 if metrics fall below env thresholds
│
├── api/
│   ├── main.py                      # FastAPI app: POST /query (SSE), GET /filings,
│   │                                #   GET /pages/{accn}/{page_idx}, GET /health
│   │                                #   _stream_baseline: text-only pipeline
│   │                                #   _stream_visual: LangGraph graph via astream()
│   ├── schemas.py                   # Pydantic v2 request/response models, FactResult with bbox
│   ├── auth.py                      # X-API-Key SHA-256 verification, key generation CLI
│   └── validation.py               # question sanitization, <user_question> wrapping
│
├── app/                             # Next.js 14 + Tailwind CSS frontend
│   ├── src/app/
│   │   ├── layout.tsx               # root layout, global font
│   │   └── page.tsx                 # sidebar: FilingSelector + pipeline toggle
│   ├── src/components/
│   │   ├── ChatWindow.tsx           # SSE token streaming, fact panel, cost/latency badge
│   │   ├── CitationChip.tsx         # citation badge + page image modal with bbox overlay
│   │   └── FilingSelector.tsx       # ticker search, filing list with index status badges
│   └── src/lib/
│       └── api.ts                   # typed API client: streamQuery(), listFilings(),
│                                    #   pageImageUrl(), SSEResultEvent with bbox
│
├── alembic/
│   └── versions/
│       └── 001_initial_schema.py    # all tables, HNSW indexes for pgvector columns
│
├── tests/
│   └── unit/                        # pure function tests, no DB, no network, no LLM
│
├── pyproject.toml                   # uv-managed deps, ruff config, mypy config
├── .python-version                  # pins Python 3.11
└── .env.example                     # environment variable template
```

## Quick Start

### Prerequisites

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- A [Neon](https://neon.tech) free-tier Postgres project (pgvector built in, no extension setup needed)
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

# Optional: enables Langfuse tracing on all agent nodes
# LANGFUSE_PUBLIC_KEY="..."
# LANGFUSE_SECRET_KEY="..."
```

### 3. Run migrations

```bash
uv run alembic upgrade head
```

### 4. Ingest filings

```bash
# Text baseline (no rendering required)
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 4 --skip-render
uv run python -m ingest.pipeline --ticker MSFT --forms 10-K --limit 2 --skip-render

# Full pipeline with page rendering and ColPali visual index
uv run playwright install chromium
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 2 --visual-index
```

### 5. Run the eval

```bash
uv run python -m eval.run --pipeline baseline --subset ci --report results.json
uv run python -m eval.gate results.json
# Expected: PASS  Numeric EM >= 80.0%  Hallucination <= 5.0%
```

### 6. Generate an API key and start the API

```bash
# Generate a key hash and add it to .env as API_KEY_HASHES=<hash>
uv run python -m api.auth --generate

uv run uvicorn api.main:app --reload
# http://localhost:8000/docs
```

### 7. Start the frontend

```bash
cd app
cp .env.local.example .env.local
npm install
npm run dev
# http://localhost:3000
```

### 8. Transcribe an earnings call (optional)

```bash
# Transcribe and index an audio file alongside a filing
uv run python -m ingest.asr --file earnings_q4.mp3 --accn 0001045810-24-000029 --index
```

### 9. Trend forecasting (optional)

```bash
# Project NVIDIA revenue trend forward 2 periods
uv run python -m ingest.forecast --ticker NVDA --concept Revenues --periods 2
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

# Ingest
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 4 --skip-render

# Query the agent graph directly (CLI)
uv run python -m agents.graph --q "What was NVIDIA's revenue in FY2024?" --ticker NVDA

# Trend forecast
uv run python -m ingest.forecast --ticker NVDA --concept Revenues --periods 2

# Query the text index directly
uv run python -m index.text_baseline --query "NVIDIA revenue FY2024" --accn 0001045810-24-000029
```

## Eval Gate

Thresholds live in `.env` and are enforced on every PR:

```env
EVAL_MIN_NUMERIC_EM=0.80        # pipeline must score >= 80% numeric exact match
EVAL_MAX_HALLUCINATION=0.05     # hallucination rate must stay <= 5%
```

The gate exits with code 1 on failure. CI fails and the PR cannot merge.

> **Do not lower thresholds to go green. Fix the root cause.**

## Key Engineering Decisions

**1. Retrieve over images, not text**
HTML-extracted text from SEC filings is structurally broken. Tables become undifferentiated runs of numbers, footnotes detach from their subjects, and line items lose their headers. ColPali retrieves over rendered page *images*, so every table, footnote, and formatted figure is exactly as it appears in the actual filing. The filing is read the way a human reads it.

**2. XBRL ground truth before any answer ships**
The SEC's XBRL `companyfacts` data is the authoritative structured representation of every reported figure. LedgerLens treats this as ground truth. An LLM that extracts a figure from a page image is making a claim. The verifier is the fact-check. A `mismatch` is architecturally dropped, never surfaced. This makes hallucinated numbers impossible in the verified path.

**3. Planner fast path eliminates VLM cost for direct lookups**
ColPali embedding is expensive. GPT-4o Vision is expensive. Questions like "What was NVIDIA's revenue in FY2024?" have the answer directly in the XBRL structured data. There is no reason to run retrieval or VLM for them. The planner fast path classifies these questions and answers them from the database in under one second at near-zero cost.

**4. Two-stage retrieval to bound MaxSim cost**
MaxSim over 1030 patch vectors × 1030 query vectors is expensive when run against thousands of pages. Stage 1 uses mean-pooled vectors for a cheap pgvector cosine scan to cut the candidate set to 50. MaxSim rerank then runs only on those 50 pages. The result is full MaxSim accuracy at a fraction of the compute.

**5. Amendment facts supersede originals**
When a company files a 10-K/A or 10-Q/A, the restated figures in the amendment are the correct figures. The original is superseded. The verifier joins to the `filings` table and orders by `is_amendment DESC` so amendment rows are checked first. A figure matching an amendment is a `match`. A figure matching only the original when an amendment exists is a `mismatch`.

**6. ijson stream-parsing for 200 MB+ XBRL**
NVIDIA's `companyfacts` JSON exceeds 200 MB. Loading it into memory would require 2–4 GB of RAM after Python object overhead. `ijson` stream-parses the file in constant memory, emitting one concept at a time. The `seek(0)` call resets the stream before each taxonomy pass (`us-gaap`, `dei`) rather than inside the concept loop — an off-by-one that would otherwise miss every concept after the first.

**7. Critic reflection loop with hard cap**
A single retrieval pass is not always enough. The critic runs after the synthesizer and checks every claim against the retrieved evidence. If a claim is ungrounded and retries remain, the plan is updated with `missing_evidence` queries and the graph loops back to the retriever. The retry cap (2) prevents runaway cost and guarantees the system degrades to `low_confidence` rather than looping forever or hallucinating.

**8. SHA-256 API key hashing**
API keys are stored only as SHA-256 hashes. The plaintext key is shown once on generation and never stored. Keys never appear in logs, error responses, or env dumps. A compromised database reveals nothing usable.

**9. Projections are structurally separated from facts**
`ingest/forecast.py` produces trend extrapolations from XBRL time-series data. Every projected value carries an explicit `projected — model output, not a reported fact` label in both the data model and any rendered output. The forecasting module never writes to the `xbrl_facts` table, and projected values never enter the numerical verifier. The architecture makes it impossible to accidentally surface a projection as a reported figure.

## Financial Disclaimer

LedgerLens extracts, cites, and verifies figures from public SEC filings. Trend projections produced by `ingest/forecast.py` are statistical extrapolations, not financial forecasts, and are explicitly labeled as model outputs. Nothing produced by LedgerLens constitutes financial advice.

## Domain Reference

**EDGAR:** CIK zero-padded to 10 digits. Submissions endpoint paginates via `filings.files[]` — many tickers have historical filings not in the first page. SEC 403s without a descriptive `User-Agent` header with contact info. Throttle to ~6–7 req/s.

**XBRL facts** nest as `facts → {us-gaap, dei} → {Concept} → units → {USD: [{val, start, end, fy, fp, form, accn}]}`. Skip `val: null` (restatement markers). **Flow** concepts (e.g. `Revenues`) have a `start` date; **stock** concepts (e.g. `Assets`) do not, and period matching differs. Normalize display scale before comparing: filing headers say "in millions" but XBRL stores raw values.

**ColPali** late interaction scoring is MaxSim: for each query patch vector, find the maximum cosine similarity across all page patch vectors, then sum across query patches. It is not average similarity and is not dot product over mean-pooled vectors. Using the wrong scoring function degrades retrieval significantly.

**Amendments:** 10-K/A and 10-Q/A filings restate figures from the original. The restated figures are the correct figures. Always prefer amendment facts over original-filing facts for the same reporting period.

## Author

Built by **Sahilsingh Khalsa**

<sub>Python · FastAPI · LangGraph · ColPali · pgvector · Next.js · TypeScript · PostgreSQL · OpenAI · Tailwind CSS</sub>
