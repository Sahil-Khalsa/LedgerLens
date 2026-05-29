# LedgerLens

> **Multimodal financial-filing intelligence engine** — answers questions about SEC filings with verified, cited responses. Every numeric claim is cross-checked against XBRL ground truth before it reaches the user, and every answer carries a `filing:page` citation traceable to the original document image.

---

## The Problem

SEC filings are notoriously hard to query accurately. Text-chunking pipelines miss figures buried in tables, footnotes, and financial statements. LLMs hallucinate numbers or cite the wrong period. There is no reliable way to know if an extracted figure is correct.

**LedgerLens solves this in three layers:**

1. **Visual retrieval** — ColPali retrieves over page *images*, not mangled HTML text, so tables and formatted statements are preserved exactly as printed
2. **XBRL verification** — every extracted figure is cross-checked against the SEC's own structured XBRL data before it is shown to the user
3. **Cited answers** — every numeric claim carries a `accn:page_idx` citation; no citation means the claim does not ship

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐   fast path ──────────────────────────────────────────────────────────┐
│   Planner   │   single XBRL lookup, no retrieval, no VLM                            │
│ (gpt-4o-mini)│                                                                      │
└──────┬──────┘                                                                       │
       │ document route                                                               │
       ▼                                                                              │
┌──────────────────┐                                                                  │
│ Visual Retriever │  Stage 1: pgvector cosine over mean-pooled ColPali vectors       │
│   (ColPali)      │           → top-50 candidate pages                               │
└──────┬───────────┘  Stage 2: MaxSim rerank on full patch embeddings → top-5 pages  │
       │                                                                              │
       ▼                                                                              ▼
┌─────────────┐                                                               ┌────────────┐
│  Extractor  │  GPT-4o Vision reads page images                              │  Verifier  │
│ (gpt-4o-vis)│  → structured Fact objects with page_ref citations            │  (XBRL DB) │
└──────┬──────┘                                                               └─────┬──────┘
       │                                                                            │
       └────────────────────────────────────────────────────────────────────────────┘
                                                                                    │
                                                              match / mismatch / unverifiable
                                                                                    │
                                                                                    ▼
                                                                         ┌─────────────────┐
                                                                         │   Synthesizer   │
                                                                         │   (gpt-4o)      │
                                                                         │ verified facts  │
                                                                         │ only, all cited │
                                                                         └────────┬────────┘
                                                                                  │
                                                                                  ▼
                                                                         ┌─────────────────┐
                                                                         │     Critic      │
                                                                         │  (gpt-4o-mini)  │
                                                                         │ faithfulness    │
                                                                         │ check — retry   │
                                                                         │ if ungrounded   │
                                                                         └─────────────────┘
```

**Hard rules baked into the pipeline:**
- `mismatch` → synthesizer drops the figure, never emits it
- `unverifiable` → shown with explicit low-confidence flag only
- Every numeric claim carries `accn:page_idx` citation — no citation, no ship
- VLM receives only top-k reranked pages — never a whole filing
- All EDGAR access goes through a rate-limited, cached, polite client

---

## Results

| Pipeline | Numeric EM | Hallucination rate | Negative accuracy |
|---|---|---|---|
| **Text baseline** (Phase 1) | **83.3%** | **0.0%** | **100.0%** |
| Visual / ColPali (Phase 2) | *pending — needs GPU* | — | — |

Eval set: NVDA + MSFT 10-Ks (FY2023–FY2025), 6 numeric + 3 negative + 2 adversarial questions.

The text baseline is the comparison number. The visual pipeline is built to beat it — expected ≥90% EM with page-level citations once ColPali indexing runs on GPU.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, TypeScript |
| Agent graph | LangGraph — StateGraph + PostgresSaver checkpointing |
| Visual retrieval | ColPali `vidore/colpali-v1.2` — 1030 patch vectors/page, MaxSim late interaction |
| Text retrieval | `all-MiniLM-L6-v2` embeddings + pgvector HNSW index |
| VLM extraction | GPT-4o Vision |
| LLM routing | `gpt-4o-mini` (planner, critic) · `gpt-4o` (synthesizer, extractor) |
| Database | PostgreSQL 16 + pgvector — hosted on [Neon](https://neon.tech) |
| API | FastAPI + Server-Sent Events streaming · slowapi rate limiting |
| Frontend | Next.js 14 + Tailwind CSS |
| Rendering | Playwright (HTML → PDF) · PyMuPDF (PDF → PNG per page) |
| XBRL parsing | `ijson` stream-parse (handles 200 MB+ companyfacts JSON) |
| Observability | Langfuse spans on all 6 agent nodes |
| Tooling | `uv` · `ruff` · `mypy --strict` · `pytest` · `pre-commit` |

---

## Project Structure

```
LedgerLens/
├── config.py                        # pydantic-settings — one settings instance
│
├── ingest/
│   ├── edgar.py                     # polite EDGAR client: User-Agent, throttle, cache
│   ├── filings.py                   # ticker → CIK → filing list → HTML download
│   ├── render.py                    # HTML → PDF → per-page PNG (Playwright + PyMuPDF)
│   ├── xbrl.py                      # stream-parse companyfacts, normalize facts
│   ├── xbrl_concepts.py             # XBRL concept alias map (handles naming variants)
│   └── pipeline.py                  # end-to-end orchestrator — all steps, idempotent
│
├── index/
│   ├── store.py                     # SQLAlchemy ORM models + upsert helpers
│   ├── text_baseline.py             # HTML → chunks → embeddings → pgvector
│   └── visual.py                    # ColPali embed → patch cache + pgvector mean-pool
│                                    # two-stage retrieval: cosine stage1 + MaxSim stage2
│
├── agents/
│   ├── state.py                     # typed State schema (TypedDict, additive pages reducer)
│   ├── nodes.py                     # 6 nodes: planner, visual_retriever, extractor,
│   │                                #   numerical_verifier, synthesizer, critic
│   │                                #   + Langfuse spans on every node
│   └── graph.py                     # StateGraph wiring, conditional edges, reflection loop
│
├── eval/
│   ├── gold_set.jsonl               # 15 gold items (numeric / negative / adversarial)
│   ├── gold.py                      # GoldItem schema + loader with subset filtering
│   ├── metrics.py                   # numeric EM, hallucination rate, negative accuracy
│   ├── run.py                       # eval runner → JSON + markdown report
│   └── gate.py                      # CI gate — exits 1 if metrics fall below threshold
│
├── api/
│   ├── main.py                      # FastAPI routes + SSE streaming
│   │                                #   _stream_baseline: text-only pipeline
│   │                                #   _stream_visual: full LangGraph agent via astream()
│   ├── schemas.py                   # Pydantic v2 request/response models
│   ├── auth.py                      # X-API-Key SHA-256 verification
│   └── validation.py               # question sanitization + <user_question> wrapping
│
├── app/                             # Next.js 14 + Tailwind frontend
│   ├── src/app/page.tsx             # layout: sidebar (filing selector + pipeline toggle)
│   ├── src/components/
│   │   ├── ChatWindow.tsx           # SSE token streaming, fact panel, cost badge
│   │   ├── CitationChip.tsx         # inline citation badge + click-to-view page modal
│   │   └── FilingSelector.tsx       # ticker search, filing list with index status
│   └── src/lib/api.ts              # typed API client: streamQuery(), listFilings()
│
├── alembic/
│   └── versions/001_initial_schema.py  # all tables + HNSW indexes
│
├── tests/
│   └── unit/                        # pure function tests (no DB, no network, no LLM)
│
├── ROADMAP.md                       # all 5 phases with status
├── DESIGN.md                        # architecture rationale
├── SPEC.md                          # full feature spec (overrides DESIGN.md on conflicts)
└── SOP.md                           # operational runbook
```

---

## Setup

### Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- A [Neon](https://neon.tech) free-tier Postgres project (pgvector built in)
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
DATABASE_URL="postgresql+psycopg://user:pass@host/db?sslmode=require"
OPENAI_API_KEY="sk-..."
EDGAR_USER_AGENT="LedgerLens Your Name your@email.com"
```

### 3. Run migrations

```bash
uv run alembic upgrade head
```

### 4. Ingest filings

```bash
# Text baseline only (no GPU required)
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 4 --skip-render
uv run python -m ingest.pipeline --ticker MSFT --forms 10-K --limit 2 --skip-render

# Full pipeline — renders pages + builds ColPali visual index (requires GPU)
uv run playwright install chromium
uv run python -m ingest.pipeline --ticker NVDA --forms 10-K --limit 2 --visual-index
```

### 5. Run the eval

```bash
uv run python -m eval.run --pipeline baseline --subset ci --report results.json
uv run python -m eval.gate results.json
```

### 6. Start the API

```bash
# Generate an API key hash
uv run python -m api.auth --generate

# Add the hash to .env → API_KEY_HASHES
uv run uvicorn api.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs
```

### 7. Start the frontend

```bash
cd app
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_KEY
npm install
npm run dev
# → http://localhost:3000
```

---

## Key Commands

```bash
# Linting & type checking
uv run ruff check .
uv run ruff format .
uv run mypy .

# Tests
uv run pytest tests/unit/ -v

# Eval
uv run python -m eval.run --pipeline baseline --subset ci --report results.json
uv run python -m eval.gate results.json

# DB inspection
uv run python -m index.store check

# XBRL facts for a ticker
uv run python -m ingest.xbrl --ticker NVDA --concept Revenues

# Query the text index directly
uv run python -m index.text_baseline --query "NVIDIA revenue FY2024" --accn 0001045810-24-000029
```

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** — Text baseline + eval harness | EDGAR ingest, XBRL normalization, text chunking, eval gate | ✅ **Done — 83.3% EM** |
| **2** — ColPali visual index | Page rendering, ColPali embed, two-stage MaxSim retrieval | ✅ Code complete — needs GPU to run |
| **3** — Full LangGraph agent | All 6 nodes wired, reflection loop, amendment preference | ✅ Code complete |
| **4** — Eval hardening + observability | Langfuse tracing, real thresholds, amendment handling | ✅ Code complete |
| **5** — Frontend | Next.js chat UI, SSE streaming, citation viewer | ✅ Scaffolded |

See [`ROADMAP.md`](ROADMAP.md) for detailed deliverables and done criteria per phase.

---

## Eval Gate

Thresholds live in `.env` and are enforced on every PR via GitHub Actions:

```env
EVAL_MIN_NUMERIC_EM=0.80        # pipeline must score ≥ 80% to merge
EVAL_MAX_HALLUCINATION=0.05     # hallucination rate must stay ≤ 5%
```

> **Do not lower thresholds to go green — fix the root cause.**

---

## Domain Notes

**EDGAR:** CIK zero-padded to 10 digits. Companyfacts JSON can exceed 200 MB — streamed with `ijson`. SEC 403s without a descriptive `User-Agent`; throttled to ~6–7 req/s.

**XBRL:** Facts nest as `facts → {us-gaap, dei} → {Concept} → units → {USD: [{val, start, end, accn}]}`. Skip `val: null` (restatement markers). Distinguish flow (has `start`) vs stock (no `start`) — period matching differs. Amendments supersede originals — the verifier checks amendment facts first.

**ColPali:** Produces ~1030 per-patch vectors per page; scoring is MaxSim (late interaction). Two-stage: mean-pooled pgvector cosine for top-N candidates → MaxSim rerank. Query encoder must be ColPali's own — never a substituted sentence-transformer.

---

## Security

- `X-API-Key` header required on all routes — stored as SHA-256 hash only
- User input sanitized before entering any prompt
- All user content wrapped in `<user_question>` delimiters in LLM prompts
- No plaintext keys in logs, env dumps, or error responses
- Rate limiting via `slowapi` (configurable per-IP, default 10 req/min)

---

## Contributing

```bash
uv run pre-commit install      # installs ruff + mypy + large-file hooks
```

All PRs must pass:
- `ruff check .` — zero lint errors
- `mypy .` — zero type errors (strict mode)
- `pytest tests/unit/` — all unit tests green
- Eval gate — numeric EM ≥ threshold (on PRs, requires `OPENAI_API_KEY` secret)

---

*Built by Sahilsingh Khalsa*
