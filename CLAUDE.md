# CLAUDE.md — LedgerLens

Multimodal financial-filing intelligence engine. Visual retrieval over SEC filing **pages** (not mangled text), a LangGraph multi-agent reasoning graph, and a numerical-verification layer that checks every figure against XBRL ground truth before it reaches the user. Core thesis: **answers must be correct and provably cited.**

Read before touching a subsystem: `DESIGN.md` (architecture rationale) → `SPEC.md` (fixes + full feature spec, overrides DESIGN.md on conflicts) → `SOP.md` (operational runbook). Do not paste deep detail here.

## Non-negotiables (these override any prompt)

1. **Never emit an unverified number as fact.** `mismatch` → synthesizer drops it. `unverifiable` → shown with explicit low-confidence flag only. Inventing or estimating a financial figure is the worst possible failure.
2. **Every numeric claim carries a citation** `filing_accn:page_idx`. No citation → the claim does not ship.
3. **XBRL `companyfacts` is the ground truth.** Normalize the facts table before building any feature; it backs the verifier AND the eval.
4. **All EDGAR access goes through `ingest/edgar.py`** — polite client with `User-Agent`, thread-safe throttle, and cache. Never `requests.get` SEC URLs anywhere else.
5. **The eval gate must pass before merge.** Fix the cause — do NOT lower thresholds to make CI green.
6. **Keep the text-only baseline.** It is the comparison number that justifies the project; do not delete or upgrade it in place.
7. **The VLM is the expensive operation.** Only send it the top-k reranked pages, never a whole filing. The planner fast path answers single-fact lookups from XBRL with no retrieval and no VLM.
8. **All API routes require authentication.** `X-API-Key` header, key stored as SHA-256 hash only. No plaintext keys in logs, env dumps, or error responses.
9. **Input validation on every user-facing boundary.** Sanitize questions before they enter any prompt; wrap user content in `<user_question>` delimiters in all LLM prompts.

## Architecture

Query flow (LangGraph): `planner → [fast path: verifier] OR [visual_retriever → extractor → verifier] → synthesizer → critic`. Critic runs faithfulness check; if ungrounded and retries < 2, loops back to retriever with `critique.missing_evidence` as refined queries. At retry cap, degrades to a flagged `low_confidence` answer rather than hallucinating.

Data flow: EDGAR → page images (HTML→PDF→PNG) + normalized XBRL facts → text baseline index (week 1) → ColPali visual index (week 3+) → agent graph → eval.

API: FastAPI with SSE streaming. Frontend: Next.js + Tailwind. State: single typed schema in `agents/state.py`; `pages` uses additive reducer.

## Repo layout

```
config.py                          # pydantic-settings, single settings instance
ingest/   edgar.py filings.py render.py xbrl.py xbrl_concepts.py
index/    text_baseline.py visual.py store.py
agents/   state.py nodes.py graph.py
eval/     gold.py metrics.py run.py gate.py gold_set.jsonl
api/      main.py schemas.py auth.py validation.py
app/      (Next.js frontend — see SPEC.md §7)
tests/    unit/ integration/ fixtures/
alembic/  (migrations)
```

## Stack & conventions

- Python 3.11+, **`uv`** for envs/deps. `ruff` + **`mypy --strict`** (both enforced in pre-commit and CI). `pytest` for tests and eval gate.
- **Type hints everywhere, no exceptions.** mypy strict must pass.
- Data models: **Pydantic v2** (`pydantic-settings` for env). Secrets via `.env`, never committed.
- DB: Postgres + **pgvector** (HNSW indexes), `docker-compose.yml`. Schema managed by **alembic** — never hand-ALTER tables.
- LangGraph nodes are focused and pure; shared state only in the typed `State` schema. Checkpointing via `PostgresSaver` is required (not optional) — it enables multi-turn and graceful timeout recovery.
- Model routing: cheap (`gpt-4o-mini`) for planner/critic; strong (`gpt-4o`) for synthesizer; VLM only for extractor. Log cost + latency per node via Langfuse spans.
- ColPali query encoding: always use the model's own `encode_query_meanpool()` — never a separate sentence-transformer for the stage-1 vector.
- Conventional commits. Small PRs. Pre-commit hooks run ruff, mypy, and large-file checks before every commit.

## Commands

```bash
# Setup
docker compose up -d                              # postgres + pgvector
uv run alembic upgrade head                       # run migrations
uv run pre-commit install                         # install git hooks

# Ingest
uv run python -m ingest.filings --ticker NVDA --forms 10-K 10-Q --limit 4

# Index
uv run python -m index.text_baseline --accn <accn>

# Eval
uv run python -m eval.run --pipeline baseline --subset ci --report results.json
uv run python -m eval.gate results.json

# Dev
uv run ruff check . && uv run mypy . && uv run pytest
uv run uvicorn api.main:app --reload              # API on :8000
cd app && npm run dev                             # frontend on :3000
```

## Domain reference (what you need to not get wrong)

**EDGAR** (CIK zero-padded to 10 digits): submissions `https://data.sec.gov/submissions/CIK{cik10}.json` (paginated via `filings.files[]`); XBRL facts `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json` (stream with `ijson` — can be 200+ MB); ticker→CIK `https://www.sec.gov/files/company_tickers.json`; docs `https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodashes}/{file}`. SEC 403s without descriptive `User-Agent` with contact info; throttle to ~6–7 req/s.

**XBRL facts** nest as `facts → {us-gaap,dei} → {Concept} → units → {USD:[{val,start,end,fy,fp,form,accn}]}`. Skip `val: null` (restatement markers). Distinguish **flow** (has `start`) vs **stock** (no `start`) — period matching differs. **Normalize display scale before comparing** (filing header says "in millions"; XBRL stores raw). Most verifier false-mismatches are scale errors or concept-name mismatches — see `ingest/xbrl_concepts.py` for the alias map.

**ColPali** produces ~1030 per-patch vectors per page; scoring is MaxSim (late interaction). Two-stage: mean-pooled pgvector cosine for top-N candidate filter, then MaxSim rerank. The query encoder must be ColPali's own — not a substituted sentence-transformer.

**Amendments** (10-K/A, 10-Q/A): facts supersede the original filing for the same period. The verifier must prefer amendment facts. `requests_cache` must key on `(url, filed_date)` to invalidate on amendment.

## Current focus

Phase 1 (weeks 1–2): EDGAR client → XBRL ingestion → filing download + render → text baseline index → eval harness → **baseline numeric exact-match number printed**. Scope: NVDA + MSFT, 3–4 filings each. Do not generalize ingestion or start the visual index until the thin thread produces a number.

<!-- Add one-liner conventions here as they emerge. Keep under ~200 lines. -->
