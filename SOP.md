# SOP.md — LedgerLens Standard Operating Procedures

Operational runbook for building, running, evaluating, and maintaining LedgerLens. This is the **how-to-do-it** companion to `DESIGN.md` (the *what and why*) and `CLAUDE.md` (the agent's rulebook). For architecture rationale, defer to `DESIGN.md`; this document does not re-explain it.

**Status note:** the project is built incrementally. Procedures below describe the **intended interface** for each module; some CLIs are the contract a module must satisfy when implemented, not yet a guarantee it exists. Keep the commands here in sync with reality as you build.

**Conventions in this doc**
- `$` prefixes a shell command. Every command runs from the repo root unless stated.
- "Verify:" lines tell you how to confirm a step succeeded. Do not proceed past a failed verify.
- `accn` = filing accession number; `CIK` = 10-digit zero-padded company id. See the Glossary.

---

## Contents

1. SOP-1 — First-time environment setup
2. SOP-2 — Daily development startup
3. SOP-3 — Ingest a new issuer / filing
4. SOP-4 — Build the text-only baseline index
5. SOP-5 — Build / refresh the visual (ColPali) index
6. SOP-6 — Run a query
7. SOP-7 — Curate the eval gold set
8. SOP-8 — Run the eval harness and read the report
9. SOP-9 — Safely change a model or prompt
10. SOP-10 — Pull request, CI, and merge
11. SOP-11 — Observability and trace review
12. SOP-12 — Cost and latency monitoring
13. SOP-13 — Data, cache, and database management
14. SOP-14 — Secrets and configuration
15. Troubleshooting playbook
16. Incident procedures
17. Glossary and quick reference

---

## SOP-1 — First-time environment setup

**Purpose:** go from a fresh clone to a running stack. **Run when:** onboarding or rebuilding a machine.

**Prerequisites:** Docker, `uv`, Python 3.11+, ~10 GB free disk (page images + embeddings grow fast).

1. Clone and enter the repo.
2. `$ uv sync` — install dependencies into the project venv.
3. `$ cp .env.example .env`, then fill in required values (see SOP-14). The **EDGAR `User-Agent`** with real contact info is mandatory; SEC will block requests without it.
4. `$ docker compose up -d` — start Postgres + pgvector.
   - Verify: `$ docker compose ps` shows the db healthy.
5. `$ uv run python -m index.store migrate` — create the schema and HNSW indexes (or `alembic upgrade head` if migrations are used).
   - Verify: `$ uv run python -m index.store check` reports the `chunks`, `pages`, and `xbrl_facts` tables exist.
6. `$ uv run ruff check . && uv run pytest -q` — confirm the toolchain runs.

**Done when:** the db is up, schema is migrated, and tests pass on an empty corpus.

---

## SOP-2 — Daily development startup

1. `$ docker compose up -d` (no-op if already running).
2. `$ git pull` and `$ uv sync` (picks up any new deps).
3. Skim `git log` since you last worked, and re-read the "Current focus" line in `CLAUDE.md`.
4. Start the API only if you need it: `$ uv run uvicorn api.main:app --reload`.

---

## SOP-3 — Ingest a new issuer / filing

**Purpose:** pull filings + XBRL ground truth for a company. **Run when:** adding a company to the corpus. **Keep scope small early** — 2–3 issuers, a handful of filings.

1. Resolve the ticker to a CIK (the client handles zero-padding):
   `$ uv run python -m ingest.filings --ticker NVDA --list`
   - Verify: prints recent filings with `form`, `accn`, `filingDate`.
2. Ingest specific forms:
   `$ uv run python -m ingest.filings --ticker NVDA --forms 10-K 10-Q --limit 4`
   - This downloads the primary document HTML and triggers the render pipeline (HTML → PDF → per-page PNG) and the XBRL `companyfacts` pull.
   - Verify: page PNGs appear under the data dir keyed `accn/page_####.png`; a row count for `xbrl_facts` increased.
3. Confirm ground truth landed:
   `$ uv run python -m ingest.xbrl --accn <accn> --show Revenues`
   - Verify: prints the reported value(s) with `fy`/`fp`/`unit`.

**Rules:** all SEC access must route through `ingest/edgar.py` (polite client: `User-Agent`, throttle ~6–7 req/s, cache). Never call SEC URLs elsewhere. If a render produces zero pages, see Troubleshooting.

---

## SOP-4 — Build the text-only baseline index

**Purpose:** the naive comparison pipeline. **Do not optimize it** — it exists to be beaten, and its accuracy number is the headline delta for the project.

1. `$ uv run python -m index.text_baseline --accn <accn>`
   - Parses HTML text, chunks with overlap, embeds, inserts into `chunks`.
   - Verify: `chunks` row count for that `accn` > 0.
2. Smoke test retrieval: `$ uv run python -m index.text_baseline --query "total revenue" --accn <accn> --topk 5`.

**Never delete the baseline** when the visual pipeline lands — the eval runs both side by side (SOP-8).

---

## SOP-5 — Build / refresh the visual (ColPali) index

**Purpose:** the differentiating retrieval path. See `DESIGN.md` §4.2 for the two-stage storage rationale.

1. `$ uv run python -m index.visual --accn <accn>`
   - For each page PNG: compute ColPali multi-vector patch embeddings (cached to parquet/npy keyed `accn:page_idx`) **and** the mean-pooled page vector (stored in pgvector with HNSW for stage-1 filtering).
   - Verify: cached patch files exist for every page; `pages` rows have a non-null pooled embedding.
2. Smoke test the two-stage retrieval:
   `$ uv run python -m index.visual --query "data center gross margin" --topk 5`
   - Stage 1: pgvector cosine → top-N candidates. Stage 2: MaxSim rerank of those candidates → top-k.
   - Verify: returns page refs with MaxSim scores; latency is dominated by stage 2 on N candidates, not the whole corpus.

**Rules:** do not brute-force MaxSim over the entire corpus. If you change the ColPali checkpoint, the embedding dimension may change — re-index all pages and update the vector column dimension (Troubleshooting).

---

## SOP-6 — Run a query

**Terminal (fastest for iteration):**
`$ uv run python -m agents.graph --q "How did NVDA data-center gross margin trend over the last 4 quarters and what did management attribute it to?"`
- Prints the routed path (fast vs document), retrieved pages, extracted facts with verification status, the synthesized answer, and citations.

**API:** `POST /query {"question": "..."}` against the running server (SOP-2).

**Read the output critically:** every numeric claim must carry a `accn:page_idx` citation and a `verified` status. A `mismatch` figure must never appear in the answer body — if it does, that is a bug, file it and add a gold-set regression case (SOP-7).

---

## SOP-7 — Curate the eval gold set

**Purpose:** maintain `eval/gold_set.jsonl`, the committed, versioned source of truth for evaluation.

1. Auto-generate numeric questions from XBRL:
   `$ uv run python -m eval.gold --ticker NVDA --auto >> eval/gold_set.jsonl`
   - Each item: a question whose `answer_value`/`unit`/`concept`/`accn` come straight from the XBRL facts. Free, large, exact ground truth.
2. Hand-write reasoning items for multi-step questions (margins, trends, attributions). Include an expected numeric answer where one exists, plus a short rubric for the prose part. Tag `kind: reasoning`.
3. **Regression items:** every confirmed bug (a wrong number, a hallucinated figure) becomes a gold item so it can never silently return.
4. Mark a stable subset with `subset: ci` — this is what the merge gate runs (SOP-10). Keep it small and fast.
5. Commit changes to `gold_set.jsonl` in their own PR with a message describing what was added and why.

**Rule:** the gold set is versioned and reviewed. Do not edit it to make a failing run pass.

---

## SOP-8 — Run the eval harness and read the report

**Purpose:** measure the three failure surfaces (retrieval, extraction, reasoning) and produce the delta vs baseline.

1. Full run, both pipelines:
   `$ uv run python -m eval.run --pipeline visual --report results_visual.json`
   `$ uv run python -m eval.run --pipeline baseline --report results_baseline.json`
2. Read the markdown report the runner emits alongside the JSON. Key metrics:
   - **Numeric exact-match** (headline; units/scale normalized, small rounding tolerance).
   - **Retrieval recall@k** (was a page containing the ground-truth fact retrieved?).
   - **Citation faithfulness** and **hallucinated-figure rate** (target the latter toward zero).
   - **Reasoning correctness** on hand-written items (LLM-judge + exact-match where numeric).
   - **Cost / p95 latency** per query.
3. Inspect the "worst failures" section first — it is the highest-value debugging signal and the raw material for SOP-7 step 3.
4. Record the visual-vs-baseline delta; this is the number that goes on the résumé and in the README.

**LLM-as-judge:** calibrate the judge against a handful of human labels before trusting the reasoning metric. Gate on the deterministic XBRL exact-match, not the judge.

---

## SOP-9 — Safely change a model or prompt

**Purpose:** prevent silent quality regressions. **Run for:** any change to a model, prompt, chunking, retrieval params, or thresholds.

1. Capture a baseline: `$ uv run python -m eval.run --pipeline visual --report before.json`.
2. Make the change on a branch. Bump the relevant prompt/model version identifier so traces are attributable.
3. Re-run: `$ uv run python -m eval.run --pipeline visual --report after.json`.
4. Diff: `$ uv run python -m eval.run compare before.json after.json`.
   - Accept only if numeric exact-match and faithfulness hold or improve and hallucination does not rise. Cost/latency regressions need an explicit justification.
5. If a metric regresses, **fix the cause** — never lower the gate thresholds to pass.
6. Note the change and its measured effect in the PR description.

---

## SOP-10 — Pull request, CI, and merge

**Pre-PR checklist:**
- [ ] `$ uv run ruff check . && uv run pytest -q` pass locally.
- [ ] If behavior changed, SOP-9 was followed and the before/after delta is in the PR description.
- [ ] New confirmed bugs were added to the gold set (SOP-7).
- [ ] No secrets, `.env`, large data files, or cache artifacts staged (`$ git status` clean of them).

**CI gate (runs on every PR):**
`$ uv run python -m eval.run --subset ci --report results.json`
`$ uv run python -m eval.gate results.json --min-numeric-em <T> --max-hallucination <H>`
- The gate fails the build if numeric exact-match drops below `T` or hallucination rises above `H`.
- **Thresholds are set just under the current measured visual-pipeline number.** Do not commit aspirational thresholds before you have a real number, and do not weaken them to go green.

**Merge:** squash, conventional-commit title, link the eval delta.

---

## SOP-11 — Observability and trace review

1. Open the tracing UI (Langfuse/LangSmith) configured via `.env`.
2. For any query, inspect the per-node trace: planner route decision, retrieved pages + scores, extractor output, verifier match/mismatch, critic verdict, and any reflection retries.
3. **Weekly:** review a sample of low-confidence or retried queries. Recurring failure shapes become gold-set items (SOP-7) and, if systemic, a `DESIGN.md`/`CLAUDE.md` rule update.
4. Confirm every node logs cost and latency tags; missing tags are a bug (SOP-12 depends on them).

---

## SOP-12 — Cost and latency monitoring

1. `$ uv run python -m eval.run --report r.json` includes per-query cost/latency aggregates.
2. Watch these levers (in priority order):
   - Planner fast-path hit rate — single-fact questions should skip retrieval and the VLM entirely.
   - VLM page count — the extractor must only see top-k reranked pages, never whole filings.
   - Cache hit rates — page renders, page embeddings, transcripts, and (if enabled) semantic query cache.
3. If p95 latency or cost per query trends up, profile the per-node trace before changing models. Most regressions are "VLM called on too many pages" or "stage-1 candidate N too large," not the model itself.

---

## SOP-13 — Data, cache, and database management

- **Layout:** raw downloads, page PNGs, and patch-embedding caches live under the data dir, keyed by `accn`. The db holds chunks, page metadata + pooled vectors, and `xbrl_facts`.
- **Reset a single issuer:** `$ uv run python -m index.store purge --accn <accn>` then re-run SOP-3/4/5.
- **Full reset (dev only):** `$ docker compose down -v` wipes the db volume; re-run SOP-1 step 5 onward. Caches on disk persist unless deleted manually.
- **Backups:** the db is reproducible from EDGAR + caches, so back up the **gold set** and any **hand-labeled judge calibration** above all — those are the only non-reproducible assets.
- **Caches are disposable** but expensive to rebuild (embeddings); don't delete them casually.

---

## SOP-14 — Secrets and configuration

- All config via `.env` (loaded by `pydantic-settings`); **never committed**. `.env.example` documents every key with safe placeholders.
- Required keys: EDGAR `User-Agent` (real contact), DB connection string, model/provider API keys, tracing keys, and the eval thresholds `T`/`H`.
- Rotate any key that touches a paid API if it leaks; revoke first, then rotate.
- The verifier and eval depend on the XBRL table, not on any secret — keep that path credential-free.

---

## Troubleshooting playbook

| Symptom | Likely cause | Fix |
|---|---|---|
| EDGAR returns 403 | Missing/blocked `User-Agent` | Set a descriptive UA with contact in `.env`; confirm the call routes through `ingest/edgar.py`. |
| EDGAR returns 429 / slow | Exceeding rate limit | Increase the throttle interval; rely on the cache; never parallelize SEC calls hard. |
| EDGAR 404 on a known company | CIK not zero-padded to 10 digits | Use the client's CIK formatting; don't build `data.sec.gov` URLs by hand. |
| Render produces 0 pages | HTML→PDF step failed (headless browser/deps) | Check the renderer install; fall back to a simpler print path; inspect one filing manually. |
| VLM misreads small footnote numbers | Page render DPI too low | Raise render DPI (e.g., 150→200) and re-run SOP-5 for that issuer. |
| Verifier flags `mismatch` on a correct number | Unit/scale not normalized (thousands vs millions vs raw) | Fix the normalization step before comparison; this is the #1 false-mismatch cause. |
| Right page never retrieved (low recall@k) | Stage-1 candidate N too small, or bad render | Increase stage-1 N; check page image quality; confirm the page actually contains the value. |
| Hallucinated figure in the answer | Synthesizer used a `mismatch`/uncited fact | Enforce the "verified-only, cited-only" filter in the synthesizer; add a gold regression item. |
| Reflection loop never ends | Retry cap not enforced | Confirm `route_after_critic` returns END at the retry limit; degrade to a flagged answer. |
| pgvector dimension error on insert | Embedding model/checkpoint changed | Re-index affected rows; update the `vector(dim)` column to match the new dimension. |
| Eval gate fails after a change | Genuine regression | Follow SOP-9; revert or fix the cause. Never lower the thresholds. |
| Cost/latency spiked | VLM seeing too many pages, or fast path missed | Check top-k, planner routing, and stage-1 N via the trace (SOP-11/12). |

---

## Incident procedures

**Eval regression reached `main`.** 1) Revert the offending commit immediately to restore a green gate. 2) Reproduce locally with SOP-9's before/after on the reverted vs offending state. 3) Fix forward with the eval delta in the PR. 4) If the gate let it through, the CI subset missed the case — add a gold item (SOP-7) so it can't recur.

**EDGAR access blocked.** 1) Stop all ingestion. 2) Confirm the `User-Agent` and that all traffic routes through the polite client; check for an accidental tight loop or parallel calls. 3) Back off, lean on cache, resume slowly. 4) If persistent, pause ingestion and work from already-cached filings.

---

## Glossary and quick reference

- **CIK** — SEC central index key; zero-pad to 10 digits in `data.sec.gov` URLs.
- **accn** — accession number; uniquely identifies a filing and joins XBRL facts to their source filing.
- **XBRL / companyfacts** — SEC's machine-readable structured financials; the project's ground truth.
- **fy / fp** — fiscal year / fiscal period of an XBRL fact.
- **MaxSim / late interaction** — ColPali/ColBERT scoring: per query token, max similarity to any page patch, summed. pgvector cosine cannot do this directly (hence two-stage retrieval).
- **recall@k** — did retrieval surface a page containing the ground-truth fact within the top-k.
- **numeric exact-match** — extracted figure equals XBRL truth after unit/scale normalization, within tolerance. The metric the gate enforces.
- **fast path** — planner answers single-fact lookups from XBRL with no retrieval and no VLM.
- **reflection loop** — critic returns an ungrounded answer to retrieval, bounded by a retry cap.

**Key files:** `DESIGN.md` (architecture), `CLAUDE.md` (agent rules + non-negotiables), `eval/gold_set.jsonl` (eval truth), `ingest/edgar.py` (the only SEC entry point), `.env` (config; never committed).
