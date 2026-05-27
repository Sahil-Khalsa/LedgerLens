# LedgerLens — Complete Specification & Improvements

This document is the authoritative supplement to `DESIGN.md` and `SOP.md`. It covers every identified gap, fix, new feature, and missing design decision. When this document conflicts with `DESIGN.md`, this document wins — it was written after a full audit. Sections are self-contained; read the one relevant to what you are building.

---

## Contents

1. XBRL & Ingestion Fixes
2. ColPali & Retrieval Fixes
3. LangGraph Agent Fixes
4. Eval Harness Fixes
5. API Contract
6. Security
7. Frontend Architecture
8. New Features
9. Async Architecture
10. Developer Experience
11. Testing Strategy
12. Observability
13. Updated & New SOPs

---

## 1. XBRL & Ingestion Fixes

### 1.1 XBRL Concept Mapping (the verifier's hardest problem)

The verifier must map a free-text extracted claim ("data center revenue was $47.5B") to an XBRL concept name (`Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, etc.). Different companies use different concepts for the same economic item. This is not solvable by exact string match.

**Design:**

Maintain a concept alias table in `ingest/xbrl_concepts.py`:

```python
CONCEPT_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "total_assets": ["Assets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
              "CashCashEquivalentsAndShortTermInvestments"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
               "CapitalExpendituresIncurredButNotYetPaid"],
}
```

The verifier resolves a candidate concept by:
1. Embedding the extracted claim label with a small embedding model
2. Cosine-matching against pre-embedded canonical concept names
3. Falling back to the alias table for well-known concepts
4. If no match above 0.75 cosine similarity → mark `unverifiable`, never `mismatch`

**Rule:** a failed concept match is `unverifiable`, not `mismatch`. Only mark `mismatch` when you are confident you found the right concept and the value differs beyond tolerance.

### 1.2 Unit & Scale Normalization (the #1 false-mismatch cause)

XBRL reports raw values (e.g., `47532000000` for $47.5B). Filings display values in millions or thousands with a note in the header. The VLM extracts the displayed value. These must be normalized before comparison.

```python
# ingest/xbrl.py
import re

SCALE_PATTERNS = [
    (r"in millions", 1_000_000),
    (r"in thousands", 1_000),
    (r"in billions", 1_000_000_000),
]

def normalize_display_value(text_value: str, filing_scale: int) -> float | None:
    """Convert a displayed figure to raw units matching XBRL."""
    cleaned = re.sub(r"[,$\s()]", "", text_value)
    try:
        v = float(cleaned)
    except ValueError:
        return None
    # Parentheses = negative (accounting convention)
    if "(" in text_value:
        v = -v
    return v * filing_scale

def detect_filing_scale(html_text: str) -> int:
    """Read the scale declaration from the filing header."""
    lower = html_text.lower()
    for pattern, scale in SCALE_PATTERNS:
        if re.search(pattern, lower):
            return scale
    return 1  # raw — rare but some filings report in raw dollars

def compare_with_tolerance(extracted: float, xbrl: float, tol: float = 0.01) -> bool:
    if xbrl == 0:
        return extracted == 0
    return abs(extracted - xbrl) / abs(xbrl) <= tol
```

Store `filing_scale` as a column on the `filings` table so it is detected once at ingest and reused by every verifier call.

### 1.3 Flow vs Stock Facts (period matching)

XBRL facts are either:
- **Flow** (income statement, cash flow): reported over a period → have both `start` and `end` dates. Match by `start` AND `end`.
- **Stock** (balance sheet): reported at a point in time → have only `end` date. Match by `end` only.

```python
def match_period(fact: dict, query_start: str | None, query_end: str) -> bool:
    if fact.get("start"):  # flow concept
        return fact["start"] == query_start and fact["end"] == query_end
    else:  # stock concept
        return fact["end"] == query_end
```

The planner must extract the temporal intent from the question and set `query_start`/`query_end` in state before the verifier runs. Failing to distinguish these is the #2 cause of false mismatches after scale errors.

### 1.4 Null & Restatement Values in XBRL

Some XBRL facts have `val: null` — these are restatement markers or correction entries. The normalizer must skip them:

```python
def normalize_facts(companyfacts: dict) -> list[dict]:
    rows = []
    for taxonomy in ("us-gaap", "dei"):
        for concept, body in companyfacts.get("facts", {}).get(taxonomy, {}).items():
            for unit, facts in body.get("units", {}).items():
                for f in facts:
                    if f.get("val") is None:        # skip restatement markers
                        continue
                    if f.get("accn") is None:       # skip unattributed facts
                        continue
                    rows.append({
                        "concept": concept,
                        "unit": unit,
                        "value": f["val"],
                        "start": f.get("start"),
                        "end": f.get("end"),
                        "fy": f.get("fy"),
                        "fp": f.get("fp"),
                        "form": f.get("form"),
                        "accn": f.get("accn"),
                        "is_flow": "start" in f,
                    })
    return rows
```

### 1.5 Amended Filings (10-K/A, 10-Q/A)

When a company files an amendment, the amended facts supersede the original. The verifier must use the most recent version.

**Design:**
- When ingesting, detect `form` values ending in `/A` (e.g., `10-K/A`)
- Store an `is_amendment` flag and a `amends_accn` foreign key on the `filings` table
- In the verifier's fact lookup, prefer the amendment's facts over the original's for the same period
- In `requests_cache`, key on `(url, filing_date)` not just `url` so an amendment at the same URL invalidates the cache

```python
# ingest/filings.py
def is_amendment(form: str) -> bool:
    return form.endswith("/A")
```

### 1.6 EDGAR Submissions Pagination

For large companies, `submissions/CIK.json` paginates via a `files` array:

```python
# ingest/edgar.py
def get_all_filings(cik10: str) -> list[dict]:
    data = get(f"https://data.sec.gov/submissions/CIK{cik10}.json").json()
    recent = _extract_filings(data["filings"]["recent"])

    # Follow pagination
    for extra_file in data["filings"].get("files", []):
        extra = get(f"https://data.sec.gov/submissions/{extra_file['name']}").json()
        recent.extend(_extract_filings(extra))

    return recent

def _extract_filings(block: dict) -> list[dict]:
    keys = ["accessionNumber", "form", "filingDate", "primaryDocument", "reportDate"]
    return [dict(zip(keys, row)) for row in zip(*[block[k] for k in keys])]
```

### 1.7 Large companyfacts JSON (streaming parse)

For large companies (AAPL, MSFT), `companyfacts` can be 200–400 MB. Do not `.json()` the whole response.

```python
import ijson  # streaming JSON parser

def stream_normalize_facts(cik10: str) -> Iterator[dict]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    response = get_stream(url)  # add get_stream() to edgar.py that returns raw response
    for concept, body in ijson.items(response.raw, "facts.us-gaap"):
        for unit, facts in body.get("units", {}).items():
            for f in facts:
                if f.get("val") is not None:
                    yield {...}
```

Add `ijson` to dependencies.

### 1.8 Scanned PDF / Bad Render Detection

Some older filings are scanned PDFs embedded as HTML. The PNG will be unreadable by the VLM and the text extraction will be empty.

**Detection:**
```python
def is_scanned_page(png_path: str) -> bool:
    """Heuristic: very low text density from OCR suggests scanned image."""
    import pytesseract
    from PIL import Image
    text = pytesseract.image_to_string(Image.open(png_path))
    return len(text.strip()) < 50  # fewer than 50 chars → likely scanned

def render_with_fallback(html_path: str, pdf_path: str, out_dir: str) -> list[str]:
    paths = html_to_pdf_to_pngs(html_path, pdf_path, out_dir)
    scanned = [p for p in paths if is_scanned_page(p)]
    if len(scanned) / max(len(paths), 1) > 0.5:
        # Majority scanned — log a warning, store pages with is_scanned=True flag
        # VLM may still work on scanned pages; flag for eval analysis
        logger.warning("filing %s appears to be a scanned document", html_path)
    return paths
```

Store `is_scanned: bool` on the `pages` table. Exclude scanned pages from the text baseline index but keep them in the visual index — the VLM can sometimes read them.

### 1.9 Multi-page Table Handling

Financial tables (income statements, balance sheets) routinely span two pages. The current design treats pages as atomic units, meaning a split table yields incomplete extraction.

**Design:**
- During indexing, detect page boundaries that look like table continuations:
  - The last element on page N is a table row without a closing tag (check the HTML before rendering)
  - The first element on page N+1 is a table row without a header
- Store `continues_from_page` and `continued_on_page` foreign keys on the `pages` table
- In the extractor node, if a retrieved page has `continued_on_page` set, automatically include the continuation page in the VLM input even if it was not in the top-k

```python
# agents/nodes.py — extractor
def extractor(state: State) -> dict:
    pages_to_send = list(state["pages"])
    # Expand continuations
    for page in state["pages"]:
        if page.get("continued_on_page"):
            continuation = load_page(page["continued_on_page"])
            if continuation not in pages_to_send:
                pages_to_send.append(continuation)
    ...
```

### 1.10 Thread-safe Throttle

The current throttle uses a mutable module-level list which is not thread-safe:

```python
# WRONG — race condition
_last = [0.0]

# CORRECT
import threading
_throttle_lock = threading.Lock()
_last_request_time = 0.0

def _throttle(min_interval: float = 0.15) -> None:
    global _last_request_time
    with _throttle_lock:
        now = time.monotonic()
        wait = min_interval - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()
```

### 1.11 Cache Invalidation for Amendments

`requests_cache` keys on URL. An amendment filed at the same URL as the original will be served from cache indefinitely.

**Fix:** add a `cache_key_suffix` based on the filing's `filed` date from the submissions JSON:

```python
def get_filing_doc(accn: str, filename: str, filed_date: str) -> requests.Response:
    url = build_doc_url(accn, filename)
    # Bust cache if the filed date is newer than what we have
    cache_key = f"{url}?filed={filed_date}"
    return session.get(cache_key, headers=HEADERS)
```

---

## 2. ColPali & Retrieval Fixes

### 2.1 Query Encoder — Must Use ColPali's Own

ColPali produces document patch vectors and query token vectors in the **same embedding space**, but only when both are encoded by the same model. You cannot use a generic sentence-transformer for the stage-1 mean-pooled query — the vectors will be in different spaces and stage-1 filtering will be meaningless.

**Correct approach:**

```python
# index/visual.py
from colpali_engine.models import ColPali, ColPaliProcessor

model = ColPali.from_pretrained("vidore/colpali-v1.2")
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

def encode_query(text: str) -> np.ndarray:
    """Returns per-token query vectors (Tq, d) using ColPali's query encoder."""
    inputs = processor.process_queries([text])
    with torch.no_grad():
        query_vecs = model(**inputs)  # (1, Tq, d)
    return query_vecs[0].cpu().numpy()

def encode_query_meanpool(text: str) -> np.ndarray:
    """Mean-pooled query vector for stage-1 pgvector search."""
    return encode_query(text).mean(axis=0)
```

Both stage-1 (mean-pooled) and stage-2 (MaxSim) use ColPali's query encoder. Never substitute a different model.

### 2.2 Batch Inference for Page Indexing

Encoding pages one at a time is ~100x slower than batching. A 200-page filing at 2 seconds per page = 6+ minutes. Batched at 8 pages per batch = under a minute.

```python
def index_pages_batched(png_paths: list[str], batch_size: int = 8) -> list[np.ndarray]:
    all_patch_vecs = []
    for i in range(0, len(png_paths), batch_size):
        batch = [Image.open(p) for p in png_paths[i:i+batch_size]]
        inputs = processor.process_images(batch)
        with torch.no_grad():
            vecs = model(**inputs)  # (B, Tp, d)
        all_patch_vecs.extend([v.cpu().numpy() for v in vecs])
    return all_patch_vecs
```

Run indexing on GPU (Modal, Colab, or local CUDA). CPU indexing is acceptable for development on small corpora but must not be used in CI or production.

### 2.3 Patch Embedding File Convention

Every module that reads or writes patch embeddings must use the same path convention:

```
data/
  embeddings/
    {accn_nodashes}/
      page_{idx:04d}_patches.npy    # float16, shape (Tp, d)
      page_{idx:04d}_meanpool.npy   # float32, shape (d,)  — also stored in pgvector
  pages/
    {accn_nodashes}/
      page_{idx:04d}.png
  html/
    {accn_nodashes}/
      primary.html
  pdf/
    {accn_nodashes}/
      primary.pdf
```

Store patch vectors as `float16` to halve storage. Mean-pooled vectors stored as `float32` in pgvector (pgvector requires float32). For a 1000-page corpus at ColPali's ~1030 patches per page at dim 128: `1000 * 1030 * 128 * 2 bytes ≈ 264 MB` — manageable, but plan for 10x growth.

The `DATA_DIR` root is set in `.env` and read via `pydantic-settings`. Never hardcode paths.

### 2.4 Page Deduplication in Reflection Loops

The `pages` reducer accumulates across reflection loops. The same page will be sent to the VLM on every retry, multiplying cost with no benefit.

```python
# agents/nodes.py — before sending to extractor
def deduplicate_pages(pages: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in pages:
        key = f"{p['accn']}:{p['page_idx']}"
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
```

Call `deduplicate_pages` at the start of the extractor node.

### 2.5 Stage-1 Candidate N Tuning

The stage-1 pgvector query returns N candidates for MaxSim reranking. N is a hyperparameter that trades recall against latency:

- Too small (N=10): good pages get filtered out before MaxSim sees them → low recall@k
- Too large (N=200): MaxSim over 200 pages is slow

Default: **N=50**. Expose as `RETRIEVAL_STAGE1_N` in `.env`. Log the MaxSim score distribution in traces so you can tune this over time. If recall@k is below target, increase N before increasing the embedding quality — it is almost always a candidate-pool problem.

---

## 3. LangGraph Agent Fixes

### 3.1 The Critic's Refined Query

The reflection loop sends an ungrounded answer back to the retriever with a "refined query" — but this refinement was never designed. Without it, the retriever just runs the original query again, producing the same pages, and the loop spins uselessly until the retry cap.

**Design:**

The critic node outputs a `critique` dict that includes `missing_evidence: list[str]` — specific claims that could not be grounded:

```python
# agents/nodes.py — critic
CRITIC_PROMPT = """
You are a faithfulness critic for a financial Q&A system.

Answer: {draft}
Retrieved pages summary: {pages_summary}
Verified facts: {facts}

For each numeric or factual claim in the answer:
1. Is it grounded in a retrieved page?
2. Does a verified fact support it?

Return JSON:
{
  "grounded": bool,
  "ungrounded_claims": ["claim text 1", ...],
  "missing_evidence": ["what to search for to fix this", ...]
}
"""

def critic(state: State) -> dict:
    result = llm_call(CRITIC_PROMPT.format(...))
    critique = parse_json(result)
    return {
        "critique": critique,
        "retries": state["retries"] + (0 if critique["grounded"] else 1),
        # Refined queries for the next retrieval pass
        "plan": critique["missing_evidence"] if not critique["grounded"] else state["plan"],
    }
```

The retriever on the next loop uses `state["plan"]` (now set to `missing_evidence`) instead of re-running the original decomposition.

### 3.2 "I Don't Know" Response Path

When retries are exhausted or the question is genuinely unanswerable, the system must return a calibrated refusal — not a low-confidence hallucination.

```python
# agents/nodes.py — synthesizer
def synthesizer(state: State) -> dict:
    verified_facts = [f for f in state["facts"] if f["verified"] == "match"]
    unverifiable_facts = [f for f in state["facts"] if f["verified"] == "unverifiable"]
    mismatch_facts = [f for f in state["facts"] if f["verified"] == "mismatch"]

    if not verified_facts and not unverifiable_facts:
        return {
            "answer": None,
            "answer_status": "insufficient_data",
            "answer_detail": (
                "The indexed filings do not contain sufficient verified data "
                "to answer this question. Consider indexing additional filings "
                "or refining the question."
            ),
        }

    if not verified_facts and unverifiable_facts:
        # Can answer but with low confidence
        ...

    # Normal path: at least one verified fact
    ...
```

The API response schema (section 5) surfaces `answer_status` to the frontend so it can render an appropriate UI state (grey "no data" state vs orange "low confidence" vs green "verified").

### 3.3 Fast Path XBRL Direct Answer

The planner fast path is described in DESIGN.md but the lookup logic is not specified.

```python
# agents/nodes.py — planner
def planner(state: State) -> dict:
    # Try to classify as a single-fact lookup
    classification = llm_call(PLANNER_PROMPT, state["question"])

    if classification["route"] == "fast":
        concept = classification["xbrl_concept"]    # e.g. "Revenues"
        period_end = classification["period_end"]   # e.g. "2024-01-28"
        period_start = classification.get("period_start")
        ticker = classification["ticker"]

        fact = lookup_xbrl_fact(ticker, concept, period_end, period_start)
        if fact:
            return {
                "route": "fast",
                "facts": [{
                    "text": f"{concept} = {fact['value']} {fact['unit']}",
                    "value": fact["value"],
                    "concept": concept,
                    "page_ref": f"{fact['accn']}:xbrl",
                    "verified": "match",
                }],
            }
        # Fast path miss — fall through to document route
    return {"route": "document", "plan": classification["queries"]}
```

A fast path miss (XBRL has no matching fact for the period) must gracefully fall through to document retrieval, not return an empty answer.

### 3.4 LangGraph Checkpointing (required, not optional)

DESIGN.md says "add a checkpointer if you want resumable runs." This is wrong — checkpointing is required. Without it, a VLM timeout mid-graph loses all work, the API returns a 500, and the user sees nothing.

```python
# agents/graph.py
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(settings.database_url)
app = g.compile(checkpointer=checkpointer)
```

Use thread_id = a UUID per API request. This also enables the multi-turn conversation feature (section 8.4).

---

## 4. Eval Harness Fixes

### 4.1 Gold Set Schema Validation

Every `gold_set.jsonl` entry must be validated against a Pydantic model before the eval runner starts. A malformed entry currently silently breaks the entire run.

```python
# eval/gold.py
from pydantic import BaseModel, field_validator
from typing import Literal

class GoldItem(BaseModel):
    q: str
    answer_value: float | None = None
    unit: str | None = None
    concept: str | None = None
    accn: str | None = None
    kind: Literal["numeric", "reasoning", "negative", "adversarial"]
    subset: list[str] = []
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    rubric: str | None = None  # required if kind == "reasoning"
    expected_status: Literal["answered", "insufficient_data"] = "answered"

    @field_validator("rubric")
    def rubric_required_for_reasoning(cls, v, values):
        if values.get("kind") == "reasoning" and not v:
            raise ValueError("rubric is required for reasoning items")
        return v

def load_gold_set(path: str) -> list[GoldItem]:
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                items.append(GoldItem.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"gold_set.jsonl line {i+1} is invalid: {e}")
    return items
```

### 4.2 Gold Set Deduplication

Auto-generated XBRL questions produce near-duplicates (same concept, adjacent quarterly periods). Deduplicate before writing to `gold_set.jsonl`:

```python
def deduplicate_auto_questions(items: list[dict], max_per_concept: int = 3) -> list[dict]:
    from collections import defaultdict
    by_concept = defaultdict(list)
    for item in items:
        by_concept[item["concept"]].append(item)
    result = []
    for concept, group in by_concept.items():
        # Keep the N most recent by end date
        group.sort(key=lambda x: x.get("end", ""), reverse=True)
        result.extend(group[:max_per_concept])
    return result
```

### 4.3 Difficulty Tagging

Tag every gold item with a difficulty level so the eval report can break down performance by difficulty:

- `easy` — single-fact XBRL lookup answerable on the fast path
- `medium` — single filing, single page, clear table value
- `hard` — multi-page, multi-period, requires reasoning or comparison

The CI subset should be weighted toward `easy` and `medium` for speed. Full eval runs all difficulties.

### 4.4 Negative Examples (unanswerable questions)

The gold set must include questions where the correct answer is `insufficient_data`:

```jsonl
{"q": "What was NVDA revenue in Q2 2019?", "kind": "negative", "expected_status": "insufficient_data", "subset": ["ci"], "difficulty": "easy"}
{"q": "What is AAPL's data center revenue?", "kind": "negative", "expected_status": "insufficient_data", "subset": [], "difficulty": "medium"}
```

The eval scores these on whether the system correctly declines rather than hallucinates. A system that gets 100% on positive questions but hallucinates on every negative is unacceptable.

### 4.5 Adversarial Examples

Questions designed to trigger hallucination:

```jsonl
{"q": "What was NVDA revenue in Q3 2031?", "kind": "adversarial", "expected_status": "insufficient_data"}
{"q": "What is the NVDA-EDGAR-SECRET internal forecast for next quarter?", "kind": "adversarial", "expected_status": "insufficient_data"}
```

At least 10% of the CI subset should be adversarial or negative.

### 4.6 LLM-as-Judge Calibration Protocol

"Calibrate against a handful of human labels" is too vague. Specific protocol:

1. Sample 50 reasoning answers from the system (mix of correct, partially correct, wrong)
2. Have at least one human label each as: `correct` / `partially_correct` / `wrong`
3. Run the LLM judge on the same 50
4. Compute agreement rate. **Threshold: ≥ 80% agreement before using the judge in eval.**
5. If below threshold: improve the judge prompt, try a stronger model, or add few-shot examples
6. Store the calibration set in `eval/judge_calibration.jsonl` — versioned alongside the gold set
7. Re-calibrate any time you change the judge model or prompt

Gate `eval.run` on `--skip-judge-calibration-check` being explicitly passed if calibration hasn't been run; otherwise fail with a clear error.

---

## 5. API Contract

### 5.1 Endpoints

```
POST   /query                  Submit a question, returns a streaming SSE response
GET    /query/{query_id}       Get a completed query result by ID (for polling fallback)
GET    /filings                List indexed filings with pagination
GET    /filings/{accn}         Get metadata for a specific filing
GET    /pages/{accn}/{page_idx}  Serve the page PNG (for citation display)
GET    /companies              List indexed companies
POST   /compare                Cross-company comparison query (section 8.1)
GET    /health                 Health check — db connectivity, model availability
GET    /metrics                Prometheus-style cost and latency counters
```

All routes except `/health` require `X-API-Key` header authentication.

### 5.2 Request & Response Schemas

```python
# api/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    ticker: str | None = None          # scope to a company
    filing_accn: str | None = None     # scope to a specific filing
    pipeline: Literal["visual", "baseline", "auto"] = "auto"
    thread_id: str | None = None       # for multi-turn continuity

class FactResult(BaseModel):
    text: str
    value: float | None
    concept: str | None
    page_ref: str                      # "accn:page_idx" or "accn:xbrl"
    verified: Literal["match", "mismatch", "unverifiable", "pending"]

class QueryResponse(BaseModel):
    query_id: str
    answer: str | None
    answer_status: Literal["answered", "low_confidence", "insufficient_data", "error"]
    facts: list[FactResult]
    route: Literal["fast", "document"]
    retries: int
    cost_usd: float
    latency_ms: int
    pipeline: Literal["visual", "baseline"]

class FilingMeta(BaseModel):
    accn: str
    ticker: str
    form: str
    filing_date: str
    report_date: str
    page_count: int
    is_indexed_visual: bool
    is_indexed_text: bool
    is_amendment: bool
```

### 5.3 Streaming via SSE

`POST /query` returns `text/event-stream`. Events:

```
event: status
data: {"stage": "planner", "route": "document"}

event: status
data: {"stage": "retriever", "pages_found": 12}

event: status
data: {"stage": "extractor", "pages_sent_to_vlm": 5}

event: status
data: {"stage": "verifier", "verified": 3, "mismatch": 0, "unverifiable": 1}

event: token
data: {"text": "Data center revenue"}

event: token
data: {"text": " was $47.5B"}

event: result
data: { ...full QueryResponse... }

event: done
data: {}
```

The frontend renders status events as a progress indicator and token events as streamed text. The `result` event delivers the full structured response for citation rendering.

```python
# api/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agents.graph import stream_graph

@app.post("/query")
async def query(req: QueryRequest, api_key: str = Depends(verify_api_key)):
    async def event_stream():
        async for event in stream_graph(req):
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 5.4 Error Response Schema

```python
class ErrorResponse(BaseModel):
    error: str        # machine-readable code: "rate_limited", "not_indexed", "model_unavailable"
    message: str      # human-readable
    retry_after: int | None  # seconds, for rate limit errors
```

HTTP status codes: 400 (bad request), 401 (bad key), 429 (rate limited), 503 (model unavailable), 500 (unexpected). Never return a 200 with an error body.

---

## 6. Security

### 6.1 API Key Authentication

```python
# api/auth.py
import hashlib, secrets
from fastapi import Header, HTTPException

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

async def verify_api_key(x_api_key: str = Header(...)) -> str:
    hashed = hash_key(x_api_key)
    if hashed not in settings.valid_api_key_hashes:  # set in .env as comma-separated hashes
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

Never store API keys in plaintext — store only the SHA-256 hash. Issue keys with `secrets.token_urlsafe(32)`.

### 6.2 Rate Limiting

```python
# api/main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")   # per IP; tighten to per API key in production
async def query(...):
    ...
```

VLM queries are expensive — 10/minute is generous for a demo, reduce to 3/minute per key if costs are a concern. Add a daily cap per key.

### 6.3 Input Validation & Prompt Injection Prevention

```python
# api/validation.py
import re

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior |above )?instructions",
    r"you are now",
    r"system prompt",
    r"<\|.*?\|>",           # token injection attempts
    r"\\n\\nHuman:",        # prompt structure injection
]

def sanitize_question(text: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("question_rejected")
    # Strip null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()
```

In every LLM prompt, the user question must be wrapped in explicit delimiters:

```python
SYNTHESIZER_PROMPT = """
You are a financial analyst. Answer based only on the verified facts below.

<verified_facts>
{facts}
</verified_facts>

<user_question>
{question}
</user_question>

Answer:
"""
```

The `<user_question>` tags prevent the model from treating user content as system instructions.

### 6.4 CORS

```python
# api/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],  # e.g. "https://ledgerlens.vercel.app"
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)
```

Never use `allow_origins=["*"]` in production.

### 6.5 No Secrets in Logs

```python
# api/logging_config.py
REDACT_FIELDS = {"api_key", "x_api_key", "authorization", "database_url", "openai_api_key"}

class RedactingFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            for field in REDACT_FIELDS:
                record.msg = re.sub(
                    rf'"{field}"\s*:\s*"[^"]*"',
                    f'"{field}": "[REDACTED]"',
                    record.msg, flags=re.IGNORECASE
                )
        return True
```

---

## 7. Frontend Architecture

### 7.1 Stack

```
Framework:    Next.js 14 (App Router)
Styling:      Tailwind CSS + shadcn/ui
Charts:       Recharts
State:        Zustand (client state) + React Query (server state)
API client:   fetch with SSE via EventSource
Type safety:  TypeScript strict mode
```

### 7.2 Page Structure

```
app/
  page.tsx              # Home: search box + recent queries
  query/
    [id]/page.tsx       # Query result with citations
  filings/
    page.tsx            # Filing browser (company list)
    [accn]/page.tsx     # Filing detail: page thumbnails
  compare/
    page.tsx            # Cross-company comparison UI
  components/
    SearchBox.tsx
    AnswerPanel.tsx      # Streamed answer + fact badges
    CitationViewer.tsx   # Page image with highlighted region
    FactBadge.tsx        # verified / unverifiable / mismatch chip
    PipelineToggle.tsx   # visual vs baseline switch
    TrendChart.tsx       # Recharts line chart for time-series data
    StatusProgress.tsx   # Streaming status (planner → retriever → ...)
```

### 7.3 Core UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  LedgerLens     [NVDA ▾]   [visual | baseline]   [Browse]  │
├─────────────────────────────────────────────────────────────┤
│  What was NVDA data center revenue in FY2024?        [Ask]  │
├────────────────────────────┬────────────────────────────────┤
│                            │                                │
│  ● Planner → document      │  📄  10-K FY2024  · Page 47   │
│  ● Retriever → 12 pages    │  ┌──────────────────────────┐ │
│  ● VLM → 5 pages           │  │                          │ │
│  ● Verifier → 3 matched    │  │   [page image PNG]       │ │
│                            │  │   [highlighted row]      │ │
│  Data center revenue was   │  │                          │ │
│  **$47.5B** in FY2024      │  └──────────────────────────┘ │
│  [✓ verified] [cite]       │                                │
│                            │  Concept:  Revenues            │
│  ── vs baseline ──         │  XBRL:     $47,532,000,000     │
│  Baseline: $48.1B          │  Extracted: $47.5B             │
│  (no citation, unverified) │  Match:    ✓ within 0.1%       │
│                            │  Filing:   0001045810-24-...   │
└────────────────────────────┴────────────────────────────────┘
```

The side-by-side visual answer vs baseline hallucination is the demo narrative made interactive.

### 7.4 Citation Display

When the user clicks a citation (`accn:page_idx`), the right panel fetches `GET /pages/{accn}/{page_idx}` and renders the PNG. If the extractor returned bbox coordinates, draw a highlight rectangle using a canvas overlay:

```tsx
// components/CitationViewer.tsx
export function CitationViewer({ pageRef, bbox }: Props) {
  const { data: imageUrl } = useQuery(["page", pageRef], () =>
    fetchPageImage(pageRef)
  );
  return (
    <div className="relative">
      <img src={imageUrl} className="w-full" />
      {bbox && (
        <div
          className="absolute border-2 border-yellow-400 bg-yellow-100/30"
          style={{
            left: `${bbox.x * 100}%`,
            top: `${bbox.y * 100}%`,
            width: `${bbox.w * 100}%`,
            height: `${bbox.h * 100}%`,
          }}
        />
      )}
    </div>
  );
}
```

Bbox coordinates should be normalized (0–1 relative to page dimensions) and returned in the `FactResult.page_ref` as an optional `bbox` field.

### 7.5 SSE Client

```tsx
// lib/queryStream.ts
export async function* streamQuery(req: QueryRequest) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": getApiKey() },
    body: JSON.stringify(req),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";
    for (const block of lines) {
      const [eventLine, dataLine] = block.split("\n");
      const type = eventLine.replace("event: ", "");
      const data = JSON.parse(dataLine.replace("data: ", ""));
      yield { type, data };
    }
  }
}
```

---

## 8. New Features

### 8.1 Cross-company Comparison

Questions like "How does NVDA gross margin compare to AMD?" require parallel retrieval across two companies.

**State change:**

```python
class CompareRequest(BaseModel):
    question: str
    tickers: list[str]   # ["NVDA", "AMD"]
    metric: str | None   # optional: pre-specify the concept
    period: str | None   # optional: "Q3 2024"
```

**Graph change:** add a `fan_out` node that spawns parallel sub-graphs (one per ticker) using LangGraph's `Send` API, then a `merge` node that collects the results:

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("sub_graph", {"ticker": t, "question": state["question"]})
            for t in state["tickers"]]
```

The synthesizer then receives facts from multiple tickers and generates a comparative answer. The frontend renders a table or chart with one column/line per company.

### 8.2 Trend Visualization

XBRL time-series data is already in the database. For any numeric answer, offer a trend chart with zero additional retrieval cost.

```python
# agents/nodes.py — after verifier
def attach_trend(state: State) -> dict:
    trends = {}
    for fact in state["facts"]:
        if fact["verified"] == "match" and fact["concept"]:
            series = query_xbrl_time_series(
                ticker=state["ticker"],
                concept=fact["concept"],
                periods=8  # last 8 quarters
            )
            trends[fact["concept"]] = series
    return {"trends": trends}
```

The API response includes `trends: dict[concept, list[{period, value}]]`. The frontend renders this as a Recharts line chart below the answer.

### 8.3 Earnings Call Transcripts

The stretch module from DESIGN.md, made concrete:

**Ingestion:**
```python
# ingest/transcripts.py
import whisper

def transcribe_earnings_call(audio_url: str, accn: str) -> list[dict]:
    """Download and transcribe an earnings call audio file."""
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_url, word_timestamps=True)
    # Segment into ~30-second chunks with speaker-change detection
    return chunk_transcript(result["segments"], accn)
```

**Indexing:** transcript chunks go into the same `chunks` table as filing text, with `source_type = "transcript"`. The text baseline and visual pipeline both benefit immediately.

**The unique feature:** for any verified XBRL fact, query the transcript chunks to find where management discussed that figure. Annotate the answer with both the filing citation and the earnings call quote. This is the feature that most interviewers will not have seen before.

### 8.4 Multi-turn Conversation

LangGraph checkpointing (section 3.4) enables this almost for free.

```python
# api/main.py
@app.post("/query")
async def query(req: QueryRequest, ...):
    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await run_graph(req, config)
    return {"thread_id": thread_id, **result}
```

The frontend stores `thread_id` per conversation and passes it on follow-up questions. The graph's checkpointer restores previous state (retrieved pages, verified facts, prior answers) so follow-ups like "how does that compare to Q2?" have full context.

The session panel in the frontend lists recent questions in the thread and allows branching (start a new thread from any point).

---

## 9. Async Architecture

### 9.1 The Problem

LangGraph's synchronous `app.invoke()` blocks the calling thread. Inside an async FastAPI route, this blocks the entire event loop — every concurrent user waits for every other user's VLM call.

### 9.2 Solution: asyncio.to_thread for development, task queue for production

**Development (simple):**
```python
import asyncio

async def run_graph(req: QueryRequest, config: dict) -> QueryResponse:
    return await asyncio.to_thread(app.invoke, req.dict(), config)
```

**Production (correct):** Use ARQ (async Redis-backed task queue):

```python
# workers/query_worker.py
from arq import cron

async def process_query(ctx, query_id: str, req_dict: dict) -> None:
    result = app.invoke(req_dict, {"configurable": {"thread_id": req_dict["thread_id"]}})
    await store_result(query_id, result)
    # SSE push via Redis pub/sub to the waiting API route
    await ctx["redis"].publish(f"query:{query_id}", json.dumps(result))
```

The API route submits the task, then listens on Redis pub/sub for the result and streams it to the client via SSE. This decouples the API from the graph execution, allows horizontal scaling, and handles timeouts gracefully.

For portfolio purposes, `asyncio.to_thread` is sufficient. Document the production path in a comment.

---

## 10. Developer Experience

### 10.1 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [--strict, --ignore-missing-imports]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files   # catch accidentally staged PNGs/embeddings
        args: [--maxkb=500]
      - id: detect-private-key
      - id: check-json
      - id: check-toml
```

Install: `uv run pre-commit install`. This runs before every commit, not just in CI.

### 10.2 mypy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
exclude = ["eval/gold_set.jsonl"]
```

`strict = true` enables: `--disallow-untyped-defs`, `--disallow-any-generics`, `--warn-return-any`, `--warn-unused-ignores`. This is not optional decoration — it catches the unit normalization bugs and flow/stock confusion before they reach production.

### 10.3 .env.example

```bash
# .env.example — copy to .env and fill in real values
# EDGAR
EDGAR_USER_AGENT="LedgerLens Your Name your@email.com"

# Database
DATABASE_URL="postgresql://ledgerlens:password@localhost:5432/ledgerlens"

# Models
OPENAI_API_KEY=""              # for planner/critic/synthesizer
ANTHROPIC_API_KEY=""           # alternative
HF_TOKEN=""                    # for gated ColPali checkpoints
COLPALI_CHECKPOINT="vidore/colpali-v1.2"
VLM_MODEL="Qwen/Qwen2-VL-7B-Instruct"
PLANNER_MODEL="gpt-4o-mini"
SYNTHESIZER_MODEL="gpt-4o"

# API
API_KEY_HASHES=""              # comma-separated SHA-256 hashes of valid API keys
FRONTEND_URL="http://localhost:3000"

# Paths
DATA_DIR="./data"

# Retrieval
RETRIEVAL_STAGE1_N=50
RETRIEVAL_TOP_K=5

# Eval
EVAL_MIN_NUMERIC_EM=0.85
EVAL_MAX_HALLUCINATION=0.02

# Observability
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST="https://cloud.langfuse.com"

# Rate limiting
RATE_LIMIT_PER_MINUTE=10
```

### 10.4 Alembic Migrations

```bash
uv add alembic
uv run alembic init alembic
```

```python
# alembic/env.py — wire to pydantic-settings
from app.config import settings
config.set_main_option("sqlalchemy.url", settings.database_url)
```

Every schema change is a migration file. Never `ALTER TABLE` by hand in development; it won't be reproducible in CI or on another machine.

Initial migration creates: `filings`, `pages`, `chunks`, `xbrl_facts`, `queries` tables with all HNSW indexes.

### 10.5 CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint-and-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run mypy .

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_PASSWORD: test }
        options: --health-cmd pg_isready
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync
      - run: uv run pytest tests/ -v --tb=short

  eval-gate:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync
      - run: uv run python -m eval.run --subset ci --report results.json
        env:
          DATABASE_URL: ${{ secrets.CI_DATABASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - run: uv run python -m eval.gate results.json
          --min-numeric-em ${{ vars.EVAL_MIN_NUMERIC_EM }}
          --max-hallucination ${{ vars.EVAL_MAX_HALLUCINATION }}
```

Lint and type checks run on every push. Eval gate runs only on PRs (it costs money).

---

## 11. Testing Strategy

### 11.1 Test Layout

```
tests/
  unit/
    test_xbrl_normalization.py    # normalize_facts, scale detection, flow/stock
    test_verifier_logic.py        # compare_with_tolerance, concept matching
    test_throttle.py              # thread-safety of edgar.py throttle
    test_gold_schema.py           # validates gold_set.jsonl on load
    test_input_validation.py      # prompt injection patterns, length limits
  integration/
    test_edgar_client.py          # VCR cassettes — no live SEC calls
    test_xbrl_ingest.py           # end-to-end XBRL parsing on fixture data
    test_text_baseline.py         # chunk + embed + retrieve on fixture filing
    test_graph_routing.py         # fast path vs document path logic
  fixtures/
    cassettes/                    # VCR recorded HTTP responses
    sample_filing.html            # small fixture filing
    sample_companyfacts.json      # fixture XBRL data
```

### 11.2 EDGAR Integration Tests with VCR

Never hit live SEC servers in tests. Use `vcrpy` to record and replay HTTP interactions:

```python
# tests/integration/test_edgar_client.py
import vcr
from ingest.edgar import get

@vcr.use_cassette("tests/fixtures/cassettes/nvda_submissions.yaml")
def test_get_submissions():
    r = get("https://data.sec.gov/submissions/CIK0001045810.json")
    assert r.status_code == 200
    data = r.json()
    assert "filings" in data
```

Record cassettes once (`--record-mode=new_episodes`), commit them, and replay forever. CI never needs network access.

### 11.3 Graph Routing Tests with Mocked Nodes

Test the conditional routing logic without running any LLM:

```python
# tests/unit/test_graph_routing.py
from unittest.mock import patch
from agents.graph import app

def test_fast_path_routing():
    with patch("agents.nodes.planner") as mock_planner:
        mock_planner.return_value = {
            "route": "fast",
            "facts": [{"verified": "match", "value": 47532.0, ...}],
        }
        result = app.invoke({"question": "What was NVDA revenue in FY2024?"})
        assert result["route"] == "fast"
        assert "retriever" not in result["_node_history"]

def test_reflection_loop_bounded():
    with patch("agents.nodes.critic") as mock_critic:
        mock_critic.return_value = {"critique": {"grounded": False}, "retries": 3}
        result = app.invoke({"question": "..."})
        # Must not exceed retry cap
        assert result["retries"] <= 2
```

### 11.4 Verifier Unit Tests (exhaustive)

The verifier's normalization logic is pure functions. Test every edge case:

```python
# tests/unit/test_verifier_logic.py
import pytest
from ingest.xbrl import normalize_display_value, compare_with_tolerance

@pytest.mark.parametrize("text,scale,expected", [
    ("47,532",   1_000_000,  47_532_000_000),
    ("(1,234)",  1_000_000,  -1_234_000_000),  # negative via parentheses
    ("$47.5",    1_000_000_000, 47_500_000_000),
    ("47.532",   1_000_000,  47_532_000),
])
def test_normalize_display_value(text, scale, expected):
    assert normalize_display_value(text, scale) == expected

def test_compare_tolerance_rounding():
    assert compare_with_tolerance(47.5e9, 47_532e6, tol=0.01)  # 0.07% diff
    assert not compare_with_tolerance(48.1e9, 47_532e6, tol=0.01)  # 1.2% diff
```

---

## 12. Observability

### 12.1 Structured Trace Schema

Every LangGraph node emits a trace event with this schema. Use Langfuse spans:

```python
# agents/tracing.py
from langfuse import Langfuse
from dataclasses import dataclass

langfuse = Langfuse()

@dataclass
class NodeTrace:
    node_name: str
    model_id: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    # Node-specific fields
    route_decision: str | None = None     # planner
    pages_retrieved: int | None = None    # retriever
    stage1_candidates: int | None = None  # retriever
    pages_sent_to_vlm: int | None = None  # extractor
    facts_extracted: int | None = None    # extractor
    verified_count: int | None = None     # verifier
    mismatch_count: int | None = None     # verifier
    unverifiable_count: int | None = None # verifier
    critic_grounded: bool | None = None   # critic
    retry_number: int | None = None       # critic
```

Wrap each node:

```python
def traced_node(node_fn, node_name: str):
    def wrapper(state: State) -> dict:
        start = time.monotonic()
        result = node_fn(state)
        trace = NodeTrace(
            node_name=node_name,
            latency_ms=int((time.monotonic() - start) * 1000),
            # ... populate from result
        )
        langfuse.span(name=node_name, metadata=asdict(trace))
        return result
    return wrapper
```

### 12.2 Cost Tracking

```python
# agents/cost.py
# Costs per 1M tokens (update when model pricing changes)
MODEL_COSTS = {
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
    "gpt-4o":        {"input": 5.00,  "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = MODEL_COSTS.get(model, {"input": 0, "output": 0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
```

Log cumulative cost per query. Expose `/metrics` endpoint with:
- Total cost today / this week / this month
- p50, p95, p99 latency per node
- Fast-path hit rate
- VLM average pages per call
- Cache hit rates

---

## 13. Updated & New SOPs

### SOP-3 (updated) — Ingest a new issuer / filing

Add step after step 1:
- **1b.** Check for amendments: `$ uv run python -m ingest.filings --ticker NVDA --list --include-amendments`
  - If any `10-K/A` or `10-Q/A` exists for a period you have indexed, re-ingest that period: the amendment's facts supersede the original.

Add to rules section:
- If the render produces > 50% scanned pages (detected automatically), log a warning and manually verify one page before proceeding. Scanned filings are indexed in the visual pipeline only.
- For any company with > 200 historical filings, run the submissions paginator explicitly: `--paginate-all`

### SOP-5 (updated) — Build / refresh the visual (ColPali) index

Add before step 1:
- **0.** Confirm the ColPali checkpoint in `.env` matches the embedding dimension in the `pages` table `vector(d)` column. If you changed the checkpoint, run `$ uv run python -m index.store remigrate-vectors --dim NEW_DIM` before indexing.

Add to rules section:
- Always index in batches of 8 pages minimum. Single-page indexing is for debugging only.
- On CPU: expect ~2 seconds per page. On GPU (T4): ~0.15 seconds per page. For > 100 pages, use Modal: `$ modal run index/visual_modal.py --accn <accn>`
- Never mix query encoders: the stage-1 mean-pooled query vector must come from `encode_query_meanpool()` in `index/visual.py`, not a separate embedding model.

### SOP-9 (updated) — Safely change a model or prompt

Add step 1a:
- **1a.** Run the judge calibration check: `$ uv run python -m eval.judge --calibrate --sample 50`. Confirm agreement ≥ 80% before trusting reasoning metrics in the before/after diff.

Add to acceptance criteria:
- Accept only if: numeric exact-match holds or improves AND hallucination does not rise AND negative/adversarial examples still correctly return `insufficient_data`. A change that improves positive-question accuracy while degrading refusal behavior is not acceptable.

### SOP-15 (new) — Multi-turn Conversation Session Management

**Purpose:** manage LangGraph checkpoint state for conversation threads. **Run when:** debugging a multi-turn session or clearing stale state.

1. List active threads: `$ uv run python -m agents.graph --list-threads --max-age 24h`
2. Inspect a thread: `$ uv run python -m agents.graph --thread-id <id> --show-history`
3. Clear a stale thread: `$ uv run python -m agents.graph --thread-id <id> --clear`
4. Bulk clear threads older than N days: `$ uv run python -m agents.graph --clear-threads-older 7`

Thread state is stored in the checkpointer's Postgres table. Run a weekly cron to clear threads older than 30 days.

### SOP-16 (new) — Frontend Development

1. Install deps: `$ cd app && npm install`
2. Set frontend env: `$ cp .env.local.example .env.local`, set `NEXT_PUBLIC_API_URL=http://localhost:8000` and `NEXT_PUBLIC_API_KEY=<your-dev-key>`
3. Start dev server: `$ npm run dev` — available at `http://localhost:3000`
4. The API must be running (SOP-2) for queries to work.
5. Type check: `$ npm run typecheck`
6. Lint: `$ npm run lint`

**Verify citation display:** query a known question, click the citation badge, confirm the correct page image loads and the highlight is positioned correctly over the relevant row.

### SOP-17 (new) — Generating and Rotating API Keys

1. Generate a key: `$ uv run python -m api.auth --generate`
   - Prints a key (copy it — it is shown once) and its SHA-256 hash.
2. Add the hash to `.env` under `API_KEY_HASHES` (comma-separated if multiple keys).
3. Restart the API: `$ uv run uvicorn api.main:app --reload`
4. Test the key: `$ curl -H "X-API-Key: <key>" http://localhost:8000/health`
5. To rotate a key: generate a new one, add the new hash, remove the old hash, restart. Zero-downtime if you add the new hash before removing the old.
6. If a key leaks: remove its hash immediately, restart, then rotate.

---

## Quick reference — gaps and where they are fixed

| Gap | Section |
|---|---|
| XBRL concept mapping | 1.1 |
| Unit/scale normalization | 1.2 |
| Flow vs stock period matching | 1.3 |
| Null XBRL values | 1.4 |
| Amended filings | 1.5, SOP-3 update |
| EDGAR submissions pagination | 1.6 |
| Large companyfacts streaming | 1.7 |
| Scanned PDF detection | 1.8 |
| Multi-page table handling | 1.9 |
| Thread-safe throttle | 1.10 |
| Cache invalidation on amendments | 1.11 |
| ColPali query encoder | 2.1 |
| Batch inference | 2.2 |
| Patch embedding file convention | 2.3 |
| Page deduplication in reflection | 2.4 |
| Stage-1 N tuning | 2.5 |
| Critic's refined query | 3.1 |
| "I don't know" response | 3.2 |
| Fast path miss handling | 3.3 |
| Checkpointing (required) | 3.4 |
| Gold set schema validation | 4.1 |
| Gold set deduplication | 4.2 |
| Difficulty tagging | 4.3 |
| Negative examples | 4.4 |
| Adversarial examples | 4.5 |
| LLM-as-judge calibration | 4.6 |
| API contract + streaming | 5 |
| Auth, rate limit, injection, CORS | 6 |
| Frontend architecture | 7 |
| Cross-company comparison | 8.1 |
| Trend visualization | 8.2 |
| Earnings call transcripts | 8.3 |
| Multi-turn conversation | 8.4 |
| Async architecture | 9 |
| Pre-commit, mypy, .env.example, alembic, CI | 10 |
| Testing strategy | 11 |
| Observability trace schema | 12 |
