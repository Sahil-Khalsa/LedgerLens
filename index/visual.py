"""
ColPali visual index — page embedding, patch cache, and two-stage retrieval.

Two-stage retrieval (SPEC.md §2):
  Stage 1 — pgvector cosine on mean-pooled page vectors → top-N candidates
  Stage 2 — MaxSim rerank on full patch embeddings → top-k pages for VLM

Query encoding: MUST use ColPali's own encode_query_meanpool().
Never substitute a generic sentence-transformer here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Lazy model singletons ─────────────────────────────────────────────────────

_model: Any = None
_processor: Any = None


def _get_model() -> Any:
    global _model
    if _model is None:
        import torch
        from colpali_engine.models import ColPali

        _model = ColPali.from_pretrained(
            settings.colpali_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        logger.info("Loaded ColPali model: %s", settings.colpali_checkpoint)
    return _model


def _get_processor() -> Any:
    global _processor
    if _processor is None:
        from colpali_engine.models import ColPaliProcessor

        _processor = ColPaliProcessor.from_pretrained(settings.colpali_checkpoint)
    return _processor


# ── MaxSim scoring ────────────────────────────────────────────────────────────


def maxsim(query_vecs: np.ndarray, page_vecs: np.ndarray) -> float:
    """
    Late-interaction score: for each query token, max similarity to any page patch.
    query_vecs: (Tq, d)
    page_vecs:  (Tp, d)
    Returns scalar score.
    """
    sim = query_vecs @ page_vecs.T  # (Tq, Tp)
    return float(sim.max(axis=1).sum())


# ── Patch embedding cache ─────────────────────────────────────────────────────


def patch_cache_path(accn: str, page_idx: int) -> Path:
    accn_nodashes = accn.replace("-", "")
    return Path(settings.embeddings_dir) / accn_nodashes / f"page_{page_idx:04d}_patches.npy"


def meanpool_cache_path(accn: str, page_idx: int) -> Path:
    accn_nodashes = accn.replace("-", "")
    return Path(settings.embeddings_dir) / accn_nodashes / f"page_{page_idx:04d}_meanpool.npy"


def load_patch_vecs(accn: str, page_idx: int) -> np.ndarray | None:
    path = patch_cache_path(accn, page_idx)
    if not path.exists():
        return None
    result: np.ndarray = np.load(str(path)).astype(np.float32)
    return result


def save_patch_vecs(accn: str, page_idx: int, vecs: np.ndarray) -> None:
    path = patch_cache_path(accn, page_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), vecs.astype(np.float16))


# ── Index ─────────────────────────────────────────────────────────────────────


def index_filing(accn: str, png_paths: list[str]) -> int:
    """
    Embed all pages for a filing using ColPali.
    Saves patch vectors to disk (.npy) and mean-pooled vectors to pgvector.
    Returns number of pages embedded.

    Idempotent: skips pages whose patch cache already exists.
    """
    import torch
    from PIL import Image

    from index.store import Filing, Page, SessionLocal

    model = _get_model()
    processor = _get_processor()
    device = next(iter(model.parameters())).device

    db = SessionLocal()
    embedded = 0
    try:
        for png_path in png_paths:
            stem = Path(png_path).stem  # page_0000
            try:
                page_idx = int(stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning("Cannot parse page_idx from %s, skipping", png_path)
                continue

            if patch_cache_path(accn, page_idx).exists():
                logger.debug("Already embedded %s page %d, skipping", accn, page_idx)
                continue

            img = Image.open(png_path).convert("RGB")
            batch = processor.process_images([img]).to(device)

            with torch.no_grad():
                patch_vecs: np.ndarray = model(**batch).float().cpu().numpy()  # (1, T, d)

            vecs = patch_vecs[0]  # (T, d)
            save_patch_vecs(accn, page_idx, vecs)

            meanpool = vecs.mean(axis=0).tolist()  # (d,)

            page_row = (
                db.query(Page).filter(Page.filing_accn == accn, Page.page_idx == page_idx).first()
            )
            if page_row is not None:
                page_row.pooled_embedding = meanpool

            embedded += 1

        if embedded:
            filing = db.query(Filing).filter(Filing.accn == accn).first()
            if filing is not None:
                filing.is_indexed_visual = True

        db.commit()
        logger.info("Embedded %d pages for %s", embedded, accn)
        return embedded

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Retrieval ─────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    accn: str | None = None,
    stage1_n: int | None = None,
    top_k: int | None = None,
) -> list[dict[str, object]]:
    """
    Two-stage ColPali retrieval.

    Stage 1: encode query with ColPali's own encoder → pgvector cosine top-N
    Stage 2: load patch embeddings for candidates → MaxSim rerank → top-k

    Returns list of {accn, page_idx, png_path, score, continued_on_page}.
    """
    import torch

    from index.store import Page, SessionLocal

    n = stage1_n if stage1_n is not None else settings.retrieval_stage1_n
    k = top_k if top_k is not None else settings.retrieval_top_k

    model = _get_model()
    processor = _get_processor()
    device = next(iter(model.parameters())).device

    # Stage 1: encode query → mean-pooled vector for pgvector search
    q_batch = processor.process_queries([query]).to(device)
    with torch.no_grad():
        q_patch_vecs: np.ndarray = model(**q_batch).float().cpu().numpy()  # (1, Tq, d)
    q_vecs = q_patch_vecs[0]  # (Tq, d) — full patch vectors for MaxSim
    q_mean = q_vecs.mean(axis=0).tolist()  # (d,) — mean pool for stage-1 pgvector

    db = SessionLocal()
    candidates: list[tuple[str, int, str | None, float]] = []
    try:
        stmt = db.query(
            Page,
            Page.pooled_embedding.cosine_distance(q_mean).label("distance"),
        ).filter(Page.pooled_embedding.isnot(None))
        if accn:
            stmt = stmt.filter(Page.filing_accn == accn)
        rows = stmt.order_by("distance").limit(n).all()
        candidates = [
            (str(page.filing_accn), int(page.page_idx), page.png_path, float(dist))
            for page, dist in rows
        ]
    finally:
        db.close()

    if not candidates:
        logger.warning("Stage 1 returned no candidates for query: %s", query[:80])
        return []

    # Stage 2: MaxSim rerank using cached patch vectors
    scored: list[dict[str, object]] = []
    for filing_accn, page_idx, png_path, _ in candidates:
        patch_vecs = load_patch_vecs(filing_accn, page_idx)
        if patch_vecs is None:
            logger.debug("No patch cache for %s page %d, skipping rerank", filing_accn, page_idx)
            continue
        score = maxsim(q_vecs, patch_vecs)
        scored.append(
            {
                "accn": filing_accn,
                "page_idx": page_idx,
                "png_path": png_path or "",
                "score": score,
                "continued_on_page": None,
            }
        )

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    return scored[:k]


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the visual (ColPali) index")
    parser.add_argument("--accn", help="Accession number to index")
    parser.add_argument("--pngs", nargs="+", help="PNG paths (required with --accn)")
    parser.add_argument("--query", help="Retrieval query")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.accn and args.pngs:
        n = index_filing(args.accn, args.pngs)
        print(f"Embedded {n} pages for {args.accn}")

    if args.query:
        results = retrieve(args.query, accn=args.accn, top_k=args.topk)
        for i, r in enumerate(results):
            print(f"\n--- Result {i + 1} (score={r['score']:.3f}) ---")
            print(f"  {r['accn']}  page {r['page_idx']}  →  {r['png_path']}")


if __name__ == "__main__":
    _cli()
