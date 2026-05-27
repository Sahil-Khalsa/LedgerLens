"""
ColPali visual index — page embedding, patch cache, and two-stage retrieval.

Phase 3 implementation. Stubs are here so imports don't break Phase 1/2 code.

Two-stage retrieval (SPEC.md §2):
  Stage 1 — pgvector cosine on mean-pooled page vectors → top-N candidates
  Stage 2 — MaxSim rerank on full patch embeddings → top-k pages for VLM

Query encoding: MUST use ColPali's own encode_query_meanpool().
Never substitute a generic sentence-transformer here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
    return np.load(str(path)).astype(np.float32)


def save_patch_vecs(accn: str, page_idx: int, vecs: np.ndarray) -> None:
    path = patch_cache_path(accn, page_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), vecs.astype(np.float16))


# ── Index (Phase 3) ───────────────────────────────────────────────────────────


def index_filing(accn: str, png_paths: list[str]) -> None:
    """
    Embed all pages for a filing using ColPali.
    Saves patch vectors to disk and mean-pooled vectors to pgvector.

    Phase 3 — not yet implemented.
    """
    raise NotImplementedError("Visual index is implemented in Phase 3")


# ── Retrieval (Phase 3) ───────────────────────────────────────────────────────


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

    Phase 3 — not yet implemented.
    """
    raise NotImplementedError("Visual retrieval is implemented in Phase 3")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the visual (ColPali) index")
    parser.add_argument("--accn", help="Accession number to index")
    parser.add_argument("--query", help="Retrieval query")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.accn:
        print("Visual indexing is available in Phase 3.")
    if args.query:
        print("Visual retrieval is available in Phase 3.")


if __name__ == "__main__":
    _cli()
