"""Seed ranking helpers shared by the live retriever and the HPO corpus.

Phantom and clean cosine scores live in different spaces and must not be
compared as numbers. Rank-interleave keeps both lists, dropping duplicates.
"""

from __future__ import annotations

import numpy as np


def ranked_indices(sims: np.ndarray, k: int) -> np.ndarray:
    """Return the top-``k`` indices of ``sims`` in descending score order."""
    if k <= 0 or len(sims) == 0:
        return np.array([], dtype=np.int32)
    k = min(int(k), len(sims))
    top = np.argpartition(sims, -k)[-k:]
    return top[np.argsort(sims[top])[::-1]].astype(np.int32)


def interleave_ranked_indices(*orders: np.ndarray) -> np.ndarray:
    """Merge ranked id lists round-robin, skipping duplicates.

    The first list leads: ``a1, b1, a2, b2, ...``. A duplicate in one list
    is skipped and the next unused id from that same list is taken in the
    same slot, so a shared top-1 does not bury the other space's second-best.
    """
    seen: set[int] = set()
    out: list[int] = []
    pointers = [0] * len(orders)
    while True:
        progressed = False
        for i, order in enumerate(orders):
            while pointers[i] < len(order):
                idx = int(order[pointers[i]])
                pointers[i] += 1
                if idx in seen:
                    continue
                seen.add(idx)
                out.append(idx)
                progressed = True
                break
        if not progressed:
            break
    return np.asarray(out, dtype=np.int32)


def dual_seed_indices(
    primary_sims: np.ndarray,
    secondary_sims: np.ndarray | None,
    k: int,
) -> np.ndarray:
    """Union of top-``k`` from each ranking, interleaved by rank.

    ``primary_sims`` is the query-space used for expansion scores (phantom
    when dual-space is on). ``secondary_sims`` is the clean-sentence ranking
    (Fixed Window's matching). When ``secondary_sims`` is omitted the result
    is ordinary top-``k`` of the primary ranking.
    """
    primary = ranked_indices(primary_sims, k)
    if secondary_sims is None or len(secondary_sims) != len(primary_sims):
        return primary
    secondary = ranked_indices(secondary_sims, k)
    return interleave_ranked_indices(primary, secondary)
