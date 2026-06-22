from __future__ import annotations

from typing import Dict, List, Mapping

import math
import numpy as np
import pandas as pd


def module_cost(row) -> int:
    if isinstance(row, dict):
        return int(row["d_in"] + row["d_out"])
    return int(getattr(row, "d_in") + getattr(row, "d_out"))


def floor_to_grid(x: float, grid: List[int]) -> int:
    g = sorted(int(v) for v in grid)
    out = g[0]
    for v in g:
        if v <= x + 1e-12:
            out = v
        else:
            break
    return int(out)


def nearest_to_grid(x: float, grid: List[int]) -> int:
    g = sorted(int(v) for v in grid)
    return int(min(g, key=lambda v: abs(v - x)))


def _next_grid_value(current: int, grid: List[int]) -> int | None:
    g = sorted(int(v) for v in grid)
    for v in g:
        if v > current:
            return int(v)
    return None


def uniform_allocation(module_stats: pd.DataFrame, rank_grid: List[int], budget: int) -> Dict[str, int]:
    """Strict uniform rank baseline.

    This assigns the same rank to every candidate site.  It may leave budget unused
    when rank 1 for all sites is too expensive.  Keep this baseline because it is
    common, but use `uniform_fill_allocation` for a fair budget-filling baseline.
    """
    costs = {row.site_name: module_cost(row) for row in module_stats.itertuples(index=False)}
    sites = list(costs)
    best = {s: 0 for s in sites}
    for r in sorted(rank_grid):
        alloc = {s: int(r) for s in sites}
        cost = sum(costs[s] * alloc[s] for s in sites)
        if cost <= budget:
            best = alloc
    return best


def uniform_fill_allocation(module_stats: pd.DataFrame, rank_grid: List[int], budget: int) -> Dict[str, int]:
    """Budget-filling uniform-ish baseline.

    Increments the currently lowest-rank modules first, breaking ties by cost and
    site name.  This avoids the strict uniform baseline's pathology where small
    budgets may assign rank 0 everywhere.
    """
    rows = list(module_stats.itertuples(index=False))
    costs = {row.site_name: module_cost(row) for row in rows}
    alloc = {row.site_name: 0 for row in rows}
    spent = 0
    while True:
        candidates = []
        for site, cur in alloc.items():
            nr = _next_grid_value(cur, rank_grid)
            if nr is None:
                continue
            extra = (nr - cur) * costs[site]
            if spent + extra <= budget:
                candidates.append((cur, costs[site], site, nr, extra))
        if not candidates:
            break
        # Lowest current rank first, then cheapest, then deterministic site order.
        cur, _cost, site, nr, extra = min(candidates)
        alloc[site] = nr
        spent += extra
    return alloc


def score_allocation(module_stats: pd.DataFrame, rank_grid: List[int], budget: int, score_col: str) -> Dict[str, int]:
    rows = list(module_stats.itertuples(index=False))
    costs = {row.site_name: module_cost(row) for row in rows}
    scores = {row.site_name: max(float(getattr(row, score_col)), 0.0) for row in rows}
    if sum(scores.values()) <= 0:
        return {row.site_name: 0 for row in rows}

    def alloc_for(kappa: float) -> Dict[str, int]:
        return {s: floor_to_grid(kappa * scores[s], rank_grid) for s in scores}

    lo, hi = 0.0, 1.0
    while True:
        a = alloc_for(hi)
        c = sum(costs[s] * a[s] for s in a)
        if c > budget or hi > 1e6:
            break
        hi *= 2.0
    best = {s: 0 for s in scores}
    for _ in range(60):
        mid = (lo + hi) / 2.0
        a = alloc_for(mid)
        c = sum(costs[s] * a[s] for s in a)
        if c <= budget:
            best = a
            lo = mid
        else:
            hi = mid
    # Spend leftover greedily by score/cost if possible.
    spent = sum(costs[s] * best[s] for s in best)
    improved = True
    grid = sorted(int(x) for x in rank_grid)
    while improved:
        improved = False
        candidates = []
        for s in best:
            idx = grid.index(best[s]) if best[s] in grid else 0
            if idx + 1 < len(grid):
                delta_rank = grid[idx + 1] - grid[idx]
                extra = delta_rank * costs[s]
                if spent + extra <= budget:
                    candidates.append((scores[s] / max(costs[s], 1), s, grid[idx + 1], extra))
        if candidates:
            _, s, nr, extra = max(candidates)
            best[s] = nr
            spent += extra
            improved = True
    return best


def gradient_norm_allocation(module_stats: pd.DataFrame, rank_grid: List[int], budget: int) -> Dict[str, int]:
    return score_allocation(module_stats, rank_grid, budget, "gradient_norm")


def _gain_vector(singular_values: np.ndarray, edge: float, mode: str) -> np.ndarray:
    s = np.asarray(singular_values, dtype=float)
    if mode == "raw":
        return 0.5 * s**2
    if mode == "edge":
        return 0.5 * np.maximum(s**2 - float(edge) ** 2, 0.0)
    if mode == "soft":
        denom = s**2 + float(edge) ** 2 + 1e-12
        return 0.5 * s**2 * (s**2 / denom)
    raise ValueError(f"unknown marginal gain mode: {mode}")


def marginal_gain_allocation(
    module_stats: pd.DataFrame,
    singular_values: Mapping[str, np.ndarray],
    rank_grid: List[int],
    budget: int,
    mode: str = "edge",
) -> Dict[str, int]:
    """Greedy allocation by estimated marginal singular-energy gain per parameter.

    The candidate increment from rank r to the next grid value r' receives the
    summed gain of singular directions (r+1,...,r') divided by its parameter cost.
    """
    rows = list(module_stats.itertuples(index=False))
    costs = {row.site_name: module_cost(row) for row in rows}
    edges = {row.site_name: float(getattr(row, "edge")) for row in rows}
    alloc = {row.site_name: 0 for row in rows}
    gains = {}
    for row in rows:
        site = row.site_name
        sv = singular_values.get(site)
        if sv is None:
            # npz files sometimes store dots as double underscores; support that too.
            sv = singular_values.get(site.replace(".", "__"))
        if sv is None:
            sv = np.zeros(max(rank_grid) + 1, dtype=float)
        gains[site] = _gain_vector(np.asarray(sv), edges[site], mode=mode)

    spent = 0
    grid = sorted(int(v) for v in rank_grid)
    while True:
        candidates = []
        for site, cur in alloc.items():
            nr = _next_grid_value(cur, grid)
            if nr is None:
                continue
            extra_rank = nr - cur
            extra_cost = extra_rank * costs[site]
            if spent + extra_cost > budget:
                continue
            g = gains[site]
            upper = min(nr, len(g))
            lower = min(cur, len(g))
            delta_gain = float(np.sum(g[lower:upper]))
            if delta_gain <= 0:
                continue
            candidates.append((delta_gain / max(extra_cost, 1), delta_gain, site, nr, extra_cost))
        if not candidates:
            break
        _ratio, _gain, site, nr, extra = max(candidates)
        alloc[site] = nr
        spent += extra
    return alloc


def allocation_cost(module_stats: pd.DataFrame, alloc: Dict[str, int]) -> int:
    cost = 0
    for row in module_stats.itertuples(index=False):
        cost += module_cost(row) * int(alloc.get(row.site_name, 0))
    return int(cost)


def uniform_exact_cost_allocations(
    module_stats: pd.DataFrame,
    rank_grid: List[int],
    target_costs: List[int],
) -> Dict[int, Dict[str, int]]:
    """Construct deterministic uniform-ish allocations at exact parameter costs.

    Candidate methods can leave different amounts of a budget cap unused. For a
    target cost that is known to be feasible, this dynamic program finds an
    allocation on the same rank grid with exactly that cost while minimizing
    unweighted rank dispersion across modules. The objective is
    ``n * sum(r_i^2) - (sum r_i)^2``, equivalent to pairwise squared rank
    differences. Exact objective ties use lexicographic site/rank order.
    """
    rows = sorted(module_stats.itertuples(index=False), key=lambda row: str(row.site_name))
    sites = [str(row.site_name) for row in rows]
    costs = [module_cost(row) for row in rows]
    grid = sorted(set(int(value) for value in rank_grid))
    if not grid or grid[0] != 0:
        raise ValueError("rank_grid must include rank 0 for exact-cost baselines")
    requested = sorted(set(int(value) for value in target_costs))
    if any(value < 0 for value in requested):
        raise ValueError("target costs must be non-negative")
    if not requested:
        return {}
    max_target = max(requested)

    # (cost, rank_sum) -> (rank_square_sum, rank_tuple)
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    for site_cost in costs:
        next_states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
        for (spent, rank_sum), (rank_sq_sum, ranks) in states.items():
            for rank in grid:
                new_cost = spent + site_cost * rank
                if new_cost > max_target:
                    continue
                key = (new_cost, rank_sum + rank)
                value = (rank_sq_sum + rank * rank, ranks + (rank,))
                previous = next_states.get(key)
                if previous is None or value < previous:
                    next_states[key] = value
        states = next_states

    result: Dict[int, Dict[str, int]] = {}
    n_sites = len(sites)
    for target in requested:
        candidates = []
        for (spent, rank_sum), (rank_sq_sum, ranks) in states.items():
            if spent != target:
                continue
            dispersion = n_sites * rank_sq_sum - rank_sum * rank_sum
            candidates.append((dispersion, ranks))
        if not candidates:
            raise ValueError(f"no exact-cost allocation exists for target cost {target}")
        ranks = min(candidates)[-1]
        result[target] = dict(zip(sites, ranks))
    return result


def uniform_exact_cost_allocation(
    module_stats: pd.DataFrame,
    rank_grid: List[int],
    target_cost: int,
) -> Dict[str, int]:
    return uniform_exact_cost_allocations(module_stats, rank_grid, [int(target_cost)])[int(target_cost)]
