"""Выбор двух площадок под новые БС.

Задача — максимальное покрытие (maximum coverage): выбрать 2 множества из ~20 так,
чтобы объединение покрытых красных квадратов было наибольшим. Пар всего C(20,2)=190,
поэтому полный перебор даёт точный ответ мгновенно. Жадный алгоритм оставлен для
сравнения (бонус): он не всегда оптимален.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .config import Config
from .geo import distance_matrix_m

Coverage = dict[str, frozenset[str]]


@dataclass
class PairResult:
    method: str
    candidates: tuple[str, ...]
    covered: frozenset[str]
    pairs_evaluated: int = 0
    tied_with: list[tuple[str, ...]] = field(default_factory=list)
    steps: list[tuple[str, int]] = field(default_factory=list)  # жадный: (площадка, прирост) по шагам

    @property
    def covered_count(self) -> int:
        return len(self.covered)


def evaluate_candidates(candidates: pd.DataFrame, towers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Добавляет ближайшую БС, расстояние до неё и флаг `allowed` (не ближе min_tower_distance_m)."""
    result = candidates.copy()
    if towers.empty or candidates.empty:
        result["nearest_tower"] = None
        result["nearest_tower_m"] = np.inf
    else:
        dist = distance_matrix_m(candidates["lat"], candidates["lon"], towers["lat"], towers["lon"])
        nearest = dist.argmin(axis=1)
        result["nearest_tower"] = towers["id"].to_numpy()[nearest]
        result["nearest_tower_m"] = dist[np.arange(len(candidates)), nearest]
    # «Ближе 500 м не допускается» — значит ровно 500 м уже можно.
    result["allowed"] = result["nearest_tower_m"] >= cfg.min_tower_distance_m
    return result


def coverage_sets(candidates: pd.DataFrame, red_cells: pd.DataFrame, radius_m: float) -> Coverage:
    """Для каждого кандидата — множество красных квадратов, центр которых не дальше radius_m."""
    if candidates.empty:
        return {}
    if red_cells.empty:
        return {cid: frozenset() for cid in candidates["id"]}
    dist = distance_matrix_m(candidates["lat"], candidates["lon"], red_cells["center_lat"], red_cells["center_lon"])
    cell_ids = red_cells["cell_id"].to_numpy()
    return {cid: frozenset(cell_ids[dist[i] <= radius_m]) for i, cid in enumerate(candidates["id"])}


def best_pair_bruteforce(coverage: Coverage, k: int = 2) -> PairResult:
    """Полный перебор всех сочетаний. При равенстве побеждает первое по порядку в файле."""
    ids = list(coverage)
    if len(ids) <= k:
        covered = frozenset().union(*(coverage[i] for i in ids)) if ids else frozenset()
        return PairResult("bruteforce", tuple(ids), covered, pairs_evaluated=1 if ids else 0)

    best: PairResult | None = None
    evaluated = 0
    for combo in combinations(ids, k):
        evaluated += 1
        covered = frozenset().union(*(coverage[i] for i in combo))
        if best is None or len(covered) > best.covered_count:
            best = PairResult("bruteforce", combo, covered)
        elif len(covered) == best.covered_count:
            best.tied_with.append(combo)
    best.pairs_evaluated = evaluated
    return best


def greedy_pair(coverage: Coverage, k: int = 2) -> PairResult:
    """Жадно: на каждом шаге берём кандидата с наибольшим приростом покрытия."""
    chosen: list[str] = []
    steps: list[tuple[str, int]] = []
    covered: frozenset[str] = frozenset()
    evaluated = 0
    for _ in range(min(k, len(coverage))):
        best_id, best_gain = None, -1
        for cid, cells in coverage.items():
            if cid in chosen:
                continue
            evaluated += 1
            gain = len(cells - covered)
            if gain > best_gain:
                best_id, best_gain = cid, gain
        chosen.append(best_id)
        steps.append((best_id, best_gain))
        covered = covered | coverage[best_id]
    return PairResult("greedy", tuple(chosen), covered, pairs_evaluated=evaluated, steps=steps)


def pair_scores(coverage: Coverage, k: int = 2) -> list[tuple[tuple[str, ...], int]]:
    """Все сочетания площадок с числом закрытых квадратов, от лучшего к худшему (для графика)."""
    scores = [(combo, len(frozenset().union(*(coverage[i] for i in combo)))) for combo in combinations(coverage, k)]
    return sorted(scores, key=lambda item: -item[1])  # sorted устойчив: при равенстве — порядок перебора
