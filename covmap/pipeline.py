"""Сквозной расчёт: файлы → очистка → сетка → выбор площадок → result.json, details.json, карта."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .clean import CleaningReport, clean_measurements, clean_points
from .config import Config
from .grid import RED, GridSpec, aggregate_cells, sparse_cell_count, split_day_night, worst_cells
from .loader import read_measurements, read_points
from .placement import PairResult, best_pair_bruteforce, coverage_sets, evaluate_candidates, greedy_pair
from .viz import render_map_html


@dataclass(frozen=True)
class InputPaths:
    measurements: Path
    towers: Path
    candidates: Path
    bounds: Path | None = None  # bounds.json организаторов (границы, сетка, R, часовой пояс)

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "InputPaths":
        data_dir = Path(data_dir)
        bounds = data_dir / "bounds.json"
        return cls(data_dir / "measurements.csv", data_dir / "towers.csv", data_dir / "candidates.csv",
                   bounds if bounds.is_file() else None)


@dataclass
class Analysis:
    cfg: Config
    spec: GridSpec
    report: CleaningReport
    measurements: pd.DataFrame
    towers: pd.DataFrame
    candidates: pd.DataFrame  # с колонками nearest_tower, nearest_tower_m, allowed
    cells: pd.DataFrame
    worst: pd.DataFrame
    red_cells: pd.DataFrame
    coverage: dict[str, frozenset[str]]
    best: PairResult
    greedy: PairResult
    day_cells: pd.DataFrame
    night_cells: pd.DataFrame
    towers_dropped: int
    candidates_dropped: int
    sparse_cells: int = 0  # квадраты, где меньше min_measurements замеров


def analyze(paths: InputPaths, cfg: Config) -> Analysis:
    if paths.bounds is not None:
        cfg = cfg.with_bounds_file(paths.bounds)
    measurements, report = clean_measurements(read_measurements(paths.measurements), cfg)
    towers, towers_dropped = clean_points(read_points(paths.towers))
    candidates, candidates_dropped = clean_points(read_points(paths.candidates))

    spec = GridSpec.from_config(cfg)
    cells = aggregate_cells(measurements, spec, cfg)
    red_cells = cells[cells["color"] == RED]

    candidates = evaluate_candidates(candidates, towers, cfg)
    allowed = candidates[candidates["allowed"]]
    coverage = coverage_sets(allowed, red_cells, cfg.coverage_radius_m, cfg.earth_radius_m)

    day, night = split_day_night(measurements, cfg)

    return Analysis(
        cfg=cfg,
        spec=spec,
        report=report,
        measurements=measurements,
        towers=towers,
        candidates=candidates,
        cells=cells,
        worst=worst_cells(cells, cfg.worst_count),
        red_cells=red_cells,
        coverage=coverage,
        best=best_pair_bruteforce(coverage),
        greedy=greedy_pair(coverage),
        day_cells=aggregate_cells(day, spec, cfg),
        night_cells=aggregate_cells(night, spec, cfg),
        towers_dropped=towers_dropped,
        candidates_dropped=candidates_dropped,
        sparse_cells=sparse_cell_count(measurements, spec, cfg),
    )


def build_result(a: Analysis) -> dict:
    """Ровно формат из ТЗ — его сверяет автопроверка."""
    return {
        "worst_cells": [{"cell_id": r.cell_id, "rsrp_median": round(float(r.median), 6)} for r in a.worst.itertuples()],
        "best_candidates": list(a.best.candidates),
        "dropped_rows": a.report.dropped,
        "covered_weak_cells": a.best.covered_count,
    }


def _color_counts(cells: pd.DataFrame) -> dict[str, int]:
    counts = cells["color"].value_counts()
    return {c: int(counts.get(c, 0)) for c in ("green", "yellow", "red")}


def _pair_dict(p: PairResult) -> dict:
    return {
        "candidates": list(p.candidates),
        "covered_weak_cells": p.covered_count,
        "evaluated": p.pairs_evaluated,
    }


def day_night_changes(a: Analysis) -> int:
    """Сколько квадратов (есть и днём, и ночью) меняют цвет между днём и ночью."""
    merged = a.day_cells[["cell_id", "color"]].merge(a.night_cells[["cell_id", "color"]], on="cell_id",
                                                      suffixes=("_day", "_night"))
    return int((merged["color_day"] != merged["color_night"]).sum())


def build_details(a: Analysis) -> dict:
    """Всё, что нужно для демо и разбора: причины отбраковки, сравнение алгоритмов, день/ночь."""
    red_total = len(a.red_cells)
    covered = a.best.covered_count
    return {
        "cleaning": a.report.as_dict(),
        "towers": {"count": len(a.towers), "dropped": a.towers_dropped},
        "candidates": {
            "count": len(a.candidates),
            "dropped": a.candidates_dropped,
            "rejected_near_tower": [
                {"id": r.id, "nearest_tower": r.nearest_tower, "distance_m": round(float(r.nearest_tower_m), 1)}
                for r in a.candidates[~a.candidates["allowed"]].itertuples()
            ],
            "coverage": {cid: len(cells) for cid, cells in a.coverage.items()},
        },
        "grid": {
            "cell_size_m": a.cfg.cell_size_m,
            "origin": {"lat": a.spec.origin_lat, "lon": a.spec.origin_lon},
            "cells": len(a.cells),
            "cells_too_sparse": a.sparse_cells,
            "colors": _color_counts(a.cells),
        },
        "worst_cells": [
            {
                "cell_id": r.cell_id,
                "median_rsrp": float(r.median),
                "count": int(r.count),
                "center": [round(float(r.center_lat), 6), round(float(r.center_lon), 6)],
            }
            for r in a.worst.itertuples()
        ],
        "selection": {
            "bruteforce": {**_pair_dict(a.best), "tied_pairs": [list(t) for t in a.best.tied_with]},
            "greedy": _pair_dict(a.greedy),
            "greedy_is_optimal": a.greedy.covered_count == a.best.covered_count,
            "weak_cells_before": red_total,
            "weak_cells_after": red_total - covered,
            "weak_cells_reduction_pct": round(100 * covered / red_total, 1) if red_total else 0.0,
        },
        "day_night": {
            "day_hours": list(a.cfg.day_hours),
            "day": _color_counts(a.day_cells),
            "night": _color_counts(a.night_cells),
            "cells_changed_class": day_night_changes(a),
        },
    }


def write_outputs(a: Analysis, out_dir: str | Path, make_map: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "result": out_dir / "result.json",
        "details": out_dir / "details.json",
        "dropped": out_dir / "dropped_rows.csv",
    }
    details = build_details(a)
    written["result"].write_text(json.dumps(build_result(a), ensure_ascii=False, indent=2), encoding="utf-8")
    written["details"].write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    # utf-8-sig — чтобы Excel сразу открыл кириллицу в причинах.
    a.report.rejected.to_csv(written["dropped"], index=False, encoding="utf-8-sig")
    if make_map:
        written["map"] = out_dir / "coverage_map.html"
        written["map"].write_text(render_map_html(a, details), encoding="utf-8")
        # Отдельная точка входа сразу в режим тура: браузеры теряют #якорь у file:// при открытии из ОС.
        written["tour"] = out_dir / "tour.html"
        written["tour"].write_text(
            '<!doctype html><meta charset="utf-8"><title>Тур по расчёту</title>'
            '<script>location.replace("coverage_map.html#tour")</script>'
            '<a href="coverage_map.html#tour">Открыть тур</a>',
            encoding="utf-8",
        )
    return written
