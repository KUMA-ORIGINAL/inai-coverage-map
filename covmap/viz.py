"""Интерактивная HTML-карта на Leaflet.

Шаблон `templates/coverage_map.html` содержит весь интерфейс; сюда передаются только
данные в виде JSON. Покрытие площадок считает Python (те же числа, что и в result.json),
браузер лишь объединяет готовые множества при выборе своей пары.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TEMPLATE = Path(__file__).with_name("templates") / "coverage_map.html"
PLACEHOLDER = "__COVMAP_DATA__"


def _cells(cells: pd.DataFrame) -> list[list]:
    return [
        [r.cell_id, round(r.south, 6), round(r.west, 6), round(r.north, 6), round(r.east, 6),
         round(float(r.median), 1), int(r.count), r.color]
        for r in cells.itertuples()
    ]


def _candidates(a) -> list[dict]:
    out = []
    for order, r in enumerate(a.candidates.itertuples()):
        out.append(
            {
                "id": r.id,
                "lat": round(float(r.lat), 6),
                "lon": round(float(r.lon), 6),
                "order": order,
                "allowed": bool(r.allowed),
                "nearest": r.nearest_tower,
                "dist": round(float(r.nearest_tower_m)) if pd.notna(r.nearest_tower_m) and r.nearest_tower is not None else None,
                "covers": sorted(a.coverage.get(r.id, ())),
            }
        )
    return out


def _heat(measurements: pd.DataFrame) -> list[list[float]]:
    if measurements.empty:
        return []
    weakness = ((-measurements["rsrp_dbm"] - 80) / 40).clip(0.05, 1.0)
    return [[round(la, 5), round(lo, 5), round(w, 2)]
            for la, lo, w in zip(measurements["lat"], measurements["lon"], weakness)]


def map_payload(a, details: dict) -> dict:
    cfg = a.cfg
    return {
        "th": {"green": cfg.green_min, "red": cfg.red_below, "min": cfg.rsrp_min, "max": cfg.rsrp_max},
        "dayHours": list(cfg.day_hours),
        "radius": cfg.coverage_radius_m,
        "exclusion": cfg.min_tower_distance_m,
        "center": list(cfg.bounds.center),
        "cells": {"all": _cells(a.cells), "day": _cells(a.day_cells), "night": _cells(a.night_cells)},
        "towers": [[r.id, round(float(r.lat), 6), round(float(r.lon), 6)] for r in a.towers.itertuples()],
        "candidates": _candidates(a),
        "worst": a.worst["cell_id"].tolist(),
        "best": {
            "ids": list(a.best.candidates),
            "covered": a.best.covered_count,
            "pairs": a.best.pairs_evaluated,
            "ties": len(a.best.tied_with),
        },
        "greedy": {"ids": list(a.greedy.candidates), "covered": a.greedy.covered_count},
        "cleaning": details["cleaning"],
        "heat": _heat(a.measurements),
    }


def render_map_html(a, details: dict) -> str:
    data = json.dumps(map_payload(a, details), ensure_ascii=False, separators=(",", ":"))
    # Не даём данным закрыть тег <script> раньше времени.
    data = data.replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, data)
