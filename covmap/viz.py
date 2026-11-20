"""Интерактивная HTML-карта на Leaflet.

Шаблон `templates/coverage_map.html` содержит весь интерфейс; сюда передаются только
данные в виде JSON. Покрытие площадок и рейтинг пар считает Python (те же числа, что
и в result.json), браузер только рисует и объединяет готовые множества.

Leaflet и плагин тепловой карты встраиваются прямо в HTML из `templates/vendor`,
поэтому карта работает и без интернета (кроме подложки OpenStreetMap).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clean import to_number
from .placement import pair_scores

TEMPLATES = Path(__file__).with_name("templates")
TEMPLATE = TEMPLATES / "coverage_map.html"
VENDOR = TEMPLATES / "vendor"


def _cells(cells: pd.DataFrame) -> list[list]:
    return [
        [r.cell_id, round(r.south, 6), round(r.west, 6), round(r.north, 6), round(r.east, 6),
         round(float(r.median), 1), int(r.count), r.color]
        for r in cells.itertuples()
    ]


def _candidates(a) -> list[dict]:
    out = []
    for order, r in enumerate(a.candidates.itertuples()):
        has_tower = r.nearest_tower is not None and np.isfinite(r.nearest_tower_m)
        out.append(
            {
                "id": r.id,
                "lat": round(float(r.lat), 6),
                "lon": round(float(r.lon), 6),
                "order": order,
                "allowed": bool(r.allowed),
                "nearest": r.nearest_tower if has_tower else None,
                "dist": round(float(r.nearest_tower_m)) if has_tower else None,
                "covers": sorted(a.coverage.get(r.id, ())),
            }
        )
    return out


def _points(measurements: pd.DataFrame) -> list[list[float]]:
    return [[round(la, 5), round(lo, 5), round(v, 1)]
            for la, lo, v in zip(measurements["lat"], measurements["lon"], measurements["rsrp_dbm"])]


def _dropped_points(a) -> list[list]:
    """Выброшенные строки, у которых есть координаты внутри района, — чтобы показать их на карте в туре."""
    rej = a.report.rejected
    if rej.empty:
        return []
    lat, lon = to_number(rej["lat"]), to_number(rej["lon"])
    visible = lat.notna() & lon.notna() & a.cfg.bounds.contains(lat, lon)
    return [[round(la, 5), round(lo, 5), reason]
            for la, lo, reason in zip(lat[visible], lon[visible], rej.loc[visible, "reason"])]


def map_payload(a, details: dict) -> dict:
    cfg = a.cfg
    return {
        "th": {"green": cfg.green_min, "red": cfg.red_below, "min": cfg.rsrp_min, "max": cfg.rsrp_max},
        "dayHours": list(cfg.day_hours),
        "radius": cfg.coverage_radius_m,
        "exclusion": cfg.min_tower_distance_m,
        "minCount": cfg.min_measurements,
        "center": list(cfg.bounds.center),
        "grid": {
            "originLat": a.spec.origin_lat,
            "originLon": a.spec.origin_lon,
            "dlat": a.spec.dlat,
            "dlon": a.spec.dlon,
            "size": cfg.cell_size_m,
            "sparse": a.sparse_cells,
        },
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
        "greedy": {
            "ids": list(a.greedy.candidates),
            "covered": a.greedy.covered_count,
            "steps": [[cid, gain] for cid, gain in a.greedy.steps],
        },
        "pairs": [[*combo, n] for combo, n in pair_scores(a.coverage)],
        "cleaning": details["cleaning"],
        "points": _points(a.measurements),
        "dropped": _dropped_points(a),
        "files": {"dropped": "dropped_rows.csv"},
    }


def _inline(code: str) -> str:
    """Защищаем встраиваемый код от преждевременного закрытия тега."""
    return code.replace("</script", "<\\/script").replace("</style", "<\\/style")


def render_map_html(a, details: dict) -> str:
    data = json.dumps(map_payload(a, details), ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")  # не даём данным закрыть тег <script>
    vendor_css = (VENDOR / "leaflet.css").read_text(encoding="utf-8")
    vendor_js = "\n".join((VENDOR / name).read_text(encoding="utf-8") for name in ("leaflet.js", "leaflet-heat.js"))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__VENDOR_CSS__*/", _inline(vendor_css))
    html = html.replace("/*__VENDOR_JS__*/", _inline(vendor_js))
    return html.replace("__COVMAP_DATA__", data)
