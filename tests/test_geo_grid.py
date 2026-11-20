import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from covmap.config import Bounds, Config
from covmap.geo import METERS_PER_DEGREE, haversine_m
from covmap.grid import GREEN, RED, YELLOW, GridSpec, aggregate_cells, cell_id, classify, split_day_night, worst_cells

CFG = Config()


def test_haversine_zero_and_one_degree():
    assert haversine_m(42.87, 74.6, 42.87, 74.6) == pytest.approx(0)
    assert haversine_m(42.0, 74.6, 43.0, 74.6) == pytest.approx(METERS_PER_DEGREE, rel=1e-9)


def test_haversine_known_distance_bishkek_almaty():
    # Бишкек — Алматы ~ 190 км по прямой.
    assert haversine_m(42.8746, 74.5698, 43.2220, 76.8512) / 1000 == pytest.approx(189, abs=10)


def test_haversine_vectorized():
    d = haversine_m(np.array([42.0, 42.0]), np.array([74.0, 74.0]), np.array([42.0, 43.0]), np.array([74.0, 74.0]))
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0)


@pytest.mark.parametrize(
    "median, color",
    [(-90.0, GREEN), (-60.0, GREEN), (-90.01, YELLOW), (-105.0, YELLOW), (-105.01, RED), (-130, RED)],
)
def test_classify_thresholds(median, color):
    assert classify(median, CFG) == color


def test_cell_is_200m():
    spec = GridSpec.from_config(CFG)
    lat0, lon0 = spec.origin_lat, spec.origin_lon
    assert haversine_m(lat0, lon0, lat0 + spec.dlat, lon0) == pytest.approx(200, rel=1e-6)
    assert haversine_m(lat0, lon0, lat0, lon0 + spec.dlon) == pytest.approx(200, rel=1e-3)


def test_grid_matches_organizers_formula_and_id_format():
    # Уточнения, п. 7: x = R·cos(lat0)·rad(lon−lon0), y = R·rad(lat−lat0), cell = floor(/200).
    cfg = replace(CFG, bounds=Bounds(42.84, 42.876, 74.56, 74.61), grid_origin=(42.84, 74.56))
    spec = GridSpec.from_config(cfg)
    lat, lon = 42.8653, 74.5781
    x = 6_371_000 * math.cos(math.radians(42.84)) * math.radians(lon - 74.56)
    y = 6_371_000 * math.radians(lat - 42.84)
    iy, ix = spec.index(lat, lon)
    assert (int(ix), int(iy)) == (math.floor(x / 200), math.floor(y / 200))
    assert cell_id(iy, ix) == f"X{math.floor(x / 200):03d}_Y{math.floor(y / 200):03d}"
    clat, clon = spec.cell_center(iy, ix)
    cx, cy = spec.to_xy(clat, clon)
    assert (float(cx), float(cy)) == pytest.approx(((int(ix) + 0.5) * 200, (int(iy) + 0.5) * 200))


def _points(lat, lon, values):
    return pd.DataFrame({"lat": [lat] * len(values), "lon": [lon] * len(values), "rsrp_dbm": values})


def test_aggregate_median_and_min_count():
    spec = GridSpec.from_config(replace(CFG, bounds=Bounds(42.0, 43.0, 74.0, 75.0)))
    a = _points(42.5, 74.5, [-100, -110, -120, -80, -90])  # 5 замеров — квадрат учитывается
    b = _points(42.6, 74.6, [-130, -130, -130, -130])  # 4 замера — отбрасывается
    cells = aggregate_cells(pd.concat([a, b]), spec, CFG)
    assert len(cells) == 1
    assert cells.iloc[0]["median"] == -100
    assert cells.iloc[0]["count"] == 5
    assert cells.iloc[0]["color"] == YELLOW


def test_worst_cells_order():
    # Уточнения: сортировка по rsrp_median, затем по cell_id.
    cells = pd.DataFrame(
        {"cell_id": ["X001_Y001", "X003_Y000", "X002_Y005", "X000_Y009"], "median": [-100, -120, -110, -120],
         "count": [5, 6, 9, 9]}
    )
    assert worst_cells(cells, 3)["cell_id"].tolist() == ["X000_Y009", "X003_Y000", "X002_Y005"]


def test_split_day_night():
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-11-01 07:59", "2026-11-01 08:00", "2026-11-01 19:59", "2026-11-01 20:00", None])}
    )
    day, night = split_day_night(df, CFG)
    assert len(day) == 2 and len(night) == 2
