"""Сетка квадратов 200×200 м и агрегация RSRP по квадратам.

Формулы ровно как в уточнениях организаторов — локальная равнопромежуточная модель:

    x = R · cos(lat0) · rad(lon − lon0)        y = R · rad(lat − lat0)
    ix = floor(x / 200)                        iy = floor(y / 200)
    cell_id = X{ix:03d}_Y{iy:03d}

(lat0, lon0) — начало сетки из bounds.json. Центр клетки (ix+0.5, iy+0.5)·200 переводится
обратно в широту/долготу. Это не Web Mercator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

GREEN, YELLOW, RED = "green", "yellow", "red"


@dataclass(frozen=True)
class GridSpec:
    origin_lat: float
    origin_lon: float
    cell_size_m: float
    radius_m: float

    @classmethod
    def from_config(cls, cfg: Config) -> "GridSpec":
        lat0, lon0 = cfg.origin
        return cls(lat0, lon0, cfg.cell_size_m, cfg.earth_radius_m)

    @property
    def _cos0(self) -> float:
        return math.cos(math.radians(self.origin_lat))

    @property
    def dlat(self) -> float:
        """Размер клетки по широте, градусы."""
        return math.degrees(self.cell_size_m / self.radius_m)

    @property
    def dlon(self) -> float:
        """Размер клетки по долготе, градусы."""
        return math.degrees(self.cell_size_m / (self.radius_m * self._cos0))

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Метры от начала сетки: x — на восток, y — на север."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = self.radius_m * self._cos0 * np.radians(lon - self.origin_lon)
        y = self.radius_m * np.radians(lat - self.origin_lat)
        return x, y

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Обратное преобразование метров сетки в широту/долготу."""
        lat = self.origin_lat + np.degrees(np.asarray(y, dtype=float) / self.radius_m)
        lon = self.origin_lon + np.degrees(np.asarray(x, dtype=float) / (self.radius_m * self._cos0))
        return lat, lon

    def index(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """(iy, ix): номер строки (с юга на север) и столбца (с запада на восток)."""
        x, y = self.to_xy(lat, lon)
        return np.floor(y / self.cell_size_m).astype(int), np.floor(x / self.cell_size_m).astype(int)

    def cell_box(self, iy, ix):
        """(юг, запад, север, восток) клетки."""
        s = self.cell_size_m
        south, west = self.to_latlon(np.asarray(ix) * s, np.asarray(iy) * s)
        north, east = self.to_latlon((np.asarray(ix) + 1) * s, (np.asarray(iy) + 1) * s)
        return south, west, north, east

    def cell_center(self, iy, ix):
        s = self.cell_size_m
        return self.to_latlon((np.asarray(ix) + 0.5) * s, (np.asarray(iy) + 0.5) * s)


def cell_id(iy: int, ix: int) -> str:
    return f"X{int(ix):03d}_Y{int(iy):03d}"


def classify(median: float, cfg: Config) -> str:
    if median >= cfg.green_min:
        return GREEN
    if median < cfg.red_below:
        return RED
    return YELLOW


def aggregate_cells(measurements: pd.DataFrame, spec: GridSpec, cfg: Config) -> pd.DataFrame:
    """Медиана RSRP по клеткам; клетки с числом замеров < min_measurements отбрасываются."""
    columns = ["cell_id", "row", "col", "median", "count", "south", "west", "north", "east",
               "center_lat", "center_lon", "color"]
    if measurements.empty:
        return pd.DataFrame(columns=columns)

    row, col = spec.index(measurements["lat"], measurements["lon"])
    grouped = (
        measurements.assign(row=row, col=col)
        .groupby(["row", "col"])["rsrp_dbm"]
        .agg(median="median", count="size")
        .reset_index()
    )
    cells = grouped[grouped["count"] >= cfg.min_measurements].copy()

    iy, ix = cells["row"].to_numpy(), cells["col"].to_numpy()
    cells["south"], cells["west"], cells["north"], cells["east"] = spec.cell_box(iy, ix)
    cells["center_lat"], cells["center_lon"] = spec.cell_center(iy, ix)
    cells["cell_id"] = [cell_id(r, c) for r, c in zip(iy, ix)]
    cells["color"] = [classify(m, cfg) for m in cells["median"]]
    return cells[columns].sort_values(["row", "col"]).reset_index(drop=True)


def sparse_cell_count(measurements: pd.DataFrame, spec: GridSpec, cfg: Config) -> int:
    """Сколько клеток отброшено, потому что в них меньше min_measurements замеров."""
    if measurements.empty:
        return 0
    row, col = spec.index(measurements["lat"], measurements["lon"])
    counts = pd.DataFrame({"row": row, "col": col}).value_counts()
    return int((counts < cfg.min_measurements).sum())


def worst_cells(cells: pd.DataFrame, n: int) -> pd.DataFrame:
    """N клеток с минимальной медианой; при равенстве — по cell_id (как в уточнениях)."""
    return cells.sort_values(["median", "cell_id"], ascending=[True, True]).head(n)


def split_day_night(measurements: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит замеры на дневные и ночные по местному времени."""
    timed = measurements[measurements["timestamp"].notna()]
    hour = timed["timestamp"].dt.hour
    start, end = cfg.day_hours
    is_day = (hour >= start) & (hour < end)
    return timed[is_day], timed[~is_day]
