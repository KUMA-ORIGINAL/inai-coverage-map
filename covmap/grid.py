"""Сетка квадратов 200×200 м и агрегация RSRP по квадратам.

Проекция — локальная равнопромежуточная: начало отсчёта в юго-западном углу района,
по долготе масштаб cos(широты центра района). На районе размером ~20 км
искажение меньше 0,1%, зато формулу легко проверить руками.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Bounds, Config
from .geo import METERS_PER_DEGREE

GREEN, YELLOW, RED = "green", "yellow", "red"


@dataclass(frozen=True)
class GridSpec:
    origin_lat: float
    origin_lon: float
    ref_lat: float
    cell_size_m: float

    @classmethod
    def from_bounds(cls, bounds: Bounds, cell_size_m: float) -> "GridSpec":
        return cls(bounds.lat_min, bounds.lon_min, bounds.center[0], cell_size_m)

    @property
    def dlat(self) -> float:
        """Размер квадрата по широте, градусы."""
        return self.cell_size_m / METERS_PER_DEGREE

    @property
    def dlon(self) -> float:
        """Размер квадрата по долготе, градусы."""
        return self.cell_size_m / (METERS_PER_DEGREE * math.cos(math.radians(self.ref_lat)))

    def index(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Номер строки (с юга на север) и столбца (с запада на восток) квадрата."""
        row = np.floor((np.asarray(lat, dtype=float) - self.origin_lat) / self.dlat).astype(int)
        col = np.floor((np.asarray(lon, dtype=float) - self.origin_lon) / self.dlon).astype(int)
        return row, col

    def cell_box(self, row, col):
        """(юг, запад, север, восток) квадрата."""
        south = self.origin_lat + row * self.dlat
        west = self.origin_lon + col * self.dlon
        return south, west, south + self.dlat, west + self.dlon


def cell_id(row: int, col: int) -> str:
    return f"r{int(row)}_c{int(col)}"


def classify(median: float, cfg: Config) -> str:
    if median >= cfg.green_min:
        return GREEN
    if median < cfg.red_below:
        return RED
    return YELLOW


def aggregate_cells(measurements: pd.DataFrame, spec: GridSpec, cfg: Config) -> pd.DataFrame:
    """Медиана RSRP по квадратам; квадраты с числом замеров < min_measurements отбрасываются."""
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

    south, west, north, east = spec.cell_box(cells["row"].to_numpy(), cells["col"].to_numpy())
    cells["south"], cells["west"], cells["north"], cells["east"] = south, west, north, east
    cells["center_lat"] = (south + north) / 2
    cells["center_lon"] = (west + east) / 2
    cells["cell_id"] = [cell_id(r, c) for r, c in zip(cells["row"], cells["col"])]
    cells["color"] = [classify(m, cfg) for m in cells["median"]]
    return cells[columns].sort_values(["row", "col"]).reset_index(drop=True)


def sparse_cell_count(measurements: pd.DataFrame, spec: GridSpec, cfg: Config) -> int:
    """Сколько квадратов отброшено, потому что в них меньше min_measurements замеров."""
    if measurements.empty:
        return 0
    row, col = spec.index(measurements["lat"], measurements["lon"])
    counts = pd.DataFrame({"row": row, "col": col}).value_counts()
    return int((counts < cfg.min_measurements).sum())


def worst_cells(cells: pd.DataFrame, n: int) -> pd.DataFrame:
    """N квадратов с минимальной медианой. При равенстве — где больше замеров, затем по id."""
    return cells.sort_values(["median", "count", "cell_id"], ascending=[True, False, True]).head(n)


def split_day_night(measurements: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит замеры с распознанным временем на дневные и ночные."""
    timed = measurements[measurements["timestamp"].notna()]
    hour = timed["timestamp"].dt.hour
    start, end = cfg.day_hours
    is_day = (hour >= start) & (hour < end)
    return timed[is_day], timed[~is_day]
