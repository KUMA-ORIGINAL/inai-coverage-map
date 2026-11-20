"""Параметры расчёта. Все пороги из ТЗ и уточнений организаторов собраны здесь.

Если рядом с данными лежит bounds.json (как в пакете организаторов), границы района,
начало сетки, размер клетки, радиус Земли и часовой пояс берутся из него.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Bounds:
    """Прямоугольные границы района (включительно)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (self.lat_min < self.lat_max and self.lon_min < self.lon_max):
            raise ValueError(f"Некорректные границы: {self}")

    @classmethod
    def parse(cls, text: str) -> "Bounds":
        """Разбирает строку `lat_min,lat_max,lon_min,lon_max`."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("Границы задаются как lat_min,lat_max,lon_min,lon_max")
        return cls(*(float(p) for p in parts))

    def contains(self, lat, lon):
        """Векторная проверка попадания точек в границы (работает с numpy/pandas)."""
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)

    @property
    def center(self) -> tuple[float, float]:
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2


# Бишкек целиком с небольшим запасом — только если bounds.json не передан.
BISHKEK_BOUNDS = Bounds(lat_min=42.78, lat_max=42.95, lon_min=74.45, lon_max=74.75)


@dataclass(frozen=True)
class Config:
    bounds: Bounds = BISHKEK_BOUNDS
    # Начало сетки; None — юго-западный угол границ.
    grid_origin: tuple[float, float] | None = None
    # Радиус Земли из уточнений организаторов (и для сетки, и для гаверсинуса).
    earth_radius_m: float = 6_371_000.0
    # Местное время для дня/ночи; время со смещением приводится к этому поясу.
    timezone: str = "Asia/Bishkek"
    # Допустимый диапазон RSRP, границы включительно.
    rsrp_min: float = -140.0
    rsrp_max: float = -44.0
    # По каким нормализованным полям строки считаются дублями. По уточнениям — все четыре.
    dedup_on: tuple[str, ...] = ("lat", "lon", "rsrp_dbm", "timestamp")
    # Сетка.
    cell_size_m: float = 200.0
    min_measurements: int = 5
    # Цвета: зелёный >= green_min, красный < red_below, жёлтый — между.
    green_min: float = -90.0
    red_below: float = -105.0
    worst_count: int = 3
    # Выбор площадок.
    coverage_radius_m: float = 1000.0
    min_tower_distance_m: float = 500.0
    # День — часы [start, end) по местному времени, остальное — ночь.
    day_hours: tuple[int, int] = (8, 20)

    @property
    def origin(self) -> tuple[float, float]:
        return self.grid_origin or (self.bounds.lat_min, self.bounds.lon_min)

    def with_bounds_file(self, path: str | Path) -> "Config":
        """Применяет bounds.json организаторов: границы, начало сетки, клетка, R, часовой пояс."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = replace(self, bounds=Bounds(data["lat_min"], data["lat_max"], data["lon_min"], data["lon_max"]))
        if "grid_origin_lat" in data and "grid_origin_lon" in data:
            cfg = replace(cfg, grid_origin=(float(data["grid_origin_lat"]), float(data["grid_origin_lon"])))
        if "cell_size_m" in data:
            cfg = replace(cfg, cell_size_m=float(data["cell_size_m"]))
        if "earth_radius_m" in data:
            cfg = replace(cfg, earth_radius_m=float(data["earth_radius_m"]))
        if "timezone" in data:
            cfg = replace(cfg, timezone=str(data["timezone"]))
        return cfg
