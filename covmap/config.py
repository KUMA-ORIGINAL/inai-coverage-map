"""Параметры расчёта. Все пороги из ТЗ собраны здесь, чтобы их можно было менять из CLI."""

from __future__ import annotations

from dataclasses import dataclass


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


# Бишкек целиком с небольшим запасом. Если организаторы дадут точные границы района,
# передайте их через --bounds.
BISHKEK_BOUNDS = Bounds(lat_min=42.78, lat_max=42.95, lon_min=74.45, lon_max=74.75)


@dataclass(frozen=True)
class Config:
    bounds: Bounds = BISHKEK_BOUNDS
    # Допустимый диапазон RSRP, границы включительно.
    rsrp_min: float = -140.0
    rsrp_max: float = -44.0
    # По каким колонкам строки считаются дублями. По умолчанию — все поля замера.
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
    # День — часы [start, end), остальное — ночь.
    day_hours: tuple[int, int] = (8, 20)
