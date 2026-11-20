"""Геометрия: расстояние по формуле гаверсинуса.

Радиус Земли по уточнениям организаторов — 6 371 000 м; его можно переопределить
через bounds.json (поле earth_radius_m).
"""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
# Длина одного градуса дуги большого круга при этом радиусе, метры.
METERS_PER_DEGREE = EARTH_RADIUS_M * math.pi / 180


def haversine_m(lat1, lon1, lat2, lon2, radius: float = EARTH_RADIUS_M):
    """Расстояние между точками в метрах. Принимает числа или numpy-массивы (с broadcasting)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_matrix_m(lat_a, lon_a, lat_b, lon_b, radius: float = EARTH_RADIUS_M) -> np.ndarray:
    """Матрица расстояний: строки — точки A, столбцы — точки B."""
    lat_a = np.asarray(lat_a, dtype=float)[:, None]
    lon_a = np.asarray(lon_a, dtype=float)[:, None]
    lat_b = np.asarray(lat_b, dtype=float)[None, :]
    lon_b = np.asarray(lon_b, dtype=float)[None, :]
    return haversine_m(lat_a, lon_a, lat_b, lon_b, radius)
