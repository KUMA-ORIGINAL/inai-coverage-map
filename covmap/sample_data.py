"""Генератор тестового датасета с заранее известным «мусором».

Нужен, пока нет настоящих данных, и для проверки очистки: в expected.json записано,
сколько строк должно быть отброшено и по каким причинам.

    python -m covmap.sample_data --out data/sample
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .geo import METERS_PER_DEGREE, distance_matrix_m

TOWERS = [
    ("T01", 42.8750, 74.5900),
    ("T02", 42.8700, 74.6200),
    ("T03", 42.8850, 74.5600),
    ("T04", 42.8600, 74.5750),
    ("T05", 42.8900, 74.6150),
    ("T06", 42.8500, 74.6100),
    ("T07", 42.8800, 74.6400),
]

# Область замеров (внутри границ Бишкека по умолчанию).
AREA = (42.832, 42.902, 74.530, 74.665)

# Слабые зоны далеко от БС — туда же ставим часть кандидатов.
WEAK_SPOTS = [(42.838, 74.540), (42.897, 74.536), (42.836, 74.655), (42.842, 74.640)]


def _offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    return (
        lat + north_m / METERS_PER_DEGREE,
        lon + east_m / (METERS_PER_DEGREE * math.cos(math.radians(lat))),
    )


def _signal(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Простая модель затухания от ближайшей БС, дБм."""
    t_lat = [t[1] for t in TOWERS]
    t_lon = [t[2] for t in TOWERS]
    d = distance_matrix_m(lat, lon, t_lat, t_lon).min(axis=1)
    return -50 - 32 * np.log10(1 + d / 60)


def _candidates(rng: np.random.Generator) -> list[tuple[str, float, float]]:
    points = []
    for lat, lon in WEAK_SPOTS:
        for _ in range(2):
            points.append(_offset(lat, lon, rng.normal(0, 500), rng.normal(0, 500)))
    for _, lat, lon in TOWERS[:3]:  # слишком близко к существующим БС — должны отсеяться
        points.append(_offset(lat, lon, 250, 180))
    t_lat, t_lon = [t[1] for t in TOWERS], [t[2] for t in TOWERS]
    while len(points) < 20:  # остальные — случайно, но гарантированно дальше 700 м от БС
        p = (rng.uniform(AREA[0], AREA[1]), rng.uniform(AREA[2], AREA[3]))
        if distance_matrix_m([p[0]], [p[1]], t_lat, t_lon).min() > 700:
            points.append(p)
    order = rng.permutation(len(points))
    return [(f"C{i + 1:02d}", *points[j]) for i, j in enumerate(order)]


def generate(out_dir: str | Path, seed: int = 42, total: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_dup, n_range, n_bounds, n_invalid = 80, 40, 30, 30
    n_edge, n_bad_time = 6, 5
    n_valid = total - n_dup - n_range - n_bounds - n_invalid - n_edge - n_bad_time

    # Замеры кластерами (маршруты/дворы), чтобы в квадратах набиралось >= 5 замеров.
    clusters = 140
    c_lat = rng.uniform(AREA[0], AREA[1], clusters)
    c_lon = rng.uniform(AREA[2], AREA[3], clusters)
    c_shadow = rng.normal(0, 3, clusters)
    which = rng.integers(0, clusters, n_valid)
    lat = c_lat[which] + rng.normal(0, 120, n_valid) / METERS_PER_DEGREE
    lon = c_lon[which] + rng.normal(0, 120, n_valid) / (METERS_PER_DEGREE * 0.733)
    lat, lon = np.clip(lat, AREA[0], AREA[1]), np.clip(lon, AREA[2], AREA[3])

    start = datetime(2026, 11, 1)
    times = [start + timedelta(seconds=int(s)) for s in rng.integers(0, 14 * 86400, n_valid)]
    night = np.array([t.hour < 8 or t.hour >= 20 for t in times])
    rsrp = _signal(lat, lon) + c_shadow[which] + rng.normal(0, 4, n_valid) + np.where(night, 3.0, 0.0)
    rsrp = np.clip(rsrp, -139.9, -44.1)

    fmt = "%Y-%m-%d %H:%M:%S"
    valid_rows = [[f"{a:.6f}", f"{b:.6f}", f"{r:.1f}", t.strftime(fmt)] for a, b, r, t in zip(lat, lon, rsrp, times)]
    ts = start.strftime(fmt)
    mid_lat, mid_lon = (AREA[0] + AREA[1]) / 2, (AREA[2] + AREA[3]) / 2

    def near_center(i: int) -> tuple[str, str]:
        return f"{mid_lat + i * 1e-4:.6f}", f"{mid_lon + i * 1e-4:.6f}"

    # Пограничные значения диапазона — валидны и должны остаться.
    edge_rows = [[*near_center(i), "-140" if i % 2 else "-44", ts] for i in range(n_edge)]
    bad_time_rows = [[*near_center(100 + i), "-95.5", "not-a-date"] for i in range(n_bad_time)]

    garbage = []
    out_of_range = ["-150", "-141", "-43", "-30", "0", "15", "-200", "-140.5"]
    garbage += [[*near_center(200 + i), out_of_range[i % len(out_of_range)], ts] for i in range(n_range)]
    oob = [("0", "0"), ("74.590000", "42.875000"), ("55.751244", "37.618423"), ("43.500000", "74.600000"),
           ("42.875000", "76.900000")]
    # Уникальный RSRP в каждой строке, чтобы они не схлопнулись как дубли друг друга.
    garbage += [[*oob[i % len(oob)], f"-{60 + i}", ts] for i in range(n_bounds)]
    bad_values = [
        ["abc", "74.6", "-90", ts], ["42.87", "", "-90", ts], ["42.87", "74.6", "", ts],
        ["42.87", "74.6", "NaN", ts], ["42.87", "74.6", "n/a", ts], ["42.87", "74.6", "−95", ts],
        ["42,87", "74.6", "-90", ts], ["42.87", "74.6", "inf", ts], ["42.87", "74.6"], ["", "", "", ""],
    ]
    garbage += [bad_values[i % len(bad_values)] for i in range(n_invalid)]

    dup_rows = [list(valid_rows[i]) for i in rng.choice(n_valid, n_dup, replace=False)]

    rows = valid_rows + edge_rows + bad_time_rows + garbage + dup_rows
    rows = [rows[i] for i in rng.permutation(len(rows))]

    with (out_dir / "measurements.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon", "rsrp_dbm", "timestamp"])
        w.writerows(rows)
    with (out_dir / "towers.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "lat", "lon"])
        w.writerows(TOWERS)
    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "lat", "lon"])
        w.writerows([(cid, f"{a:.6f}", f"{b:.6f}") for cid, a, b in _candidates(rng)])

    expected = {
        "total_rows": len(rows),
        "dropped_rows": n_dup + n_range + n_bounds + n_invalid,
        "duplicates": n_dup,
        "rsrp_out_of_range": n_range,
        "out_of_bounds": n_bounds,
        "invalid_values": n_invalid,
        "bad_timestamp_kept": n_bad_time,
    }
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return expected


def main() -> None:
    p = argparse.ArgumentParser(description="Сгенерировать тестовый датасет с мусором")
    p.add_argument("--out", default="data/sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total", type=int, default=5000)
    args = p.parse_args()
    print(json.dumps(generate(args.out, args.seed, args.total), indent=2))


if __name__ == "__main__":
    main()
