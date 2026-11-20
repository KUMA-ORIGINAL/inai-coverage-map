"""Точка входа: python main.py --data data/ --out out/"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from covmap.config import Bounds, Config
from covmap.loader import DataError
from covmap.pipeline import InputPaths, analyze, build_details, build_result, write_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Карта покрытия: слабые зоны и площадки для новых БС")
    p.add_argument("--data", default="data", help="папка с measurements.csv, towers.csv, candidates.csv")
    p.add_argument("--measurements", help="путь к measurements.csv (перекрывает --data)")
    p.add_argument("--towers", help="путь к towers.csv")
    p.add_argument("--candidates", help="путь к candidates.csv")
    p.add_argument("--out", default="out", help="куда сохранить result.json и карту")
    p.add_argument("--bounds", type=Bounds.parse, help="границы района: lat_min,lat_max,lon_min,lon_max")
    p.add_argument("--dedup-on", help="колонки для поиска дублей через запятую (по умолчанию все поля)")
    p.add_argument("--day-hours", help="дневные часы, например 8,20")
    p.add_argument("--no-map", action="store_true", help="не строить HTML-карту")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.bounds:
        cfg = replace(cfg, bounds=args.bounds)
    if args.dedup_on:
        cfg = replace(cfg, dedup_on=tuple(c.strip() for c in args.dedup_on.split(",") if c.strip()))
    if args.day_hours:
        start, end = (int(x) for x in args.day_hours.split(","))
        cfg = replace(cfg, day_hours=(start, end))
    return cfg


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8 — не падаем на кириллице и «−».
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    defaults = InputPaths.from_dir(args.data)
    paths = InputPaths(
        measurements=Path(args.measurements) if args.measurements else defaults.measurements,
        towers=Path(args.towers) if args.towers else defaults.towers,
        candidates=Path(args.candidates) if args.candidates else defaults.candidates,
    )
    try:
        analysis = analyze(paths, build_config(args))
    except DataError as exc:
        print(f"Ошибка входных данных: {exc}", file=sys.stderr)
        return 2

    written = write_outputs(analysis, args.out, make_map=not args.no_map)
    details = build_details(analysis)
    sel, cleaning = details["selection"], details["cleaning"]

    print(json.dumps(build_result(analysis), ensure_ascii=False, indent=2))
    print(
        f"\nОтброшено {cleaning['dropped']} из {cleaning['total_rows']}: нечисловые {cleaning['invalid_values']}, "
        f"дубли {cleaning['duplicates']}, RSRP вне диапазона {cleaning['rsrp_out_of_range']}, "
        f"вне района {cleaning['out_of_bounds']}"
    )
    print(
        f"Слабых зон: {sel['weak_cells_before']} → {sel['weak_cells_after']} (−{sel['weak_cells_reduction_pct']}%). "
        f"Жадный алгоритм: {sel['greedy']['covered_weak_cells']}, перебор: {sel['bruteforce']['covered_weak_cells']}"
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
