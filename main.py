"""Точка входа.

    python main.py                      # данные из папки data/
    python main.py путь/к/данным        # любая папка с measurements.csv, towers.csv, candidates.csv
    python main.py путь/к/данным --open # и сразу открыть карту в браузере
    python main.py путь/к/данным --tour # открыть карту в режиме тура по ТЗ (для демо)
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

try:
    from covmap.config import Bounds, Config
    from covmap.loader import DataError
    from covmap.pipeline import InputPaths, analyze, build_details, build_result, write_outputs
except ImportError as exc:  # чаще всего не установлены зависимости
    sys.stderr.write(
        f"Не найден модуль: {exc.name}.\n"
        "Установите зависимости: pip install -r requirements.txt\n"
        "или запустите run.bat (Windows) / ./run.sh (Linux, macOS) — они сделают это сами.\n"
    )
    sys.exit(3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Карта покрытия: чистит замеры, строит сетку 200 м и выбирает 2 площадки для новых БС.",
    )
    p.add_argument("data_dir", nargs="?", help="папка с measurements.csv, towers.csv, candidates.csv (по умолчанию data)")
    p.add_argument("--data", default="data", help="то же, что data_dir")
    p.add_argument("--measurements", help="путь к файлу замеров (перекрывает папку)")
    p.add_argument("--towers", help="путь к файлу существующих БС")
    p.add_argument("--candidates", help="путь к файлу площадок-кандидатов")
    p.add_argument("--out", default="out", help="куда сохранить result.json и карту (по умолчанию out)")
    p.add_argument("--bounds", type=Bounds.parse, help="границы района: lat_min,lat_max,lon_min,lon_max")
    p.add_argument("--dedup-on", help="колонки для поиска дублей через запятую (по умолчанию все поля)")
    p.add_argument("--day-hours", help="дневные часы, например 8,20")
    p.add_argument("--no-map", action="store_true", help="не строить HTML-карту")
    p.add_argument("--open", action="store_true", help="открыть карту в браузере после расчёта")
    p.add_argument("--tour", action="store_true", help="открыть карту сразу в режиме пошагового тура по ТЗ (для демо)")
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


def resolve_paths(args: argparse.Namespace) -> InputPaths:
    defaults = InputPaths.from_dir(args.data_dir or args.data)
    return InputPaths(
        measurements=Path(args.measurements) if args.measurements else defaults.measurements,
        towers=Path(args.towers) if args.towers else defaults.towers,
        candidates=Path(args.candidates) if args.candidates else defaults.candidates,
    )


def print_summary(analysis, written: dict[str, Path]) -> None:
    details = build_details(analysis)
    sel, c = details["selection"], details["cleaning"]
    print(json.dumps(build_result(analysis), ensure_ascii=False, indent=2))
    print()
    print(f"Строк в файле замеров: {c['total_rows']}, отброшено {c['dropped']}:")
    print(f"  не число или пусто   {c['invalid_values']}")
    print(f"  дубли                {c['duplicates']}")
    print(f"  RSRP вне диапазона   {c['rsrp_out_of_range']}")
    print(f"  вне границ района    {c['out_of_bounds']}")
    print(
        f"Квадратов: {details['grid']['cells']}, слабых {sel['weak_cells_before']}. "
        f"Пара {' + '.join(sel['bruteforce']['candidates'])} закрывает {sel['bruteforce']['covered_weak_cells']} "
        f"(−{sel['weak_cells_reduction_pct']}%), жадный алгоритм — {sel['greedy']['covered_weak_cells']}."
    )
    print()
    for name, path in written.items():
        print(f"{name:>8}: {path.resolve()}")


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8 — не падаем на кириллице и «−».
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    paths = resolve_paths(args)
    try:
        analysis = analyze(paths, build_config(args))
    except DataError as exc:
        print(f"Ошибка входных данных: {exc}", file=sys.stderr)
        print("Укажите папку с данными: python main.py путь/к/папке", file=sys.stderr)
        return 2

    written = write_outputs(analysis, args.out, make_map=not args.no_map)
    print_summary(analysis, written)
    page = "tour" if args.tour else "map"
    if (args.open or args.tour) and page in written:
        webbrowser.open(written[page].resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
