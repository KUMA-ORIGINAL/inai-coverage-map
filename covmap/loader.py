"""Надёжное чтение CSV с «мусором».

Читаем всё как строки через модуль csv, а не pandas.read_csv: так строка с лишними
или недостающими полями не роняет программу и не теряется молча, а честно
попадает в отброшенные на этапе очистки.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

MEASUREMENT_COLUMNS = ("lat", "lon", "rsrp_dbm", "timestamp")
POINT_COLUMNS = ("id", "lat", "lon")

_ALIASES = {
    "latitude": "lat",
    "lng": "lon",
    "long": "lon",
    "longitude": "lon",
    "rsrp": "rsrp_dbm",
    "time": "timestamp",
    "datetime": "timestamp",
}


class DataError(Exception):
    """Входной файл нельзя прочитать: нет файла, нет заголовка или нужных колонок."""


def _normalize_header(name: str) -> str:
    name = name.replace("﻿", "").strip().lower()
    return _ALIASES.get(name, name)


def _detect_delimiter(header_line: str) -> str:
    counts = {d: header_line.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_table(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Читает CSV и возвращает DataFrame строк с колонками `required` (все значения — str)."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Файл не найден: {path}")

    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        first_line = fh.readline()
        if not first_line.strip():
            raise DataError(f"Пустой файл или нет заголовка: {path}")
        delimiter = _detect_delimiter(first_line)
        fh.seek(0)
        reader = csv.reader(fh, delimiter=delimiter)

        header = [_normalize_header(h) for h in next(reader)]
        missing = [c for c in required if c not in header]
        if missing:
            raise DataError(f"{path.name}: нет колонок {missing}, есть {header}")
        positions = [header.index(c) for c in required]

        rows, lines = [], []
        for raw in reader:
            if not raw or (len(raw) == 1 and not raw[0].strip()):
                raw = []  # пустая физическая строка — по уточнениям она тоже считается отброшенной
            rows.append([raw[i].strip() if i < len(raw) else "" for i in positions])
            lines.append(reader.line_num)  # номер строки в файле — для отчёта об отброшенных

    df = pd.DataFrame(rows, columns=list(required), dtype=object)
    df["source_line"] = pd.Series(lines, dtype="int64")
    return df


def read_measurements(path: str | Path) -> pd.DataFrame:
    return read_table(path, MEASUREMENT_COLUMNS)


def read_points(path: str | Path) -> pd.DataFrame:
    """Башни и площадки-кандидаты: id, lat, lon."""
    return read_table(path, POINT_COLUMNS)
