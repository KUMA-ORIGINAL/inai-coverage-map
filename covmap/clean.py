"""Очистка замеров и справочников.

Порядок шагов: нечисловые/пустые → дубли → RSRP вне диапазона → координаты вне района.
Итоговое число отброшенных строк от порядка не зависит (остаются ровно уникальные
валидные строки), порядок влияет только на разбивку по причинам.

Для каждой выброшенной строки сохраняется номер строки в исходном файле и причина —
их можно выгрузить в dropped_rows.csv и сверить с эталоном построчно.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

REJECTED_COLUMNS = ["source_line", "reason", "detail", "lat", "lon", "rsrp_dbm", "timestamp"]


@dataclass
class CleaningReport:
    total_rows: int
    invalid_values: int = 0
    duplicates: int = 0
    rsrp_out_of_range: int = 0
    out_of_bounds: int = 0
    bad_timestamp_kept: int = 0
    # Выброшенные строки: номер строки в файле, код причины, пояснение, исходные значения.
    rejected: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=REJECTED_COLUMNS),
                                   repr=False, compare=False)

    @property
    def dropped(self) -> int:
        return self.invalid_values + self.duplicates + self.rsrp_out_of_range + self.out_of_bounds

    @property
    def kept(self) -> int:
        return self.total_rows - self.dropped

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "invalid_values": self.invalid_values,
            "duplicates": self.duplicates,
            "rsrp_out_of_range": self.rsrp_out_of_range,
            "out_of_bounds": self.out_of_bounds,
            "bad_timestamp_kept": self.bad_timestamp_kept,
            "dropped": self.dropped,
            "kept": self.kept,
        }


def to_number(series: pd.Series) -> pd.Series:
    """Строки → float64. Всё, что не число (или ±inf), становится NaN."""
    values = pd.to_numeric(series.astype(object), errors="coerce").astype("float64")
    return values.where(np.isfinite(values))


def to_timestamp(series: pd.Series) -> pd.Series:
    """Строки → datetime. Нераспознанное — NaT. Часовые пояса отбрасываются (берём местное время)."""

    def parse_one(value) -> pd.Timestamp:
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError, OverflowError):
            return pd.NaT
        if ts is pd.NaT:
            return pd.NaT
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = [parse_one(v) if str(v).strip() else pd.NaT for v in series]
    return pd.Series(pd.to_datetime(parsed), index=series.index)


def _invalid_detail(raw_row: pd.Series, parsed_row: pd.Series) -> str:
    problems = []
    for col in ("lat", "lon", "rsrp_dbm"):
        if pd.isna(parsed_row[col]):
            value = str(raw_row[col])
            problems.append(f"{col} пусто" if not value.strip() else f"{col}='{value}' не число")
    return "; ".join(problems)


def _rejected(raw: pd.DataFrame, lines: pd.Series, index, reason: str, details) -> pd.DataFrame:
    part = raw.loc[index, ["lat", "lon", "rsrp_dbm", "timestamp"]].copy()
    part.insert(0, "detail", list(details))
    part.insert(0, "reason", reason)
    part.insert(0, "source_line", lines.loc[index].to_numpy())
    return part


def clean_measurements(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, CleaningReport]:
    """Возвращает чистые замеры (lat, lon, rsrp_dbm, timestamp, timestamp_raw) и отчёт."""
    report = CleaningReport(total_rows=len(raw))
    if "source_line" in raw:
        lines = raw["source_line"].astype(int)
    else:  # данные пришли не из файла (тесты) — нумеруем как строки CSV после заголовка
        lines = pd.Series(np.arange(2, len(raw) + 2), index=raw.index)

    df = pd.DataFrame(
        {
            "lat": to_number(raw["lat"]),
            "lon": to_number(raw["lon"]),
            "rsrp_dbm": to_number(raw["rsrp_dbm"]),
            "timestamp_raw": raw["timestamp"].astype(str).str.strip(),
        },
        index=raw.index,
    )
    rejected: list[pd.DataFrame] = []

    # 1. Нечисловые и пустые значения.
    valid = df[["lat", "lon", "rsrp_dbm"]].notna().all(axis=1)
    bad = df.index[~valid]
    report.invalid_values = len(bad)
    rejected.append(_rejected(raw, lines, bad, "invalid_values",
                              (_invalid_detail(raw.loc[i], df.loc[i]) for i in bad)))
    df = df[valid]

    # 2. Дубли: оставляем первое вхождение, для повторов запоминаем номер исходной строки.
    subset = ["timestamp_raw" if c == "timestamp" else c for c in cfg.dedup_on]
    is_dup = df.duplicated(subset=subset, keep="first")
    if is_dup.any():
        first = pd.Series(df.index, index=df.index).groupby([df[c] for c in subset], dropna=False).transform("first")
        dup = df.index[is_dup]
        rejected.append(_rejected(raw, lines, dup, "duplicates",
                                  (f"повтор строки {lines[first[i]]}" for i in dup)))
    report.duplicates = int(is_dup.sum())
    df = df[~is_dup]

    # 3. RSRP вне допустимого диапазона (границы включительно).
    in_range = df["rsrp_dbm"].between(cfg.rsrp_min, cfg.rsrp_max, inclusive="both")
    out = df.index[~in_range]
    report.rsrp_out_of_range = len(out)
    rejected.append(_rejected(raw, lines, out, "rsrp_out_of_range",
                              (f"RSRP {df.at[i, 'rsrp_dbm']:g} вне [{cfg.rsrp_min:g}; {cfg.rsrp_max:g}]" for i in out)))
    df = df[in_range]

    # 4. Координаты вне границ района.
    inside = cfg.bounds.contains(df["lat"], df["lon"])
    outside = df.index[~inside]
    report.out_of_bounds = len(outside)
    rejected.append(_rejected(raw, lines, outside, "out_of_bounds",
                              (f"({df.at[i, 'lat']:g}, {df.at[i, 'lon']:g}) вне района" for i in outside)))
    df = df[inside].copy()

    # Битое время не повод выбрасывать замер (в ТЗ такого правила нет), но для
    # сравнения «день/ночь» такие строки не используются.
    df["timestamp"] = to_timestamp(df["timestamp_raw"])
    report.bad_timestamp_kept = int(df["timestamp"].isna().sum())

    non_empty = [part for part in rejected if len(part)]
    if non_empty:
        report.rejected = pd.concat(non_empty).sort_values("source_line").reset_index(drop=True)
    return df.reset_index(drop=True), report


def clean_points(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Башни/кандидаты: убираем строки без id или координат и повторы id. Возвращает (df, отброшено)."""
    df = pd.DataFrame(
        {
            "id": raw["id"].astype(str).str.strip(),
            "lat": to_number(raw["lat"]),
            "lon": to_number(raw["lon"]),
        }
    )
    ok = (df["id"] != "") & df["lat"].notna() & df["lon"].notna()
    cleaned = df[ok].drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return cleaned, len(df) - len(cleaned)
