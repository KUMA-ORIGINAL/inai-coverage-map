"""Очистка замеров и справочников.

Порядок шагов: нечисловые/пустые → дубли → RSRP вне диапазона → координаты вне района.
Итоговое число отброшенных строк от порядка не зависит (остаются ровно уникальные
валидные строки), порядок влияет только на разбивку по причинам.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class CleaningReport:
    total_rows: int
    invalid_values: int = 0
    duplicates: int = 0
    rsrp_out_of_range: int = 0
    out_of_bounds: int = 0
    bad_timestamp_kept: int = 0

    @property
    def dropped(self) -> int:
        return self.invalid_values + self.duplicates + self.rsrp_out_of_range + self.out_of_bounds

    @property
    def kept(self) -> int:
        return self.total_rows - self.dropped

    def as_dict(self) -> dict:
        return {**asdict(self), "dropped": self.dropped, "kept": self.kept}


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


def clean_measurements(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, CleaningReport]:
    """Возвращает чистые замеры (lat, lon, rsrp_dbm, timestamp, timestamp_raw) и отчёт."""
    report = CleaningReport(total_rows=len(raw))
    df = pd.DataFrame(
        {
            "lat": to_number(raw["lat"]),
            "lon": to_number(raw["lon"]),
            "rsrp_dbm": to_number(raw["rsrp_dbm"]),
            "timestamp_raw": raw["timestamp"].astype(str).str.strip(),
        },
        index=raw.index,
    )

    valid = df[["lat", "lon", "rsrp_dbm"]].notna().all(axis=1)
    report.invalid_values = int((~valid).sum())
    df = df[valid]

    subset = ["timestamp_raw" if c == "timestamp" else c for c in cfg.dedup_on]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    report.duplicates = before - len(df)

    in_range = df["rsrp_dbm"].between(cfg.rsrp_min, cfg.rsrp_max, inclusive="both")
    report.rsrp_out_of_range = int((~in_range).sum())
    df = df[in_range]

    inside = cfg.bounds.contains(df["lat"], df["lon"])
    report.out_of_bounds = int((~inside).sum())
    df = df[inside].copy()

    # Битое время не повод выбрасывать замер (в ТЗ такого правила нет), но для
    # сравнения «день/ночь» такие строки не используются.
    df["timestamp"] = to_timestamp(df["timestamp_raw"])
    report.bad_timestamp_kept = int(df["timestamp"].isna().sum())

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
