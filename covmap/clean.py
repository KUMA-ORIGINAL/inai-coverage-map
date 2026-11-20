"""Очистка замеров и справочников — по уточнениям организаторов (CLARIFICATIONS_RU, п. 7).

Порядок: обязательные поля и конечность чисел → timestamp ISO 8601 → RSRP в −140…−44 →
географические границы (включительно) → дедупликация по четырём нормализованным полям
(числа как числа, время как момент времени). Пустая физическая строка тоже отброшенная.

Итоговое число отброшенных строк от порядка не зависит (остаются ровно уникальные
валидные строки), порядок влияет только на разбивку по причинам.

Для каждой выброшенной строки сохраняется номер строки в исходном файле и причина —
их можно выгрузить в dropped_rows.csv и сверить с эталоном построчно.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo

import numpy as np
import pandas as pd

from .config import Config

REJECTED_COLUMNS = ["source_line", "reason", "detail", "lat", "lon", "rsrp_dbm", "timestamp"]
REASONS = ("invalid_values", "invalid_timestamp", "rsrp_out_of_range", "out_of_bounds", "duplicates")


@dataclass
class CleaningReport:
    total_rows: int
    invalid_values: int = 0
    invalid_timestamp: int = 0
    rsrp_out_of_range: int = 0
    out_of_bounds: int = 0
    duplicates: int = 0
    # Выброшенные строки: номер строки в файле, код причины, пояснение, исходные значения.
    rejected: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=REJECTED_COLUMNS),
                                   repr=False, compare=False)

    @property
    def dropped(self) -> int:
        return sum(getattr(self, r) for r in REASONS)

    @property
    def kept(self) -> int:
        return self.total_rows - self.dropped

    def as_dict(self) -> dict:
        return {"total_rows": self.total_rows, **{r: getattr(self, r) for r in REASONS},
                "dropped": self.dropped, "kept": self.kept}


def to_number(series: pd.Series) -> pd.Series:
    """Строки → float64. Всё, что не число (или ±inf, NaN), становится NaN."""
    values = pd.to_numeric(series.astype(object), errors="coerce").astype("float64")
    return values.where(np.isfinite(values))


def local_timezone(name: str) -> tzinfo:
    """Часовой пояс по имени; если базы поясов нет (Windows без tzdata) — фиксированное смещение."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — любая проблема с базой поясов
        fixed = {"Asia/Bishkek": 6}
        return timezone(timedelta(hours=fixed.get(name, 0)), name)


def parse_iso_timestamp(value: str, tz: tzinfo) -> pd.Timestamp:
    """ISO 8601 → местное время без смещения. Неразбираемое — NaT.

    Время со смещением (+06:00, Z) переводится в местный пояс, поэтому один и тот же момент,
    записанный по-разному, после нормализации совпадает (это важно для дублей).
    """
    text = str(value).strip()
    if not text:
        return pd.NaT
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return pd.NaT
    if dt.tzinfo is not None:
        dt = dt.astimezone(tz).replace(tzinfo=None)
    return pd.Timestamp(dt)


def _invalid_detail(raw_row: pd.Series, parsed_row: pd.Series) -> str:
    if not any(str(raw_row[c]).strip() for c in ("lat", "lon", "rsrp_dbm", "timestamp")):
        return "пустая строка"
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
    """Возвращает чистые замеры (lat, lon, rsrp_dbm, timestamp — местное время, timestamp_raw) и отчёт."""
    report = CleaningReport(total_rows=len(raw))
    if "source_line" in raw:
        lines = raw["source_line"].astype(int)
    else:  # данные пришли не из файла (тесты) — нумеруем как строки CSV после заголовка
        lines = pd.Series(np.arange(2, len(raw) + 2), index=raw.index)

    tz = local_timezone(cfg.timezone)
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

    # 1. Обязательные поля и конечность чисел (пустые, NaN, Infinity, текст).
    valid = df[["lat", "lon", "rsrp_dbm"]].notna().all(axis=1)
    bad = df.index[~valid]
    report.invalid_values = len(bad)
    rejected.append(_rejected(raw, lines, bad, "invalid_values",
                              (_invalid_detail(raw.loc[i], df.loc[i]) for i in bad)))
    df = df[valid].copy()

    # 2. Время в формате ISO 8601.
    df["timestamp"] = pd.to_datetime(pd.Series([parse_iso_timestamp(v, tz) for v in df["timestamp_raw"]],
                                               index=df.index, dtype=object))
    bad_time = df.index[df["timestamp"].isna()]
    report.invalid_timestamp = len(bad_time)
    rejected.append(_rejected(raw, lines, bad_time, "invalid_timestamp",
                              (f"timestamp пусто" if not df.at[i, "timestamp_raw"] else
                               f"timestamp='{df.at[i, 'timestamp_raw']}' не ISO 8601" for i in bad_time)))
    df = df[df["timestamp"].notna()]

    # 3. RSRP вне допустимого диапазона (границы включительно).
    in_range = df["rsrp_dbm"].between(cfg.rsrp_min, cfg.rsrp_max, inclusive="both")
    out = df.index[~in_range]
    report.rsrp_out_of_range = len(out)
    rejected.append(_rejected(raw, lines, out, "rsrp_out_of_range",
                              (f"RSRP {df.at[i, 'rsrp_dbm']:g} вне [{cfg.rsrp_min:g}; {cfg.rsrp_max:g}]" for i in out)))
    df = df[in_range]

    # 4. Координаты вне границ района (границы включительно).
    inside = cfg.bounds.contains(df["lat"], df["lon"])
    outside = df.index[~inside]
    report.out_of_bounds = len(outside)
    rejected.append(_rejected(raw, lines, outside, "out_of_bounds",
                              (f"({df.at[i, 'lat']:g}, {df.at[i, 'lon']:g}) вне района" for i in outside)))
    df = df[inside]

    # 5. Дубли по нормализованным полям: оставляем первое вхождение.
    subset = list(cfg.dedup_on)
    is_dup = df.duplicated(subset=subset, keep="first")
    if is_dup.any():
        first = pd.Series(df.index, index=df.index).groupby([df[c] for c in subset], dropna=False).transform("first")
        dup = df.index[is_dup]
        rejected.append(_rejected(raw, lines, dup, "duplicates",
                                  (f"повтор строки {lines[first[i]]}" for i in dup)))
    report.duplicates = int(is_dup.sum())
    df = df[~is_dup]

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
