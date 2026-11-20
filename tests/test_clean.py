from dataclasses import replace

import pandas as pd

from covmap.clean import clean_measurements, clean_points, local_timezone, parse_iso_timestamp
from covmap.config import Config
from covmap.loader import read_measurements

CFG = Config()
IN = ("42.87", "74.60")  # точка внутри Бишкека
TS = "2026-11-01T10:00:00+06:00"
TS2 = "2026-11-01T11:00:00+06:00"


def raw(rows):
    return pd.DataFrame(rows, columns=["lat", "lon", "rsrp_dbm", "timestamp"], dtype=object)


def test_range_bounds_are_inclusive():
    df, rep = clean_measurements(raw([[*IN, "-140", TS], [*IN, "-44", TS2], [*IN, "-140.1", TS], [*IN, "-43.9", TS]]), CFG)
    assert len(df) == 2
    assert rep.rsrp_out_of_range == 2


def test_invalid_values_dropped():
    rows = [[*IN, "abc", TS], ["", "74.6", "-90", TS], [*IN, "NaN", TS], [*IN, "Infinity", TS], [*IN, "-90", TS]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert len(df) == 1
    assert rep.invalid_values == 4


def test_invalid_timestamp_dropped():
    rows = [[*IN, "-90", "bad-time"], [*IN, "-91", ""], [*IN, "-92", "2026-13-01T10:00:00"], [*IN, "-93", TS]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.invalid_timestamp == 3
    assert len(df) == 1


def test_timestamp_normalized_to_local_time():
    tz = local_timezone("Asia/Bishkek")
    local = parse_iso_timestamp("2026-11-16T13:58:30+06:00", tz)
    assert local == parse_iso_timestamp("2026-11-16T07:58:30Z", tz)
    assert local == parse_iso_timestamp("2026-11-16 13:58:30", tz)  # без смещения — уже местное
    assert local.hour == 13


def test_duplicates_compare_normalized_values():
    rows = [
        [*IN, "-90", TS],
        [*IN, "-90", TS],                                   # полный повтор
        ["42.870", "74.6000", "-90.0", TS],                 # те же числа в другой записи
        [*IN, "-90", "2026-11-01T04:00:00Z"],               # тот же момент в UTC
        [*IN, "-90", TS2],                                  # другое время — не дубль
    ]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.duplicates == 3
    assert len(df) == 2


def test_dedup_on_custom_columns():
    rows = [[*IN, "-90", TS], [*IN, "-95", TS]]
    _, rep = clean_measurements(raw(rows), replace(CFG, dedup_on=("lat", "lon", "timestamp")))
    assert rep.duplicates == 1


def test_out_of_bounds_and_swapped_coordinates():
    rows = [["0", "0", "-90", TS], ["74.60", "42.87", "-90", TS], [*IN, "-90", TS]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.out_of_bounds == 2
    assert len(df) == 1


def test_bounds_are_inclusive():
    b = CFG.bounds
    rows = [[str(b.lat_min), str(b.lon_min), "-90", TS], [str(b.lat_max), str(b.lon_max), "-90", TS]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.out_of_bounds == 0 and len(df) == 2


def test_report_totals_add_up():
    rows = [[*IN, "-90", TS], [*IN, "-90", TS], [*IN, "x", TS], [*IN, "-200", TS], ["0", "0", "-90", TS], [*IN, "-91", "?"]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.dropped == 5
    assert rep.kept == len(df) == 1
    assert sorted(rep.rejected["reason"]) == sorted(
        ["duplicates", "invalid_values", "rsrp_out_of_range", "out_of_bounds", "invalid_timestamp"])


def test_loader_counts_blank_lines_and_handles_short_rows_semicolon_bom(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(f"﻿Lat;Lon;RSRP_dBm;timestamp\n42.87;74.6;-90;{TS}\n42.87;74.6\n\n42.88;74.6;-91;{TS};extra\n",
                    encoding="utf-8")
    df = read_measurements(path)
    assert len(df) == 4  # пустая физическая строка тоже строка данных — она будет отброшена
    cleaned, rep = clean_measurements(df, CFG)
    assert rep.invalid_values == 2 and len(cleaned) == 2
    assert "пустая строка" in rep.rejected["detail"].tolist()


def test_clean_points_drops_bad_and_duplicate_ids():
    df = pd.DataFrame({"id": ["T1", "T1", "", "T2", "T3"], "lat": ["42.8", "42.9", "42.8", "x", "42.8"],
                       "lon": ["74.6", "74.6", "74.6", "74.6", "74.6"]}, dtype=object)
    points, dropped = clean_points(df)
    assert points["id"].tolist() == ["T1", "T3"]
    assert dropped == 3
