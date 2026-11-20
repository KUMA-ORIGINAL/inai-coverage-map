from dataclasses import replace

import pandas as pd

from covmap.clean import clean_measurements, clean_points
from covmap.config import Config
from covmap.loader import read_measurements

CFG = Config()
IN = ("42.87", "74.60")  # точка внутри Бишкека
TS = "2026-11-01 10:00:00"


def raw(rows):
    return pd.DataFrame(rows, columns=["lat", "lon", "rsrp_dbm", "timestamp"], dtype=object)


def test_range_bounds_are_inclusive():
    df, rep = clean_measurements(raw([[*IN, "-140", TS], [*IN, "-44", "t2"], [*IN, "-140.1", TS], [*IN, "-43.9", TS]]), CFG)
    assert len(df) == 2
    assert rep.rsrp_out_of_range == 2


def test_invalid_values_dropped():
    rows = [[*IN, "abc", TS], ["", "74.6", "-90", TS], [*IN, "NaN", TS], [*IN, "inf", TS], [*IN, "-90", TS]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert len(df) == 1
    assert rep.invalid_values == 4


def test_duplicates_dropped_once():
    rows = [[*IN, "-90", TS], [*IN, "-90", TS], [*IN, "-90.0", TS], [*IN, "-90", "2026-11-01 11:00:00"]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.duplicates == 2  # "-90.0" после разбора — то же число
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


def test_bad_timestamp_is_kept():
    df, rep = clean_measurements(raw([[*IN, "-90", "not-a-date"]]), CFG)
    assert len(df) == 1
    assert rep.bad_timestamp_kept == 1
    assert pd.isna(df.loc[0, "timestamp"])


def test_report_totals_add_up():
    rows = [[*IN, "-90", TS], [*IN, "-90", TS], [*IN, "x", TS], [*IN, "-200", TS], ["0", "0", "-90", TS]]
    df, rep = clean_measurements(raw(rows), CFG)
    assert rep.dropped == 4
    assert rep.kept == len(df) == 1


def test_loader_handles_short_rows_semicolon_and_bom(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("﻿Lat;Lon;RSRP_dBm;timestamp\n42.87;74.6;-90;t\n42.87;74.6\n\n42.88;74.6;-91;t;extra\n",
                    encoding="utf-8")
    df = read_measurements(path)
    assert len(df) == 3
    cleaned, rep = clean_measurements(df, CFG)
    assert rep.invalid_values == 1 and len(cleaned) == 2


def test_clean_points_drops_bad_and_duplicate_ids():
    df = pd.DataFrame({"id": ["T1", "T1", "", "T2", "T3"], "lat": ["42.8", "42.9", "42.8", "x", "42.8"],
                       "lon": ["74.6", "74.6", "74.6", "74.6", "74.6"]}, dtype=object)
    points, dropped = clean_points(df)
    assert points["id"].tolist() == ["T1", "T3"]
    assert dropped == 3
