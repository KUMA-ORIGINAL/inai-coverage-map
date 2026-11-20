"""Выходные файлы для жюри: dropped_rows.csv, карта без внешних скриптов, данные для тура."""

import csv
import json
import re

import pytest

from covmap.config import Config
from covmap.pipeline import InputPaths, analyze, build_details
from covmap.placement import pair_scores
from covmap.sample_data import generate
from covmap.viz import map_payload
from main import main


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    expected = generate(data, seed=21, total=2500)
    out = tmp_path_factory.mktemp("out")
    assert main([str(data), "--out", str(out)]) == 0
    return data, out, expected


def test_dropped_rows_csv_has_every_row_with_reason(run):
    _, out, expected = run
    with (out / "dropped_rows.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == expected["dropped_rows"]
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        assert r["detail"], r
        assert int(r["source_line"]) >= 2
    for key in ("invalid_values", "duplicates", "rsrp_out_of_range", "out_of_bounds"):
        assert reasons.get(key, 0) == expected[key], key
    lines = [int(r["source_line"]) for r in rows]
    assert lines == sorted(lines) and len(set(lines)) == len(lines)


def test_duplicate_points_to_first_occurrence(run):
    data, out, _ = run
    with (out / "dropped_rows.csv").open(encoding="utf-8-sig", newline="") as fh:
        dup = next(r for r in csv.DictReader(fh) if r["reason"] == "duplicates")
    first_line = int(re.search(r"\d+", dup["detail"]).group())
    source = (data / "measurements.csv").read_text(encoding="utf-8").splitlines()
    assert first_line < int(dup["source_line"])
    assert source[first_line - 1] == source[int(dup["source_line"]) - 1]


def test_map_works_offline_and_tour_page_exists(run):
    _, out, _ = run
    html = (out / "coverage_map.html").read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", html), "скрипты должны быть встроены, а не грузиться из сети"
    assert "Leaflet 1.9.4" in html and "simpleheat" in html
    assert "__COVMAP_DATA__" not in html and "__VENDOR_JS__" not in html
    assert "coverage_map.html#tour" in (out / "tour.html").read_text(encoding="utf-8")


def test_payload_pairs_ranked_and_greedy_steps(run):
    data, _, _ = run
    a = analyze(InputPaths.from_dir(data), Config())
    payload = map_payload(a, build_details(a))
    scores = [p[2] for p in payload["pairs"]]
    assert scores == sorted(scores, reverse=True)
    assert payload["pairs"][0][:2] == payload["best"]["ids"]
    assert len(payload["pairs"]) == payload["best"]["pairs"]
    assert sum(gain for _, gain in payload["greedy"]["steps"]) == payload["greedy"]["covered"]
    json.dumps(payload)  # сериализуется без ошибок


def test_pair_scores_order_matches_bruteforce_tiebreak():
    cov = {"A": frozenset({1, 2}), "B": frozenset({3}), "C": frozenset({3, 4})}
    assert pair_scores(cov)[0] == (("A", "C"), 4)
    assert [n for _, n in pair_scores(cov)] == [4, 3, 2]
