"""Устойчивость к мусору: на разных случайных наборах число отброшенных строк должно совпадать точно."""

import json

import pytest

from check import check
from covmap.config import Config
from covmap.pipeline import InputPaths, analyze, build_result
from covmap.sample_data import generate
from main import main


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_cleaning_counts_exact_on_random_garbage(tmp_path, seed):
    expected = generate(tmp_path, seed=seed, total=2000)
    a = analyze(InputPaths.from_dir(tmp_path), Config())
    report = a.report.as_dict()
    for key in ("invalid_values", "duplicates", "rsrp_out_of_range", "out_of_bounds", "invalid_timestamp"):
        assert report[key] == expected[key], key
    assert build_result(a)["dropped_rows"] == expected["dropped_rows"]
    assert len(a.best.candidates) == 2


def test_check_passes_and_fails_like_jury():
    result = {"best_candidates": ["C07", "C12"], "dropped_rows": 214, "worst_cells": ["a", "b", "c"]}
    ok_ref = {"best_candidates": ["C12", "C07"], "dropped_rows": 224, "worst_cells": ["a", "b", "x"]}
    assert all(passed for _, passed, _ in check(result, ok_ref))

    bad_ref = {"best_candidates": ["C07", "C01"], "dropped_rows": 240, "worst_cells": ["a", "y", "z"]}
    assert not any(passed for _, passed, _ in check(result, bad_ref))


def test_check_only_uses_fields_present_in_reference():
    assert [name for name, _, _ in check({"dropped_rows": 10}, {"dropped_rows": 10})] == ["Отброшенные строки ±5%"]


def test_positional_data_dir_and_self_check(tmp_path):
    data = tmp_path / "hidden"
    expected = generate(data, seed=11, total=1500)
    out = tmp_path / "out"
    assert main([str(data), "--out", str(out), "--no-map"]) == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert all(passed for _, passed, _ in check(result, expected))
