"""Сквозная проверка на сгенерированном датасете с известным количеством мусора."""

import json

import pytest

from covmap.sample_data import generate
from main import main


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    expected = generate(data, seed=7, total=3000)
    return data, expected


def test_end_to_end_result_format(sample, tmp_path):
    data, expected = sample
    assert main(["--data", str(data), "--out", str(tmp_path)]) == 0

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert set(result) == {"worst_cells", "best_candidates", "dropped_rows", "covered_weak_cells"}
    assert result["dropped_rows"] == expected["dropped_rows"]
    assert len(result["worst_cells"]) == 3
    assert len(result["best_candidates"]) == 2
    assert result["covered_weak_cells"] > 0
    assert (tmp_path / "coverage_map.html").stat().st_size > 10_000

    details = json.loads((tmp_path / "details.json").read_text(encoding="utf-8"))
    for key in ("duplicates", "rsrp_out_of_range", "out_of_bounds", "invalid_values", "invalid_timestamp"):
        assert details["cleaning"][key] == expected[key], key
    rejected = {r["id"] for r in details["candidates"]["rejected_near_tower"]}
    assert not rejected & set(result["best_candidates"])
    assert len(rejected) == 3


def test_missing_file_returns_error_code(tmp_path):
    assert main(["--data", str(tmp_path), "--out", str(tmp_path / "out"), "--no-map"]) == 2
