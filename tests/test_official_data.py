"""Открытые данные организаторов (data/) должны давать ровно их эталон data/expected.json."""

import json
from pathlib import Path

import pytest

from check import check, extras
from main import main

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "expected.json").is_file(), reason="нет открытых данных организаторов")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    assert main([str(DATA), "--out", str(out), "--no-map"]) == 0
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def test_matches_official_expected_exactly(result):
    expected = json.loads((DATA / "expected.json").read_text(encoding="utf-8"))
    assert result == expected


def test_passes_jury_criteria(result):
    expected = json.loads((DATA / "expected.json").read_text(encoding="utf-8"))
    assert all(ok for _, ok, _ in check(result, expected))
    assert all(ok for _, ok, _ in extras(result, expected))
