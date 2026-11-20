import pandas as pd
import pytest

from covmap.config import Config
from covmap.geo import METERS_PER_DEGREE
from covmap.placement import best_pair_bruteforce, coverage_sets, evaluate_candidates, greedy_pair

CFG = Config()
LAT, LON = 42.87, 74.60


def north(meters: float) -> float:
    return LAT + meters / METERS_PER_DEGREE


def points(rows):
    return pd.DataFrame(rows, columns=["id", "lat", "lon"])


def test_candidate_closer_than_500m_rejected():
    towers = points([("T1", LAT, LON)])
    cands = points([("A", north(499), LON), ("B", north(501), LON)])
    result = evaluate_candidates(cands, towers, CFG).set_index("id")
    assert not result.loc["A", "allowed"]
    assert result.loc["B", "allowed"]
    assert result.loc["A", "nearest_tower"] == "T1"


def test_no_towers_means_all_allowed():
    result = evaluate_candidates(points([("A", LAT, LON)]), points([]), CFG)
    assert result["allowed"].all()


def test_coverage_radius_uses_cell_centers():
    red = pd.DataFrame({"cell_id": ["near", "far"], "center_lat": [north(999), north(1001)], "center_lon": [LON, LON]})
    cov = coverage_sets(points([("A", LAT, LON)]), red, 1000)
    assert cov == {"A": frozenset({"near"})}


def test_bruteforce_counts_union_without_double_counting():
    cov = {"A": frozenset({1, 2, 3}), "B": frozenset({2, 3, 4}), "C": frozenset({5})}
    best = best_pair_bruteforce(cov)
    assert best.covered_count == 4  # A+B = {1,2,3,4}; A+C тоже 4
    assert best.candidates == ("A", "B")  # при равенстве — первая пара по порядку
    assert ("A", "C") in best.tied_with
    assert best.pairs_evaluated == 3


def test_greedy_can_be_suboptimal():
    # Классический контрпример: жадный берёт самого «большого» X и проигрывает паре Y+Z.
    cov = {"X": frozenset({2, 3, 4, 5}), "Y": frozenset({1, 2, 3}), "Z": frozenset({4, 5, 6})}
    greedy = greedy_pair(cov)
    best = best_pair_bruteforce(cov)
    assert greedy.candidates[0] == "X"
    assert greedy.covered_count == 5
    assert best.candidates == ("Y", "Z")
    assert best.covered_count == 6


@pytest.mark.parametrize("cov, expected", [({}, ()), ({"A": frozenset({1})}, ("A",))])
def test_fewer_than_two_candidates(cov, expected):
    assert best_pair_bruteforce(cov).candidates == expected
    assert greedy_pair(cov).candidates == expected
