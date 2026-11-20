"""Самопроверка по правилам автопроверки жюри из ТЗ 7.

Зачёт, если одновременно:
  * пара площадок совпадает с эталоном (порядок не важен);
  * число отброшенных строк отличается от эталона не больше чем на 5%;
  * совпадают хотя бы 2 из 3 худших квадратов.

Дополнительно (на зачёт не влияет) показывается совпадение covered_weak_cells и медиан.
Проверяются только поля, которые есть в эталоне. worst_cells может быть списком ID
или списком объектов {cell_id, rsrp_median}, как в expected.json организаторов.

    python check.py out/result.json data/expected.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DROPPED_TOLERANCE = 0.05
WORST_MIN_MATCH = 2


def _cell_ids(cells) -> list[str]:
    return [c["cell_id"] if isinstance(c, dict) else str(c) for c in cells or []]


def _medians(cells) -> dict[str, float]:
    return {c["cell_id"]: c.get("rsrp_median") for c in cells or [] if isinstance(c, dict)}


def check(result: dict, reference: dict) -> list[tuple[str, bool, str]]:
    """Критерии зачёта: список (критерий, пройден, пояснение)."""
    checks = []
    if "best_candidates" in reference:
        got, want = result.get("best_candidates", []), reference["best_candidates"]
        checks.append(("Пара площадок", set(got) == set(want), f"получено {got}, эталон {want}"))
    if "dropped_rows" in reference:
        got, want = result.get("dropped_rows"), reference["dropped_rows"]
        ok = isinstance(got, int) and abs(got - want) <= DROPPED_TOLERANCE * max(want, 0)
        diff = f"{(got - want) / want:+.1%}" if isinstance(got, int) and want else "—"
        checks.append(("Отброшенные строки ±5%", ok, f"получено {got}, эталон {want}, отклонение {diff}"))
    if "worst_cells" in reference:
        got, want = _cell_ids(result.get("worst_cells")), _cell_ids(reference["worst_cells"])
        match = len(set(got) & set(want))
        checks.append((f"Худшие квадраты ≥ {WORST_MIN_MATCH} из 3", match >= WORST_MIN_MATCH, f"совпало {match}: {got} / {want}"))
    return checks


def extras(result: dict, reference: dict) -> list[tuple[str, bool, str]]:
    """Дополнительные сверки, не влияющие на зачёт."""
    out = []
    if "covered_weak_cells" in reference:
        got, want = result.get("covered_weak_cells"), reference["covered_weak_cells"]
        out.append(("covered_weak_cells", got == want, f"получено {got}, эталон {want}"))
    want_med = _medians(reference.get("worst_cells"))
    if want_med:
        got_med = _medians(result.get("worst_cells"))
        same = all(cid in got_med and abs(got_med[cid] - m) < 1e-6 for cid, m in want_med.items())
        out.append(("Медианы худших квадратов", same, f"получено {got_med}, эталон {want_med}"))
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Использование: python check.py out/result.json эталон.json", file=sys.stderr)
        return 2
    result, reference = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv)
    checks = check(result, reference)
    if not checks:
        print("В эталоне нет ни одного проверяемого поля (best_candidates, dropped_rows, worst_cells).")
        return 2
    for name, ok, info in checks:
        print(f"{'ЗАЧЁТ  ' if ok else 'НЕЗАЧЁТ'}  {name}: {info}")
    for name, ok, info in extras(result, reference):
        print(f"{'совпало ' if ok else 'отличие '} {name}: {info}")
    passed = all(ok for _, ok, _ in checks)
    print("\nИтог:", "ЗАЧЁТ" if passed else "НЕЗАЧЁТ")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
