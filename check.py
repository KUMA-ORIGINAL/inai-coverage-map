"""Самопроверка по правилам автопроверки жюри из ТЗ 7.

Зачёт, если одновременно:
  * пара площадок совпадает с эталоном (порядок не важен);
  * число отброшенных строк отличается от эталона не больше чем на 5%;
  * совпадают хотя бы 2 из 3 худших квадратов.

Проверяются только поля, которые есть в эталоне.

    python check.py out/result.json эталон.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DROPPED_TOLERANCE = 0.05
WORST_MIN_MATCH = 2


def check(result: dict, reference: dict) -> list[tuple[str, bool, str]]:
    """Возвращает список (критерий, пройден, пояснение)."""
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
        got, want = result.get("worst_cells", []), reference["worst_cells"]
        match = len(set(got) & set(want))
        checks.append((f"Худшие квадраты ≥ {WORST_MIN_MATCH} из 3", match >= WORST_MIN_MATCH, f"совпало {match}: {got} / {want}"))
    return checks


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
    passed = all(ok for _, ok, _ in checks)
    print("\nИтог:", "ЗАЧЁТ" if passed else "НЕЗАЧЁТ")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
