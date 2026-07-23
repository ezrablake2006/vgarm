from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Iterable

from .models import EpisodeResult


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(results: list[EpisodeResult]) -> dict:
    total = len(results)
    pick_entered = [r for r in results if r.pick_attempted]
    place_entered = [r for r in results if r.place_attempted]
    failures = Counter(r.failure_category for r in results if r.failure_category)
    durations = [r.duration_seconds for r in results]
    return {
        "episodes": total,
        "successful_episodes": sum(r.task_success for r in results),
        "overall_success_rate": _ratio(sum(r.task_success for r in results), total),
        "parse_success_rate": _ratio(sum(r.parse_success for r in results), total),
        "pick_success_rate": _ratio(sum(r.pick_success is True for r in pick_entered), len(pick_entered)),
        "pick_denominator": len(pick_entered),
        "place_success_rate": _ratio(sum(r.place_success is True for r in place_entered), len(place_entered)),
        "place_denominator": len(place_entered),
        "collision_rate": _ratio(failures["collision"], total),
        "timeout_rate": _ratio(failures["timeout"], total),
        "average_duration_seconds": mean(durations) if durations else None,
        "median_duration_seconds": median(durations) if durations else None,
        "failures": {
            key: {"count": count, "rate": _ratio(count, total)}
            for key, count in sorted(failures.items())
        },
    }


def aggregate_results(results: Iterable[EpisodeResult]) -> dict:
    items = list(results)
    robots: dict[str, list[EpisodeResult]] = {}
    for result in items:
        robots.setdefault(result.robot, []).append(result)
    return {
        "overall": _summarize(items),
        "by_robot": {name: _summarize(group) for name, group in sorted(robots.items())},
        "by_task": {
            name: _summarize([item for item in items if item.task_id == name])
            for name in sorted({item.task_id for item in items})
        },
        "by_robot_task": {
            robot: {
                task: _summarize([
                    item for item in items
                    if item.robot == robot and item.task_id == task
                ])
                for task in sorted({item.task_id for item in group})
            }
            for robot, group in sorted(robots.items())
        },
    }
