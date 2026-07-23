from __future__ import annotations

import math
from typing import Mapping

from vgarm.nlu.parser import Intent

from .models import VerificationResult


def _delta(predicate: str, source: tuple[float, float], target: tuple[float, float]) -> float:
    if predicate == "left_of":
        return source[1] - target[1]
    if predicate == "right_of":
        return target[1] - source[1]
    if predicate == "front_of":
        return source[0] - target[0]
    if predicate == "behind":
        return target[0] - source[0]
    raise ValueError(predicate)


def verify_task(
    intent: Intent,
    source_name: str,
    initial: Mapping[str, tuple[float, float]],
    final: Mapping[str, tuple[float, float]],
    *,
    reference_name: str | None = None,
    margin: float = 0.05,
    center_tolerance: float = 0.04,
) -> VerificationResult:
    source = final[source_name]
    direction_map = {
        "左边": "left_of",
        "右边": "right_of",
        "前面": "front_of",
        "后面": "behind",
    }
    if intent.kind == "swap":
        assert reference_name is not None
        first_error = math.dist(final[source_name], initial[reference_name])
        second_error = math.dist(final[reference_name], initial[source_name])
        measured = max(first_error, second_error)
        return VerificationResult(
            "swap", measured <= center_tolerance, measured, center_tolerance,
            {"first_error": first_error, "second_error": second_error},
        )
    if intent.kind == "lift":
        measured = math.dist(source, initial[source_name])
        return VerificationResult("lift", measured <= center_tolerance, measured, center_tolerance)
    if intent.kind == "place_to_center" or (
        intent.kind == "move_to_dir" and intent.target_keyword == "中间"
    ):
        measured = math.dist(source, (0.55, 0.0))
        return VerificationResult("center", measured <= center_tolerance, measured, center_tolerance)
    predicate = direction_map.get(intent.target_keyword or "")
    if predicate is None:
        return VerificationResult("unsupported", False, None, margin)
    target = final[reference_name] if reference_name else (0.55, 0.0)
    measured = _delta(predicate, source, target)
    return VerificationResult(predicate, measured >= margin, measured, margin)
