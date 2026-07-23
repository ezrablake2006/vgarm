from __future__ import annotations

from dataclasses import dataclass, field
import math

from vgarm.reconstruction.types import SceneLayout, SimObject

from .randomization import WORKSPACE_X, WORKSPACE_Y


@dataclass
class PathDiagnostics:
    direct_path_clear: bool | None
    target_region_clear: bool | None
    minimum_clearance: float | None
    required_clearance: float


@dataclass
class TargetPlan:
    original_target: tuple[float, float]
    target: tuple[float, float]
    obstacle_objects: list[str] = field(default_factory=list)
    diagnostics: PathDiagnostics | None = None


class NoSafeTargetError(ValueError):
    pass


def footprint(obj: SimObject) -> tuple[float, float]:
    if obj.geom_type == "sphere":
        return obj.size_xyz[0], obj.size_xyz[0]
    return obj.size_xyz[0], obj.size_xyz[1]


def clearance_at(
    target: tuple[float, float],
    moving: SimObject,
    obstacles: list[SimObject],
) -> tuple[float, list[str]]:
    mx, my = footprint(moving)
    minimum = math.inf
    colliding: list[str] = []
    for obstacle in obstacles:
        ox, oy = footprint(obstacle)
        dx = abs(target[0] - obstacle.pos_xyz[0]) - (mx + ox)
        dy = abs(target[1] - obstacle.pos_xyz[1]) - (my + oy)
        separation = max(dx, dy)
        minimum = min(minimum, separation)
        if dx < 0 and dy < 0:
            colliding.append(obstacle.name)
    return minimum, colliding


def _direct_path_clear(
    start: tuple[float, float],
    target: tuple[float, float],
    moving: SimObject,
    obstacles: list[SimObject],
    clearance: float,
) -> bool:
    # Deterministic sampling at 5 mm intervals is sufficient for the
    # axis-aligned primitive footprints used by SceneLayout.
    distance = math.dist(start, target)
    samples = max(1, int(math.ceil(distance / 0.005)))
    for index in range(samples + 1):
        ratio = index / samples
        point = (
            start[0] + (target[0] - start[0]) * ratio,
            start[1] + (target[1] - start[1]) * ratio,
        )
        measured, colliding = clearance_at(point, moving, obstacles)
        if colliding or measured < clearance:
            return False
    return True


def _semantic_ok(
    target: tuple[float, float],
    keyword: str,
    anchor: tuple[float, float],
    margin: float,
) -> bool:
    return {
        "左边": target[1] - anchor[1] >= margin,
        "右边": anchor[1] - target[1] >= margin,
        "前面": target[0] - anchor[0] >= margin,
        "后面": anchor[0] - target[0] >= margin,
        "中间": math.dist(target, anchor) <= 0.04,
    }.get(keyword, True)


def plan_safe_target(
    layout: SceneLayout,
    moving_name: str,
    original_target: tuple[float, float],
    keyword: str,
    *,
    anchor: tuple[float, float] = (0.55, 0.0),
    clearance: float = 0.005,
    predicate_margin: float = 0.05,
    search_radius: float = 0.12,
    step: float = 0.01,
) -> TargetPlan:
    moving = next(obj for obj in layout.objects if obj.name == moving_name)
    obstacles = [obj for obj in layout.objects if obj.name != moving_name]
    rx, ry = footprint(moving)
    candidates: list[tuple[float, float]] = []
    cells = int(round(search_radius / step))
    for ix in range(-cells, cells + 1):
        for iy in range(-cells, cells + 1):
            candidate = (original_target[0] + ix * step, original_target[1] + iy * step)
            candidates.append(candidate)
    candidates.sort(
        key=lambda point: (
            round(math.dist(point, original_target), 12),
            point[0],
            point[1],
        )
    )
    original_clearance, original_obstacles = clearance_at(
        original_target, moving, obstacles
    )
    for candidate in candidates:
        if not (
            WORKSPACE_X[0] + rx <= candidate[0] <= WORKSPACE_X[1] - rx
            and WORKSPACE_Y[0] + ry <= candidate[1] <= WORKSPACE_Y[1] - ry
        ):
            continue
        if not _semantic_ok(candidate, keyword, anchor, predicate_margin):
            continue
        measured, colliding = clearance_at(candidate, moving, obstacles)
        if not colliding and measured >= clearance:
            return TargetPlan(
                original_target,
                candidate,
                original_obstacles,
                PathDiagnostics(
                    direct_path_clear=_direct_path_clear(
                        (moving.pos_xyz[0], moving.pos_xyz[1]),
                        candidate,
                        moving,
                        obstacles,
                        clearance,
                    ),
                    target_region_clear=not original_obstacles and original_clearance >= clearance,
                    minimum_clearance=measured,
                    required_clearance=clearance,
                ),
            )
    raise NoSafeTargetError(
        f"no safe target for {moving_name} in {keyword} region"
    )
