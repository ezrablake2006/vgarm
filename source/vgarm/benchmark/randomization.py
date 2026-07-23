from __future__ import annotations

from dataclasses import replace
import random

from vgarm.reconstruction.types import SceneLayout, SimObject


WORKSPACE_X = (0.35, 0.75)
WORKSPACE_Y = (-0.30, 0.30)


def _radius(obj: SimObject) -> tuple[float, float]:
    if obj.geom_type == "sphere":
        return obj.size_xyz[0], obj.size_xyz[0]
    return obj.size_xyz[0], obj.size_xyz[1]


def _overlaps(candidate: SimObject, placed: list[SimObject], clearance: float) -> bool:
    cx, cy, _ = candidate.pos_xyz
    crx, cry = _radius(candidate)
    for other in placed:
        ox, oy, _ = other.pos_xyz
        orx, ory = _radius(other)
        if abs(cx - ox) < crx + orx + clearance and abs(cy - oy) < cry + ory + clearance:
            return True
    return False


def jitter_scene(
    scene: SceneLayout,
    amount: float,
    seed: int,
    *,
    clearance: float = 0.005,
    max_attempts: int = 200,
) -> SceneLayout:
    if amount < 0:
        raise ValueError("position jitter must be non-negative")
    if amount == 0:
        return scene
    rng = random.Random(seed)
    placed: list[SimObject] = []
    for obj in scene.objects:
        rx, ry = _radius(obj)
        for _ in range(max_attempts):
            x = obj.pos_xyz[0] + rng.uniform(-amount, amount)
            y = obj.pos_xyz[1] + rng.uniform(-amount, amount)
            x = min(max(x, WORKSPACE_X[0] + rx), WORKSPACE_X[1] - rx)
            y = min(max(y, WORKSPACE_Y[0] + ry), WORKSPACE_Y[1] - ry)
            candidate = replace(obj, pos_xyz=(x, y, obj.pos_xyz[2]))
            if not _overlaps(candidate, placed, clearance):
                placed.append(candidate)
                break
        else:
            raise ValueError(f"could not place {obj.name} without overlap")
    return replace(scene, objects=placed)
