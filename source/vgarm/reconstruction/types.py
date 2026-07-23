from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GeomType = Literal["box", "sphere", "cylinder"]


@dataclass(frozen=True)
class SimCamera:
    name: str
    pos_xyz: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float = 45.0


@dataclass(frozen=True)
class SimObject:
    name: str
    category: str
    geom_type: GeomType
    pos_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)


@dataclass(frozen=True)
class SceneLayout:
    objects: list[SimObject] = field(default_factory=list)
    cameras: list[SimCamera] = field(default_factory=list)
    floor_plane: bool = True
    floor_rgba: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 1.0)

