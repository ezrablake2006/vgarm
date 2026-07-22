from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import SceneLayout, SimObject
from .schema import validate_scene_payload


def _parse_object(obj: dict[str, Any]) -> SimObject:
    name = str(obj["name"])
    category = str(obj.get("category", "object"))
    geom_type = str(obj.get("geom_type", "box"))
    pos = tuple(float(x) for x in obj["pos_xyz"])
    size = tuple(float(x) for x in obj["size_xyz"])
    rgba = tuple(float(x) for x in obj.get("rgba", (0.7, 0.7, 0.7, 1.0)))
    friction = tuple(float(x) for x in obj.get("friction", (1.0, 0.005, 0.0001)))
    return SimObject(
        name=name,
        category=category,
        geom_type=geom_type,  # type: ignore[arg-type]
        pos_xyz=pos,  # type: ignore[arg-type]
        size_xyz=size,  # type: ignore[arg-type]
        rgba=rgba,  # type: ignore[arg-type]
        friction=friction,  # type: ignore[arg-type]
    )


def reconstruct_scene(image_path: str | None = None, scene_json_path: str | None = None) -> SceneLayout:
    if scene_json_path is None and image_path is not None:
        p = Path(image_path)
        guess = p.with_suffix(".json")
        if guess.exists():
            scene_json_path = str(guess)

    if scene_json_path is None:
        return SceneLayout(
            objects=[
                SimObject(
                    name="cube_red",
                    category="cube",
                    geom_type="box",
                    pos_xyz=(0.55, -0.10, 0.03),
                    size_xyz=(0.02, 0.02, 0.02),
                    rgba=(0.9, 0.2, 0.2, 1.0),
                ),
                SimObject(
                    name="cube_yellow",
                    category="cube",
                    geom_type="box",
                    pos_xyz=(0.60, 0.10, 0.03),
                    size_xyz=(0.02, 0.02, 0.02),
                    rgba=(0.95, 0.85, 0.1, 1.0),
                ),
                SimObject(
                    name="cube_blue",
                    category="cube",
                    geom_type="box",
                    pos_xyz=(0.45, 0.00, 0.03),
                    size_xyz=(0.02, 0.02, 0.02),
                    rgba=(0.25, 0.45, 0.95, 1.0),
                ),
            ]
        )

    with open(scene_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    validate_scene_payload(payload)

    objects = [_parse_object(o) for o in payload.get("objects", [])]
    floor_plane = bool(payload.get("floor_plane", True))
    floor_rgba = tuple(float(x) for x in payload.get("floor_rgba", (0.2, 0.3, 0.4, 1.0)))
    return SceneLayout(objects=objects, floor_plane=floor_plane, floor_rgba=floor_rgba)  # type: ignore[arg-type]

