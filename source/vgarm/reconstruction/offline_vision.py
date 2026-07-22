from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import SceneLayout, SimObject


_CAT_MAP = {
    "cube": ("cube", "box"),
    "box": ("cube", "box"),
    "bottle": ("bottle", "cylinder"),
    "cup": ("cup", "cylinder"),
    "sphere": ("ball", "sphere"),
}


def _infer_rgba(color: str | None) -> tuple[float, float, float, float]:
    if not color:
        return (0.7, 0.7, 0.7, 1.0)
    m = {
        "red": (0.9, 0.2, 0.2, 1.0),
        "yellow": (0.95, 0.85, 0.1, 1.0),
        "blue": (0.25, 0.45, 0.95, 1.0),
        "green": (0.2, 0.8, 0.3, 1.0),
        "white": (0.9, 0.9, 0.9, 1.0),
        "black": (0.2, 0.2, 0.2, 1.0),
    }
    return m.get(color, (0.7, 0.7, 0.7, 1.0))


def build_scene_from_vision(outputs_dir: str) -> SceneLayout:
    od = Path(outputs_dir)
    det = od / "detections.json"
    with open(det, "r", encoding="utf-8") as f:
        items = json.load(f)
    objs = []
    for i, it in enumerate(items):
        label = str(it.get("label", "object"))
        color = it.get("color")
        bbox = it.get("bbox", [0, 0, 100, 100])
        x, y, w, h = bbox
        cat, gtype = _CAT_MAP.get(label, ("object", "box"))
        cx = float(x + w / 2)
        cy = float(y + h / 2)
        X = 0.4 + 0.002 * cx
        Y = -0.2 + 0.002 * (cy - 240)
        Z = 0.03
        size = (max(0.015, 0.0008 * w), max(0.015, 0.0008 * h), 0.02)
        objs.append(
            SimObject(
                name=f"{label}_{i}",
                category=cat,
                geom_type=gtype,  # type: ignore[arg-type]
                pos_xyz=(X, Y, Z),
                size_xyz=size,
                rgba=_infer_rgba(color),
            )
        )
    return SceneLayout(objects=objs, floor_plane=True)

