from __future__ import annotations

from typing import Any


def validate_scene_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("scene must be object")
    if "objects" not in payload:
        raise KeyError("objects missing")
    objs = payload["objects"]
    if not isinstance(objs, list):
        raise TypeError("objects must be array")
    for o in objs:
        if not isinstance(o, dict):
            raise TypeError("object must be object")
        for k in ("name", "geom_type", "pos_xyz", "size_xyz"):
            if k not in o:
                raise KeyError(f"{k} missing")
        if o["geom_type"] not in ("box", "sphere", "cylinder"):
            raise ValueError("geom_type invalid")
        pos = o["pos_xyz"]
        size = o["size_xyz"]
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            raise TypeError("pos_xyz must be length-3 array")
        if not (isinstance(size, (list, tuple)) and len(size) == 3):
            raise TypeError("size_xyz must be length-3 array")
    if "floor_plane" in payload and not isinstance(payload["floor_plane"], bool):
        raise TypeError("floor_plane must be bool")
    cameras = payload.get("cameras", [])
    if not isinstance(cameras, list):
        raise TypeError("cameras must be array")
    names = set()
    for camera in cameras:
        if not isinstance(camera, dict):
            raise TypeError("camera must be object")
        for key, length in (("pos_xyz", 3), ("xyaxes", 6)):
            value = camera.get(key)
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise TypeError(f"camera {key} must be length-{length} array")
        name = camera.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("camera name must be non-empty string")
        if name in names:
            raise ValueError(f"duplicate camera name: {name}")
        names.add(name)
