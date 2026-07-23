from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from vgarm.reconstruction.types import SceneLayout, SimObject

from .robots import RobotSpec


@dataclass(frozen=True)
class BuiltScene:
    xml_text: str
    object_names: tuple[str, ...]


def _fmt_floats(xs: tuple[float, ...]) -> str:
    return " ".join(f"{x:.6g}" for x in xs)


def _geom_size_attr(obj: SimObject) -> str:
    if obj.geom_type == "box":
        return _fmt_floats(obj.size_xyz)
    if obj.geom_type == "sphere":
        return f"{obj.size_xyz[0]:.6g}"
    if obj.geom_type == "cylinder":
        radius = obj.size_xyz[0]
        half_length = obj.size_xyz[2]
        return _fmt_floats((radius, half_length))
    raise ValueError(f"unsupported geom_type: {obj.geom_type}")


def build_scene_xml(
    scene: SceneLayout,
    robot: RobotSpec,
    model_name: str = "vgarm_scene",
    xml_base_dir: Path | None = None,
) -> BuiltScene:
    include_file = Path(robot.include_xml_path).resolve()
    if xml_base_dir is None:
        include_path = include_file.as_posix()
    else:
        rel_inc = os.path.relpath(str(include_file), str(xml_base_dir.resolve()))
        include_path = Path(rel_inc).as_posix()
    lines: list[str] = []
    lines.append(f'<mujoco model="{model_name}">')
    lines.append(f'  <include file="{include_path}"/>')
    lines.append("  <option timestep=\"0.002\"/>")
    lines.append("  <worldbody>")
    lines.append("    <light pos=\"0 0 1.5\" dir=\"0 0 -1\" directional=\"true\"/>")
    for camera in scene.cameras:
        lines.append(
            f'    <camera name="{camera.name}" '
            f'pos="{_fmt_floats(camera.pos_xyz)}" '
            f'xyaxes="{_fmt_floats(camera.xyaxes)}" '
            f'fovy="{camera.fovy:.6g}"/>'
        )
    if scene.floor_plane:
        lines.append(f'    <geom name="floor" size="0 0 0.05" type="plane" rgba="{_fmt_floats(scene.floor_rgba)}"/>')

    object_names: list[str] = []
    for obj in scene.objects:
        object_names.append(obj.name)
        free_name = f"{obj.name}_free"
        geom_name = f"{obj.name}_geom"
        grab_site = f"{obj.name}_grab"
        lines.append(f'    <body name="{obj.name}" pos="{_fmt_floats(obj.pos_xyz)}">')
        lines.append(f'      <freejoint name="{free_name}"/>')
        lines.append(
            f'      <geom name="{geom_name}" type="{obj.geom_type}" size="{_geom_size_attr(obj)}" '
            f'rgba="{_fmt_floats(obj.rgba)}" friction="{_fmt_floats(obj.friction)}"/>'
        )
        lines.append(f'      <site name="{grab_site}" pos="0 0 {max(obj.size_xyz[2], 1e-3):.6g}" size="0.003" rgba="0 1 0 0.6" group="4"/>')
        lines.append("    </body>")

    lines.append("  </worldbody>")
    lines.append("  <equality>")
    for name in object_names:
        lines.append(
            f'    <connect name="attach_{name}" site1="{robot.attachment_site_name}" site2="{name}_grab" active="false"/>'
        )
    lines.append("  </equality>")
    lines.append("</mujoco>")
    return BuiltScene(xml_text="\n".join(lines) + "\n", object_names=tuple(object_names))
