"""Headless end-to-end smoke test for the reliable VGArm controller."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SOURCE_ROOT.parents[1]
TEST_DEPENDENCIES = WORKSPACE_ROOT / ".codex_testdeps"
if TEST_DEPENDENCIES.is_dir():
    sys.path.insert(0, str(TEST_DEPENDENCIES))
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault(
    "VGARM_MENAGERIE_ROOT",
    str(
        WORKSPACE_ROOT
        / ".venv"
        / "Lib"
        / "site-packages"
        / "mujoco_menagerie-main"
        / "mujoco_menagerie-main"
    ),
)

import numpy as np
import mujoco

from vgarm.mjc import PickPlaceExecutor, available_robots, build_scene_xml
from vgarm.reconstruction import reconstruct_scene


def main() -> None:
    scene_path = WORKSPACE_ROOT / "vgarm" / "examples" / "basic_scene.json"
    layout = reconstruct_scene(scene_json_path=str(scene_path))
    robot = available_robots()["franka_fr3"]
    robot_directory = robot.include_xml_path.resolve().parent
    built = build_scene_xml(layout, robot, xml_base_dir=robot_directory)
    xml_path = robot_directory / "_vgarm_reliable_smoke.xml"
    xml_path.write_text(built.xml_text, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        executor = PickPlaceExecutor(model, data, robot, built.object_names)
        red_start = np.asarray(executor.body_xy("cube_red"))
        blue_start = np.asarray(executor.body_xy("cube_blue"))
        executor.swap_objects("cube_red", "cube_blue")
        red_final = np.asarray(executor.body_xy("cube_red"))
        blue_final = np.asarray(executor.body_xy("cube_blue"))
        red_error = float(np.linalg.norm(red_final - blue_start))
        blue_error = float(np.linalg.norm(blue_final - red_start))
        if red_error > 0.04 or blue_error > 0.04:
            raise RuntimeError(
                f"swap verification failed: red={red_error:.4f}, blue={blue_error:.4f}"
            )
        print(
            "reliable_control_smoke_ok",
            f"red_error={red_error:.4f}",
            f"blue_error={blue_error:.4f}",
        )
    finally:
        if xml_path.exists():
            xml_path.unlink()


if __name__ == "__main__":
    main()
