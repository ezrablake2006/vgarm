from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

from vgarm.mjc import PickPlaceExecutor, available_robots, build_scene_xml
from vgarm.mjc.controller import ControlFailure
from vgarm.nlu import parse_cn
from vgarm.reconstruction import reconstruct_scene
from vgarm.reconstruction.types import SimObject


def _infer_color(rgba: tuple[float, float, float, float]) -> str | None:
    red, green, blue, _ = rgba
    if red > 0.65 and green < 0.55 and blue < 0.55:
        return "red"
    if red > 0.65 and green > 0.65 and blue < 0.55:
        return "yellow"
    if blue > 0.65 and red < 0.6:
        return "blue"
    if green > 0.65 and red < 0.6:
        return "green"
    return None


def _select_object(
    objects: list[SimObject],
    color: str | None,
    category: str | None,
) -> SimObject:
    candidates = objects
    if category is not None:
        candidates = [item for item in candidates if item.category == category]
    if color is not None:
        color_matches = [
            item for item in candidates if _infer_color(item.rgba) == color
        ]
        candidates = color_matches or [
            item for item in candidates if color in item.name.lower()
        ]
    if not candidates:
        raise ValueError("no matching object in scene")
    return candidates[0]


def _execute_intent(executor, intent, layout) -> None:
    selected = None
    if intent.object_color is not None or intent.object_category is not None:
        selected = _select_object(
            layout.objects,
            intent.object_color,
            intent.object_category,
        )
    anchor_xy = (0.55, 0.0)
    if intent.kind == "move_to_dir":
        target_xy = executor.resolve_target_xy(
            intent.target_keyword or "中间",
            anchor_xy=anchor_xy,
        )
        executor.pick_and_place(selected.name, target_xy=target_xy)
    elif intent.kind == "lift":
        executor.lift_object(selected.name)
    elif intent.kind == "move_next_to_object":
        reference = _select_object(layout.objects, intent.ref_color, None)
        executor.move_next_to(
            selected.name,
            reference.name,
            intent.target_keyword or "左边",
        )
    elif intent.kind == "swap":
        first = _select_object(layout.objects, intent.object_color, None)
        second = _select_object(layout.objects, intent.ref_color, None)
        executor.swap_objects(first.name, second.name)
    elif intent.kind == "place_to_center":
        executor.pick_and_place(selected.name, anchor_xy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vgarm")
    parser.add_argument(
        "--robot",
        default="franka_fr3",
        help="franka_fr3 | franka_panda | ur5e | kinova_gen3",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="可选：输入环境照片路径（当前未启用视觉模型）",
    )
    parser.add_argument("--scene", default=None, help="场景 JSON 路径")
    parser.add_argument("--cmd", required=True, help="中文操作指令")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="不启动图形窗口，执行后直接退出",
    )
    args = parser.parse_args(argv)

    robots = available_robots()
    if args.robot not in robots:
        parser.error(
            f"unknown robot: {args.robot}. available: {', '.join(sorted(robots))}"
        )
    robot = robots[args.robot]
    if not robot.include_xml_path.is_file():
        parser.error(
            "robot model is missing: "
            f"{robot.include_xml_path}. Set VGARM_MENAGERIE_ROOT or install "
            "MuJoCo Menagerie assets."
        )

    layout = (
        reconstruct_scene(scene_json_path=str(Path(args.scene)))
        if args.scene is not None
        else reconstruct_scene(image_path=args.image)
    )
    robot_directory = Path(robot.include_xml_path).resolve().parent
    built = build_scene_xml(layout, robot, xml_base_dir=robot_directory)
    xml_path = robot_directory / f"_vgarm_scene_{robot.robot_id}.xml"
    xml_path.write_text(built.xml_text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    intent = parse_cn(args.cmd)

    try:
        if args.no_viewer:
            executor = PickPlaceExecutor(
                model,
                data,
                robot,
                built.object_names,
            )
            _execute_intent(executor, intent, layout)
            executor.step(200)
            return 0

        from mujoco import viewer

        with viewer.launch_passive(model, data) as passive_viewer:
            executor = PickPlaceExecutor(
                model,
                data,
                robot,
                built.object_names,
                sync=passive_viewer.sync,
            )
            _execute_intent(executor, intent, layout)
            while passive_viewer.is_running():
                executor.step()
        return 0
    except ControlFailure as error:
        print(f"VGArm control aborted safely: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
