from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

from vgarm.mjc import (
    PickPlaceExecutor,
    SimulationSession,
    ViewerClosed,
    available_robots,
    build_scene_xml,
)
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


def _execute_intent(executor, intent, layout, planned_target_xy=None):
    from dataclasses import replace
    from vgarm.benchmark.planning import plan_safe_target

    selected = None
    if intent.object_color is not None or intent.object_category is not None:
        selected = _select_object(
            layout.objects,
            intent.object_color,
            intent.object_category,
        )
    anchor_xy = (0.55, 0.0)
    if intent.kind == "move_to_dir":
        original_target = executor.resolve_target_xy(
            intent.target_keyword or "中间",
            anchor_xy=anchor_xy,
        )
        plan = plan_safe_target(
            layout, selected.name, original_target,
            intent.target_keyword or "中间", anchor=anchor_xy,
        )
        if planned_target_xy is not None:
            plan = replace(plan, target=tuple(planned_target_xy[:2]))
        executor.pick_and_place(selected.name, target_xy=plan.target)
        return plan
    elif intent.kind == "lift":
        executor.lift_object(selected.name)
    elif intent.kind == "move_next_to_object":
        reference = _select_object(layout.objects, intent.ref_color, None)
        reference_xy = executor.body_xy(reference.name)
        original_target = executor.resolve_target_xy(
            intent.target_keyword or "左边", anchor_xy=reference_xy
        )
        plan = plan_safe_target(
            layout, selected.name, original_target,
            intent.target_keyword or "左边", anchor=reference_xy,
        )
        if planned_target_xy is not None:
            plan = replace(plan, target=tuple(planned_target_xy[:2]))
        executor.pick_and_place(selected.name, plan.target)
        return plan
    elif intent.kind == "swap":
        first = _select_object(layout.objects, intent.object_color, None)
        second = _select_object(layout.objects, intent.ref_color, None)
        executor.swap_objects(first.name, second.name)
    elif intent.kind == "place_to_center":
        executor.pick_and_place(selected.name, anchor_xy)
    return None


def _benchmark_main(argv: list[str]) -> int:
    if argv and argv[0] == "compare":
        from vgarm.benchmark.compare import compare_results

        compare_parser = argparse.ArgumentParser(prog="vgarm benchmark compare")
        compare_parser.add_argument("first", type=Path)
        compare_parser.add_argument("second", type=Path)
        compare_args = compare_parser.parse_args(argv[1:])
        try:
            equal, first_hash, second_hash = compare_results(
                compare_args.first, compare_args.second
            )
        except OSError as error:
            compare_parser.error(str(error))
        print(f"first:  {first_hash}")
        print(f"second: {second_hash}")
        print("Deterministic results match" if equal else "Deterministic results differ")
        return 0 if equal else 1
    if argv and argv[0] == "replay":
        from vgarm.benchmark.replay import ReplayDataError, replay_episode

        replay_parser = argparse.ArgumentParser(prog="vgarm benchmark replay")
        replay_parser.add_argument("episodes", type=Path)
        replay_parser.add_argument("--episode-id", type=int, required=True)
        viewer_group = replay_parser.add_mutually_exclusive_group()
        viewer_group.add_argument("--viewer", dest="no_viewer", action="store_false")
        viewer_group.add_argument("--no-viewer", dest="no_viewer", action="store_true")
        replay_parser.set_defaults(no_viewer=True)
        replay_parser.add_argument("--speed", type=float, default=1.0)
        replay_args = replay_parser.parse_args(argv[1:])
        if replay_args.speed <= 0:
            replay_parser.error("--speed must be greater than zero")
        try:
            original, replay, match = replay_episode(
                replay_args.episodes,
                replay_args.episode_id,
                no_viewer=replay_args.no_viewer,
                speed=replay_args.speed,
            )
        except (OSError, ReplayDataError, ValueError) as error:
            replay_parser.error(str(error))
        print(f"Original task success: {str(original['task_success']).lower()}")
        print(f"Replay task success: {str(replay['task_success']).lower()}")
        print("Original verification: " + json.dumps(
            original["verification"], ensure_ascii=False, sort_keys=True
        ))
        print("Replay verification: " + json.dumps(
            replay["verification"], ensure_ascii=False, sort_keys=True
        ))
        print(f"Deterministic match: {str(match).lower()}")
        return 0 if match else 1
    from vgarm.benchmark.models import BenchmarkConfig
    from vgarm.benchmark.runner import BenchmarkRunner, load_tasks

    parser = argparse.ArgumentParser(prog="vgarm benchmark")
    parser.add_argument("--scene", required=True, help="场景 JSON 路径")
    parser.add_argument("--robots", default="franka_fr3", help="逗号分隔的机器人名称")
    parser.add_argument("--episodes", type=int, default=3, help="每个机器人的 episode 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=Path, default=None, help="任务 JSON 文件")
    parser.add_argument("--position-jitter", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--viewer", dest="no_viewer", action="store_false",
        help="显示每个 episode 的真实 MuJoCo 仿真",
    )
    viewer_group.add_argument(
        "--no-viewer", dest="no_viewer", action="store_true",
        help="无窗口运行（默认）",
    )
    parser.set_defaults(no_viewer=True)
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="viewer 播放速度；只改变墙钟节奏，不改变仿真步数",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.position_jitter < 0:
        parser.error("--position-jitter must be non-negative")
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    if not args.no_viewer and args.episodes > 10:
        print(
            "warning: viewer mode with more than 10 episodes may run slowly",
            file=sys.stderr,
        )
    scene = Path(args.scene)
    if not scene.is_file():
        parser.error(f"scene file does not exist: {scene}")
    try:
        tasks = load_tasks(args.tasks)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid tasks file: {error}")
    robot_names = tuple(name.strip() for name in args.robots.split(",") if name.strip())
    known = available_robots()
    unknown = sorted(set(robot_names) - set(known))
    if not robot_names:
        parser.error("--robots must contain at least one robot")
    if unknown:
        parser.error(f"unknown robot(s): {', '.join(unknown)}")
    missing = [name for name in robot_names if not known[name].include_xml_path.is_file()]
    if missing:
        parser.error(
            "robot model assets are missing for: "
            f"{', '.join(missing)}. Set VGARM_MENAGERIE_ROOT."
        )
    config = BenchmarkConfig(
        scene=scene.resolve(),
        robots=robot_names,
        episodes=args.episodes,
        seed=args.seed,
        output=args.output.resolve(),
        tasks=tasks,
        position_jitter=args.position_jitter,
        no_viewer=args.no_viewer,
        viewer_speed=args.speed,
        quiet=args.quiet,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
    )
    try:
        summary = BenchmarkRunner(config).run()
    except FileExistsError as error:
        parser.error(f"{error}; pass --overwrite to replace result files")
    if not args.quiet:
        rate = summary["overall"]["overall_success_rate"]
        print("\nBenchmark completed")
        print(f"Results: {config.output}")
        print(f"Overall success rate: {rate:.1%}" if rate is not None else "Overall success rate: N/A")
    overall = summary["overall"]
    setup_failures = sum(
        item["count"]
        for category, item in overall["failures"].items()
        if category in {"scene_load_failed", "unexpected_error"}
    )
    if (
        overall["episodes"] > 0
        and setup_failures == overall["episodes"]
        and overall["pick_denominator"] == 0
    ):
        if not args.quiet:
            print("Benchmark initialization failed for every episode", file=sys.stderr)
        return 3
    return 0


def _dataset_main(argv: list[str]) -> int:
    if not argv:
        argv = ["--help"]
    command = argv[0]
    if command == "generate":
        from vgarm.trajectory.dataset import generate_dataset
        from vgarm.trajectory.models import DatasetConfig
        from vgarm.trajectory.util import atomic_json

        parser = argparse.ArgumentParser(prog="vgarm dataset generate")
        parser.add_argument("--scene", type=Path, required=True)
        parser.add_argument("--tasks", type=Path, required=True)
        parser.add_argument("--robots", default="franka_fr3")
        parser.add_argument("--episodes", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--position-jitter", type=float, default=0.0)
        parser.add_argument("--modalities", default="state")
        viewer = parser.add_mutually_exclusive_group()
        viewer.add_argument("--viewer", dest="no_viewer", action="store_false")
        viewer.add_argument("--no-viewer", dest="no_viewer", action="store_true")
        parser.set_defaults(no_viewer=True)
        parser.add_argument("--output", type=Path, required=True)
        output_mode = parser.add_mutually_exclusive_group()
        output_mode.add_argument("--overwrite", action="store_true")
        output_mode.add_argument("--resume", action="store_true")
        parser.add_argument("--fail-fast", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args(argv[1:])
        if args.episodes <= 0:
            parser.error("--episodes must be greater than zero")
        if args.position_jitter < 0:
            parser.error("--position-jitter must be non-negative")
        modalities = tuple(
            item.strip() for item in args.modalities.split(",") if item.strip()
        )
        unsupported = set(modalities) - {"state"}
        if unsupported:
            parser.error(
                "this build currently supports state recording only; unsupported: "
                + ", ".join(sorted(unsupported))
            )
        robots = tuple(
            item.strip() for item in args.robots.split(",") if item.strip()
        )
        known = available_robots()
        unknown = set(robots) - set(known)
        if unknown:
            parser.error("unknown robot(s): " + ", ".join(sorted(unknown)))
        root = args.output.resolve()
        results = {}
        for robot in robots:
            child = root / robot if len(robots) > 1 else root
            config = DatasetConfig(
                root=child,
                scene=args.scene.resolve(),
                robot=robot,
                tasks_file=args.tasks.resolve(),
                episodes=args.episodes,
                seed=args.seed,
                position_jitter=args.position_jitter,
                modalities=modalities,
                no_viewer=args.no_viewer,
                overwrite=args.overwrite,
                resume=args.resume,
                fail_fast=args.fail_fast,
                quiet=args.quiet,
                verbose=args.verbose,
            )
            try:
                results[robot] = generate_dataset(config)
            except (FileExistsError, ValueError) as error:
                parser.error(str(error))
        if len(robots) > 1:
            atomic_json(root / "manifest.json", {
                "format": "VGArm multi-robot trajectory dataset",
                "robots": list(robots),
                "subdatasets": {robot: robot for robot in robots},
            })
        if not args.quiet:
            for robot, stats in results.items():
                print(
                    f"{robot}: {stats['episodes']} episode(s), "
                    f"{stats['total_physics_steps']} steps"
                )
            print(f"Dataset: {root}")
        return 0
    if command in {"inspect", "stats", "validate"}:
        parser = argparse.ArgumentParser(prog=f"vgarm dataset {command}")
        parser.add_argument("dataset", type=Path)
        args = parser.parse_args(argv[1:])
        if command == "inspect":
            from vgarm.trajectory.reader import read_episodes

            dataset = json.loads(
                (args.dataset / "meta" / "dataset.json").read_text()
            )
            episodes = read_episodes(args.dataset)
            print(json.dumps({
                "dataset": dataset,
                "episodes": len(episodes),
                "completed_episode_ids": [item["episode_id"] for item in episodes],
            }, ensure_ascii=False, indent=2))
            return 0
        if command == "stats":
            from vgarm.trajectory.stats import compute_stats

            print(json.dumps(
                compute_stats(args.dataset), ensure_ascii=False, indent=2
            ))
            return 0
        from vgarm.trajectory.validator import validate_dataset

        report = validate_dataset(args.dataset)
        print("PASS" if report["passed"] else "FAIL")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    if command == "replay":
        from vgarm.trajectory.replay import replay_trajectory

        parser = argparse.ArgumentParser(prog="vgarm dataset replay")
        parser.add_argument("dataset", type=Path)
        parser.add_argument("--episode-id", type=int, required=True)
        viewer = parser.add_mutually_exclusive_group()
        viewer.add_argument("--viewer", dest="no_viewer", action="store_false")
        viewer.add_argument("--no-viewer", dest="no_viewer", action="store_true")
        parser.set_defaults(no_viewer=True)
        parser.add_argument("--speed", type=float, default=1.0)
        args = parser.parse_args(argv[1:])
        if args.speed <= 0:
            parser.error("--speed must be greater than zero")
        try:
            result = replay_trajectory(
                args.dataset,
                args.episode_id,
                no_viewer=args.no_viewer,
                speed=args.speed,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["matched"] else 1
    if command == "export-lerobot":
        from vgarm.trajectory.lerobot_export import export_lerobot

        parser = argparse.ArgumentParser(prog="vgarm dataset export-lerobot")
        parser.add_argument("dataset", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        try:
            result = export_lerobot(args.dataset, args.output)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser = argparse.ArgumentParser(prog="vgarm dataset")
    parser.print_help()
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        return _benchmark_main(argv[1:])
    if argv and argv[0] == "dataset":
        return _dataset_main(argv[1:])
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
        with SimulationSession(model, data, enabled=True) as session:
            executor = PickPlaceExecutor(
                model,
                data,
                robot,
                built.object_names,
                sync=session.sync,
            )
            _execute_intent(executor, intent, layout)
            session.wait_until_closed(lambda: executor.step())
        return 0
    except ViewerClosed:
        return 0
    except ControlFailure as error:
        print(f"VGArm control aborted safely: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
