from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import traceback as traceback_module
from typing import Callable

import mujoco

from vgarm import __version__
from vgarm.cli import _execute_intent, _select_object
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

from .metrics import aggregate_results
from .models import BenchmarkConfig, EpisodeResult, TaskSpec
from .planning import NoSafeTargetError
from .planning import footprint
from .randomization import jitter_scene
from .verification import verify_task
from .writer import ResultWriter


DEFAULT_TASKS = (
    TaskSpec("move_red_left", "把红色方块移到左边"),
    TaskSpec("move_blue_right", "把蓝色方块移到右边"),
    TaskSpec("place_red_left_of_blue", "把红色方块放在蓝色方块左边"),
    TaskSpec("place_blue_right_of_red", "把蓝色方块放在红色方块右边"),
    TaskSpec("swap_red_blue", "交换红色方块和蓝色方块位置"),
)


class SceneLoadFailure(RuntimeError):
    pass


def load_tasks(path: Path | None) -> tuple[TaskSpec, ...]:
    if path is None:
        return DEFAULT_TASKS
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("tasks file must contain a non-empty JSON array")
    tasks: list[TaskSpec] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("id") or not item.get("instruction"):
            raise ValueError("each task requires non-empty id and instruction")
        tasks.append(TaskSpec(str(item["id"]), str(item["instruction"])))
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("task ids must be unique")
    return tuple(tasks)


def classify_exception(error: BaseException) -> str:
    if isinstance(error, (SceneLoadFailure, FileNotFoundError, json.JSONDecodeError)):
        return "scene_load_failed"
    if isinstance(error, ControlFailure) and error.category:
        return error.category
    if isinstance(error, NoSafeTargetError):
        return "no_safe_target"
    if isinstance(error, ViewerClosed):
        return "viewer_closed"
    if isinstance(error, (KeyError, ValueError)) and "matching object" in str(error):
        return "object_not_found"
    message = str(error).lower()
    if "collision" in message:
        return "collision"
    if "timeout" in message:
        return "timeout"
    if "unreachable" in message:
        return "unreachable"
    return "unexpected_error"


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        episode_executor: Callable | None = None,
        viewer_factory: Callable | None = None,
    ):
        self.config = config
        self._episode_executor = episode_executor
        self._viewer_factory = viewer_factory

    def _task_order(self, robot_index: int) -> list[TaskSpec]:
        rng = random.Random(self.config.seed + robot_index)
        tasks: list[TaskSpec] = []
        while len(tasks) < self.config.episodes:
            batch = list(self.config.tasks)
            rng.shuffle(batch)
            tasks.extend(batch)
        return tasks[: self.config.episodes]

    @staticmethod
    def _positions(layout) -> dict[str, list[float]]:
        return {obj.name: list(obj.pos_xyz) for obj in layout.objects}

    @staticmethod
    def _geometry(layout) -> dict:
        return {
            obj.name: {
                "geom_type": obj.geom_type,
                "size_xyz": list(obj.size_xyz),
                "footprint_half_extents": list(footprint(obj)),
            }
            for obj in layout.objects
        }

    def _run_actual_episode(self, robot_name: str, task: TaskSpec, episode_seed: int):
        try:
            layout = reconstruct_scene(scene_json_path=str(self.config.scene))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SceneLoadFailure(str(error)) from error
        layout = jitter_scene(layout, self.config.position_jitter, episode_seed)
        return self._run_layout_episode(robot_name, task, episode_seed, layout)

    def _run_layout_episode(
        self,
        robot_name: str,
        task: TaskSpec,
        episode_seed: int,
        layout,
        *,
        planned_target_override=None,
    ):
        intent = parse_cn(task.instruction)
        robots = available_robots()
        robot = robots[robot_name]
        robot_directory = robot.include_xml_path.resolve().parent
        built = build_scene_xml(layout, robot, xml_base_dir=robot_directory)
        xml_path = robot_directory / f"_vgarm_benchmark_{robot.robot_id}_{episode_seed}.xml"
        xml_path.write_text(built.xml_text, encoding="utf-8")
        executor = None
        plan = None
        initial = {}
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            with SimulationSession(
                model,
                data,
                enabled=not self.config.no_viewer,
                speed=self.config.viewer_speed,
                viewer_factory=self._viewer_factory,
            ) as session:
                executor = PickPlaceExecutor(
                    model,
                    data,
                    robot,
                    built.object_names,
                    sync=session.sync if not self.config.no_viewer else None,
                )
                initial = {name: executor.body_xy(name) for name in built.object_names}
                selected = _select_object(layout.objects, intent.object_color, intent.object_category)
                reference = None
                if intent.kind in ("move_next_to_object", "swap"):
                    reference = _select_object(layout.objects, intent.ref_color, None)
                plan = _execute_intent(
                    executor,
                    intent,
                    layout,
                    planned_target_xy=planned_target_override,
                )
                executor.step(200)
                final = {name: executor.body_xy(name) for name in built.object_names}
                verification = verify_task(
                    intent, selected.name, initial, final,
                    reference_name=reference.name if reference else None,
                )
                return layout, executor.execution_trace, verification, initial, final, plan
        except Exception as error:
            error.benchmark_layout = layout
            error.benchmark_trace = executor.execution_trace if executor is not None else []
            error.benchmark_plan = plan
            error.benchmark_initial = initial
            if executor is not None:
                error.benchmark_final = {
                    name: executor.body_xy(name) for name in built.object_names
                }
            raise
        finally:
            try:
                xml_path.unlink()
            except FileNotFoundError:
                pass

    def _run_one(
        self, episode_id: int, robot: str, task: TaskSpec, episode_seed: int
    ) -> EpisodeResult:
        started = datetime.now(timezone.utc)
        start_clock = time.monotonic()
        result = EpisodeResult(
            episode_id=episode_id,
            episode_seed=episode_seed,
            robot=robot,
            task_id=task.id,
            instruction=task.instruction,
            scene=str(self.config.scene),
            program_version=__version__,
            started_at=started.isoformat(),
            duration_seconds=0.0,
            object_positions={},
        )
        trace = []
        try:
            # Parse separately so parse failures have a stable stage.
            parse_cn(task.instruction)
            result.parse_success = True
            executor = self._episode_executor or self._run_actual_episode
            payload = executor(robot, task, episode_seed)
            if len(payload) == 3:  # compatibility for injected lightweight test executors
                layout, trace, verification = payload
                initial = self._positions(layout)
                final = {}
                plan = None
            else:
                layout, trace, verification, initial_xy, final_xy, plan = payload
                initial = {name: [xy[0], xy[1]] for name, xy in initial_xy.items()}
                final = {name: [xy[0], xy[1]] for name, xy in final_xy.items()}
            result.object_positions = self._positions(layout)
            result.object_geometry = self._geometry(layout)
            result.initial_object_positions = initial
            result.final_object_positions = final
            if plan is not None:
                result.original_target_position = list(plan.original_target)
                result.planned_target_position = list(plan.target)
                result.obstacle_objects = plan.obstacle_objects
                result.path_diagnostics = asdict(plan.diagnostics)
            result.verification = verification
            result.task_success = verification.passed
            if not verification.passed:
                result.failure_stage = "verification"
                result.failure_category = "verification_failed"
                result.failure_reason = f"{verification.predicate} predicate failed"
        except KeyboardInterrupt:
            raise
        except Exception as error:
            trace = getattr(error, "benchmark_trace", trace)
            error_layout = getattr(error, "benchmark_layout", None)
            if error_layout is not None:
                result.object_positions = self._positions(error_layout)
                result.initial_object_positions = self._positions(error_layout)
                result.object_geometry = self._geometry(error_layout)
            error_final = getattr(error, "benchmark_final", None)
            if error_final:
                result.final_object_positions = {
                    name: [xy[0], xy[1]] for name, xy in error_final.items()
                }
            error_plan = getattr(error, "benchmark_plan", None)
            if error_plan is not None:
                result.original_target_position = list(error_plan.original_target)
                result.planned_target_position = list(error_plan.target)
                result.obstacle_objects = error_plan.obstacle_objects
                result.path_diagnostics = asdict(error_plan.diagnostics)
            result.failure_stage = (
                error.stage if isinstance(error, ControlFailure) and error.stage
                else ("scene_or_execution" if result.parse_success else "parse")
            )
            result.failure_category = (
                classify_exception(error) if result.parse_success else "task_parse_failed"
            )
            result.failure_reason = str(error)
            if result.failure_category == "unexpected_error":
                result.traceback = traceback_module.format_exc()
        finally:
            result.duration_seconds = time.monotonic() - start_clock
            result.pick_attempted = any(item.pick_attempted for item in trace)
            if result.pick_attempted:
                result.pick_success = all(item.pick_success for item in trace)
            result.place_attempted = any(item.place_attempted for item in trace)
            if result.place_attempted:
                result.place_success = all(item.place_success for item in trace)
            result.grasp_diagnostics = [
                item.grasp_diagnostics for item in trace if item.grasp_diagnostics
            ]
            if trace:
                result.transport_waypoints = [
                    waypoint for item in trace for waypoint in item.transport_waypoints
                ]
                result.planned_target_position = (
                    trace[-1].planned_target_position or result.planned_target_position
                )
                result.held_object = next(
                    (item.object_name for item in reversed(trace) if item.pick_success and not item.place_success),
                    None,
                )
                result.collision_diagnostics = next(
                    (item.collision_diagnostics for item in reversed(trace) if item.collision_diagnostics),
                    None,
                )
                if result.collision_diagnostics is not None:
                    collision = result.collision_diagnostics
                    collision["planned_target_position"] = result.planned_target_position
                    collision["required_clearance"] = result.path_diagnostics.get(
                        "required_clearance"
                    )
                    held_position = collision.get("held_object_position")
                    obstacle_position = collision.get("obstacle_position")
                    if held_position and obstacle_position:
                        collision["target_obstacle_distance"] = (
                            (held_position[0] - obstacle_position[0]) ** 2
                            + (held_position[1] - obstacle_position[1]) ** 2
                        ) ** 0.5
        return result

    def run(self) -> dict:
        results: list[EpisodeResult] = []
        with ResultWriter(self.config) as writer:
            if not self.config.quiet:
                print(
                    "VGArm Benchmark\n"
                    f"Robots: {len(self.config.robots)}\n"
                    f"Tasks: {len(self.config.tasks)}\n"
                    f"Episodes per robot: {self.config.episodes}\n"
                    f"Seed: {self.config.seed}\n"
                )
            episode_id = 0
            for robot_index, robot in enumerate(self.config.robots):
                successes = collisions = timeouts = 0
                for local_index, task in enumerate(self._task_order(robot_index)):
                    episode_seed = self.config.seed + episode_id
                    result = self._run_one(episode_id, robot, task, episode_seed)
                    results.append(result)
                    writer.write_episode(result)
                    writer.log(json.dumps(result.to_dict(), ensure_ascii=False))
                    successes += int(result.task_success)
                    collisions += int(result.failure_category == "collision")
                    timeouts += int(result.failure_category == "timeout")
                    if not self.config.quiet:
                        print(
                            f"[{robot}] {local_index + 1}/{self.config.episodes} | "
                            f"success {successes} | collision {collisions} | timeout {timeouts}"
                        )
                    if self.config.verbose and result.failure_reason:
                        print(f"  {result.failure_category}: {result.failure_reason}")
                    if self.config.fail_fast and not result.task_success:
                        summary = aggregate_results(results)
                        writer.write_summary(summary)
                        return summary
                    if result.failure_category == "viewer_closed":
                        summary = aggregate_results(results)
                        writer.write_summary(summary)
                        return summary
                    episode_id += 1
            summary = aggregate_results(results)
            writer.write_summary(summary)
        return summary
