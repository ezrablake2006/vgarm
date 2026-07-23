from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

from vgarm import __version__
from vgarm.reconstruction import reconstruct_scene

from .models import BenchmarkConfig, TaskSpec
from .runner import BenchmarkRunner


REQUIRED_FIELDS = (
    "episode_id",
    "episode_seed",
    "robot",
    "task_id",
    "instruction",
    "scene",
    "program_version",
    "initial_object_positions",
    "planned_target_position",
    "transport_waypoints",
    "path_diagnostics",
    "task_success",
    "verification",
)

MATCH_FIELDS = (
    "episode_seed",
    "robot",
    "task_id",
    "instruction",
    "initial_object_positions",
    "planned_target_position",
    "transport_waypoints",
    "final_object_positions",
    "task_success",
    "failure_category",
    "verification",
)


class ReplayDataError(ValueError):
    pass


def load_episode(path: Path, episode_id: int) -> dict[str, Any]:
    found = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReplayDataError(
                    f"invalid JSON on line {line_number}: {error}"
                ) from error
            if episode.get("episode_id") == episode_id:
                found = episode
                break
    if found is None:
        raise ReplayDataError(f"episode id {episode_id} was not found in {path}")
    missing = [field for field in REQUIRED_FIELDS if field not in found]
    if missing:
        raise ReplayDataError(
            "episode uses an older schema and cannot be replayed exactly; "
            f"missing fields: {', '.join(missing)}"
        )
    if not found["initial_object_positions"]:
        raise ReplayDataError("initial_object_positions is empty")
    if found["program_version"] != __version__:
        raise ReplayDataError(
            f"episode version {found['program_version']} does not match installed "
            f"VGArm {__version__}"
        )
    return found


def restore_layout(episode: dict[str, Any]):
    scene_path = Path(episode["scene"])
    if not scene_path.is_file():
        raise ReplayDataError(f"scene file does not exist: {scene_path}")
    layout = reconstruct_scene(scene_json_path=str(scene_path))
    positions = episode["initial_object_positions"]
    scene_names = {obj.name for obj in layout.objects}
    if set(positions) != scene_names:
        raise ReplayDataError(
            "initial object names do not match scene: "
            f"episode={sorted(positions)}, scene={sorted(scene_names)}"
        )
    objects = []
    for obj in layout.objects:
        position = positions[obj.name]
        if not isinstance(position, list) or len(position) not in (2, 3):
            raise ReplayDataError(
                f"invalid initial position for {obj.name}: expected XY or XYZ"
            )
        z = float(position[2]) if len(position) == 3 else obj.pos_xyz[2]
        objects.append(
            replace(
                obj,
                pos_xyz=(float(position[0]), float(position[1]), z),
            )
        )
    return replace(layout, objects=objects)


def _normalized(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in sorted(value.items())}
    return value


def replay_match(original: dict[str, Any], replay: dict[str, Any]) -> bool:
    return all(
        _normalized(original.get(field)) == _normalized(replay.get(field))
        for field in MATCH_FIELDS
    )


def replay_episode(
    episodes_path: Path,
    episode_id: int,
    *,
    no_viewer: bool = True,
    speed: float = 1.0,
    viewer_factory: Callable | None = None,
):
    original = load_episode(episodes_path, episode_id)
    config_path = episodes_path.parent / "config.json"
    if not config_path.is_file():
        raise ReplayDataError(
            f"benchmark config is required for exact replay: {config_path}"
        )
    benchmark_config = json.loads(config_path.read_text(encoding="utf-8"))
    if "position_jitter" not in benchmark_config:
        raise ReplayDataError("config.json is missing position_jitter")
    layout = restore_layout(original)
    task = TaskSpec(original["task_id"], original["instruction"])
    config = BenchmarkConfig(
        scene=Path(original["scene"]),
        robots=(original["robot"],),
        episodes=1,
        seed=int(original["episode_seed"]),
        output=episodes_path.parent,
        tasks=(task,),
        position_jitter=float(benchmark_config["position_jitter"]),
        no_viewer=no_viewer,
        viewer_speed=speed,
        quiet=True,
    )
    runner = BenchmarkRunner(config, viewer_factory=viewer_factory)
    runner._episode_executor = lambda robot, replay_task, seed: runner._run_layout_episode(
        robot,
        replay_task,
        seed,
        layout,
        planned_target_override=original["planned_target_position"],
    )
    result = runner._run_one(
        int(original["episode_id"]),
        original["robot"],
        task,
        int(original["episode_seed"]),
    )
    replay = result.to_dict()
    return original, replay, replay_match(original, replay)
