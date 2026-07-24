from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from vgarm import __version__
from vgarm.benchmark.models import BenchmarkConfig
from vgarm.benchmark.runner import BenchmarkRunner, load_tasks
from vgarm.mjc import available_robots, build_scene_xml, compile_scene_model
from vgarm.reconstruction import reconstruct_scene

from .cameras import (
    DatasetConfigurationError,
    require_rgb_dependencies,
    validate_camera_names,
    validate_visual_configuration,
)
from .models import DatasetConfig, FORMAT_NAME, SCHEMA_VERSION
from .recorder import TrajectoryRecorder
from .stats import compute_stats
from .util import atomic_json, atomic_text, sha256_file, stable_hash


def config_payload(config: DatasetConfig, tasks) -> dict:
    return {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene": str(config.scene.resolve()),
        "scene_sha256": sha256_file(config.scene),
        "robot": config.robot,
        "tasks": [asdict(task) for task in tasks],
        "seed": config.seed,
        "position_jitter": config.position_jitter,
        "modalities": list(config.modalities),
        "cameras": list(config.cameras),
        "visual_width": config.visual_width if len(config.modalities) > 1 else None,
        "visual_height": config.visual_height if len(config.modalities) > 1 else None,
        "visual_fps": config.visual_fps if len(config.modalities) > 1 else None,
        "visual_chunk_frames": config.visual_chunk_frames,
        "visual_schedule": "first physics row at or after n/visual_fps",
        "depth_encoding": "npz/float32/meter" if "depth" in config.modalities else None,
        "segmentation_encoding": "npz/int32 object_id,object_type" if "segmentation" in config.modalities else None,
        "vgarm_version": __version__,
    }


def prepare_dataset(config: DatasetConfig, tasks) -> tuple[dict, set[int]]:
    root = config.root
    payload = config_payload(config, tasks)
    fingerprint = stable_hash(payload)
    dataset_file = root / "meta" / "dataset.json"
    if root.exists() and any(root.iterdir()):
        if config.overwrite:
            shutil.rmtree(root)
        elif not config.resume:
            raise FileExistsError(f"dataset directory is not empty: {root}")
    if config.resume:
        if not dataset_file.is_file():
            raise DatasetConfigurationError(
                "cannot resume: meta/dataset.json is missing"
            )
        existing = json.loads(dataset_file.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise DatasetConfigurationError(
                "cannot resume: dataset configuration fingerprint differs"
            )
    for directory in ("meta", "data", "states", "videos", "arrays", "logs", ".incomplete"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    if not dataset_file.exists():
        atomic_json(dataset_file, {**payload, "config_fingerprint": fingerprint})
        with (root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as stream:
            for task in tasks:
                stream.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")
    completed = set()
    episodes_file = root / "meta" / "episodes.jsonl"
    if episodes_file.is_file():
        for line in episodes_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("completed"):
                    completed.add(int(item["episode_id"]))
    return payload, completed


def preflight_dataset(config: DatasetConfig) -> None:
    """Validate visual configuration before creating output or running physics."""
    validate_visual_configuration(
        config.modalities,
        config.cameras,
        config.visual_width,
        config.visual_height,
        config.visual_fps,
        config.visual_chunk_frames,
    )
    if len(config.modalities) == 1:
        return
    layout = reconstruct_scene(scene_json_path=str(config.scene))
    scene_cameras = {item.name for item in layout.cameras}
    missing = [item for item in config.cameras if item not in scene_cameras]
    if missing:
        choices = ", ".join(sorted(scene_cameras)) or "(none)"
        raise DatasetConfigurationError(
            f"unknown visual camera '{missing[0]}'; "
            f"unknown RGB camera '{missing[0]}'; available cameras: {choices}"
        )
    if "rgb" in config.modalities:
        require_rgb_dependencies()
    robot = available_robots()[config.robot]
    if not robot.include_xml_path.is_file():
        raise DatasetConfigurationError(
            f"robot model assets are missing for {config.robot}; "
            "set VGARM_MENAGERIE_ROOT"
        )
    robot_directory = robot.include_xml_path.resolve().parent
    built = build_scene_xml(layout, robot, xml_base_dir=robot_directory)
    # MuJoCo's virtual-file assets preserve the include/mesh/texture relative
    # layout without writing a top-level XML into the external Menagerie.
    model = compile_scene_model(built, robot)
    validate_camera_names(model, config.cameras)


def generate_dataset(config: DatasetConfig, *, viewer_factory=None) -> dict:
    preflight_dataset(config)
    tasks = load_tasks(config.tasks_file)
    _, completed = prepare_dataset(config, tasks)
    robot_spec = available_robots()[config.robot]
    recorders = {}
    recorder_errors = {}

    def recorder_factory(robot, task, seed, executor, layout, initial, xml_text):
        episode_id = seed - config.seed
        incomplete = config.root / ".incomplete" / f"episode_{episode_id:06d}"
        if incomplete.exists():
            shutil.rmtree(incomplete)
        try:
            recorder = TrajectoryRecorder(
                config.root,
                episode_id,
                seed,
                task,
                executor,
                config.scene,
                initial_object_positions={
                    name: [xy[0], xy[1]] for name, xy in initial.items()
                },
                viewer_enabled=not config.no_viewer,
                modalities=config.modalities,
                cameras=config.cameras,
                visual_width=config.visual_width,
                visual_height=config.visual_height,
                visual_fps=config.visual_fps,
                visual_chunk_frames=config.visual_chunk_frames,
            )
        except Exception as error:
            recorder_errors[episode_id] = error
            raise
        recorder.model_xml_hash = stable_hash(xml_text)
        recorder.asset_manifest_hash = sha256_file(robot_spec.include_xml_path)
        recorders[episode_id] = recorder
        if not (config.root / "meta" / "schema.json").exists():
            atomic_json(config.root / "meta" / "schema.json", recorder.schema)
        return recorder

    benchmark_config = BenchmarkConfig(
        scene=config.scene,
        robots=(config.robot,),
        episodes=config.episodes,
        seed=config.seed,
        output=config.root / "logs" / "benchmark",
        tasks=tasks,
        position_jitter=config.position_jitter,
        no_viewer=config.no_viewer,
        quiet=True,
    )
    runner = BenchmarkRunner(
        benchmark_config,
        viewer_factory=viewer_factory,
        recorder_factory=recorder_factory,
    )
    order = runner._task_order(0)
    episodes_path = config.root / "meta" / "episodes.jsonl"
    for episode_id, task in enumerate(order):
        if episode_id in completed:
            continue
        seed = config.seed + episode_id
        try:
            result = runner._run_one(episode_id, config.robot, task, seed)
            recorder = recorders.get(episode_id)
            if recorder is None:
                initialization_error = recorder_errors.get(episode_id)
                if initialization_error is not None:
                    raise initialization_error
                raise RuntimeError(
                    f"trajectory recorder was not initialized for episode {episode_id}"
                )
            metadata = recorder.commit(
                result,
                model_xml_hash=recorder.model_xml_hash,
                asset_manifest_hash=recorder.asset_manifest_hash,
            )
            existing_lines = (
                episodes_path.read_text(encoding="utf-8")
                if episodes_path.exists() else ""
            )
            atomic_text(
                episodes_path,
                existing_lines + json.dumps(metadata, ensure_ascii=False) + "\n",
            )
            recorder.finalize_commit()
            completed.add(episode_id)
            if config.verbose:
                print(
                    f"[{config.robot}] episode {episode_id} "
                    f"steps={metadata['num_steps']} success={metadata['success']}"
                )
            if config.fail_fast and not result.task_success:
                break
        except BaseException as error:
            recorder = recorders.get(episode_id)
            if recorder is not None:
                if getattr(recorder, "_commit_moves", None):
                    recorder.rollback_commit(type(error).__name__)
                else:
                    recorder.abort(type(error).__name__)
            raise
    stats = compute_stats(config.root)
    atomic_json(config.root / "manifest.json", {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "robot": config.robot,
        "episodes": len(completed),
        "config_fingerprint": json.loads(
            (config.root / "meta" / "dataset.json").read_text()
        )["config_fingerprint"],
    })
    return stats
