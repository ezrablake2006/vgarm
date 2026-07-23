from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

from vgarm import __version__
from vgarm.benchmark.models import BenchmarkConfig
from vgarm.benchmark.runner import BenchmarkRunner, load_tasks
from vgarm.mjc import available_robots

from .models import DatasetConfig, FORMAT_NAME, SCHEMA_VERSION
from .recorder import TrajectoryRecorder
from .stats import compute_stats
from .util import atomic_json, sha256_file, stable_hash


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
            raise ValueError("cannot resume: meta/dataset.json is missing")
        existing = json.loads(dataset_file.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("cannot resume: dataset configuration fingerprint differs")
    for directory in ("meta", "data", "states", "videos", "logs", ".incomplete"):
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


def generate_dataset(config: DatasetConfig, *, viewer_factory=None) -> dict:
    tasks = load_tasks(config.tasks_file)
    _, completed = prepare_dataset(config, tasks)
    robot_spec = available_robots()[config.robot]
    recorders = {}

    def recorder_factory(robot, task, seed, executor, layout, initial, xml_text):
        episode_id = seed - config.seed
        incomplete = config.root / ".incomplete" / f"episode_{episode_id:06d}"
        if incomplete.exists():
            shutil.rmtree(incomplete)
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
        )
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
            recorder = recorders[episode_id]
            metadata = recorder.commit(
                result,
                model_xml_hash=recorder.model_xml_hash,
                asset_manifest_hash=recorder.asset_manifest_hash,
            )
            with episodes_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            completed.add(episode_id)
            if config.verbose:
                print(
                    f"[{config.robot}] episode {episode_id} "
                    f"steps={metadata['num_steps']} success={metadata['success']}"
                )
            if config.fail_fast and not result.task_success:
                break
        except KeyboardInterrupt:
            recorder = recorders.get(episode_id)
            if recorder is not None:
                recorder.abort("KeyboardInterrupt")
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
