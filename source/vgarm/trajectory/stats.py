from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .reader import read_episode_rows, read_episodes
from .util import atomic_json


def compute_stats(root: Path) -> dict:
    episodes = read_episodes(root)
    phase_counts = Counter()
    task_counts = Counter()
    task_success = Counter()
    joint_values = []
    action_values = []
    object_positions = []
    total_steps = 0
    per_camera_frames = Counter()
    video_storage = 0
    modality_frames = Counter()
    modality_bytes = Counter()
    depth_count = 0
    depth_sum = depth_sum2 = 0.0
    depth_min, depth_max = float("inf"), float("-inf")
    segmentation_pairs = set()
    total_visual_instants = 0
    for episode in episodes:
        episode_camera_frames = Counter()
        rows = read_episode_rows(root, episode)
        total_steps += len(rows)
        task_counts[episode["task_id"]] += 1
        task_success[episode["task_id"]] += int(episode["success"])
        for camera, video in episode.get("video_files", {}).items():
            episode_camera_frames[camera] = max(
                episode_camera_frames[camera], int(video["frame_count"]))
            video_storage += int(video["bytes"])
        if episode.get("video_files"):
            modality_frames["rgb"] += sum(
                item["frame_count"] for item in episode["video_files"].values())
            modality_bytes["rgb"] += sum(x["bytes"] for x in episode["video_files"].values())
        for camera, modalities in episode.get("array_files", {}).items():
            for modality, shards in modalities.items():
                episode_camera_frames[camera] = max(
                    episode_camera_frames[camera],
                    sum(x["frame_count"] for x in shards))
                modality_frames[modality] += sum(x["frame_count"] for x in shards)
                modality_bytes[modality] += sum(x["bytes"] for x in shards)
                for shard in shards:
                    path = root / shard["path"]
                    with path.open("rb") as stream, np.load(
                        stream, allow_pickle=False) as data:
                        if modality == "depth":
                            values = data["depth_m"].astype(np.float64, copy=False)
                            depth_count += values.size
                            depth_sum += float(values.sum())
                            depth_sum2 += float(np.square(values).sum())
                            depth_min = min(depth_min, float(values.min()))
                            depth_max = max(depth_max, float(values.max()))
                        elif modality == "segmentation":
                            segmentation_pairs.update(zip(
                                data["object_id"].reshape(-1).tolist(),
                                data["object_type"].reshape(-1).tolist()))
        per_camera_frames.update(episode_camera_frames)
        total_visual_instants += max(episode_camera_frames.values(), default=0)
        for row in rows:
            phase_counts[str(row["control"].get("phase"))] += 1
            joint_values.append(row["observation"]["joint_position"])
            action_values.append(row["action"]["ctrl"])
            for state in row["observation"]["objects"].values():
                object_positions.append(state["position"])
    def describe(values):
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        return {
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
        }
    total_sim = sum(item["sim_duration_seconds"] for item in episodes)
    total_wall = sum(item["wall_duration_seconds"] for item in episodes)
    storage_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    total_rgb_frames = modality_frames["rgb"]
    stats = {
        "episodes": len(episodes),
        "successful_episodes": sum(item["success"] for item in episodes),
        "failed_episodes": sum(not item["success"] for item in episodes),
        "total_physics_steps": total_steps,
        "total_simulation_duration": total_sim,
        "total_wall_duration": total_wall,
        "steps_per_second": total_steps / total_wall if total_wall else None,
        "storage_bytes": storage_bytes,
        "total_rgb_frames": total_rgb_frames,
        "total_visual_frames": total_visual_instants,
        "per_modality_frame_count": dict(modality_frames),
        "per_camera_frame_count": dict(per_camera_frames),
        "video_storage_bytes": video_storage,
        "per_modality_storage_bytes": dict(modality_bytes),
        "rgb_storage_bytes": modality_bytes["rgb"],
        "depth_storage_bytes": modality_bytes["depth"],
        "segmentation_storage_bytes": modality_bytes["segmentation"],
        "per_modality_storage_ratio": {
            name: size / storage_bytes if storage_bytes else 0.0
            for name, size in modality_bytes.items()
        },
        "depth": ({
            "min": depth_min, "max": depth_max,
            "mean": depth_sum / depth_count,
            "std": max(0.0, depth_sum2 / depth_count - (depth_sum / depth_count) ** 2) ** .5,
            "finite_ratio": 1.0,
        } if depth_count else None),
        "segmentation_observed_object_pair_count": len(segmentation_pairs),
        "average_video_bytes_per_episode": (
            video_storage / len(episodes) if episodes else None
        ),
        "effective_rgb_fps": (
            total_visual_instants / total_sim if total_sim else None
        ),
        "effective_visual_fps": (
            total_visual_instants / total_sim if total_sim else None
        ),
        "rgb_storage_ratio": (
            video_storage / storage_bytes if storage_bytes else 0.0
        ),
        "trajectory_storage_ratio": (
            sum(
                (root / item["trajectory_file"]).stat().st_size
                for item in episodes
            ) / storage_bytes if storage_bytes else 0.0
        ),
        "per_task_episode_count": dict(task_counts),
        "per_task_success": dict(task_success),
        "per_phase_step_count": dict(phase_counts),
        "joint": describe(joint_values),
        "action": describe(action_values),
        "object_workspace": describe(object_positions),
        "average_episode_length": total_steps / len(episodes) if episodes else None,
    }
    atomic_json(root / "meta" / "stats.json", stats)
    lines = [
        "# VGArm Trajectory Dataset",
        "",
        f"- Episodes: {stats['episodes']}",
        f"- Successful: {stats['successful_episodes']}",
        f"- Physics steps: {stats['total_physics_steps']}",
        f"- Simulation duration: {stats['total_simulation_duration']:.3f}s",
        f"- Storage: {stats['storage_bytes']} bytes",
        f"- RGB frames: {stats['total_rgb_frames']}",
        f"- Video storage: {stats['video_storage_bytes']} bytes",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats
