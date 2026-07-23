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
    for episode in episodes:
        rows = read_episode_rows(root, episode)
        total_steps += len(rows)
        task_counts[episode["task_id"]] += 1
        task_success[episode["task_id"]] += int(episode["success"])
        for camera, video in episode.get("video_files", {}).items():
            per_camera_frames[camera] += int(video["frame_count"])
            video_storage += int(video["bytes"])
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
    total_rgb_frames = sum(per_camera_frames.values())
    sampled_instants = (
        max(per_camera_frames.values()) if per_camera_frames else 0
    )
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
        "per_camera_frame_count": dict(per_camera_frames),
        "video_storage_bytes": video_storage,
        "average_video_bytes_per_episode": (
            video_storage / len(episodes) if episodes else None
        ),
        "effective_rgb_fps": (
            sampled_instants / total_sim if total_sim else None
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
