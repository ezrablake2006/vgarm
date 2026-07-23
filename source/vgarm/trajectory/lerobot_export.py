from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from .reader import read_episode_rows, read_episodes


def export_lerobot(root: Path, output: Path) -> dict:
    """Export through the official LeRobotDataset v3 public API.

    LeRobot is deliberately imported lazily: its PyTorch/Hugging Face stack is
    not a dependency of native VGArm recording.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is not installed. Use a supported Python environment and "
            "install the optional 'vgarm[lerobot]' extra."
        ) from error

    episodes = read_episodes(root)
    if not episodes:
        raise ValueError("native dataset contains no completed episodes")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    first_rows = read_episode_rows(root, episodes[0])
    state_size = len(first_rows[0]["observation"]["canonical_state"])
    action_size = len(first_rows[0]["action"]["ctrl"])
    fps = round(1.0 / float(first_rows[0]["physics_timestep"]))
    repo_id = f"local/{output.name}"
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (state_size,),
            "names": None,
        },
        "action": {
            "dtype": "float32", "shape": (action_size,),
            "names": episodes[0]["actuator_names"],
        },
    }
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output,
            fps=fps,
            robot_type=episodes[0]["robot"],
            features=features,
            use_videos=False,
        )
        frame_count = 0
        for episode in episodes:
            rows = read_episode_rows(root, episode)
            for row in rows:
                dataset.add_frame({
                    "observation.state": np.asarray(
                        row["observation"]["canonical_state"], dtype=np.float32
                    ),
                    "action": np.asarray(row["action"]["ctrl"], dtype=np.float32),
                    "task": episode["instruction"],
                })
                frame_count += 1
            dataset.save_episode()
        dataset.finalize()
        loaded = LeRobotDataset(repo_id=repo_id, root=output)
        if len(loaded) != frame_count:
            raise RuntimeError(
                f"official loader returned {len(loaded)} frames; expected {frame_count}"
            )
        first = loaded[0]
        last = loaded[len(loaded) - 1]
        for name, expected in (
            ("observation.state", state_size),
            ("action", action_size),
        ):
            if tuple(first[name].shape) != (expected,) or tuple(last[name].shape) != (expected,):
                raise RuntimeError(f"official loader returned an invalid {name} shape")
        return {
            "episodes": len(episodes),
            "frames": frame_count,
            "first_task": episodes[0]["instruction"],
            "first_state_shape": list(first["observation.state"].shape),
            "last_action_shape": list(last["action"].shape),
            "official_loader_verified": True,
        }
    except Exception:
        # Never leave a directory that could be mistaken for a finalized
        # LeRobot dataset.
        if output.exists():
            shutil.rmtree(output)
        raise
