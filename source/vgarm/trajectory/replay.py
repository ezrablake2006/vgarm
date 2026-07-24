from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from vgarm.mjc import (
    SimulationSession, available_robots, build_scene_xml, compile_scene_model)
from vgarm.benchmark.randomization import jitter_scene
from vgarm.reconstruction import reconstruct_scene

from .reader import find_episode, read_episode_rows
from .util import stable_hash


# Low-level replay does not rerun controller-side mj_forward/IK calls.  A
# strict 0.15 mm tolerance covers MuJoCo solver warm-start differences while
# remaining small relative to the 4.5 mm manipulation tolerance.
REPLAY_TOLERANCE = 1.5e-4


def _restore_layout(metadata, dataset_config=None):
    layout = reconstruct_scene(scene_json_path=metadata["scene_path"])
    if (
        dataset_config is not None
        and dataset_config.get("schema_version") == "1.0"
    ):
        # Camera-only XML additions in schema 1.1 must not change the model
        # fingerprint of an existing v0.3/state-only dataset.
        layout = replace(layout, cameras=[])
    # The XML hash covers the deterministic pre-simulation scene layout.
    # Initial object positions in episode metadata are measured after robot
    # homing and can contain harmless solver-scale displacement (e.g. 1e-12 m);
    # the complete NPZ state below restores those measured positions exactly.
    if dataset_config is not None:
        return jitter_scene(
            layout,
            float(dataset_config["position_jitter"]),
            int(metadata["episode_seed"]),
        )
    positions = metadata["initial_object_positions"]
    return replace(
        layout,
        objects=[
            replace(
                obj,
                pos_xyz=(
                    float(positions[obj.name][0]),
                    float(positions[obj.name][1]),
                    obj.pos_xyz[2],
                ),
            )
            for obj in layout.objects
        ],
    )


def _state(model, data, schema, robot, object_names):
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, robot.attachment_site_name
    )
    return {
        "joint_position": np.asarray(data.qpos)[schema["joint_qpos_addresses"]].copy(),
        "joint_velocity": np.asarray(data.qvel)[schema["joint_dof_addresses"]].copy(),
        "eef_position": np.asarray(data.site_xpos[site_id]).copy(),
        "objects": {
            name: np.asarray(
                data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
            ).copy()
            for name in object_names
        },
    }


def replay_trajectory(
    root: Path,
    episode_id: int,
    *,
    no_viewer: bool = True,
    speed: float = 1.0,
    viewer_factory=None,
) -> dict:
    metadata = find_episode(root, episode_id)
    dataset_config = json.loads(
        (root / "meta" / "dataset.json").read_text(encoding="utf-8")
    )
    schema = json.loads((root / "meta" / "schema.json").read_text(encoding="utf-8"))
    rows = read_episode_rows(root, metadata)
    robot = available_robots()[metadata["robot"]]
    layout = _restore_layout(metadata, dataset_config)
    # Keep the generated XML beside the robot include, exactly as the live
    # benchmark runner does.  Resolving first is important when the packaged
    # robot manifest uses a path relative to the current working directory:
    # MuJoCo resolves nested includes relative to the generated XML file.
    directory = robot.include_xml_path.resolve().parent
    built = build_scene_xml(layout, robot, xml_base_dir=directory)
    if stable_hash(built.xml_text) != metadata["model_xml_hash"]:
        raise ValueError("rebuilt model XML hash differs from recorded episode")
    model = compile_scene_model(built, robot)
    try:
        data = mujoco.MjData(model)
        with np.load(
            root / metadata["initial_state_file"], allow_pickle=False
        ) as loaded:
            initial = {key: loaded[key].copy() for key in loaded.files}
        specification = mujoco.mjtState(int(initial["state_spec"][0]))
        mujoco.mj_setState(model, data, initial["state_vector"], specification)
        data.ctrl[:] = initial["ctrl"]
        data.eq_active[:] = initial["eq_active"]
        mujoco.mj_forward(model, data)
        # mj_forward reconstructs derived kinematics but also overwrites
        # integration warm-start fields. Restore the recorded integration
        # state once more while retaining those derived arrays.
        mujoco.mj_setState(model, data, initial["state_vector"], specification)
        data.ctrl[:] = initial["ctrl"]
        data.eq_active[:] = initial["eq_active"]
        maxima = {"qpos": 0.0, "qvel": 0.0, "eef": 0.0, "object": 0.0}
        with SimulationSession(
            model, data, enabled=not no_viewer, speed=speed,
            viewer_factory=viewer_factory,
        ) as session:
            for index, row in enumerate(rows):
                equality = np.asarray(row["action"]["equality_command"])
                if not np.array_equal(data.eq_active, equality):
                    data.eq_active[:] = equality
                    # The live controller calls mj_forward immediately when a
                    # grasp constraint is activated or released.
                    mujoco.mj_forward(model, data)
                current = _state(model, data, schema, robot, metadata["object_names"])
                expected = row["observation"]
                maxima["qpos"] = max(
                    maxima["qpos"],
                    float(np.max(np.abs(
                        current["joint_position"] - expected["joint_position"]
                    ))),
                )
                maxima["qvel"] = max(
                    maxima["qvel"],
                    float(np.max(np.abs(
                        current["joint_velocity"] - expected["joint_velocity"]
                    ))),
                )
                maxima["eef"] = max(
                    maxima["eef"],
                    float(np.max(np.abs(
                        current["eef_position"] - expected["eef_position"]
                    ))),
                )
                for name, position in current["objects"].items():
                    maxima["object"] = max(
                        maxima["object"],
                        float(np.max(np.abs(
                            position - expected["objects"][name]["position"]
                        ))),
                    )
                data.ctrl[:] = row["action"]["ctrl"]
                mujoco.mj_step(model, data)
                session.sync()
            with np.load(
                root / metadata["final_state_file"], allow_pickle=False
            ) as loaded:
                final_expected = json.loads(str(loaded["observation_json"]))
            final_current = _state(
                model, data, schema, robot, metadata["object_names"]
            )
            maxima["qpos"] = max(
                maxima["qpos"],
                float(np.max(np.abs(
                    final_current["joint_position"]
                    - final_expected["joint_position"]
                ))),
            )
            maxima["qvel"] = max(
                maxima["qvel"],
                float(np.max(np.abs(
                    final_current["joint_velocity"]
                    - final_expected["joint_velocity"]
                ))),
            )
            maxima["eef"] = max(
                maxima["eef"],
                float(np.max(np.abs(
                    final_current["eef_position"] - final_expected["eef_position"]
                ))),
            )
            for name, position in final_current["objects"].items():
                maxima["object"] = max(
                    maxima["object"],
                    float(np.max(np.abs(
                        position - final_expected["objects"][name]["position"]
                    ))),
                )
        matched = all(value <= REPLAY_TOLERANCE for value in maxima.values())
        return {
            "episode_id": episode_id,
            "max_qpos_drift": maxima["qpos"],
            "max_qvel_drift": maxima["qvel"],
            "max_eef_position_drift": maxima["eef"],
            "max_object_position_drift": maxima["object"],
            "tolerance": REPLAY_TOLERANCE,
            "original_success": metadata["success"],
            "replay_success": bool(metadata["success"] and matched),
            "matched": matched,
            "final_task_verification": metadata["verification"],
            "has_rgb": "rgb" in metadata.get("recording_modalities", []),
            "has_depth": "depth" in metadata.get("recording_modalities", []),
            "has_segmentation": "segmentation" in metadata.get("recording_modalities", []),
            "rgb_frame_count": (
                next(iter(metadata.get("video_files", {}).values()))[
                    "frame_count"
                ]
                if metadata.get("video_files") else 0
            ),
            "depth_frame_count": sum(
                shard["frame_count"]
                for shard in next(iter(
                    metadata.get("array_files", {}).values()), {}).get(
                        "depth", [])
            ),
            "segmentation_frame_count": sum(
                shard["frame_count"]
                for shard in next(iter(
                    metadata.get("array_files", {}).values()), {}).get(
                        "segmentation", [])
            ),
        }
    finally:
        # SimulationSession and NPZ contexts own all external resources.
        pass
