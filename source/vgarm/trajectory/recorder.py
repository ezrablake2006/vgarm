from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import duckdb
import mujoco
import numpy as np

from vgarm import __version__

from .schema import build_schema
from .util import atomic_json, sha256_file


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _quat_from_matrix(matrix) -> list[float]:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix).reshape(9))
    return quaternion.tolist()


class TrajectoryRecorder:
    def __init__(
        self,
        dataset_root: Path,
        episode_id: int,
        episode_seed: int,
        task,
        executor,
        scene_path: Path,
        *,
        initial_object_positions: dict,
        viewer_enabled: bool,
    ):
        self.root = dataset_root
        self.episode_id = episode_id
        self.episode_seed = episode_seed
        self.task = task
        self.executor = executor
        self.model = executor.model
        self.data = executor.data
        self.scene_path = scene_path
        self.object_names = tuple(executor.object_names)
        self.initial_object_positions = initial_object_positions
        self.viewer_enabled = viewer_enabled
        self.rows: list[dict] = []
        self.started = time.monotonic()
        self.schema = build_schema(executor, self.object_names)
        self.incomplete = self.root / ".incomplete" / f"episode_{episode_id:06d}"
        self.incomplete.mkdir(parents=True, exist_ok=False)
        self.initial_state = self._capture_state()
        self.initial_state["observation_json"] = np.asarray(
            json.dumps(self._observation(), ensure_ascii=False)
        )
        np.savez_compressed(
            self.incomplete / "initial.npz",
            **self.initial_state,
        )

    def _capture_state(self) -> dict:
        size = mujoco.mj_stateSize(self.model, STATE_SPEC)
        vector = np.empty(size)
        mujoco.mj_getState(self.model, self.data, vector, STATE_SPEC)
        return {
            "state_spec": np.asarray([int(STATE_SPEC)], dtype=np.int64),
            "state_vector": vector,
            "ctrl": np.asarray(self.data.ctrl).copy(),
            "eq_active": np.asarray(self.data.eq_active).copy(),
            "qpos": np.asarray(self.data.qpos).copy(),
            "qvel": np.asarray(self.data.qvel).copy(),
            "time": np.asarray([self.data.time]),
            "object_names": np.asarray(self.object_names),
        }

    def _observation(self) -> dict:
        executor = self.executor
        site = executor.attachment_site
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site,
            velocity,
            0,
        )
        eef_position = np.asarray(self.data.site_xpos[site]).tolist()
        eef_quaternion = _quat_from_matrix(self.data.site_xmat[site])
        object_state = {}
        canonical_objects = []
        for name in self.object_names:
            body_id = executor.object_body_ids[name]
            body_velocity = np.asarray(self.data.cvel[body_id])
            position = np.asarray(self.data.xpos[body_id]).tolist()
            quaternion = np.asarray(self.data.xquat[body_id]).tolist()
            linear = body_velocity[3:].tolist()
            angular = body_velocity[:3].tolist()
            object_state[name] = {
                "position": position,
                "quaternion": quaternion,
                "linear_velocity": linear,
                "angular_velocity": angular,
            }
            canonical_objects.extend(position + quaternion + linear + angular)
        actuated = executor._actuated
        joint_position = [
            float(self.data.qpos[item.qpos_address]) for item in actuated
        ]
        joint_velocity = [
            float(self.data.qvel[item.dof_address]) for item in actuated
        ]
        canonical_state = (
            eef_position
            + eef_quaternion
            + velocity[3:].tolist()
            + velocity[:3].tolist()
            + canonical_objects
        )
        return {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "actuator_state": (
                np.asarray(self.data.act).tolist() if self.model.na else None
            ),
            "ctrl": np.asarray(self.data.ctrl).tolist(),
            "eef_position": eef_position,
            "eef_quaternion": eef_quaternion,
            "eef_linear_velocity": velocity[3:].tolist(),
            "eef_angular_velocity": velocity[:3].tolist(),
            "gripper_state": None,
            "held_object": executor._held_object,
            "objects": object_state,
            "canonical_state": canonical_state,
        }

    def record_pre_step(self, executor) -> None:
        observation = self._observation()
        target_position = getattr(executor, "_current_eef_target", None)
        target_orientation = getattr(
            executor, "_current_eef_orientation_target", None
        )
        action = {
            "ctrl": np.asarray(self.data.ctrl).tolist(),
            "joint_target": np.asarray(executor._commanded_qpos).tolist(),
            "eef_target_position": (
                np.asarray(target_position).tolist()
                if target_position is not None else None
            ),
            "eef_target_quaternion": (
                _quat_from_matrix(target_orientation)
                if target_orientation is not None else None
            ),
            "equality_command": np.asarray(self.data.eq_active, dtype=int).tolist(),
            "gripper_command": None,
            "canonical_action": (
                (np.asarray(target_position).tolist() if target_position is not None else [None] * 3)
                + (_quat_from_matrix(target_orientation) if target_orientation is not None else [None] * 4)
            ),
        }
        self.rows.append({
            "episode_id": self.episode_id,
            "frame_index": len(self.rows),
            "sim_step": len(self.rows),
            "timestamp": float(self.data.time),
            "physics_timestep": float(self.model.opt.timestep),
            "observation": observation,
            "action": action,
            "control": {
                "skill": getattr(executor, "_control_skill", None),
                "phase": getattr(executor, "_motion_phase", None),
                "waypoint_index": getattr(executor, "_motion_waypoint_index", None),
                "ik_iteration": getattr(executor, "_ik_iteration", None),
                "ik_stopping_reason": getattr(executor, "_ik_stopping_reason", None),
                "position_error": getattr(executor, "_position_error", None),
                "orientation_error": getattr(executor, "_orientation_error", None),
                "collision_active": False,
                "grasp_active": executor._held_object is not None,
            },
            "terminated": False,
            "truncated": False,
            "success": None,
            "failure_category": None,
        })

    def abort(self, reason: str) -> None:
        (self.incomplete / "ABORTED").write_text(reason, encoding="utf-8")

    def commit(self, result, *, model_xml_hash: str, asset_manifest_hash: str) -> dict:
        if not self.rows:
            raise ValueError("cannot commit empty trajectory episode")
        self.rows[-1]["terminated"] = True
        self.rows[-1]["success"] = bool(result.task_success)
        self.rows[-1]["failure_category"] = result.failure_category
        final_state = self._capture_state()
        final_state["observation_json"] = np.asarray(
            json.dumps(self._observation(), ensure_ascii=False)
        )
        np.savez_compressed(self.incomplete / "final.npz", **final_state)
        jsonl = self.incomplete / "trajectory.jsonl"
        with jsonl.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        parquet = self.incomplete / "trajectory.parquet"
        connection = duckdb.connect()
        try:
            source = str(jsonl).replace("'", "''")
            target = str(parquet).replace("'", "''")
            connection.execute(
                f"COPY (SELECT * FROM read_json_auto('{source}', "
                "maximum_object_size=104857600)) "
                f"TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            connection.close()
        episode_name = f"episode_{self.episode_id:06d}"
        paths = {
            "trajectory": self.root / "data" / f"{episode_name}.parquet",
            "initial": self.root / "states" / f"{episode_name}_initial.npz",
            "final": self.root / "states" / f"{episode_name}_final.npz",
        }
        for directory in ("data", "states"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        os.replace(parquet, paths["trajectory"])
        os.replace(self.incomplete / "initial.npz", paths["initial"])
        os.replace(self.incomplete / "final.npz", paths["final"])
        shutil.rmtree(self.incomplete)
        metadata = {
            "dataset_schema_version": "1.0",
            "episode_id": self.episode_id,
            "robot": result.robot,
            "task_id": result.task_id,
            "instruction": result.instruction,
            "scene_path": str(self.scene_path),
            "scene_sha256": sha256_file(self.scene_path),
            "episode_seed": self.episode_seed,
            "vgarm_version": __version__,
            "mujoco_version": mujoco.__version__,
            "asset_manifest_hash": asset_manifest_hash,
            "model_xml_hash": model_xml_hash,
            "physics_timestep": float(self.model.opt.timestep),
            "num_steps": len(self.rows),
            "sim_duration_seconds": len(self.rows) * float(self.model.opt.timestep),
            "wall_duration_seconds": time.monotonic() - self.started,
            "success": result.task_success,
            "failure_stage": result.failure_stage,
            "failure_category": result.failure_category,
            "failure_reason": result.failure_reason,
            "initial_state_file": str(paths["initial"].relative_to(self.root)),
            "final_state_file": str(paths["final"].relative_to(self.root)),
            "trajectory_file": str(paths["trajectory"].relative_to(self.root)),
            "video_files": {},
            "initial_object_positions": result.initial_object_positions,
            "planned_target_position": result.planned_target_position,
            "target_adjusted": (
                result.original_target_position is not None
                and result.planned_target_position is not None
                and result.original_target_position[:2] != result.planned_target_position[:2]
            ),
            "final_object_positions": result.final_object_positions,
            "verification": (
                result.verification.__dict__ if result.verification else None
            ),
            "transport_waypoints": result.transport_waypoints,
            "trajectory_sha256": sha256_file(paths["trajectory"]),
            "initial_state_sha256": sha256_file(paths["initial"]),
            "final_state_sha256": sha256_file(paths["final"]),
            "joint_names": self.schema["joint_names"],
            "actuator_names": self.schema["actuator_names"],
            "object_names": list(self.object_names),
            "attachment_site": self.executor.robot.attachment_site_name,
            "action_representation": self.schema["action_representation"],
            "observation_schema": self.schema["dimensions"],
            "camera_schema": {},
            "viewer_enabled": self.viewer_enabled,
            "recording_modalities": ["state"],
            "completed": True,
        }
        return metadata
