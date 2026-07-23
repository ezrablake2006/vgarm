from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .reader import read_episode_rows, read_episodes
from .util import sha256_file


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def validate_dataset(root: Path) -> dict:
    errors = []
    warnings = []
    dataset_file = root / "meta" / "dataset.json"
    schema_file = root / "meta" / "schema.json"
    if not dataset_file.is_file() or not schema_file.is_file():
        return {
            "passed": False, "episodes": 0, "steps": 0,
            "invalid_episodes": [], "errors": ["dataset or schema metadata missing"],
            "warnings": [],
        }
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != "1.0" or schema.get("dataset_schema_version") != "1.0":
        errors.append("unsupported schema version")
    episodes = read_episodes(root)
    ids = [item["episode_id"] for item in episodes]
    if len(ids) != len(set(ids)):
        errors.append("duplicate episode ids")
    invalid = []
    total_steps = 0
    total_duration = 0.0
    for episode in episodes:
        episode_errors = []
        required_files = {
            "trajectory_file": "trajectory_sha256",
            "initial_state_file": "initial_state_sha256",
            "final_state_file": "final_state_sha256",
        }
        for file_field, hash_field in required_files.items():
            path = root / episode[file_field]
            if not path.is_file():
                episode_errors.append(f"missing {file_field}")
            elif sha256_file(path) != episode[hash_field]:
                episode_errors.append(f"checksum mismatch: {file_field}")
        if episode_errors:
            invalid.append(episode["episode_id"])
            errors.extend(f"episode {episode['episode_id']}: {item}" for item in episode_errors)
            continue
        rows = read_episode_rows(root, episode)
        total_steps += len(rows)
        total_duration += episode["sim_duration_seconds"]
        if len(rows) != episode["num_steps"] or not rows:
            episode_errors.append("trajectory row count mismatch or empty")
        for index, row in enumerate(rows):
            if row["frame_index"] != index or row["sim_step"] != index:
                episode_errors.append("frame index or sim step is not continuous")
                break
            if not _finite(row):
                episode_errors.append(f"non-finite value at frame {index}")
                break
            if len(row["observation"]["joint_position"]) != schema["dimensions"]["joint"]:
                episode_errors.append("joint shape mismatch")
                break
            if len(row["action"]["ctrl"]) != schema["dimensions"]["actuator"]:
                episode_errors.append("action shape mismatch")
                break
            if list(row["observation"]["objects"]) != schema["object_names"]:
                episode_errors.append("object ordering mismatch")
                break
            if index:
                delta = row["timestamp"] - rows[index - 1]["timestamp"]
                if delta <= 0 or not math.isclose(
                    delta, schema["physics_timestep"], rel_tol=0, abs_tol=1e-9
                ):
                    episode_errors.append("timestamp is not strictly timestep-aligned")
                    break
        if rows:
            if not rows[-1]["terminated"] or rows[-1]["truncated"]:
                episode_errors.append("last row termination flags are invalid")
            if bool(rows[-1]["success"]) != bool(episode["success"]):
                episode_errors.append("success does not match episode metadata")
            initial = np.load(root / episode["initial_state_file"])
            final = np.load(root / episode["final_state_file"])
            if "observation_json" not in initial or "observation_json" not in final:
                episode_errors.append("initial or final observation missing")
            else:
                initial_observation = json.loads(str(initial["observation_json"]))
                first_observation = dict(rows[0]["observation"])
                # Controller writes a_0 after the initial state snapshot and
                # before the pre-step hook.  observation.ctrl therefore
                # intentionally equals a_0 rather than the snapshot ctrl.
                initial_observation.pop("ctrl", None)
                first_observation.pop("ctrl", None)
                if initial_observation != first_observation:
                    episode_errors.append("first observation does not match initial state")
        if episode["joint_names"] != schema["joint_names"]:
            episode_errors.append("joint ordering mismatch")
        if episode["actuator_names"] != schema["actuator_names"]:
            episode_errors.append("actuator ordering mismatch")
        if episode["object_names"] != schema["object_names"]:
            episode_errors.append("episode object ordering mismatch")
        if episode_errors:
            invalid.append(episode["episode_id"])
            errors.extend(f"episode {episode['episode_id']}: {item}" for item in episode_errors)
    incomplete = list((root / ".incomplete").glob("episode_*"))
    if incomplete:
        warnings.append(f"{len(incomplete)} incomplete episode(s) excluded")
    return {
        "passed": not errors,
        "episodes": len(episodes),
        "steps": total_steps,
        "duration": total_duration,
        "storage_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "invalid_episodes": sorted(set(invalid)),
        "errors": errors,
        "warnings": warnings,
    }
