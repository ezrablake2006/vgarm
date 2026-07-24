from __future__ import annotations

import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .reader import read_episode_rows, read_episodes
from .util import sha256_file
from .cameras import RgbDependencyError, probe_video


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
    supported = {"1.0", "1.1", "1.2"}
    if (
        dataset.get("schema_version") not in supported
        or schema.get("dataset_schema_version") not in supported
    ):
        errors.append("unsupported schema version")
    dataset_modalities = dataset.get("modalities", ["state"])
    dataset_cameras = dataset.get("cameras", [])
    declared_video_paths = set()
    declared_array_paths = set()
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
        episode_modalities = episode.get("recording_modalities", ["state"])
        videos = episode.get("video_files", {})
        if "rgb" in episode_modalities:
            if set(videos) != set(dataset_cameras):
                episode_errors.append(
                    "RGB camera set differs from dataset configuration"
                )
            frame_counts = set()
            for camera, video in videos.items():
                path = root / video.get("path", "")
                declared_video_paths.add(path.resolve())
                prefix = f"camera {camera}"
                if not path.is_file():
                    episode_errors.append(f"{prefix}: video file missing")
                    continue
                if sha256_file(path) != video.get("sha256"):
                    episode_errors.append(f"{prefix}: video checksum mismatch")
                    continue
                if path.stat().st_size != video.get("bytes"):
                    episode_errors.append(f"{prefix}: video byte size mismatch")
                try:
                    probe = probe_video(path)
                except (OSError, RuntimeError, ValueError, RgbDependencyError) as error:
                    episode_errors.append(f"{prefix}: video decode failed: {error}")
                    continue
                for field in ("frame_count", "width", "height"):
                    if probe[field] != video.get(field):
                        episode_errors.append(
                            f"{prefix}: decoded {field} differs from metadata"
                        )
                frame_counts.add(probe["frame_count"])
            if len(frame_counts) > 1:
                episode_errors.append("RGB cameras have different frame counts")
        elif videos:
            episode_errors.append("state-only episode unexpectedly declares videos")
        array_sequences = []
        observed_segmentation_pairs = set()
        for camera, modalities in episode.get("array_files", {}).items():
            for modality, shards in modalities.items():
                expected_chunk = 0
                sequence = []
                for shard in shards:
                    path = root / shard.get("path", "")
                    declared_array_paths.add(path.resolve())
                    if path.name != f"chunk_{expected_chunk:06d}.npz":
                        episode_errors.append(f"{camera}/{modality}: non-contiguous chunk names")
                    expected_chunk += 1
                    if not path.is_file():
                        episode_errors.append(f"{camera}/{modality}: missing chunk")
                        continue
                    if sha256_file(path) != shard.get("sha256"):
                        episode_errors.append(f"{camera}/{modality}: chunk checksum mismatch")
                        continue
                    if path.stat().st_size != shard.get("bytes"):
                        episode_errors.append(f"{camera}/{modality}: chunk byte size mismatch")
                    try:
                        with path.open("rb") as stream, np.load(
                            stream, allow_pickle=False) as data:
                            required = {"frame_index", "timestamp", "physics_row"}
                            required |= {"depth_m"} if modality == "depth" else {
                                "object_id", "object_type"}
                            if not required.issubset(data.files):
                                raise ValueError("required arrays missing")
                            if data["frame_index"].dtype != np.int64:
                                raise ValueError("frame_index dtype is not int64")
                            count = len(data["frame_index"])
                            if any(len(data[key]) != count for key in required):
                                raise ValueError("array lengths differ")
                            if modality == "depth":
                                value = data["depth_m"]
                                if value.dtype != np.float32:
                                    raise ValueError("depth dtype is not float32")
                                if value.ndim != 3:
                                    raise ValueError("depth shape is invalid")
                                expected_shape = (
                                    int(dataset.get("visual_height", value.shape[1])),
                                    int(dataset.get("visual_width", value.shape[2])),
                                )
                                if value.shape[1:] != expected_shape:
                                    raise ValueError(
                                        "depth shape differs from visual resolution")
                                if not np.isfinite(value).all():
                                    raise ValueError("depth contains NaN or infinity")
                            else:
                                for key in ("object_id", "object_type"):
                                    if data[key].dtype != np.int32:
                                        raise ValueError(f"{key} dtype is not int32")
                                    if data[key].ndim != 3:
                                        raise ValueError(f"{key} shape is invalid")
                                    expected_shape = (
                                        int(dataset.get(
                                            "visual_height", data[key].shape[1])),
                                        int(dataset.get(
                                            "visual_width", data[key].shape[2])),
                                    )
                                    if data[key].shape[1:] != expected_shape:
                                        raise ValueError(
                                            f"{key} shape differs from visual resolution")
                                observed_segmentation_pairs.update(zip(
                                    data["object_id"].reshape(-1).tolist(),
                                    data["object_type"].reshape(-1).tolist()))
                            sequence.extend(zip(
                                data["frame_index"].astype(int).tolist(),
                                data["physics_row"].astype(int).tolist(),
                                data["timestamp"].astype(float).tolist()))
                    except (
                        OSError, ValueError, EOFError, zipfile.BadZipFile
                    ) as error:
                        episode_errors.append(f"{camera}/{modality}: invalid chunk: {error}")
                array_sequences.append((camera, modality, sequence))
        if "segmentation" in episode_modalities:
            index = episode.get("segmentation_index_file")
            if not index or not (root / index).is_file():
                episode_errors.append("segmentation label manifest missing")
            else:
                try:
                    label_data = json.loads((root / index).read_text(encoding="utf-8"))
                    declared = {
                        (int(x["object_id"]), int(x["object_type"]))
                        for x in label_data["labels"] if x.get("observed")
                    }
                    if declared != observed_segmentation_pairs:
                        episode_errors.append(
                            "segmentation observed pairs differ from label manifest")
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    episode_errors.append("segmentation label manifest is invalid")
        if episode_errors:
            invalid.append(episode["episode_id"])
            errors.extend(
                f"episode {episode['episode_id']}: {item}"
                for item in episode_errors
            )
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
        if any(item != "state" for item in episode_modalities):
            mappings = [
                (index, row.get("visual_observation"))
                for index, row in enumerate(rows)
                if row.get("visual_observation") is not None
            ]
            index_key = "frame_index"
            timestamp_key = "timestamp"
            mapped_indices = [int(mapping[index_key]) for _, mapping in mappings]
            counts = [item["frame_count"] for item in videos.values()]
            counts += [len(sequence) for _, _, sequence in array_sequences]
            expected_count = counts[0] if counts else 0
            if len(set(counts)) > 1:
                episode_errors.append("visual modalities/cameras have different frame counts")
            if mapped_indices != list(range(expected_count)):
                episode_errors.append(
                    "RGB frame mapping is not unique and continuous from zero"
                )
            previous_timestamp = None
            for row_index, mapping in mappings:
                timestamp = float(mapping[timestamp_key])
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    episode_errors.append("RGB timestamps are not strictly increasing")
                    break
                if not math.isclose(
                    timestamp,
                    float(rows[row_index]["timestamp"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    episode_errors.append(
                        "RGB timestamp differs from mapped physics row"
                    )
                    break
                previous_timestamp = timestamp
            expected_sequence = [
                (int(mapping[index_key]), row_index, float(mapping[timestamp_key]))
                for row_index, mapping in mappings
            ]
            for camera, modality, sequence in array_sequences:
                if sequence != expected_sequence:
                    episode_errors.append(
                        f"{camera}/{modality}: frame mapping differs from Parquet"
                    )
        elif schema.get("dataset_schema_version") == "1.1":
            if any(row.get("visual_observation") is not None for row in rows):
                episode_errors.append("state-only trajectory contains RGB mappings")
        if rows:
            if not rows[-1]["terminated"] or rows[-1]["truncated"]:
                episode_errors.append("last row termination flags are invalid")
            if bool(rows[-1]["success"]) != bool(episode["success"]):
                episode_errors.append("success does not match episode metadata")
            with np.load(
                root / episode["initial_state_file"], allow_pickle=False
            ) as initial, np.load(
                root / episode["final_state_file"], allow_pickle=False
            ) as final:
                if "observation_json" not in initial or "observation_json" not in final:
                    episode_errors.append("initial or final observation missing")
                else:
                    initial_observation = json.loads(str(initial["observation_json"]))
                    first_observation = dict(rows[0]["observation"])
                    # Controller writes a_0 after the initial state snapshot and
                    # before the pre-step hook. observation.ctrl therefore
                    # intentionally equals a_0 rather than snapshot ctrl.
                    initial_observation.pop("ctrl", None)
                    first_observation.pop("ctrl", None)
                    if initial_observation != first_observation:
                        episode_errors.append(
                            "first observation does not match initial state")
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
    actual_videos = {
        path.resolve() for path in (root / "videos").rglob("*.mp4")
        if path.is_file()
    }
    orphaned = actual_videos - declared_video_paths
    if orphaned:
        errors.append(
            "undeclared video file(s): "
            + ", ".join(str(path.relative_to(root.resolve())) for path in sorted(orphaned))
        )
    actual_arrays = {
        path.resolve() for path in (root / "arrays").rglob("*.npz")
        if path.is_file()
    }
    orphan_arrays = actual_arrays - declared_array_paths
    if orphan_arrays:
        errors.append("undeclared array chunk(s): " + ", ".join(
            str(path.relative_to(root.resolve())) for path in sorted(orphan_arrays)))
    if "rgb" in dataset_modalities and not dataset_cameras:
        errors.append("RGB dataset does not declare cameras")
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
