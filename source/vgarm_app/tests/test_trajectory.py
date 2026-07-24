import json
import os
import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from vgarm.benchmark.models import TaskSpec
from vgarm.cli import main
from vgarm.mjc import controller as control
from vgarm.trajectory.dataset import prepare_dataset, preflight_dataset
from vgarm.trajectory.models import DatasetConfig
from vgarm.trajectory.recorder import TrajectoryRecorder
from vgarm.trajectory.util import atomic_json, atomic_text, sha256_file
from vgarm.trajectory.validator import _finite, validate_dataset
from vgarm.trajectory.cameras import (
    EpisodeRgbRecorder,
    EpisodeVisualRecorder,
    DatasetConfigurationError,
    RationalFrameScheduler,
    named_cameras,
    probe_video,
    validate_camera_names,
    validate_rgb_configuration,
    DEFAULT_VISUAL_CHUNK_FRAMES,
)
from vgarm.trajectory.cameras import _ChunkWriter
from vgarm.trajectory.reader import open_episode, VisualDataError, read_episode_rows
from vgarm.trajectory.stats import compute_stats


SCENE = Path(__file__).resolve().parents[3] / "examples" / "basic_scene.json"
TASKS = Path(__file__).resolve().parents[3] / "examples" / "benchmark_tasks.json"


class StepAlignmentTests(unittest.TestCase):
    def test_recorder_runs_before_the_actual_mj_step(self):
        events = []
        executor = object.__new__(control.PickPlaceExecutor)
        executor.model = object()
        executor.data = object()
        executor._sync = lambda: events.append("sync")
        executor._step_recorder = types.SimpleNamespace(
            record_pre_step=lambda observed: events.append(("record", observed))
        )
        original = control.mujoco
        control.mujoco = types.SimpleNamespace(
            mj_step=lambda model, data: events.append(("step", model, data))
        )
        try:
            executor.step()
        finally:
            control.mujoco = original
        self.assertEqual(events[0], ("record", executor))
        self.assertEqual(events[1], ("step", executor.model, executor.data))
        self.assertEqual(events[2], "sync")


class DatasetSafetyTests(unittest.TestCase):
    def _config(self, root, **changes):
        values = dict(
            root=Path(root),
            scene=SCENE,
            robot="franka_fr3",
            tasks_file=TASKS,
            episodes=1,
            seed=42,
            position_jitter=0.03,
        )
        values.update(changes)
        return DatasetConfig(**values)

    def test_atomic_json_leaves_complete_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta" / "value.json"
            atomic_json(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_empty_episode_is_rejected(self):
        recorder = object.__new__(TrajectoryRecorder)
        recorder.rows = []
        with self.assertRaisesRegex(ValueError, "empty trajectory"):
            recorder.commit(object(), model_xml_hash="", asset_manifest_hash="")

    def test_non_finite_values_are_rejected_by_validator_primitive(self):
        self.assertFalse(_finite({"value": float("nan")}))
        self.assertFalse(_finite([float("inf")]))

    def test_resume_requires_matching_fingerprint(self):
        tasks = (TaskSpec("task", "把红色方块移到左边"),)
        with tempfile.TemporaryDirectory() as tmp:
            prepare_dataset(self._config(tmp), tasks)
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                prepare_dataset(
                    self._config(tmp, seed=43, resume=True), tasks
                )
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                prepare_dataset(
                    self._config(
                        tmp, visual_chunk_frames=32, resume=True), tasks
                )

    def test_overwrite_and_resume_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as caught:
            main([
                "dataset", "generate", "--scene", str(SCENE),
                "--tasks", str(TASKS), "--output", "unused",
                "--overwrite", "--resume",
            ])
        self.assertEqual(caught.exception.code, 2)

    def test_missing_dataset_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = validate_dataset(Path(tmp))
        self.assertFalse(report["passed"])
        self.assertIn("metadata missing", report["errors"][0])

    def test_visual_preflight_uses_asset_dictionary_without_temp_xml(self):
        config = self._config(
            Path(tempfile.gettempdir()) / "unused-vgarm-preflight",
            modalities=("state", "depth"), cameras=("camera_front",),
            visual_width=16, visual_height=15, visual_fps=1)
        robot_dir = (
            __import__("vgarm.mjc", fromlist=["available_robots"])
            .available_robots()["franka_fr3"].include_xml_path.resolve().parent
        )
        before = set(robot_dir.glob("_vgarm_preflight_*.xml"))
        preflight_dataset(config)
        after = set(robot_dir.glob("_vgarm_preflight_*.xml"))
        self.assertEqual(after, before)


class OptionalExportTests(unittest.TestCase):
    def test_lerobot_missing_dependency_is_friendly(self):
        if __import__("importlib").util.find_spec("lerobot") is not None:
            self.skipTest("LeRobot is installed in this environment")
        with self.assertRaises(SystemExit) as caught:
            main([
                "dataset", "export-lerobot", "missing",
                "--output", "unused",
            ])
        self.assertEqual(caught.exception.code, 2)


class RgbSchedulerTests(unittest.TestCase):
    def _samples(self, fps, steps):
        scheduler = RationalFrameScheduler(0.002, fps)
        return [
            (step, index)
            for step in range(steps)
            if (index := scheduler.sample_index(step)) is not None
        ]

    def test_exact_20_hz_schedule(self):
        self.assertEqual(
            self._samples(20, 101),
            [(0, 0), (25, 1), (50, 2), (75, 3), (100, 4)],
        )

    def test_non_divisible_30_and_17_hz_schedules(self):
        for fps in (30, 17):
            samples = self._samples(fps, 500_000)
            expected = int(((500_000 - 1) * 0.002) * fps) + 1
            self.assertEqual(len(samples), expected)
            for step, frame in samples[::max(1, len(samples) // 100)]:
                error = step * 0.002 - frame / fps
                self.assertGreaterEqual(error, -1e-12)
                self.assertLess(error, 0.002 + 1e-12)

    def test_scheduler_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            RationalFrameScheduler(0.002, 0)


class RgbCameraTests(unittest.TestCase):
    @staticmethod
    def _model():
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option timestep="0.002"/>
              <worldbody>
                <camera name="camera_front" pos="0 -1 1"
                        xyaxes="1 0 0 0 .70710678 .70710678"/>
                <geom type="box" size=".05 .05 .05"/>
              </worldbody>
            </mujoco>
            """
        )
        return model, mujoco.MjData(model)

    def test_named_camera_validation_and_missing_message(self):
        model, _ = self._model()
        self.assertEqual(named_cameras(model), {"camera_front": 0})
        with self.assertRaisesRegex(
            DatasetConfigurationError, "available cameras: camera_front"
        ):
            validate_camera_names(model, ("missing",))

    def test_headless_rgb_stream_is_decodable_and_aligned(self):
        model, data = self._model()
        mujoco.mj_forward(model, data)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRgbRecorder(
                model,
                data,
                Path(tmp),
                cameras=("camera_front",),
                width=64,
                height=48,
                fps=20,
            )
            mappings = []
            for step in range(101):
                mapping = recorder.sample(step, float(data.time))
                if mapping is not None:
                    mappings.append(mapping)
                mujoco.mj_step(model, data)
            metadata = recorder.close()["camera_front"]
            self.assertEqual(
                [item["rgb_frame_index"] for item in mappings],
                list(range(5)),
            )
            self.assertEqual(metadata["frame_count"], 5)
            decoded = probe_video(Path(metadata["temporary_path"]))
            self.assertEqual((decoded["height"], decoded["width"]), (48, 64))

    def test_320_by_240_h264_frame_encodes(self):
        model, data = self._model()
        mujoco.mj_forward(model, data)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRgbRecorder(
                model,
                data,
                Path(tmp),
                cameras=("camera_front",),
                width=320,
                height=240,
                fps=10,
            )
            self.assertEqual(
                recorder.sample(0, float(data.time))["rgb_frame_index"], 0
            )
            metadata = recorder.close()["camera_front"]
            self.assertEqual(metadata["frame_count"], 1)
            self.assertEqual((metadata["width"], metadata["height"]), (320, 240))


class DepthSegmentationTests(unittest.TestCase):
    @staticmethod
    def _model():
        model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <visual><map znear=".01" zfar="10"/></visual>
          <option timestep=".002"/>
          <worldbody>
            <camera name="cam" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
            <geom name="known_plane" type="plane" size="2 2 .1"/>
          </worldbody>
        </mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_metric_depth_segmentation_and_chunk_boundary(self):
        model, data = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeVisualRecorder(
                model, data, Path(tmp),
                modalities=("state", "depth", "segmentation"),
                cameras=("cam",), width=33, height=25, fps=20,
                chunk_frames=2)
            for step in range(51):
                recorder.sample(step, float(data.time))
                mujoco.mj_step(model, data)
            result = recorder.close()["arrays"]["cam"]
            self.assertEqual([x["frame_count"] for x in result["depth"]], [2, 1])
            with np.load(result["depth"][0]["path"], allow_pickle=False) as chunk:
                self.assertEqual(chunk["depth_m"].dtype, np.float32)
                self.assertEqual(chunk["depth_m"].shape, (2, 25, 33))
                self.assertAlmostEqual(float(chunk["depth_m"][0, 12, 16]), 1.0, places=5)
                self.assertEqual(chunk["physics_row"].tolist(), [0, 25])
            with np.load(result["segmentation"][0]["path"], allow_pickle=False) as chunk:
                self.assertEqual(chunk["object_id"].dtype, np.int32)
                self.assertEqual(chunk["object_type"].dtype, np.int32)
                self.assertEqual(int(chunk["object_id"][0, 12, 16]), 0)
                self.assertEqual(
                    int(chunk["object_type"][0, 12, 16]),
                    int(mujoco.mjtObj.mjOBJ_GEOM))

    def test_depth_and_segmentation_allow_odd_dimensions(self):
        validate_rgb_configuration(
            ("state", "depth", "segmentation"), ("cam",), 33, 25, 17)

    def test_new_and_old_visual_options_must_not_conflict(self):
        with self.assertRaises(SystemExit) as caught:
            main([
                "dataset", "generate", "--scene", str(SCENE),
                "--tasks", str(TASKS), "--output", "unused",
                "--modalities", "state,depth", "--cameras", "camera_front",
                "--visual-width", "64", "--rgb-width", "66"])
        self.assertEqual(caught.exception.code, 2)

    def test_default_chunk_size_is_64_and_181_frames_make_three_chunks(self):
        self.assertEqual(DEFAULT_VISUAL_CHUNK_FRAMES, 64)
        self.assertEqual(DatasetConfig(
            root=Path("unused"), scene=SCENE, robot="franka_fr3",
            tasks_file=TASKS, episodes=1, seed=42, position_jitter=0,
        ).visual_chunk_frames, 64)
        with tempfile.TemporaryDirectory() as tmp:
            writer = _ChunkWriter(
                Path(tmp), "depth", DEFAULT_VISUAL_CHUNK_FRAMES)
            for index in range(181):
                writer.append({
                    "frame_index": np.int64(index),
                    "timestamp": np.float64(index / 10),
                    "physics_row": np.int64(index * 50),
                }, depth_m=np.zeros((2, 3), dtype=np.float32))
            shards = writer.close()
            self.assertEqual([item["frame_count"] for item in shards], [64, 64, 53])
            self.assertEqual(
                [Path(item["path"]).name for item in shards],
                ["chunk_000000.npz", "chunk_000001.npz", "chunk_000002.npz"])


class VisualValidatorTamperTests(unittest.TestCase):
    def _fixture(self, root: Path):
        for directory in ("meta", "data", "states", "arrays", ".incomplete"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        observation = {
            "joint_position": [0.0], "joint_velocity": [0.0],
            "ctrl": [0.0], "objects": {}, "canonical_state": []}
        row = {
            "episode_id": 0, "frame_index": 0, "sim_step": 0,
            "timestamp": 0.0, "physics_timestep": 0.002,
            "observation": observation,
            "action": {"ctrl": [0.0]},
            "control": {"phase": "test"},
            "visual_observation": {"frame_index": 0, "timestamp": 0.0},
            "terminated": True, "truncated": False, "success": True,
            "failure_category": None}
        jsonl = root / "row.jsonl"
        jsonl.write_text(json.dumps(row) + "\n")
        parquet = root / "data" / "episode_000000.parquet"
        con = __import__("duckdb").connect()
        try:
            source = str(jsonl).replace("'", "''")
            target = str(parquet).replace("'", "''")
            con.execute(
                f"COPY (SELECT * FROM read_json_auto('{source}')) "
                f"TO '{target}' (FORMAT PARQUET)")
        finally:
            con.close()
        jsonl.unlink()
        for suffix in ("initial", "final"):
            np.savez_compressed(
                root / "states" / f"episode_000000_{suffix}.npz",
                observation_json=np.asarray(json.dumps(observation)))
        array_files = {}
        for modality in ("depth", "segmentation"):
            directory = root / "arrays" / "episode_000000" / "cam" / modality
            writer = _ChunkWriter(directory.parent, modality, 64)
            values = {
                "depth_m": np.ones((2, 3), np.float32)
            } if modality == "depth" else {
                "object_id": np.full((2, 3), -1, np.int32),
                "object_type": np.full((2, 3), -1, np.int32)}
            writer.append({
                "frame_index": np.int64(0), "timestamp": np.float64(0),
                "physics_row": np.int64(0)}, **values)
            shards = writer.close()
            for item in shards:
                item["path"] = Path(item["path"]).relative_to(root).as_posix()
            array_files[modality] = shards
        index = root / "meta" / "episode_000000_segmentation_index.json"
        atomic_json(index, {
            "background": {"object_id": -1, "object_type": -1},
            "labels": [{"object_id": -1, "object_type": -1,
                        "object_type_name": "background", "name": None,
                        "observed": True}]})
        schema = {
            "dataset_schema_version": "1.2", "physics_timestep": 0.002,
            "dimensions": {"joint": 1, "actuator": 1},
            "object_names": [], "joint_names": ["j"], "actuator_names": ["a"]}
        atomic_json(root / "meta" / "dataset.json", {
            "schema_version": "1.2",
            "modalities": ["state", "depth", "segmentation"],
            "cameras": ["cam"], "visual_width": 3, "visual_height": 2,
            "visual_fps": 10, "visual_chunk_frames": 64,
            "config_fingerprint": "fixture"})
        atomic_json(root / "meta" / "schema.json", schema)
        episode = {
            "dataset_schema_version": "1.2", "episode_id": 0,
            "recording_modalities": ["state", "depth", "segmentation"],
            "trajectory_file": "data/episode_000000.parquet",
            "initial_state_file": "states/episode_000000_initial.npz",
            "final_state_file": "states/episode_000000_final.npz",
            "trajectory_sha256": sha256_file(parquet),
            "initial_state_sha256": sha256_file(
                root / "states" / "episode_000000_initial.npz"),
            "final_state_sha256": sha256_file(
                root / "states" / "episode_000000_final.npz"),
            "video_files": {}, "array_files": {"cam": array_files},
            "segmentation_index_file":
                "meta/episode_000000_segmentation_index.json",
            "num_steps": 1, "sim_duration_seconds": .002,
            "wall_duration_seconds": .01, "success": True,
            "joint_names": ["j"], "actuator_names": ["a"],
            "object_names": []}
        self._write_episode(root, episode)
        return episode

    @staticmethod
    def _write_episode(root, episode):
        (root / "meta" / "episodes.jsonl").write_text(
            json.dumps(episode) + "\n")

    def _rewrite_chunk(self, root, episode, modality, transform):
        shard = episode["array_files"]["cam"][modality][0]
        path = root / shard["path"]
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key].copy() for key in loaded.files}
        transform(arrays)
        np.savez_compressed(path, **arrays)
        shard["sha256"] = sha256_file(path)
        shard["bytes"] = path.stat().st_size
        self._write_episode(root, episode)

    def test_real_npz_tamper_matrix(self):
        cases = {
            "duplicate frame index": lambda a: a.__setitem__(
                "frame_index", np.asarray([0, 0], np.int64)),
            "timestamp": lambda a: a.__setitem__(
                "timestamp", np.asarray([1.0], np.float64)),
            "dtype": lambda a: a.__setitem__(
                "depth_m", a["depth_m"].astype(np.float64)),
            "shape": lambda a: a.__setitem__(
                "depth_m", a["depth_m"][..., :1]),
        }
        for label, transform in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                episode = self._fixture(root)
                self._rewrite_chunk(root, episode, "depth", transform)
                self.assertFalse(validate_dataset(root)["passed"])

    def test_missing_orphan_truncated_and_checksum_tamper(self):
        for kind in ("missing", "orphan", "truncated", "checksum"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                episode = self._fixture(root)
                path = root / episode["array_files"]["cam"]["depth"][0]["path"]
                if kind == "missing":
                    path.unlink()
                elif kind == "orphan":
                    (path.parent / "chunk_999999.npz").write_bytes(path.read_bytes())
                elif kind == "truncated":
                    path.write_bytes(path.read_bytes()[:20])
                    episode["array_files"]["cam"]["depth"][0]["sha256"] = sha256_file(path)
                    episode["array_files"]["cam"]["depth"][0]["bytes"] = path.stat().st_size
                    self._write_episode(root, episode)
                else:
                    with path.open("ab") as stream:
                        stream.write(b"tamper")
                self.assertFalse(validate_dataset(root)["passed"])


class VisualAtomicFailureTests(unittest.TestCase):
    def _append(self, writer, index):
        writer.append({
            "frame_index": np.int64(index),
            "timestamp": np.float64(index * .1),
            "physics_row": np.int64(index),
        }, depth_m=np.ones((2, 3), np.float32))

    def test_real_chunk_rename_then_second_depth_chunk_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = _ChunkWriter(root, "depth", 2)
            original = np.savez_compressed
            calls = 0
            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected depth failure")
                return original(*args, **kwargs)
            with mock.patch(
                "vgarm.trajectory.cameras.np.savez_compressed",
                side_effect=fail_second,
            ):
                self._append(writer, 0)
                self._append(writer, 1)
                self._append(writer, 2)
                with self.assertRaisesRegex(OSError, "injected depth"):
                    self._append(writer, 3)
            chunks = sorted((root / "depth").glob("chunk_*.npz"))
            self.assertEqual([path.name for path in chunks], ["chunk_000000.npz"])
            self.assertFalse(list((root / "depth").glob("*.tmp.npz")))

    def test_segmentation_middle_chunk_failure_preserves_only_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = _ChunkWriter(root / ".incomplete", "segmentation", 1)
            original = np.savez_compressed
            calls = 0
            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected segmentation failure")
                return original(*args, **kwargs)
            metadata = {
                "frame_index": np.int64(0), "timestamp": np.float64(0),
                "physics_row": np.int64(0)}
            with mock.patch(
                "vgarm.trajectory.cameras.np.savez_compressed",
                side_effect=fail_second,
            ):
                writer.append(
                    metadata, object_id=np.zeros((2, 3), np.int32),
                    object_type=np.full((2, 3), 5, np.int32))
                with self.assertRaisesRegex(OSError, "segmentation"):
                    writer.append(
                        {**metadata, "frame_index": np.int64(1)},
                        object_id=np.ones((2, 3), np.int32),
                        object_type=np.full((2, 3), 5, np.int32))
            self.assertTrue(
                (root / ".incomplete" / "segmentation" /
                 "chunk_000000.npz").is_file())
            self.assertFalse((root / "arrays").exists())

    def test_rgb_frame_succeeds_then_depth_writer_failure_aborts_cleanly(self):
        model, data = DepthSegmentationTests._model()
        class Writer:
            def __init__(self):
                self.frames = 0
                self.closed = False
            def append_data(self, frame):
                self.frames += 1
            def close(self):
                self.closed = True
        writer = Writer()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeVisualRecorder(
                model, data, Path(tmp),
                modalities=("state", "rgb", "depth"), cameras=("cam",),
                width=32, height=24, fps=20, chunk_frames=2,
                writer_factory=lambda path: writer)
            with mock.patch.object(
                recorder.chunks["cam", "depth"], "append",
                side_effect=OSError("injected depth failure"),
            ), self.assertRaisesRegex(OSError, "depth failure"):
                recorder.sample(0, 0.0)
            recorder.abort()
            self.assertEqual(writer.frames, 1)
            self.assertTrue(writer.closed)
            self.assertFalse((Path(tmp) / "arrays" / "episode_000000").exists())

    def test_depth_frame_succeeds_then_segmentation_failure_aborts_cleanly(self):
        model, data = DepthSegmentationTests._model()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeVisualRecorder(
                model, data, Path(tmp),
                modalities=("state", "depth", "segmentation"),
                cameras=("cam",), width=33, height=25, fps=20,
                chunk_frames=2)
            depth_writer = recorder.chunks["cam", "depth"]
            with mock.patch.object(
                recorder.chunks["cam", "segmentation"], "append",
                side_effect=OSError("injected segmentation failure"),
            ), self.assertRaisesRegex(OSError, "segmentation failure"):
                recorder.sample(0, 0.0)
            self.assertEqual(len(depth_writer.items), 1)
            recorder.abort()
            self.assertFalse((Path(tmp) / "arrays" / "episode_000000").exists())

    def test_commit_metadata_failure_rolls_every_artifact_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / ".incomplete" / "episode_000000"
            incomplete.mkdir(parents=True)
            pairs = []
            for relative in (
                "data/episode_000000.parquet",
                "videos/episode_000000/cam.mp4",
                "arrays/episode_000000/cam/depth/chunk_000000.npz",
                "arrays/episode_000000/cam/segmentation/chunk_000000.npz",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(relative.encode())
                source = incomplete / relative
                pairs.append((source, target))
            recorder = object.__new__(TrajectoryRecorder)
            recorder.root = root
            recorder.episode_id = 0
            recorder.incomplete = incomplete
            recorder._commit_moves = pairs
            recorder._segmentation_index_path = None
            recorder.rollback_commit("injected metadata failure")
            self.assertTrue((incomplete / "ABORTED").is_file())
            self.assertTrue(all(source.is_file() for source, _ in pairs))
            self.assertTrue(all(not target.exists() for _, target in pairs))
            self.assertFalse((root / "meta" / "episodes.jsonl").exists())

    def test_episode_metadata_atomic_replace_failure_keeps_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta" / "episodes.jsonl"
            path.parent.mkdir()
            path.write_text('{"episode_id": 0}\\n')
            with mock.patch(
                "vgarm.trajectory.util.os.replace",
                side_effect=OSError("injected metadata replace failure"),
            ), self.assertRaisesRegex(OSError, "metadata replace"):
                atomic_text(path, '{"episode_id": 0}\\n{"episode_id": 1}\\n')
            self.assertEqual(path.read_text(), '{"episode_id": 0}\\n')
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())


class Schema11MappingFixtureTests(unittest.TestCase):
    def test_legacy_rgb_mapping_is_normalized_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            source = root / "legacy.jsonl"
            source.write_text(json.dumps({
                "frame_index": 0,
                "visual_observation": {
                    "rgb_frame_index": 0, "rgb_timestamp": 0.0},
            }) + "\n")
            parquet = root / "data" / "episode_000000.parquet"
            con = __import__("duckdb").connect()
            try:
                con.execute(
                    f"COPY (SELECT * FROM read_json_auto('{source}')) "
                    f"TO '{parquet}' (FORMAT PARQUET)")
            finally:
                con.close()
            before = sha256_file(parquet)
            rows = read_episode_rows(
                root, {"trajectory_file": "data/episode_000000.parquet"})
            self.assertEqual(rows[0]["visual_observation"], {
                "frame_index": 0, "timestamp": 0.0})
            self.assertEqual(sha256_file(parquet), before)

class RgbCliAndCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _generate_args(output, *extra):
        return [
            "dataset", "generate",
            "--scene", str(SCENE),
            "--tasks", str(
                Path(__file__).resolve().parents[3]
                / "examples" / "rgb_trajectory_tasks.json"
            ),
            "--robots", "franka_fr3",
            "--episodes", "1",
            "--modalities", "state,rgb",
            "--cameras", "camera_front",
            "--output", str(output),
            *extra,
        ]

    def test_rgb_requires_camera(self):
        with self.assertRaises(SystemExit) as caught:
            main([
                "dataset", "generate", "--scene", str(SCENE),
                "--tasks", str(TASKS), "--output", "unused",
                "--modalities", "state,rgb",
            ])
        self.assertEqual(caught.exception.code, 2)

    def test_state_modality_is_required(self):
        with self.assertRaises(SystemExit) as caught:
            main([
                "dataset", "generate", "--scene", str(SCENE),
                "--tasks", str(TASKS), "--output", "unused",
                "--modalities", "rgb", "--cameras", "camera_front",
            ])
        self.assertEqual(caught.exception.code, 2)

    def test_schema_10_empty_dataset_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "videos").mkdir()
            (root / ".incomplete").mkdir()
            (root / "meta" / "dataset.json").write_text(
                json.dumps({"schema_version": "1.0", "modalities": ["state"]})
            )
            (root / "meta" / "schema.json").write_text(
                json.dumps({"dataset_schema_version": "1.0"})
            )
            report = validate_dataset(root)
            self.assertTrue(report["passed"], report)

    def test_state_only_empty_stats_have_zero_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            stats = compute_stats(root)
            self.assertEqual(stats["total_rgb_frames"], 0)
            self.assertEqual(stats["video_storage_bytes"], 0)

    def test_keyboard_interrupt_has_resume_message_and_exit_130(self):
        error = io.StringIO()
        with mock.patch(
            "vgarm.trajectory.dataset.generate_dataset",
            side_effect=KeyboardInterrupt,
        ), redirect_stderr(error):
            code = main([
                "dataset", "generate", "--scene", str(SCENE),
                "--tasks", str(TASKS), "--output", "unused",
            ])
        self.assertEqual(code, 130)
        self.assertIn("Completed episodes were preserved", error.getvalue())

    def test_unknown_camera_fails_before_runner_and_lists_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            error = io.StringIO()
            with mock.patch(
                "vgarm.trajectory.dataset.BenchmarkRunner"
            ) as runner, redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                main(self._generate_args(
                    output, "--cameras", "camera_does_not_exist"
                ))
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(runner.called)
            self.assertIn(
                "unknown RGB camera 'camera_does_not_exist'", error.getvalue()
            )
            self.assertIn(
                "available cameras: camera_front, camera_top", error.getvalue()
            )
            self.assertNotIn("Traceback", error.getvalue())
            self.assertFalse(output.exists())

    def test_one_unknown_camera_rejects_entire_multi_camera_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            error = io.StringIO()
            with mock.patch(
                "vgarm.trajectory.dataset.BenchmarkRunner"
            ) as runner, redirect_stderr(error), self.assertRaises(SystemExit):
                main(self._generate_args(
                    output,
                    "--cameras", "camera_front,camera_does_not_exist",
                ))
            self.assertFalse(runner.called)
            self.assertFalse(output.exists())

    def test_invalid_h264_dimensions_fail_before_generate(self):
        invalid = ((321, 240), (320, 241), (0, 240), (-2, 240), (320, 0))
        for width, height in invalid:
            with self.subTest(width=width, height=height):
                with tempfile.TemporaryDirectory() as tmp:
                    error = io.StringIO()
                    with mock.patch(
                        "vgarm.trajectory.dataset.generate_dataset"
                    ) as generate, redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                        main(self._generate_args(
                            Path(tmp) / "dataset",
                            "--rgb-width", str(width),
                            "--rgb-height", str(height),
                        ))
                    self.assertEqual(caught.exception.code, 2)
                    self.assertFalse(generate.called)
                    self.assertIn(
                        "positive even integers for H.264 yuv420p",
                        error.getvalue(),
                    )
                    self.assertNotIn("Traceback", error.getvalue())

    def test_invalid_rgb_fps_fails_before_generate(self):
        for fps in (0, -1):
            with self.subTest(fps=fps), mock.patch(
                "vgarm.trajectory.dataset.generate_dataset"
            ) as generate, self.assertRaises(SystemExit) as caught:
                main(self._generate_args("unused", "--rgb-fps", str(fps)))
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(generate.called)

    def test_duplicate_camera_fails_before_generate(self):
        with mock.patch(
            "vgarm.trajectory.dataset.generate_dataset"
        ) as generate, self.assertRaises(SystemExit) as caught:
            main(self._generate_args(
                "unused", "--cameras", "camera_front,camera_front"
            ))
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(generate.called)

    def test_320_by_240_is_valid_and_state_only_ignores_rgb_dimensions(self):
        validate_rgb_configuration(
            ("state", "rgb"), ("camera_front",), 320, 240, 10
        )
        validate_rgb_configuration(("state",), (), 321, 241, -1)

    def test_missing_recorder_is_not_reported_as_key_error(self):
        class RunnerWithoutRecorder:
            def __init__(self, config, **kwargs):
                self.config = config

            def _task_order(self, robot_index):
                return [self.config.tasks[0]]

            def _run_one(self, *args):
                return mock.Mock()

        with tempfile.TemporaryDirectory() as tmp:
            config = DatasetConfig(
                root=Path(tmp) / "dataset",
                scene=SCENE,
                robot="franka_fr3",
                tasks_file=TASKS,
                episodes=1,
                seed=42,
                position_jitter=0,
            )
            with mock.patch(
                "vgarm.trajectory.dataset.preflight_dataset"
            ), mock.patch(
                "vgarm.trajectory.dataset.BenchmarkRunner",
                RunnerWithoutRecorder,
            ):
                from vgarm.trajectory.dataset import generate_dataset

                with self.assertRaisesRegex(
                    RuntimeError, "recorder was not initialized"
                ) as caught:
                    generate_dataset(config)
            self.assertNotIsInstance(caught.exception, KeyError)


if __name__ == "__main__":
    unittest.main()
