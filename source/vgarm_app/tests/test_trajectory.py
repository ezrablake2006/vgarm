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
from vgarm.trajectory.dataset import prepare_dataset
from vgarm.trajectory.models import DatasetConfig
from vgarm.trajectory.recorder import TrajectoryRecorder
from vgarm.trajectory.util import atomic_json
from vgarm.trajectory.validator import _finite, validate_dataset
from vgarm.trajectory.cameras import (
    EpisodeRgbRecorder,
    DatasetConfigurationError,
    RationalFrameScheduler,
    named_cameras,
    probe_video,
    validate_camera_names,
    validate_rgb_configuration,
)
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
