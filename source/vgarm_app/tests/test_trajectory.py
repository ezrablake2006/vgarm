import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from vgarm.benchmark.models import TaskSpec
from vgarm.cli import main
from vgarm.mjc import controller as control
from vgarm.trajectory.dataset import prepare_dataset
from vgarm.trajectory.models import DatasetConfig
from vgarm.trajectory.recorder import TrajectoryRecorder
from vgarm.trajectory.util import atomic_json
from vgarm.trajectory.validator import _finite, validate_dataset


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


if __name__ == "__main__":
    unittest.main()
