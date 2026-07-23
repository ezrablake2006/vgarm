import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from vgarm.benchmark.replay import (
    ReplayDataError,
    load_episode,
    replay_match,
    restore_layout,
)
from vgarm.cli import main
from vgarm.mjc.session import SimulationSession, ViewerClosed


SCENE = Path(__file__).resolve().parents[3] / "examples" / "basic_scene.json"


class FakeViewer:
    def __init__(self, running=True):
        self.running = running
        self.sync_count = 0
        self.closed = False

    def is_running(self):
        return self.running

    def sync(self):
        self.sync_count += 1

    def close(self):
        self.closed = True


class FakeViewerContext:
    def __init__(self, viewer):
        self.viewer = viewer
        self.exited = False

    def __enter__(self):
        return self.viewer

    def __exit__(self, *_args):
        self.exited = True
        self.viewer.close()


class ViewerSessionTests(unittest.TestCase):
    def test_headless_does_not_create_viewer(self):
        factory = mock.Mock()
        model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.002))
        with SimulationSession(model, object(), enabled=False) as session:
            session.sync()
        factory.assert_not_called()

    def test_viewer_is_created_synced_and_closed(self):
        viewer = FakeViewer()
        context = FakeViewerContext(viewer)
        factory = mock.Mock(return_value=context)
        model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.002))
        with SimulationSession(
            model, object(), enabled=True, viewer_factory=factory,
            clock=mock.Mock(side_effect=[0.0, 0.01]),
            sleeper=mock.Mock(),
        ) as session:
            session.sync()
        factory.assert_called_once_with(model, mock.ANY)
        self.assertEqual(viewer.sync_count, 1)
        self.assertTrue(context.exited)
        self.assertTrue(viewer.closed)

    def test_user_closed_viewer_raises_and_context_closes(self):
        viewer = FakeViewer(running=False)
        context = FakeViewerContext(viewer)
        model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.002))
        with self.assertRaises(ViewerClosed):
            with SimulationSession(
                model, object(), enabled=True,
                viewer_factory=lambda *_args: context,
            ) as session:
                session.sync()
        self.assertTrue(context.exited)


class BenchmarkViewerCliTests(unittest.TestCase):
    def _summary(self):
        return {
            "overall": {
                "overall_success_rate": 1.0,
                "episodes": 1,
                "pick_denominator": 1,
                "failures": {},
            }
        }

    def _run_and_config(self, extra):
        fake_robot = types.SimpleNamespace(include_xml_path=SCENE)
        with mock.patch(
            "vgarm.cli.available_robots",
            return_value={"franka_fr3": fake_robot},
        ), mock.patch(
            "vgarm.benchmark.runner.BenchmarkRunner"
        ) as runner_class:
            runner_class.return_value.run.return_value = self._summary()
            code = main([
                "benchmark", "--scene", str(SCENE), "--episodes", "1",
                "--output", "unused", *extra,
            ])
            config = runner_class.call_args.args[0]
        self.assertEqual(code, 0)
        return config

    def test_benchmark_defaults_to_headless(self):
        self.assertTrue(self._run_and_config([]).no_viewer)

    def test_viewer_enters_viewer_path_independent_of_verbose(self):
        config = self._run_and_config(["--viewer", "--verbose"])
        self.assertFalse(config.no_viewer)
        self.assertTrue(config.verbose)

    def test_viewer_flags_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as caught:
            main([
                "benchmark", "--scene", str(SCENE), "--output", "unused",
                "--viewer", "--no-viewer",
            ])
        self.assertEqual(caught.exception.code, 2)


class ReplayDataTests(unittest.TestCase):
    def _episode(self):
        return {
            "episode_id": 7,
            "episode_seed": 49,
            "robot": "franka_fr3",
            "task_id": "move_red_left",
            "instruction": "把红色方块移到左边",
            "scene": str(SCENE),
            "program_version": "0.2.0",
            "initial_object_positions": {
                "cube_red": [0.52, -0.10],
                "cube_yellow": [0.58, 0.11],
                "cube_blue": [0.45, -0.03],
            },
            "planned_target_position": [0.53, 0.15, 0.044],
            "transport_waypoints": [[0.53, 0.15, 0.25]],
            "path_diagnostics": {"required_clearance": 0.005},
            "final_object_positions": {"cube_red": [0.53, 0.15]},
            "task_success": True,
            "failure_category": None,
            "verification": {"predicate": "left_of", "passed": True},
        }

    def test_replay_restores_robot_task_seed_and_positions(self):
        episode = self._episode()
        layout = restore_layout(episode)
        positions = {obj.name: obj.pos_xyz[:2] for obj in layout.objects}
        self.assertEqual(episode["robot"], "franka_fr3")
        self.assertEqual(episode["task_id"], "move_red_left")
        self.assertEqual(episode["episode_seed"], 49)
        self.assertEqual(positions["cube_red"], (0.52, -0.10))

    def test_replay_deterministic_fields_ignore_runtime_values(self):
        original = self._episode()
        replay = dict(original, started_at="later", duration_seconds=999)
        self.assertTrue(replay_match(original, replay))
        replay["planned_target_position"] = [0.54, 0.15, 0.044]
        self.assertFalse(replay_match(original, replay))

    def test_invalid_episode_id_is_friendly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(json.dumps(self._episode()) + "\n")
            with self.assertRaisesRegex(ReplayDataError, "was not found"):
                load_episode(path, 999)

    def test_old_episode_missing_fields_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(json.dumps({"episode_id": 1, "robot": "franka_fr3"}) + "\n")
            with self.assertRaisesRegex(ReplayDataError, "older schema"):
                load_episode(path, 1)


if __name__ == "__main__":
    unittest.main()
