import json
from pathlib import Path
import tempfile
import unittest

from vgarm.benchmark.metrics import aggregate_results
from vgarm.benchmark.models import (
    BenchmarkConfig,
    EpisodeResult,
    TaskSpec,
    VerificationResult,
)
from vgarm.benchmark.randomization import jitter_scene
from vgarm.benchmark.planning import NoSafeTargetError, plan_safe_target
from vgarm.benchmark.compare import compare_results
from vgarm.benchmark.runner import BenchmarkRunner
from vgarm.benchmark.verification import verify_task
from vgarm.benchmark.writer import ResultWriter
from vgarm.cli import main
from vgarm.mjc.controller import ActionStageResult
from vgarm.nlu import parse_cn
from vgarm.reconstruction import reconstruct_scene


SCENE = Path(__file__).resolve().parents[3] / "examples" / "basic_scene.json"


def episode(episode_id=0, robot="franka_fr3", **changes):
    values = dict(
        episode_id=episode_id,
        episode_seed=42 + episode_id,
        robot=robot,
        task_id="task",
        instruction="把红色方块移到左边",
        scene=str(SCENE),
        program_version="test",
        started_at="2026-01-01T00:00:00+00:00",
        duration_seconds=1.0,
        object_positions={},
    )
    values.update(changes)
    return EpisodeResult(**values)


class MetricsTests(unittest.TestCase):
    def test_success_and_failure_metrics(self):
        results = [
            episode(parse_success=True, pick_attempted=True, pick_success=True,
                    place_attempted=True, place_success=True, task_success=True),
            episode(1, parse_success=True, pick_attempted=True, pick_success=False,
                    place_attempted=False, place_success=None, failure_category="collision"),
        ]
        summary = aggregate_results(results)["overall"]
        self.assertEqual(summary["overall_success_rate"], 0.5)
        self.assertEqual(summary["pick_success_rate"], 0.5)
        self.assertEqual(summary["place_success_rate"], 1.0)
        self.assertEqual(summary["place_denominator"], 1)
        self.assertEqual(summary["collision_rate"], 0.5)

    def test_robot_groups_and_empty_results(self):
        summary = aggregate_results([
            episode(robot="franka_fr3", task_success=True),
            episode(1, robot="ur5e", failure_category="timeout"),
        ])
        self.assertEqual(set(summary["by_robot"]), {"franka_fr3", "ur5e"})
        self.assertIsNone(aggregate_results([])["overall"]["overall_success_rate"])


class RandomizationTests(unittest.TestCase):
    def setUp(self):
        self.scene = reconstruct_scene(scene_json_path=str(SCENE))

    def test_seed_is_reproducible_and_different_seeds_differ(self):
        first = jitter_scene(self.scene, 0.03, 42)
        again = jitter_scene(self.scene, 0.03, 42)
        other = jitter_scene(self.scene, 0.03, 43)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_jitter_does_not_overlap_objects(self):
        randomized = jitter_scene(self.scene, 0.03, 42)
        for index, obj in enumerate(randomized.objects):
            for other in randomized.objects[index + 1:]:
                self.assertTrue(
                    abs(obj.pos_xyz[0] - other.pos_xyz[0]) >= obj.size_xyz[0] + other.size_xyz[0] + 0.005
                    or abs(obj.pos_xyz[1] - other.pos_xyz[1]) >= obj.size_xyz[1] + other.size_xyz[1] + 0.005
                )


class TargetPlanningTests(unittest.TestCase):
    def setUp(self):
        self.scene = jitter_scene(
            reconstruct_scene(scene_json_path=str(SCENE)), 0.03, 49
        )

    def test_seed_49_target_is_adjusted_and_safe(self):
        plan = plan_safe_target(
            self.scene, "cube_red", (0.55, 0.15), "左边"
        )
        self.assertFalse(plan.diagnostics.target_region_clear)
        self.assertEqual(plan.target, (0.53, 0.15))
        self.assertGreaterEqual(
            plan.diagnostics.minimum_clearance,
            plan.diagnostics.required_clearance,
        )
        self.assertGreaterEqual(plan.target[1], 0.05)
        self.assertEqual(
            plan,
            plan_safe_target(self.scene, "cube_red", (0.55, 0.15), "左边"),
        )

    def test_clear_target_is_unchanged(self):
        plan = plan_safe_target(
            self.scene, "cube_blue", (0.55, -0.25), "右边"
        )
        self.assertEqual(plan.target, (0.55, -0.25))

    def test_no_safe_target_is_explicit(self):
        with self.assertRaises(NoSafeTargetError):
            plan_safe_target(
                self.scene, "cube_red", (0.55, 0.15), "左边",
                clearance=1.0, search_radius=0.01,
            )


class VerificationTests(unittest.TestCase):
    def test_direction_center_swap_and_lift_predicates(self):
        initial = {"red": (0.5, -0.1), "blue": (0.45, 0.0)}
        left = verify_task(
            parse_cn("把红色方块放在蓝色方块左边"), "red", initial,
            {"red": (0.45, 0.10), "blue": (0.45, 0.0)}, reference_name="blue",
        )
        self.assertTrue(left.passed)
        self.assertEqual(left.predicate, "left_of")
        center = verify_task(
            parse_cn("把红色方块移到中间"), "red", initial,
            {"red": (0.55, 0.0), "blue": (0.45, 0.0)},
        )
        self.assertTrue(center.passed)
        swap = verify_task(
            parse_cn("交换红色方块和蓝色方块位置"), "red", initial,
            {"red": initial["blue"], "blue": initial["red"]}, reference_name="blue",
        )
        self.assertTrue(swap.passed)
        lift = verify_task(
            parse_cn("把红色方块抬起"), "red", initial, initial,
        )
        self.assertTrue(lift.passed)


class WriterAndRunnerTests(unittest.TestCase):
    def _config(self, output, episodes=1, tasks=None):
        return BenchmarkConfig(
            scene=SCENE,
            robots=("franka_fr3",),
            episodes=episodes,
            seed=42,
            output=Path(output),
            tasks=tasks or (TaskSpec("move", "把红色方块移到左边"),),
        )

    def test_jsonl_csv_json_and_markdown_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "results")
            result = episode(parse_success=True, task_success=True)
            with ResultWriter(config) as writer:
                writer.write_episode(result)
                writer.write_summary(aggregate_results([result]))
            for name in (
                "config.json", "episodes.jsonl", "episodes.csv", "summary.json",
                "summary.csv", "summary.md", "benchmark.log",
                "summary_by_task.json", "summary_by_task.csv",
                "summary_by_task.md", "summary_robot_task_matrix.csv",
            ):
                self.assertTrue((config.output / name).is_file())
            payload = json.loads((config.output / "episodes.jsonl").read_text())
            self.assertEqual(payload["episode_seed"], 42)
            self.assertIn("| Robot |", (config.output / "summary.md").read_text())

    def test_deterministic_comparison_ignores_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            first.mkdir()
            second.mkdir()
            base = episode(parse_success=True, task_success=True).to_dict()
            changed = dict(base, started_at="later", duration_seconds=99.0)
            (first / "episodes.jsonl").write_text(json.dumps(base) + "\n")
            (second / "episodes.jsonl").write_text(json.dumps(changed) + "\n")
            equal, first_hash, second_hash = compare_results(first, second)
            self.assertTrue(equal)
            self.assertEqual(first_hash, second_hash)

    def test_existing_nonempty_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            output.mkdir()
            (output / "keep.txt").write_text("existing")
            with self.assertRaises(FileExistsError):
                ResultWriter(self._config(output))

    def test_episode_exception_does_not_stop_benchmark(self):
        calls = 0
        layout = reconstruct_scene(scene_json_path=str(SCENE))

        def fake_executor(_robot, _task, _seed):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic failure")
            trace = [ActionStageResult("cube_red", True, True, True, True)]
            return layout, trace, VerificationResult("left_of", True, 0.15, 0.05)

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "results", episodes=2)
            summary = BenchmarkRunner(config, episode_executor=fake_executor).run()
            self.assertEqual(summary["overall"]["episodes"], 2)
            self.assertEqual(summary["overall"]["successful_episodes"], 1)
            lines = (config.output / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["failure_category"], "unexpected_error")


class CliValidationTests(unittest.TestCase):
    def test_invalid_episode_count(self):
        with self.assertRaises(SystemExit) as caught:
            main(["benchmark", "--scene", str(SCENE), "--episodes", "0", "--output", "unused"])
        self.assertEqual(caught.exception.code, 2)

    def test_invalid_robot(self):
        with self.assertRaises(SystemExit) as caught:
            main([
                "benchmark", "--scene", str(SCENE), "--robots", "not_a_robot",
                "--episodes", "1", "--output", "unused",
            ])
        self.assertEqual(caught.exception.code, 2)


class HeadlessBenchmarkSmokeTest(unittest.TestCase):
    def test_real_mujoco_episode(self):
        from vgarm.mjc import available_robots

        model = available_robots()["franka_fr3"].include_xml_path
        if not model.is_file():
            self.skipTest(f"MuJoCo Menagerie asset missing: {model}")
        with tempfile.TemporaryDirectory() as tmp:
            config = BenchmarkConfig(
                scene=SCENE,
                robots=("franka_fr3",),
                episodes=1,
                seed=42,
                output=Path(tmp) / "smoke",
                tasks=(TaskSpec("move_red_left", "把红色方块移到左边"),),
                no_viewer=True,
                quiet=True,
            )
            summary = BenchmarkRunner(config).run()
            self.assertEqual(summary["overall"]["episodes"], 1)
            self.assertEqual(len((config.output / "episodes.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
