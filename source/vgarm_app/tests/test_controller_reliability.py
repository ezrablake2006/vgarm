import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


if "mujoco" not in sys.modules:
    sys.modules["mujoco"] = types.ModuleType("mujoco")

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

from vgarm.mjc import controller as control


class ControllerMathTests(unittest.TestCase):
    def test_rotation_error_controls_orientation(self):
        current = np.eye(3)
        target = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(
            control.rotation_error(current, target),
            [0.0, 0.0, 1.0],
        )

    def test_joint_targets_keep_a_limit_margin(self):
        values = control.clamp_to_joint_limits(
            [-2.0, 0.5, 4.0],
            [-1.0, -math.inf, -3.0],
            [1.0, math.inf, 3.0],
            margin=0.1,
        )
        np.testing.assert_allclose(values, [-0.9, 0.5, 2.9])


class ControllerRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_mujoco = control.mujoco

    def tearDown(self):
        control.mujoco = self.original_mujoco

    def test_home_keyframe_initializes_only_robot_joints(self):
        class ObjectType:
            mjOBJ_KEY = 1

        fake_mujoco = types.SimpleNamespace(
            mjtObj=ObjectType,
            mj_name2id=lambda *_args: 0,
            mj_forward=lambda *_args: None,
        )
        control.mujoco = fake_mujoco
        executor = object.__new__(control.PickPlaceExecutor)
        executor.model = types.SimpleNamespace(
            key_qpos=np.asarray([[0.25, -0.75, 99.0, 99.0]]),
        )
        executor.data = types.SimpleNamespace(
            qpos=np.asarray([0.0, 0.0, 0.55, -0.20]),
            qvel=np.ones(2),
            ctrl=np.zeros(2),
        )
        executor.robot = types.SimpleNamespace(default_home_qpos=None)
        executor.config = control.ControllerConfig(joint_limit_margin=0.05)
        executor._actuated = [
            control.ActuatedJoint(0, 0, 0, 0, -1.0, 1.0),
            control.ActuatedJoint(1, 1, 1, 1, -1.0, 1.0),
        ]

        executor._initialize_home_pose()

        np.testing.assert_allclose(executor.data.qpos[:2], [0.25, -0.75])
        np.testing.assert_allclose(executor.data.qpos[2:], [0.55, -0.20])
        np.testing.assert_allclose(executor.data.ctrl, [0.25, -0.75])

    def test_release_height_keeps_box_above_floor(self):
        fake_mujoco = types.SimpleNamespace(
            mjtObj=types.SimpleNamespace(mjOBJ_SITE=1),
            mj_name2id=lambda *_args: 4,
            mj_forward=lambda *_args: None,
        )
        control.mujoco = fake_mujoco
        executor = object.__new__(control.PickPlaceExecutor)
        executor.model = object()
        executor.data = types.SimpleNamespace(
            site_xpos=np.asarray(
                [[0.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 0.04]]
            ),
            xpos=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]]),
        )
        executor.config = control.ControllerConfig(release_gap=0.004)
        executor.object_body_ids = {"cube": 1}
        executor._object_half_height = lambda _name: 0.02

        release_height = executor._release_site_height("cube", support_z=0.0)

        self.assertAlmostEqual(release_height, 0.044)

    def test_new_robot_object_contact_is_reported(self):
        contact = types.SimpleNamespace(dist=-0.001, geom1=0, geom2=1)
        fake_mujoco = types.SimpleNamespace(
            mjtObj=types.SimpleNamespace(mjOBJ_BODY=1),
            mj_id2name=lambda _model, _kind, body_id: f"body_{body_id}",
        )
        control.mujoco = fake_mujoco
        executor = object.__new__(control.PickPlaceExecutor)
        executor.model = types.SimpleNamespace(geom_bodyid=np.asarray([1, 3]))
        executor.data = types.SimpleNamespace(ncon=1, contact=[contact])
        executor._robot_body_ids = {1}
        executor.object_body_ids = {"target": 2, "obstacle": 3}
        executor._held_object = None
        executor._baseline_robot_contacts = set()

        reason = executor._unexpected_collision(("target",))

        self.assertEqual(reason, "robot collision with body_3")

    def test_safe_path_uses_precision_only_near_target(self):
        executor = object.__new__(control.PickPlaceExecutor)
        executor.config = control.ControllerConfig()
        executor.current_pose = lambda: (np.asarray([0.4, 0.0, 0.25]), np.eye(3))
        calls = []

        def move_linear(target, **kwargs):
            calls.append((np.asarray(target), kwargs["profile"]))
            return control.MotionResult(True, 0.001, 0.001, 1)

        executor.move_linear = move_linear
        result = executor.move_via_safe_height([0.55, -0.1, 0.04])

        self.assertTrue(result.success)
        self.assertEqual(len(calls), 4)
        self.assertIs(calls[0][1], executor.config.transit_profile)
        self.assertIs(calls[2][1], executor.config.transit_profile)
        self.assertAlmostEqual(calls[2][0][2], 0.10)
        self.assertIs(calls[3][1], executor.config.precision_profile)
        self.assertAlmostEqual(calls[3][0][2], 0.04)

    def test_sources_compile(self):
        for relative in (
            "vgarm/mjc/controller.py",
            "vgarm/mjc/robots.py",
            "vgarm/cli.py",
        ):
            source = SOURCE_ROOT / relative
            compile(source.read_text(encoding="utf-8"), str(source), "exec")


if __name__ == "__main__":
    unittest.main()
