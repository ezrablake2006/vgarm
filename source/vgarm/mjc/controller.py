from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional, Sequence

import mujoco
import numpy as np

from .robots import RobotSpec


@dataclass(frozen=True)
class TargetOffset:
    dx: float
    dy: float


@dataclass(frozen=True)
class MotionProfile:
    position_tolerance: float
    orientation_tolerance: float
    max_joint_speed: float
    max_tracking_error: float
    stable_steps: int
    max_steps: int
    waypoint_size: float


@dataclass(frozen=True)
class ControllerConfig:
    damping: float = 0.04
    orientation_weight: float = 0.35
    joint_limit_margin: float = 0.03
    safe_height: float = 0.25
    approach_clearance: float = 0.08
    precision_clearance: float = 0.06
    release_gap: float = 0.004
    settle_steps: int = 60
    transit_profile: MotionProfile = field(
        default_factory=lambda: MotionProfile(
            position_tolerance=0.009,
            orientation_tolerance=0.10,
            max_joint_speed=1.2,
            max_tracking_error=0.10,
            stable_steps=2,
            max_steps=900,
            waypoint_size=0.035,
        )
    )
    carry_profile: MotionProfile = field(
        default_factory=lambda: MotionProfile(
            position_tolerance=0.007,
            orientation_tolerance=0.08,
            max_joint_speed=0.95,
            max_tracking_error=0.075,
            stable_steps=2,
            max_steps=1100,
            waypoint_size=0.025,
        )
    )
    precision_profile: MotionProfile = field(
        default_factory=lambda: MotionProfile(
            position_tolerance=0.0045,
            orientation_tolerance=0.06,
            max_joint_speed=0.60,
            max_tracking_error=0.05,
            stable_steps=4,
            max_steps=1400,
            waypoint_size=0.015,
        )
    )


@dataclass(frozen=True)
class MotionResult:
    success: bool
    position_error: float
    orientation_error: float
    steps: int
    reason: str | None = None


@dataclass(frozen=True)
class ActuatedJoint:
    actuator_id: int
    joint_id: int
    qpos_address: int
    dof_address: int
    lower: float
    upper: float


@dataclass
class ActionStageResult:
    object_name: str
    pick_attempted: bool = False
    pick_success: bool = False
    place_attempted: bool = False
    place_success: bool = False
    failure_stage: str | None = None
    planned_target_position: list[float] | None = None
    transport_waypoints: list[list[float]] = field(default_factory=list)
    grasp_diagnostics: dict | None = None
    collision_diagnostics: dict | None = None


class ControlFailure(RuntimeError):
    """Raised when a motion, grasp, placement, or recovery check fails."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        category: str | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.category = category


_TARGET_OFFSETS: dict[str, TargetOffset] = {
    "左边": TargetOffset(dx=0.0, dy=0.15),
    "右边": TargetOffset(dx=0.0, dy=-0.15),
    "前面": TargetOffset(dx=0.10, dy=0.0),
    "后面": TargetOffset(dx=-0.10, dy=0.0),
    "中间": TargetOffset(dx=0.0, dy=0.0),
}


def rotation_error(current: Sequence[Sequence[float]], target: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a world-frame small-angle error from current to target."""
    current_matrix = np.asarray(current, dtype=float).reshape(3, 3)
    target_matrix = np.asarray(target, dtype=float).reshape(3, 3)
    return 0.5 * sum(
        np.cross(current_matrix[:, axis], target_matrix[:, axis])
        for axis in range(3)
    )


def clamp_to_joint_limits(
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    margin: float,
) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    lower_array = np.asarray(lower, dtype=float).copy()
    upper_array = np.asarray(upper, dtype=float).copy()
    finite_lower = np.isfinite(lower_array)
    finite_upper = np.isfinite(upper_array)
    lower_array[finite_lower] += margin
    upper_array[finite_upper] -= margin
    inverted = lower_array > upper_array
    if np.any(inverted):
        midpoint = (lower_array[inverted] + upper_array[inverted]) / 2.0
        lower_array[inverted] = midpoint
        upper_array[inverted] = midpoint
    return np.minimum(np.maximum(values_array, lower_array), upper_array)


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise KeyError(f"missing site: {name}")
    return site_id


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError(f"missing body: {name}")
    return body_id


def _eq_id(model: mujoco.MjModel, name: str) -> int:
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if equality_id < 0:
        raise KeyError(f"missing equality: {name}")
    return equality_id


def _ancestor_bodies(model: mujoco.MjModel, body_id: int) -> set[int]:
    ancestors: set[int] = set()
    current = body_id
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        current = int(model.body_parentid[current])
    return ancestors


def _actuated_joints(model: mujoco.MjModel, site_id: int) -> list[ActuatedJoint]:
    site_body = int(model.site_bodyid[site_id])
    ancestors = _ancestor_bodies(model, site_body)
    joints: list[ActuatedJoint] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or joint_id >= model.njnt:
            continue
        if int(model.jnt_bodyid[joint_id]) not in ancestors:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        if bool(model.jnt_limited[joint_id]):
            lower, upper = (float(value) for value in model.jnt_range[joint_id])
        else:
            lower, upper = -np.inf, np.inf
        joints.append(
            ActuatedJoint(
                actuator_id=actuator_id,
                joint_id=joint_id,
                qpos_address=int(model.jnt_qposadr[joint_id]),
                dof_address=int(model.jnt_dofadr[joint_id]),
                lower=lower,
                upper=upper,
            )
        )
    return joints


class PickPlaceExecutor:
    """Pose-controlled, joint-limited and collision-aware manipulation executor."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: RobotSpec,
        object_names: Iterable[str],
        sync: Callable[[], None] | None = None,
        config: ControllerConfig | None = None,
        step_recorder=None,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.config = config or ControllerConfig()
        if robot.transit_max_steps is not None:
            self.config = replace(
                self.config,
                transit_profile=replace(
                    self.config.transit_profile,
                    max_steps=robot.transit_max_steps,
                ),
            )
        self.attachment_site = _site_id(model, robot.attachment_site_name)
        self.object_names = tuple(object_names)
        self.eq_for_object = {
            name: _eq_id(model, f"attach_{name}") for name in self.object_names
        }
        self.object_body_ids = {
            name: _body_id(model, name) for name in self.object_names
        }
        self._object_name_by_body = {
            body_id: name for name, body_id in self.object_body_ids.items()
        }
        self._actuated = _actuated_joints(model, self.attachment_site)
        if not self._actuated:
            raise ControlFailure("no actuated joints found between robot base and attachment site")
        self._sync = sync
        self._step_recorder = step_recorder
        self._held_object: str | None = None
        self.execution_trace: list[ActionStageResult] = []
        self._initialize_home_pose()
        self.step(self.config.settle_steps)
        mujoco.mj_forward(self.model, self.data)
        self._safe_orientation = np.asarray(
            self.data.site_xmat[self.attachment_site], dtype=float
        ).reshape(3, 3).copy()
        self._home_qpos = self._joint_positions()
        self._robot_body_ids = _ancestor_bodies(
            model, int(model.site_bodyid[self.attachment_site])
        )
        self._baseline_robot_contacts = self._robot_contact_pairs()
        self._active_action: ActionStageResult | None = None
        self._motion_phase: str | None = None
        self._motion_waypoint_index: int | None = None
        if robot.grasp_orientation is not None:
            self._safe_orientation = np.asarray(
                robot.grasp_orientation, dtype=float
            ).reshape(3, 3)
            alignment = replace(
                self.config.precision_profile,
                max_steps=max(
                    robot.orientation_alignment_steps,
                    self.config.precision_profile.max_steps,
                ),
                max_joint_speed=0.9,
                orientation_tolerance=0.08,
            )
            position, _ = self.current_pose()
            result = self.move_attachment_to(
                position,
                self._safe_orientation,
                check_collisions=False,
                profile=alignment,
            )
            if not result.success:
                raise ControlFailure(
                    f"tool orientation alignment failed: {result.reason}",
                    stage="initialization",
                    category="ik_failed",
                )

    def _initialize_home_pose(self) -> None:
        home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        configured_home = self.robot.default_home_qpos
        for index, joint in enumerate(self._actuated):
            if home_key >= 0:
                value = float(self.model.key_qpos[home_key, joint.qpos_address])
            elif configured_home is not None and index < len(configured_home):
                value = float(configured_home[index])
            else:
                value = float(self.data.qpos[joint.qpos_address])
            value = float(
                clamp_to_joint_limits(
                    [value], [joint.lower], [joint.upper], self.config.joint_limit_margin
                )[0]
            )
            self.data.qpos[joint.qpos_address] = value
            self.data.qvel[joint.dof_address] = 0.0
            self.data.ctrl[joint.actuator_id] = value
        mujoco.mj_forward(self.model, self.data)
        self._commanded_qpos = self._joint_positions()

    def _joint_positions(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[joint.qpos_address] for joint in self._actuated],
            dtype=float,
        )

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([joint.lower for joint in self._actuated], dtype=float),
            np.asarray([joint.upper for joint in self._actuated], dtype=float),
        )

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            if self._step_recorder is not None:
                self._step_recorder.record_pre_step(self)
            mujoco.mj_step(self.model, self.data)
            if self._sync is not None:
                self._sync()

    def set_step_recorder(self, recorder) -> None:
        self._step_recorder = recorder

    def _body_name(self, body_id: int) -> str:
        if body_id == 0:
            return "world/floor"
        name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return name or f"body_{body_id}"

    def _contact_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            pairs.add(tuple(sorted((geom_a, geom_b))))
        return pairs

    def _robot_contact_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for pair in self._contact_pairs():
            body_a = int(self.model.geom_bodyid[pair[0]])
            body_b = int(self.model.geom_bodyid[pair[1]])
            if body_a in self._robot_body_ids or body_b in self._robot_body_ids:
                pairs.add(pair)
        return pairs

    def _unexpected_collision(
        self,
        allowed_object_names: Iterable[str] = (),
        allow_held_environment_contact: bool = False,
    ) -> str | None:
        allowed_bodies = {
            self.object_body_ids[name]
            for name in allowed_object_names
            if name in self.object_body_ids
        }
        held_body = (
            self.object_body_ids[self._held_object]
            if self._held_object is not None
            else None
        )
        for pair in self._contact_pairs():
            body_a = int(self.model.geom_bodyid[pair[0]])
            body_b = int(self.model.geom_bodyid[pair[1]])
            robot_a = body_a in self._robot_body_ids
            robot_b = body_b in self._robot_body_ids
            if robot_a or robot_b:
                if pair in self._baseline_robot_contacts:
                    continue
                if robot_a and robot_b:
                    message = (
                        f"robot self-collision: {self._body_name(body_a)} / "
                        f"{self._body_name(body_b)}"
                    )
                    self._record_collision(body_a, body_b)
                    return message
                other_body = body_b if robot_a else body_a
                if other_body == held_body or other_body in allowed_bodies:
                    continue
                self._record_collision(body_a, body_b)
                return f"robot collision with {self._body_name(other_body)}"
            if held_body is not None and held_body in (body_a, body_b):
                other_body = body_b if body_a == held_body else body_a
                if other_body not in self._robot_body_ids:
                    if allow_held_environment_contact:
                        continue
                    self._record_collision(body_a, body_b)
                    return (
                        f"held object collision: {self._body_name(held_body)} / "
                        f"{self._body_name(other_body)}"
                    )
        return None

    def _record_collision(self, body_a: int, body_b: int) -> None:
        if getattr(self, "_active_action", None) is None:
            return
        held = self._held_object
        obstacle_body = body_b if held and body_a == self.object_body_ids.get(held) else body_a
        obstacle = self._object_name_by_body.get(obstacle_body, self._body_name(obstacle_body))
        def position(body_id: int) -> list[float]:
            return [float(value) for value in self.data.xpos[body_id]]
        self._active_action.collision_diagnostics = {
            "held_object": held,
            "obstacle": obstacle,
            "phase": self._motion_phase,
            "waypoint_index": self._motion_waypoint_index,
            "held_object_position": position(self.object_body_ids[held]) if held else None,
            "obstacle_position": position(obstacle_body),
        }

    def current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        position = np.asarray(
            self.data.site_xpos[self.attachment_site], dtype=float
        ).copy()
        orientation = np.asarray(
            self.data.site_xmat[self.attachment_site], dtype=float
        ).reshape(3, 3).copy()
        return position, orientation

    def move_attachment_to(
        self,
        target_xyz: Sequence[float],
        target_orientation: Sequence[Sequence[float]] | None = None,
        *,
        allowed_object_names: Iterable[str] = (),
        check_collisions: bool = True,
        allow_held_environment_contact: bool = False,
        profile: MotionProfile | None = None,
        max_steps: int | None = None,
    ) -> MotionResult:
        motion_profile = profile or self.config.transit_profile
        target_position = np.asarray(target_xyz, dtype=float)
        orientation_target = (
            self._safe_orientation
            if target_orientation is None
            else np.asarray(target_orientation, dtype=float).reshape(3, 3)
        )
        allowed = tuple(allowed_object_names)
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        dof_indices = [joint.dof_address for joint in self._actuated]
        lower, upper = self._joint_limits()
        stable = 0
        position_norm = float("inf")
        orientation_norm = float("inf")
        limit = max_steps or motion_profile.max_steps

        for step_index in range(1, limit + 1):
            mujoco.mj_forward(self.model, self.data)
            if check_collisions:
                collision = self._unexpected_collision(
                    allowed,
                    allow_held_environment_contact=allow_held_environment_contact,
                )
                if collision is not None:
                    return MotionResult(
                        False,
                        position_norm,
                        orientation_norm,
                        step_index - 1,
                        collision,
                    )

            current_position = np.asarray(
                self.data.site_xpos[self.attachment_site], dtype=float
            )
            current_orientation = np.asarray(
                self.data.site_xmat[self.attachment_site], dtype=float
            ).reshape(3, 3)
            position_delta = target_position - current_position
            orientation_delta = rotation_error(
                current_orientation, orientation_target
            )
            position_norm = float(np.linalg.norm(position_delta))
            orientation_norm = float(np.linalg.norm(orientation_delta))
            self._current_eef_target = target_position.copy()
            self._current_eef_orientation_target = orientation_target.copy()
            self._ik_iteration = step_index
            self._position_error = position_norm
            self._orientation_error = orientation_norm
            self._ik_stopping_reason = None
            if (
                position_norm <= motion_profile.position_tolerance
                and orientation_norm <= motion_profile.orientation_tolerance
            ):
                stable += 1
                if stable >= motion_profile.stable_steps:
                    return MotionResult(
                        True,
                        position_norm,
                        orientation_norm,
                        step_index - 1,
                    )
            else:
                stable = 0

            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.attachment_site,
            )
            task_jacobian = np.vstack(
                (
                    jacobian_position[:, dof_indices],
                    self.config.orientation_weight
                    * jacobian_rotation[:, dof_indices],
                )
            )
            task_error = np.concatenate(
                (
                    position_delta,
                    self.config.orientation_weight * orientation_delta,
                )
            )
            regularized = (
                task_jacobian @ task_jacobian.T
                + self.config.damping**2 * np.eye(6)
            )
            task_velocity = task_jacobian.T @ np.linalg.solve(
                regularized, task_error
            )
            pseudo_inverse = task_jacobian.T @ np.linalg.solve(
                regularized, np.eye(6)
            )
            nullspace = np.eye(len(self._actuated)) - pseudo_inverse @ task_jacobian
            current_joints = self._joint_positions()
            centering_velocity = 0.15 * (self._home_qpos - current_joints)
            joint_velocity = task_velocity + nullspace @ centering_velocity
            joint_velocity = np.clip(
                joint_velocity,
                -motion_profile.max_joint_speed,
                motion_profile.max_joint_speed,
            )
            joint_targets = clamp_to_joint_limits(
                self._commanded_qpos
                + joint_velocity * float(self.model.opt.timestep),
                lower,
                upper,
                self.config.joint_limit_margin,
            )
            joint_targets = np.minimum(
                np.maximum(
                    joint_targets,
                    current_joints - motion_profile.max_tracking_error,
                ),
                current_joints + motion_profile.max_tracking_error,
            )
            self._commanded_qpos = joint_targets
            for joint, target in zip(self._actuated, joint_targets):
                self.data.ctrl[joint.actuator_id] = float(target)
            self.step()

        return MotionResult(
            False,
            position_norm,
            orientation_norm,
            limit,
            "pose convergence timeout",
        )

    def move_linear(
        self,
        target_xyz: Sequence[float],
        *,
        allowed_object_names: Iterable[str] = (),
        check_collisions: bool = True,
        allow_held_environment_contact: bool = False,
        profile: MotionProfile | None = None,
    ) -> MotionResult:
        motion_profile = profile or self.config.transit_profile
        start, _ = self.current_pose()
        target = np.asarray(target_xyz, dtype=float)
        distance = float(np.linalg.norm(target - start))
        waypoint_count = max(
            1, int(np.ceil(distance / motion_profile.waypoint_size))
        )
        total_steps = 0
        last = MotionResult(False, float("inf"), float("inf"), 0, "no waypoint")
        for index in range(1, waypoint_count + 1):
            waypoint = start + (target - start) * (index / waypoint_count)
            last = self.move_attachment_to(
                waypoint,
                allowed_object_names=allowed_object_names,
                check_collisions=check_collisions,
                allow_held_environment_contact=allow_held_environment_contact,
                profile=motion_profile,
            )
            total_steps += last.steps
            if not last.success:
                return MotionResult(
                    False,
                    last.position_error,
                    last.orientation_error,
                    total_steps,
                    f"cartesian waypoint {index}/{waypoint_count}: {last.reason}",
                )
        return MotionResult(
            True,
            last.position_error,
            last.orientation_error,
            total_steps,
        )

    def move_via_safe_height(
        self,
        target_xyz: Sequence[float],
        *,
        allowed_object_names: Iterable[str] = (),
        allowed_at_target: Iterable[str] = (),
        transit_profile: MotionProfile | None = None,
        target_profile: MotionProfile | None = None,
    ) -> MotionResult:
        transit_motion = transit_profile or self.config.transit_profile
        target_motion = target_profile or self.config.precision_profile
        target = np.asarray(target_xyz, dtype=float)
        current, _ = self.current_pose()
        safe_z = max(
            self.config.safe_height,
            float(current[2]),
            float(target[2]) + self.config.approach_clearance,
        )
        allowed = tuple(allowed_object_names)
        target_allowed = tuple(dict.fromkeys(allowed + tuple(allowed_at_target)))
        precision_start_z = min(
            safe_z,
            float(target[2]) + self.config.precision_clearance,
        )
        waypoints = (
            ([current[0], current[1], safe_z], allowed, transit_motion),
            ([target[0], target[1], safe_z], allowed, transit_motion),
            ([target[0], target[1], precision_start_z], allowed, transit_motion),
            (target, target_allowed, target_motion),
        )
        total_steps = 0
        last = MotionResult(False, float("inf"), float("inf"), 0, "no waypoint")
        segment_names = ("raise", "translate", "approach", "precision-descend")
        for segment_name, (waypoint, waypoint_allowed, waypoint_profile) in zip(
            segment_names, waypoints
        ):
            self._motion_phase = segment_name.replace("-", "_")
            active_action = getattr(self, "_active_action", None)
            self._motion_waypoint_index = len(
                active_action.transport_waypoints
            ) if active_action is not None else None
            if active_action is not None:
                self._active_action.transport_waypoints.append(
                    [float(value) for value in waypoint]
                )
            last = self.move_linear(
                waypoint,
                allowed_object_names=waypoint_allowed,
                profile=waypoint_profile,
            )
            total_steps += last.steps
            if not last.success:
                return MotionResult(
                    False,
                    last.position_error,
                    last.orientation_error,
                    total_steps,
                    f"safe-path {segment_name}: {last.reason}",
                )
        return MotionResult(
            True,
            last.position_error,
            last.orientation_error,
            total_steps,
        )

    def _require(self, result: MotionResult, segment: str) -> None:
        if not result.success:
            reason = result.reason or "motion failed"
            if "collision" in reason:
                category = "collision"
            elif "timeout" in reason:
                category = "timeout"
            else:
                category = "ik_failed"
            # Collision geometry is captured when contact is detected; attach
            # numerical solver state without parsing console output.
            active = getattr(self, "_active_action", None)
            if active is not None and active.collision_diagnostics is not None:
                active.collision_diagnostics["position_error"] = result.position_error
                active.collision_diagnostics["orientation_error"] = result.orientation_error
            raise ControlFailure(
                f"{segment} failed: {reason}; "
                f"position_error={result.position_error:.4f}, "
                f"orientation_error={result.orientation_error:.4f}",
                stage=segment,
                category=category,
            )

    def body_xy(self, body_name: str) -> tuple[float, float]:
        body_id = _body_id(self.model, body_name)
        mujoco.mj_forward(self.model, self.data)
        position = self.data.xpos[body_id]
        return float(position[0]), float(position[1])

    def _object_half_height(self, object_name: str) -> float:
        body_id = self.object_body_ids[object_name]
        heights: list[float] = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != body_id:
                continue
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                heights.append(float(self.model.geom_size[geom_id, 0]))
            elif geom_type in (
                int(mujoco.mjtGeom.mjGEOM_BOX),
                int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            ):
                heights.append(float(self.model.geom_size[geom_id, 2]))
            elif geom_type in (
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            ):
                heights.append(float(self.model.geom_size[geom_id, 1]))
        if not heights:
            raise ControlFailure(f"object has no supported collision geometry: {object_name}")
        return max(heights)

    def _release_site_height(self, object_name: str, support_z: float = 0.0) -> float:
        body_id = self.object_body_ids[object_name]
        grab_site = _site_id(self.model, f"{object_name}_grab")
        mujoco.mj_forward(self.model, self.data)
        grab_offset = float(
            self.data.site_xpos[grab_site, 2] - self.data.xpos[body_id, 2]
        )
        return (
            support_z
            + self._object_half_height(object_name)
            + grab_offset
            + self.config.release_gap
        )

    def _recover_upward(self, allowed_object_names: Iterable[str] = ()) -> None:
        current, _ = self.current_pose()
        target = [
            current[0],
            current[1],
            max(self.config.safe_height, float(current[2]) + 0.12),
        ]
        self.move_linear(
            target,
            allowed_object_names=allowed_object_names,
            check_collisions=False,
            profile=self.config.carry_profile,
        )

    def _recover_attached_object(
        self,
        object_name: str,
        original_xy: tuple[float, float],
        release_height: float,
    ) -> None:
        equality_id = self.eq_for_object[object_name]
        try:
            self._recover_upward((object_name,))
            recovery_target = [original_xy[0], original_xy[1], release_height]
            recovery = self.move_via_safe_height(
                recovery_target,
                allowed_object_names=(object_name,),
                allowed_at_target=(object_name,),
                transit_profile=self.config.carry_profile,
                target_profile=self.config.precision_profile,
            )
            if not recovery.success:
                raise ControlFailure(recovery.reason or "recovery motion failed")
        finally:
            self.data.eq_active[equality_id] = 0
            self._held_object = None
            mujoco.mj_forward(self.model, self.data)
            self.step(30)
            self._recover_upward()

    def pick_and_place(
        self,
        object_name: str,
        target_xy: tuple[float, float],
        base_h: float = 0.02,
    ) -> None:
        del base_h  # Object geometry determines the correct release height.
        if object_name not in self.object_body_ids:
            raise KeyError(object_name)
        grab_site = _site_id(self.model, f"{object_name}_grab")
        equality_id = self.eq_for_object[object_name]
        original_xy = self.body_xy(object_name)
        body_id = self.object_body_ids[object_name]
        release_height = self._release_site_height(object_name, support_z=0.0)
        attached = False
        action = ActionStageResult(object_name=object_name, pick_attempted=True)
        self.execution_trace.append(action)
        self._active_action = action
        self._control_skill = "pick_place"

        try:
            mujoco.mj_forward(self.model, self.data)
            grasp_position = np.asarray(
                self.data.site_xpos[grab_site], dtype=float
            ).copy()
            approach = self.move_via_safe_height(
                grasp_position,
                allowed_at_target=(object_name,),
                transit_profile=self.config.transit_profile,
                target_profile=self.config.precision_profile,
            )
            self._require(approach, "grasp approach")
            refinement = self.move_attachment_to(
                grasp_position,
                allowed_object_names=(object_name,),
                profile=self.config.precision_profile,
            )
            self._require(refinement, "grasp refinement")
            current_position, _ = self.current_pose()
            mujoco.mj_forward(self.model, self.data)
            actual_position = np.asarray(current_position, dtype=float)
            live_target = np.asarray(self.data.site_xpos[grab_site], dtype=float)
            grasp_error = float(
                np.linalg.norm(actual_position - live_target)
            )
            lower, upper = self._joint_limits()
            joints = self._joint_positions()
            actual_orientation = self.current_pose()[1]
            action.grasp_diagnostics = {
                "robot": self.robot.robot_id,
                "robot_model_path": str(self.robot.include_xml_path),
                "attachment_site_name": self.robot.attachment_site_name,
                "object_grab_site_name": f"{object_name}_grab",
                "target_tcp_position": live_target.tolist(),
                "actual_tcp_position": actual_position.tolist(),
                "position_error_vector": (live_target - actual_position).tolist(),
                "position_error_norm": grasp_error,
                "target_orientation": self._safe_orientation.tolist(),
                "actual_orientation": actual_orientation.tolist(),
                "orientation_error": float(
                    np.linalg.norm(rotation_error(actual_orientation, self._safe_orientation))
                ),
                "joint_positions": joints.tolist(),
                "joint_limits": list(zip(lower.tolist(), upper.tolist())),
                "at_joint_limit": bool(
                    np.any(joints <= lower + self.config.joint_limit_margin + 1e-6)
                    or np.any(joints >= upper - self.config.joint_limit_margin - 1e-6)
                ),
                "ik_iterations": refinement.steps,
                "ik_stopping_reason": refinement.reason or "converged",
                "position_task_weight": 1.0,
                "orientation_task_weight": self.config.orientation_weight,
                "damping": self.config.damping,
            }
            if grasp_error > 0.006:
                raise ControlFailure(
                    f"grasp verification failed: site error={grasp_error:.4f}",
                    stage="grasp",
                    category="grasp_failed",
                )

            initial_object_z = float(self.data.xpos[body_id, 2])
            self.data.eq_active[equality_id] = 1
            self._held_object = object_name
            self._motion_phase = "grasp"
            attached = True
            mujoco.mj_forward(self.model, self.data)
            self.step(20)

            current_position, _ = self.current_pose()
            lift_target = [
                current_position[0],
                current_position[1],
                max(
                    self.config.safe_height,
                    float(current_position[2]) + 0.15,
                ),
            ]
            lift = self.move_linear(
                lift_target,
                allowed_object_names=(object_name,),
                allow_held_environment_contact=True,
                profile=self.config.carry_profile,
            )
            self._require(lift, "lift")
            mujoco.mj_forward(self.model, self.data)
            if float(self.data.xpos[body_id, 2]) < initial_object_z + 0.05:
                raise ControlFailure(
                    "lift verification failed: object did not rise",
                    stage="lift",
                    category="object_dropped",
                )
            action.pick_success = True

            place_target = [target_xy[0], target_xy[1], release_height]
            self._motion_phase = "transport"
            action.planned_target_position = [float(value) for value in place_target]
            action.place_attempted = True
            transport = self.move_via_safe_height(
                place_target,
                allowed_object_names=(object_name,),
                allowed_at_target=(object_name,),
                transit_profile=self.config.carry_profile,
                target_profile=self.config.precision_profile,
            )
            self._require(transport, "transport/place")

            self.data.eq_active[equality_id] = 0
            self._held_object = None
            self._motion_phase = "release"
            attached = False
            mujoco.mj_forward(self.model, self.data)
            self._motion_phase = "settle"
            self.step(self.config.settle_steps)

            final_xy = np.asarray(self.body_xy(object_name))
            target_error = float(
                np.linalg.norm(final_xy - np.asarray(target_xy, dtype=float))
            )
            final_z = float(self.data.xpos[body_id, 2])
            half_height = self._object_half_height(object_name)
            if target_error > 0.03 or final_z < half_height * 0.75:
                raise ControlFailure(
                    f"placement verification failed: xy_error={target_error:.4f}, "
                    f"z={final_z:.4f}",
                    stage="placement",
                    category="placement_failed",
                )
            action.place_success = True
            self._recover_upward()
        except Exception as error:
            if isinstance(error, ControlFailure):
                action.failure_stage = error.stage
            if attached:
                self._recover_attached_object(
                    object_name,
                    original_xy,
                    release_height,
                )
            else:
                self.data.eq_active[equality_id] = 0
                self._held_object = None
                mujoco.mj_forward(self.model, self.data)
                self._recover_upward()
            raise
        finally:
            self._active_action = None

    def resolve_target_xy(
        self,
        keyword: str,
        anchor_xy: tuple[float, float] = (0.55, 0.0),
    ) -> tuple[float, float]:
        offset = _TARGET_OFFSETS.get(keyword, TargetOffset(0.0, 0.0))
        return anchor_xy[0] + offset.dx, anchor_xy[1] + offset.dy

    def lift_object(self, object_name: str, base_h: float = 0.02) -> None:
        xy = self.body_xy(object_name)
        self.pick_and_place(object_name, xy, base_h=base_h)

    def move_next_to(
        self,
        object_name: str,
        ref_object: str,
        direction: str,
        base_h: float = 0.02,
    ) -> None:
        reference_xy = self.body_xy(ref_object)
        target = self.resolve_target_xy(direction, anchor_xy=reference_xy)
        self.pick_and_place(object_name, target, base_h=base_h)

    def swap_objects(
        self,
        first: str,
        second: str,
        base_h: float = 0.02,
        stage_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        first_xy = self.body_xy(first)
        second_xy = self.body_xy(second)
        if stage_xy is None:
            stage_xy = (0.55, 0.22)
        self.pick_and_place(first, stage_xy, base_h=base_h)
        self.pick_and_place(second, first_xy, base_h=base_h)
        self.pick_and_place(first, second_xy, base_h=base_h)
