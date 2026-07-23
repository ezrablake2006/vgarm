from __future__ import annotations


def build_schema(executor, object_names: tuple[str, ...]) -> dict:
    joint_names = [
        executor.model.joint(joint.joint_id).name for joint in executor._actuated
    ]
    actuator_names = [
        executor.model.actuator(joint.actuator_id).name for joint in executor._actuated
    ]
    return {
        "dataset_schema_version": "1.0",
        "time_alignment": "(o_t, a_t) -> o_{t+1}; sampled immediately before mj_step",
        "physics_timestep": float(executor.model.opt.timestep),
        "joint_names": joint_names,
        "joint_qpos_addresses": [item.qpos_address for item in executor._actuated],
        "joint_dof_addresses": [item.dof_address for item in executor._actuated],
        "actuator_names": actuator_names,
        "object_names": list(object_names),
        "attachment_site": executor.robot.attachment_site_name,
        "dimensions": {
            "joint": len(joint_names),
            "actuator": len(actuator_names),
            "qpos": int(executor.model.nq),
            "qvel": int(executor.model.nv),
            "canonical_state": 13 + 13 * len(object_names),
            "canonical_action": 7,
        },
        "action_representation": {
            "action.ctrl": "actual data.ctrl applied by the next mj_step",
            "action.joint_target": "controller commanded joint position target",
            "action.eef_target_position": "active Cartesian target or null",
            "action.eef_target_quaternion": "active Cartesian orientation target or null",
            "action.equality_command": "actual model equality activation vector",
        },
        "unavailable_fields": {
            "observation.actuator_state": "model.na == 0",
            "observation.gripper_state": "robots use equality grasp abstraction",
            "action.gripper_command": "no physical gripper actuator",
        },
    }
