# VGArm Trajectory Dataset v1.1

VGArm 0.4.0 records the same `MjModel`, `MjData`, controller outputs and physics
steps used by normal tasks and benchmarks. Optional RGB comes from named MuJoCo
cameras on that same model/data. It does not reconstruct trajectories from
waypoints or logs, and RGB does not control the robot.

## Time alignment

`PickPlaceExecutor.step()` is the single hook. For physical step `t`, the
controller first writes `data.ctrl`; the recorder then samples observation
`o_t`, the actual pending control `a_t`, Cartesian/joint targets and structured
phase fields; exactly one `mujoco.mj_step()` follows. The next row is `o_{t+1}`.
Optional RGB is rendered at that same pre-step moment. A sampled row means:

```text
(I_t, o_t, a_t) -> o_{t+1}
```

The final observation is stored separately in the final NPZ. `timestamp` is
MuJoCo simulation time and adjacent rows differ by `model.opt.timestep`.

RGB uses an exact rational schedule relative to the first recorded row. Frame
zero is sampled on row zero. For each later ideal time `n / rgb_fps`, the first
physics row at or after that time is selected. Error stays in
`[0, physics_timestep)` without cumulative drift. With 46,579 rows at 0.002
seconds and 20 Hz, this endpoint rule produces 1,864 frames.

## Generate and use

```bash
vgarm dataset generate \
  --scene examples/basic_scene.json \
  --tasks examples/trajectory_tasks.json \
  --robots franka_fr3 --episodes 3 --seed 42 \
  --position-jitter 0.03 --modalities state --no-viewer \
  --output datasets/vgarm_fr3_state_v1

vgarm dataset inspect datasets/vgarm_fr3_state_v1
vgarm dataset validate datasets/vgarm_fr3_state_v1
vgarm dataset stats datasets/vgarm_fr3_state_v1
vgarm dataset replay datasets/vgarm_fr3_state_v1 --episode-id 0 --no-viewer
```

```bash
pip install "vgarm[rgb]"
MUJOCO_GL=egl vgarm dataset generate \
  --scene examples/basic_scene.json \
  --tasks examples/trajectory_tasks.json \
  --robots franka_fr3 --episodes 1 --seed 42 \
  --position-jitter 0.03 --modalities state,rgb \
  --cameras camera_front --rgb-width 640 --rgb-height 480 --rgb-fps 20 \
  --no-viewer --output datasets/vgarm_fr3_rgb_v1
```

`--overwrite` deletes an existing output dataset; `--resume` instead requires
the exact stored configuration fingerprint and skips completed episode IDs.
They are mutually exclusive. A multi-robot request creates one independently
schematized child dataset per robot plus a top-level manifest.

## Directory and atomic completion

```text
root/
  meta/dataset.json schema.json tasks.jsonl episodes.jsonl stats.json
  data/episode_000000.parquet
  states/episode_000000_initial.npz
  states/episode_000000_final.npz
  logs/
  videos/episode_000000/camera_front.mp4
  manifest.json
  summary.md
```

An episode is first written below `.incomplete/`. Parquet and both NPZ files
are checksummed and atomically renamed before a `completed: true` metadata line
is fsynced to `episodes.jsonl`. Validation and statistics ignore incomplete
episodes. Resume removes and deterministically regenerates only the incomplete
episode.

RGB is streamed first to
`.incomplete/episode_XXXXXX/videos/<camera>.tmp.mp4`. Writers are closed and
flushed, and the first/middle/last frame plus total count are decoded before
formal files are moved. Video checksums and metadata are finalized before the
completed episode line is fsynced.

## Schema and action meaning

Rows contain frame/simulation indices, simulation timestamp and timestep;
actuated joint positions and velocities; actual `data.ctrl`; end-effector pose
and spatial velocity; ordered object poses and velocities; held-object state;
low-level and canonical actions; controller skill, phase, waypoint, IK and
error fields; and terminal status.

Schema 1.1 adds nullable `visual_observation.rgb_frame_index` and
`rgb_timestamp`. Only sampled rows contain them. Indices start at zero and are
continuous; all cameras share the same index. Video frame N therefore joins
uniquely to one physics row, `o_t` and the actual `a_t`.

- `action.ctrl`: the exact actuator vector present in `data.ctrl` immediately
  before the next `mj_step`; replay uses this field directly.
- `action.joint_target`: the controller's commanded actuated-joint target. It
  is not copied from the measured joint position.
- `action.eef_target_position/quaternion`: the active structured Cartesian
  target, or null when none exists.
- `action.equality_command`: the actual equality-constraint activation vector.
- `action.gripper_command`: null in v1 because current robots use equality
  constraints rather than physical gripper actuators.

`observation.actuator_state` is null when MuJoCo reports `model.na == 0`;
`gripper_state` is likewise declared unavailable instead of being filled with
zeros. `canonical_state` contains EEF pose/velocity followed by ordered object
pose/velocity. `canonical_action` contains the Cartesian target when one is
defined.

Each native root permits only one robot, so joint and actuator arrays have a
fixed, explicit ordering. They are never silently padded across embodiments.

## Complete physics state and replay

Initial and final NPZ files use `mj_stateSize`, `mj_getState` and
`mj_setState` with `mjSTATE_INTEGRATION`. They additionally store initial
`ctrl`, equality activation, names and a JSON observation snapshot. No Python
`MjModel` or `MjData` object is pickled.

Low-level replay rebuilds the hashed scene/model, restores the saved integration
state, applies each saved `action.ctrl`, mirrors recorded equality changes and
calls one `mj_step` per row. It never reruns IK. The current absolute drift
tolerance is 0.15 mm for qpos, qvel, EEF and object position. It covers
deterministic solver warm-start reconstruction differences while remaining far
below the manipulation verification tolerance. A mismatch returns a non-zero
exit status.

## Validation and statistics

Validation checks schema/version, files and SHA-256, unique episode IDs,
continuous indices, strictly timestep-aligned timestamps, finite values,
shapes/orderings, terminal flags, success metadata and initial/final snapshots.
It prints PASS/FAIL and returns non-zero on failure.

Statistics include episode/success counts, physics steps, simulation/wall
duration, throughput/storage, task and phase counts, joint/action moments,
object workspace and average episode length. Wall-clock values are excluded
from deterministic fingerprints.

## RGB compatibility and LeRobot

RGB uses `imageio` and its bundled `imageio-ffmpeg` executable. Output is MP4,
H.264, yuv420p. Renderer and writer are reused for the episode; only scheduled
rows render, and frames are never accumulated in Python memory.

Dataset/schema 1.0 from VGArm 0.3.0 remains readable, validatable, stat-able and
replayable. It has no visual column or video requirement. Schema 1.1 makes RGB
optional, so state-only installation still needs no codec package. Validator
checks checksums, decoding, frame count, resolution, camera sets, continuous
Parquet mappings, matching simulation timestamps and orphaned formal videos.

LeRobot is optional:

```bash
pip install "vgarm[lerobot]"
vgarm dataset export-lerobot NATIVE --output LEROBOT
```

The exporter uses the official `LeRobotDataset.create()`, `add_frame()`,
`save_episode()` and `finalize()` API, then reloads the result with the official
loader. It never constructs a look-alike directory. LeRobot is not installed
in the project Python 3.14 validation environment, so conversion should be run
in a separate LeRobot-supported environment. Native recording remains the
source of truth.
