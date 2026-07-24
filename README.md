# VGArm

VGArm 0.4.1 can record synchronized RGB, metric depth (meters), and raw
MuJoCo segmentation object IDs/types from named virtual cameras. Visual
frames, simulator state, and the action used by the next physics step are
strictly pre-step aligned. These are dataset-recording capabilities: the
controller still uses simulator ground-truth state. RGB/depth/segmentation do
not participate in control; visual closed-loop control, object detection,
behavior cloning, world models, and VLA policies are not implemented.

VGArm 是一个基于 MuJoCo 的轻量级、可复现、语言条件多机械臂操作与轨迹数据生成工具。它将场景 JSON、中文指令解析、机器人场景构建、可靠抓放控制、benchmark 和逐物理步状态—动作记录串联起来，可驱动 Franka、UR5e 和 Kinova 等机器人模型。

## 当前环境

- 项目路径：`~/projects/vgarm`
- 虚拟环境：`~/projects/vgarm/.venv`
- Python：3.14.4
- MuJoCo：3.10.0
- VGArm：0.4.0
- MuJoCo Menagerie：项目内 `third_party/mujoco_menagerie/` 快照（该快照未携带独立 commit 元数据）

激活环境并配置模型路径：

```bash
cd ~/projects/vgarm
source .venv/bin/activate
export VGARM_MENAGERIE_ROOT="$PWD/third_party/mujoco_menagerie"
```

检查环境：

```bash
python --version
vgarm --help
```

## 首次安装或重建环境

```bash
cd ~/projects/vgarm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./dist/vgarm-0.4.0-py3-none-any.whl
```

如果 `.venv` 已存在且 `vgarm --help` 正常，无需重复安装。

## 目录结构

- `dist/`：可安装的 VGArm wheel 发布包，当前版本为 `0.4.0`。
- `source/`：Python 源代码、测试和开发文档。
- `examples/`：可直接运行的场景 JSON。
- `docs/`：架构、场景格式和发布说明。
- `third_party/mujoco_menagerie/`：VGArm 使用的 MuJoCo Menagerie 机器人模型。

## 启动

启动带 MuJoCo 窗口的演示：

```bash
cd ~/projects/vgarm
source .venv/bin/activate
export VGARM_MENAGERIE_ROOT="$PWD/third_party/mujoco_menagerie"

vgarm --robot franka_fr3 \
  --scene ./examples/basic_scene.json \
  --cmd "交换红色方块和蓝色方块位置"
```

无窗口运行：

```bash
vgarm --robot franka_fr3 \
  --scene ./examples/basic_scene.json \
  --cmd "交换红色方块和蓝色方块位置" \
  --no-viewer
```

支持的机器人标识：

- `franka_fr3`
- `franka_panda`
- `ur5e`
- `kinova_gen3`

## 源码开发与测试

直接执行 `vgarm` 使用的是安装在 `.venv` 中的 wheel。需要立即运行 `source/` 中尚未重新打包的修改时，使用：

```bash
cd ~/projects/vgarm
source .venv/bin/activate
export VGARM_MENAGERIE_ROOT="$PWD/third_party/mujoco_menagerie"

PYTHONPATH=./source python -m vgarm.cli \
  --robot franka_fr3 \
  --scene ./examples/basic_scene.json \
  --cmd "把红色方块移到左边"
```

运行单元测试：

```bash
cd ~/projects/vgarm
source .venv/bin/activate
PYTHONPATH=./source python -m unittest discover \
  -s source/vgarm_app/tests \
  -p 'test_*.py'
```

控制器遇到不可达路径、碰撞风险或收敛超时时会安全终止，并输出 `VGArm control aborted safely`。

## Trajectory Dataset

v0.4.0 可以把真实 VGArm 控制循环保存为版本化的逐物理步状态—动作和
MuJoCo 虚拟相机 RGB 数据。state-only 不需要视频依赖；RGB 需要：

```bash
python -m pip install ".[rgb]"
```

```bash
vgarm dataset generate \
  --scene ./examples/basic_scene.json \
  --tasks ./examples/trajectory_tasks.json \
  --robots franka_fr3 --episodes 3 --seed 42 \
  --position-jitter 0.03 --modalities state --no-viewer \
  --output ./datasets/vgarm_fr3_state_v1

vgarm dataset inspect ./datasets/vgarm_fr3_state_v1
vgarm dataset validate ./datasets/vgarm_fr3_state_v1
vgarm dataset stats ./datasets/vgarm_fr3_state_v1
vgarm dataset replay ./datasets/vgarm_fr3_state_v1 --episode-id 0 --no-viewer
vgarm dataset export-lerobot ./datasets/vgarm_fr3_state_v1 \
  --output ./datasets/vgarm_fr3_lerobot
```

RGB 录制：

```bash
MUJOCO_GL=egl vgarm dataset generate \
  --scene examples/basic_scene.json \
  --tasks examples/trajectory_tasks.json \
  --robots franka_fr3 --episodes 1 --seed 42 \
  --position-jitter 0.03 --modalities state,rgb \
  --cameras camera_front --rgb-width 640 --rgb-height 480 --rgb-fps 20 \
  --no-viewer --output datasets/vgarm_fr3_rgb_v1
```

每行严格表示 `(o_t, a_t) → o_{t+1}`：`action.ctrl` 是下一次
`mujoco.mj_step()` 真正使用的 `data.ctrl`，不是关节观测或 waypoint。
最后 observation 与完整 `mjSTATE_INTEGRATION` 初末状态单独保存。一个原生
dataset 只包含一种机器人；多机器人命令会生成各自独立 schema 的子数据集。

`--resume` 只接受完全相同的配置 fingerprint，并跳过已经原子完成的 episode。
RGB 与对应 `o_t/a_t` 在同一个 pre-step 采样；控制器仍使用模拟器真值，
RGB 不参与控制。LeRobot 是可选依赖；导出器按官方 v3 API 实现，并设计为
在导出后使用官方 loader 重载检查，但尚未在当前 Python 3.14 环境中完成
实际兼容性验证。
完整字段、重放容差、校验规则和已知限制见
[Trajectory Dataset 文档](docs/trajectory_dataset.md)。

v0.4.0 尚未包含视觉检测、视觉闭环、世界模型、行为克隆或其他学习型策略。

## Benchmark

Benchmark 结果在本地实际执行 MuJoCo episode 后生成，不在代码或 README 中硬编码。

快速 headless smoke test：

```bash
vgarm benchmark \
  --scene ./examples/basic_scene.json \
  --robots franka_fr3 \
  --episodes 3 \
  --seed 42 \
  --no-viewer \
  --output ./benchmark_results/smoke
```

Benchmark 默认 headless；`--verbose` 只增加诊断日志，不会打开画面。人工检查真实
MuJoCo 执行过程时显式使用 `--viewer`：

```bash
vgarm benchmark \
  --scene ./examples/basic_scene.json \
  --tasks ./examples/benchmark_tasks.json \
  --robots kinova_gen3 \
  --episodes 1 \
  --seed 42 \
  --position-jitter 0.03 \
  --viewer \
  --speed 1.0 \
  --output ./benchmark_results/visual_kinova
```

viewer 使用 benchmark 实际执行的同一个 `MjModel` 和 `MjData`，在每个物理 step
同步 approach、grasp、lift、transport、precision descend、release 和 settle。
批量正式评测仍建议使用默认 headless 模式。WSL 环境显示窗口需要 WSLg。

多机器人正式评测：

```bash
vgarm benchmark \
  --scene ./examples/basic_scene.json \
  --tasks ./examples/benchmark_tasks.json \
  --robots franka_fr3,franka_panda,ur5e,kinova_gen3 \
  --episodes 20 \
  --seed 42 \
  --position-jitter 0.03 \
  --no-viewer \
  --output ./benchmark_results/run_seed42
```

输出目录包含 `config.json`、增量写入的 `episodes.jsonl` 和
`episodes.csv`、机器可读的 `summary.json`、分机器人 `summary.csv`、
按任务报告 `summary_by_task.{json,csv,md}`、机器人/任务矩阵
`summary_robot_task_matrix.csv`、可直接复制到 README 或报告的 `summary.md`，
以及 `benchmark.log`。输出目录非空
时命令默认拒绝覆盖；确认替换结果时显式使用 `--overwrite`。

指标口径：

- 总体/解析/碰撞/超时率的分母均为全部 episode。
- 抓取成功率的分母仅为实际进入抓取阶段的 episode。
- 放置成功率的分母仅为实际进入放置阶段的 episode。
- 最终成功必须通过方向、中心或交换位置的几何谓词验证。

相同代码和资产版本、场景、配置及 seed 会复现任务顺序和场景扰动。完整实验协议、
失败分类、坐标定义和已知限制参见 [Benchmark 文档](docs/benchmark.md)。

比较两次运行的确定性字段：

```bash
vgarm benchmark compare benchmark_results/repro_a benchmark_results/repro_b
```

精确重放已保存 episode：

```bash
vgarm benchmark replay \
  ./benchmark_results/v020_formal_4x20/episodes.jsonl \
  --episode-id 60 \
  --viewer \
  --speed 1.0
```

replay 直接恢复记录中的机器人、任务、episode seed、实际初始物体位置和调整后的
计划目标，不会重新随机抽样。`--speed` 只改变 viewer 墙钟同步节奏，物理 timestep
和控制 step 数保持不变。

### v0.2.0 正式回归结果

以下结果由本机 Python 3.14.4、MuJoCo 3.10.0 和项目内 Menagerie 快照真实运行
生成。配置为每机器人 20 episodes、seed 42、position jitter 0.03 m，共 80 条；
完整原始结果位于 `benchmark_results/v020_formal_4x20/`。

| Robot | Episodes | Parse success | Pick success | Place success | Overall success | Collision rate | Timeout rate | Average duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| franka_fr3 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 14.476s |
| franka_panda | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 13.957s |
| kinova_gen3 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 20.511s |
| ur5e | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 16.620s |

这些数字只描述上述固定场景、任务集和版本，不代表更广泛物体、障碍物或工作区上的性能。
