# VGArm

VGArm 是基于 MuJoCo 的视觉引导机械臂交互原型。它将场景 JSON、中文指令解析、机器人场景构建和可靠抓放控制串联起来，可驱动 Franka、UR5e 和 Kinova 等机器人模型。

## 当前环境

- 项目路径：`~/projects/vgarm`
- 虚拟环境：`~/projects/vgarm/.venv`
- Python：3.14
- VGArm：0.1.2
- 机器人模型：`third_party/mujoco_menagerie/`

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
python -m pip install ./dist/vgarm-0.1.2-py3-none-any.whl
```

如果 `.venv` 已存在且 `vgarm --help` 正常，无需重复安装。

## 目录结构

- `dist/`：可安装的 VGArm wheel 发布包，当前版本为 `0.1.2`。
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

控制器遇到不可达路径、碰撞风险或收敛超时时会安全终止，并输出 `VGArm control aborted safely`。这表示安全机制主动中止，而不是程序崩溃。

