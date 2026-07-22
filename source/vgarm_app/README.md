## 视觉引导的智能机械臂交互系统（MuJoCo环境生成与操作）- 原型骨架

这个目录提供一个可运行的最小原型（MVP）骨架，用于把“环境重建（先以JSON替代视觉模型）→ MuJoCo 场景生成 → 自然语言指令解析 → 机械臂抓取/放置执行”串起来。

### 快速运行

1. 安装依赖（建议在虚拟环境中）：

```bash
python -m pip install -r vgarm_app/requirements.txt
```

2. 运行一次演示（会启动MuJoCo窗口）：

```bash
python -m vgarm.cli --robot franka_fr3 --scene vgarm_app/examples/scenes/sample_scene.json --cmd "把红色方块移到左边"
```

### 当前实现范围（MVP）

- 环境重建：读取 `sample_scene.json`（后续可替换为 DETR/SAM/MiDaS 管线输出）
- 场景生成：自动拼接机器人模型 + 物体 + 抓取连接约束（connect equality）
- 指令理解：中文规则解析（颜色 + 物体类别 + “移到/放到” + 方位）
- 物体操作：位置/姿态阻尼最小二乘 IK、关节限位、安全高度路径、碰撞中止与连接约束抓取

### 目录结构

- `vgarm/`：核心Python包
  - `reconstruction/`：环境重建接口与数据结构（当前为JSON/占位实现）
  - `nlu/`：自然语言解析（当前为规则解析）
  - `mjc/`：MuJoCo 场景生成、IK、执行器
- `examples/`：示例场景
- `tests/`：基础单元测试（标准库 `unittest`）

