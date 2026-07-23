# VGArm Benchmark

Benchmark 复用正常的 VGArm 执行链：加载 `SceneLayout`、解析中文指令、生成
MJCF、初始化 `PickPlaceExecutor`、执行抓放动作，最后读取 MuJoCo 中的物体位置
验证任务。任务成功不是由“未抛异常”推断的。

## 实验协议

每个机器人独立执行 `--episodes` 个 episode。任务按 seed 确定的顺序循环抽取；
每个 episode 都重新加载场景和创建 MuJoCo 模型，episode seed 为全局 seed 加全局
episode 编号。默认提供五个解析器已支持的移动、相对放置和交换任务，也可通过
`--tasks` 读取 JSON 数组。

每次运行需要新的空输出目录。只有显式传入 `--overwrite` 才会覆盖同名结果文件。
`episodes.jsonl` 和 `episodes.csv` 在每个 episode 后刷新，因此后续 episode 失败或
用户中断不会丢失已经完成的记录。

Benchmark 默认使用 `--no-viewer`。`--viewer` 与 `--no-viewer` 互斥，
`--verbose` 仅控制日志详细度，与画面无关。viewer session 直接持有当前 episode
的 `MjModel`/`MjData`，其 `sync()` 从 `PickPlaceExecutor.step()` 调用，因此显示
的是实际控制和物理过程，而不是事后动画。每个 episode 使用独立 viewer context，
完成或异常后立即释放；用户关闭窗口会将当前 episode 记录为 `viewer_closed` 并
停止本次 viewer benchmark。

`--speed` 接受大于零的倍率，只调整每次 viewer sync 之间的墙钟等待，不跳过
MuJoCo step，也不改变控制器输入。高倍速仍受渲染和计算性能限制。批量正式评测
建议 headless；viewer 用于诊断和人工检查。WSL 需要 WSLg 才能显示窗口。

## 指标定义

- Overall success：通过最终几何谓词验证的 episode / 全部 episode。
- Parse success：成功解析指令的 episode / 全部 episode。
- Pick success：所有实际抓取动作完成抓取并通过抬升验证的 episode / 进入抓取阶段的 episode。
- Place success：所有实际放置动作完成控制器放置验证的 episode / 进入放置阶段的 episode。
- Collision rate：最终失败类别为 `collision` 的 episode / 全部 episode。
- Timeout rate：最终失败类别为 `timeout` 的 episode / 全部 episode。
- 平均值和中位数使用完整 episode 墙钟耗时。

没有进入相应阶段时，抓取或放置字段为 `null`，不进入该指标的分母。

## 几何验证和坐标系

MuJoCo 世界坐标为 `+X` 向前、`+Y` 向左、`+Z` 向上。相对方向谓词要求轴向
间距至少 0.05 m；中心任务要求物体到 `(0.55, 0.0)` 的距离不超过 0.04 m；
交换要求两个物体分别到对方初始位置的最大误差不超过 0.04 m。`lift` 在当前
控制器中表示抬起并放回原平面位置，因此同时依赖控制器的抬升验证和最终平面
位置误差。

## 随机种子与场景扰动

`--position-jitter N` 在每个物体原始 XY 位置附近施加最多 `N` 米的均匀扰动。
实现使用 episode seed 独立创建随机数生成器，将物体限制在
`X=[0.35, 0.75]`、`Y=[-0.30, 0.30]`，并保留 0.005 m 的物体间平面间隙。
原始 JSON 不会改变，实际位置保存在每条 episode 记录中。设为 `0` 可关闭。

相同代码版本、场景、任务、机器人、配置和 seed 会产生相同任务顺序与初始位置。
物理结果还可能受到 MuJoCo 版本、机器人资产版本、CPU 数值差异影响。

## 失败分类

稳定类别包括 `scene_load_failed`、`task_parse_failed`、`object_not_found`、
`unreachable`、`ik_failed`、`grasp_failed`、`object_dropped`、
`placement_failed`、`collision`、`timeout`、`verification_failed` 和
`unexpected_error`。结构化 `ControlFailure` 提供阶段与类别；未知异常保留
错误文本和 traceback。默认继续后续 episode，`--fail-fast` 可在首个失败后停止。
`KeyboardInterrupt` 不会被捕获。

## 输出

- `config.json`：完整实验配置和全局 seed。
- `episodes.jsonl`：逐行的完整 episode 记录。
- `episodes.csv`：便于数据分析的 episode 表。
- `object_positions`：兼容 v0.1.x 的初始位置字段，已弃用；新代码使用
  `initial_object_positions` 与 `final_object_positions`。
- `summary.json`：总体和分机器人聚合指标。
- `summary.csv`：每个机器人一行的指标。
- `summary.md`：可直接复制到 README/报告的 Markdown 表格。
- `summary_by_task.{json,csv,md}`：按任务聚合指标。
- `summary_robot_task_matrix.csv`：机器人 × 任务总体成功率矩阵。
- `benchmark.log`：逐 episode 调试记录。

`vgarm benchmark compare RUN_A RUN_B` 会忽略时间戳、耗时和输出目录，对 episode
seed、机器人、任务、初始/计划/最终位置、结果、失败类别和验证数据生成 SHA-256。

## Episode replay

```bash
vgarm benchmark replay RESULTS/episodes.jsonl --episode-id 60 --viewer
```

replay 同时读取同目录的 `config.json`，要求 episode 使用当前 schema 和相同 VGArm
版本。它从原始 scene 恢复物体定义，再以 `initial_object_positions` 覆盖 XY/XYZ，
并将已保存的 `planned_target_position` 作为执行目标；因此不会重新 jitter 或重新
搜索目标。完成后比较 seed、机器人、任务、指令、初始位置、计划目标、运输
waypoints、最终位置、结果、失败类别和 verification。旧记录缺少必要字段、场景
不存在、版本不一致或 episode id 无效时会明确拒绝重放。

## 已知限制

- 第一版只扰动物体 XY 位置，不随机化尺寸、质量、摩擦或相机。
- 工作区边界是当前示例桌面任务的保守范围，不代表所有机器人完整可达域。
- 控制器阶段轨迹以一次任务内所有抓放都成功为 episode 的阶段成功；交换任务包含三次抓放。
- benchmark 当前以 headless 方式执行；`--no-viewer` 明确表达正式评测的推荐模式。
- 复现机器人实验需要相同版本的 MuJoCo Menagerie 资产。
- 目标搜索使用轴对齐 primitive footprint 和确定性 1 cm 网格，不是通用运动规划器。
