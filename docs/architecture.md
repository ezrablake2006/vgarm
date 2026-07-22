## 架构概览

目标是把系统拆成可替换的四层：环境重建 → 场景生成/仿真 → 指令理解 → 交互层（文本/语音/UI）。

### 模块边界

- `vgarm.reconstruction`
  - 输入：照片路径（未来）或场景JSON（当前MVP）
  - 输出：`SceneLayout`（物体列表、几何类型、尺寸/位置/颜色、粗略物理参数）
- `vgarm.mjc`
  - 输入：`SceneLayout` + `RobotSpec`
  - 输出：可加载的MJCF（MuJoCo XML）与可执行的抓取/放置动作
  - 抓取策略：通过 `equality/connect` 把末端位点与物体抓取位点绑定/解除
  - 运动控制：位置与姿态联合约束的阻尼最小二乘 IK
  - 安全机制：机器人 home 初始化、关节限位、安全高度路径、碰撞中止、抓放验证与失败恢复
  - 速度策略：自由空间、持物运输、精确抓放使用独立运动参数；目标最后 6 cm 自动减速
- `vgarm.nlu`
  - 输入：自然语言文本
  - 输出：结构化命令（当前仅支持“把/将…移到/放到…(左/右/前/后/中)”）
  - 未来可替换为：LLM、BERT分类器、Rasa意图槽位等
- `vgarm.speech` / `vgarm.ui`
  - 当前为占位接口，后续接入 Whisper/Vosk、TTS 以及 PyQt/DearImGui/Web 前端

### 关键约定

- 物体命名：`<name>`（body）与 `<name>_grab`（抓取site），对应连接约束 `attach_<name>`
- 末端位点：从 `RobotSpec.attachment_site_name` 指定；不同机器人可以映射到不同site
- 场景生成落盘：为保证Menagerie资产路径解析，生成的XML写入机器人模型目录（与其资产目录同级）
