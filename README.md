# SwitchDoor 三关环境

这是一个独立、确定性的三关 SwitchDoor 环境与数据生成器，用于研究隐藏规则
发现、跨关规则使用和闭环规划。仓库只包含环境、精确求解器、数据生成器和本地
Web 体验端，不包含 JEPA、LLM 或训练代码。

这个包一次生成一个完整的三关 episode：

1. **L1（7×7）**：执行一次红开关探测，从真实门变化中区分同色/异色映射；
2. **L2（7×7）**：目标位于红门后，使用同一隐藏映射选择正确开关；
3. **L3（11×11）**：保持相同对象和规则，只增加地图尺寸与导航长度。

门只会从关闭变为永久打开。一个 episode 的三关共享同一个隐藏
`switch_mapping`，每关开始时位置与门状态重置，但公开历史可以跨关保留。

## 最短运行

```bash
git clone https://github.com/zixuangui-rgb/SwitchDoor.git
cd SwitchDoor
python -m pip install -e .
switchdoor-three-level audit-config
switchdoor-three-level generate --output /tmp/switchdoor_three_level --seed 20260826
switchdoor-three-level validate --input /tmp/switchdoor_three_level
```

要求 Python 3.11 或更高版本；环境本身没有第三方运行时依赖。

本地交互体验端不需要额外运行时依赖：

```bash
switchdoor-three-level play-web --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。默认模式会为每个新回合随机选择并隐藏
`switch_mapping`；“固定同色/异色”只用于练习。方向键或 WASD 移动，空格键
执行 `INTERACT`。L1/L2 到达目标后需点击“进入下一关”，L3 完成后网页才揭示
本回合隐藏规则。

Web JSON 明确分为两个命名空间：`observation` 只包含由注册公开观察派生的当前
PNG、可用动作、状态和上一步动作；`runner` 包含关卡号、seed、完成信号等网页
运行管理信息。隐藏映射只保存在服务器内存中，在整个 episode 完成前不会进入
响应。页面刷新会恢复服务进程中当前的回合。Web 体验是本机单用户工具，用于人工
理解和环境冒烟检查，不是正式多客户端 runner，也不改变模型实验输入合同。

默认生成一个三关 episode。批量生成：

```bash
switchdoor-three-level generate \
  --output /tmp/switchdoor_three_level_batch \
  --episode-roots 100 \
  --seed 20260826
```

生成同一布局下的同色/异色配对 twins：

```bash
switchdoor-three-level generate \
  --output /tmp/switchdoor_three_level_twins \
  --episode-roots 20 \
  --seed 20260826 \
  --paired
```

## 输出结构

```text
OUTPUT/
  generation.json
  index.jsonl
  episodes/<episode_id>/
    public.json
    supervision.jsonl
    private.json
    frames/L1/000.png ...
    frames/L2/000.png ...
    frames/L3/000.png ...
```

- `public.json`：逐关的 RGB 观察、过去动作、状态和可用动作；不含隐藏规则、
  符号地图或样本身份。
- `supervision.jsonl`：每个非终止观察对应的 oracle 下一动作，作为训练目标，
  不是模型输入。
- `private.json`：隐藏映射、符号布局、变换和 oracle 审计信息；必须与模型输入
  隔离。
- `index.jsonl`：数据集索引；不包含隐藏映射。

训练或评测 loader 在时刻 `t` 只能提供该 episode 从开头到当前观察的
`public.json` 前缀。后续观察、`target_action`、`private.json`、`generation.json`
和索引中的私有文件路径都不是模型输入。

每个 episode 都经过完整回放验证。生成器还会检查两种映射下三关均可解、L1
恰好一次探测能够区分映射，以及 L2/L3 的正确首个开关随映射翻转。配对 twins 的
三关初始 PNG 必须逐字节一致。

## 代码入口

- 科学事实源：`config/experiment.json`
- 状态转移：`src/switchdoor_three_level/model.py`
- 布局与随机变换：`src/switchdoor_three_level/layouts.py`
- 精确求解：`src/switchdoor_three_level/solver.py`
- 无第三方依赖渲染：`src/switchdoor_three_level/render.py`
- 生成与验证：`src/switchdoor_three_level/generator.py`
- 在线逐动作环境：`src/switchdoor_three_level/live_env.py`
- 本地 Web 服务：`src/switchdoor_three_level/web_app.py`

本包只建立可复现环境与监督数据，不代表任何模型已经学会读规则、预测或通关。

## 开发检查

```bash
python -m pip install -e '.[test]'
ruff check .
ruff format --check .
pytest -q
```

## 版本说明

当前版本以三关 `[7, 7, 11]` 环境替换了仓库早期的单关 7×7、C0–C3
实现。Python 包、导入名和命令行入口分别改为 `switchdoor-three-level`、
`switchdoor_three_level` 和 `switchdoor-three-level`，不保留旧 API 兼容层。
Python 最低版本由 3.10 调整为 3.11；旧数据集格式、`toggle` 规则和旧 renderer
接口均不兼容。当前生成数据使用 schema v2，并从 `public.json` 移除了可编码隐藏
规则的样本身份字段。旧实现仍可从 Git 提交 `1efc8a7` 恢复。

完整环境合同见 [`EXPERIMENT_PROTOCOL.zh.md`](EXPERIMENT_PROTOCOL.zh.md)。
