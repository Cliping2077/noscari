# Noscari — cognitive learning research · Alpha 0.1.0a5

[English](README.md) | 简体中文

**Every agent can learn. Every user can teach.**

持续学习研究原型：将教学记录、验证反馈与内部参数更新分开，验证经验是否改变
未见任务中的选择。CPU 即可运行，无 API Key、无模型下载、实验不联网。
V0.2 已并入既有工程；架构名称为 Noscari，当前模型名称为 Baby，`axiom-learning-core` 包名及
`axiom` 导入保留兼容。V0.2 是任务书版本，软件版本仍按 alpha 顺序递增。

**Noscari 是认知架构研究项目；Baby 是该架构当前的有限模型与实验实现。**
架构约定经验、技能、当前状态、候选及外部能力的边界；Baby 实现的是小规模
可训练选择器、统计状态和可选递归状态，尚未实现任务书中的全部长期目标。
数据库与模板、工程原语、内部学习参数和外部教师的贡献分别记录。外部模型
完成任务不能算作 Baby 已经学会；保存一条教学记录也不能算作参数学习。

本次 a5 只补强发布检查、复现工具与研究设计，**不改 a4 的认知算法和实验数据**。
研究 Alpha 已在 [GitHub Release](https://github.com/Cliping2077/noscari/releases/tag/v0.1.0a5) 公开，尚未发布 PyPI。
Noscari 的软件发行版本为 0.1.0a5；Baby 检查点由实验协议、训练种子和参数哈希标识，
暂不另行定义模型代次。[远程 CI](https://github.com/Cliping2077/noscari/actions/runs/39251025287)
五组任务全部通过：Windows/Ubuntu × Python 3.11/3.14 各 89 项测试通过；
无 PyTorch 组为 75 项通过、3 项跳过。CI 不重跑完整研究。
[发布检查清单](docs/github-release.md) 保留冻结时的本地准备记录，后续远程验证见 Release。
[下一轮信息价值设计](docs/help-value-design.md) 尚未开始实验。

这不是完整类人学习系统，也不是 AGI。语义匹配方法与词汇由工程实现提供，模型学习
何时使用已有方法；它尚不能从自由语言教学中创造新方法。

## 安装与演示

Python 3.11+。建议独立虚拟环境。首次安装依赖需要网络；安装后演示和测试可离线。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,torch]"
.venv/Scripts/python -m axiom.examples.teach_menu --output work/demo-1
.venv/Scripts/python -m axiom.examples.recovery --output work/recovery-demo
.venv/Scripts/python -m axiom.examples.motivation --output work/motivation-demo
.venv/Scripts/python -m pytest -q --basetemp work/pytest-1
```

Linux/macOS 将 `.venv/Scripts/python` 换为 `.venv/bin/python`。不用 MLP 时安装
`.[dev]` 即可；基础包只依赖 NumPy、Pydantic。本轮实际测试环境是 Windows、
Python 3.14.6、CPU PyTorch；上述远程 CI 已验证 Windows/Ubuntu 与 Python 3.11/3.14。

目录存在且非空时，实验命令会拒绝覆盖，换一个输出目录即可。

## 公开接口

```python
from axiom import Axiom
from axiom.environments.menu import make_demo

agent = Axiom("work/personal.db")
observation, demonstration = make_demo()
agent.observe(observation)
print(agent.suggest())
event_id = agent.teach(demonstration)  # 存储示范，不更新权重
receipt = agent.learn()               # 明确训练，返回前后摘要和参数差值
agent.save("work/checkpoint.json")
agent.rollback()
agent.close()
```

`feedback(Feedback(...))` 提交执行结果；`learn()` 重放有效验证样本；
`Axiom.load(checkpoint, new_db)` 恢复；`correct` / `retract` 修正或撤销；
`forget(subject_id)` 删除本地教学者相关记录并使旧训练状态失效。

`verified=True` 是**可信宿主验证器的断言**，不能直接转发教师、网页或 LLM 的字段。
JSON 检查点是可信本地恢复格式，不是签名文件；只能加载来源可信的文件。
可替换异步 `LLMProvider` 协议位于 `tools/providers.py`，默认实现完全离线；当前
程序不调用外部 LLM API。首次师徒实验由开发会话中的 Codex 助手通过回复文件担任外部教师。
真实搜索、浏览器和商业 API 适配器尚未实现。

## 实验

最新 Cognitive Model Research 已完成固定协议的 5 种子研究：4,078 参数的
PyTorch GRU 用实际反馈更新持续状态，在新任务中改变方法选择，但成功率
82.92%，未超过显式统计＋MLP 的 83.33%。主要效用差 −0.00542 的 95% 配对
种子区间跨零，**没有新结构优越性证据，也未解决混乱分类负迁移**。此组件保留为
可选研究实现，原稳定接口不变。这里是新的任务协议，不能与旧 59%→76% 跨协议比较。

同一固定测试集上的全部最终策略如下（五个训练种子；每策略 1,920 个任务）：

| 策略 | 成功率 | 平均净效用 | 求助次数/任务 |
|---|---:|---:|---:|
| 遍历 | 71.51% | 0.2897 | 0.000 |
| 规则记忆 | 83.80% | 0.5332 | 0.000 |
| EMA 统计 | 83.07% | 0.5221 | 0.245 |
| 上下文线性 | 82.92% | 0.5107 | 0.176 |
| 无持续历史 MLP | 83.33% | 0.5116 | 0.244 |
| 显式统计＋MLP | 83.33% | 0.5224 | 0.224 |
| GRU | 82.92% | 0.5170 | 0.202 |
| 未训练 GRU | 74.06% | 0.3395 | 0.408 |
| 每任务重置 GRU 状态 | 83.18% | 0.5109 | 0.261 |
| GRU 禁止求助 | 83.85% | 0.5294 | 0.000 |
| GRU 不回放旧阶段 | 82.50% | 0.5048 | 0.219 |

主要比较 GRU 减显式统计＋MLP 的效用区间为 **[−0.01265, 0.00182]**。
没有证据证明 Noscari 的新结构比强简单方法更好；规则记忆的结果也必须保留。
禁止求助后的改善是策略消融，不能直接解释为每次提问均有害。
分环境、早/晚阶段、旧分布保持、阶段 A 模型和所有参数变化见完整报告及原始数据。

- [模型审计与研究依据](docs/cognitive-research.md)
- [完整结果、消融与失败分析](docs/cognitive-report.md)
- [可重放的固定首会话成长轨迹](docs/cognitive-growth-trace.md)
- [认知组件接入与复现命令](docs/cognitive-api.md)
- [GitHub 研究 alpha 发布说明](docs/github-release.md)

```powershell
python -m axiom.benchmarks.cognitive freeze --output work/cognitive-smoke --smoke
python -m axiom.benchmarks.cognitive train --output work/cognitive-smoke
python -m axiom.benchmarks.cognitive evaluate --output work/cognitive-smoke
python -m axiom.benchmarks.cognitive replay --output work/cognitive-smoke
```

需要安装 `.[dev,torch]`。去掉 `--smoke` 才运行完整固定研究；小规模演示不用于
性能结论。当前权重/状态和原始回放记录都可独立于外部 LLM 保存恢复。

首次真实师徒交互已记录：预训练恢复选择器发出请求，开发会话中的 Codex 助手提供
一条建议，模拟器执行验证后才训练原有 Axiom 选择模块。300 个提前封存的相关
任务中，成功率 59% → 76%；规则记忆基线也为 76%，混乱分类下出现负迁移。
它是在尝试之前询问，且教学实例本来可由第一步遍历解决，因此尚未证明能准确判断
何时必须求助。[完整师徒实验报告](docs/first-teacher-report.md) 保留这些限制。

人工教师与记录重放的运行方法见 [教师接口](docs/teacher-bridge.md)。它没有常驻
连接到 Codex；首次回答由当前会话读取请求后提供。独立运行时需要人工回复文件，
或未来实现的可选 API 适配器。模板负责措辞，参数负责当前的求助选择。

Core Lab 第二轮加入逐步恢复：在共同语义失败样本上，学习恢复成功率 91.39%，
固定遍历回退 85.74%；但去掉动态评分特征仍有 91.46%，因此尚无动态特征额外收益
证据。[第二轮报告](docs/round2-report.md) 保留此负结果及强基线比较。

Frontier Lab 最小动机模块会由观察生成模板问题，保留目标，并显式训练探索偏好。
兴趣实验中初态相同的学习者随经历分化、在条件反转后修正偏好；独立测试测得预测
进展提高。第一阶段历史均值基线同样好。这是固定目标定义下的估计学习，尚未实现
自创奖励、自由提问、主观好奇心或意识验证。[V0.2 报告](docs/v02-report.md)。

```powershell
python -m axiom.benchmarks.run --output work/benchmark-1 --seeds 5 --train-tasks 240 --tasks-per-cell 20
python -m axiom.benchmarks.retention --output work/retention-1
python -m axiom.benchmarks.recovery --output work/recovery-1 --seeds 5 --train-tasks 480 --tasks-per-cell 30
python -m axiom.frontier.interest --output work/interest-1 --seeds 5 --train-per-topic 24 --eval-tasks 60
```

不安装 PyTorch 时，第一条加 `--without-mlp`。
恢复实验也支持 `--without-mlp`；动机实验不依赖 PyTorch。

研究路线与明确的未实现项见 [正式补充](docs/brief-v02.md)、[V0.2 计划](docs/v02-plan.md)。
两类实验规范见 [Core Lab](experiments/core/README.md)、[Frontier Lab](experiments/frontier/README.md)。

实验包含 A 串行、B 同经验记忆检索、C 固定多候选、D AXIOM，LinUCB/MLP 对照，
以及冻结参数、关闭联想、关闭求助三个消融。六个类别属于固定玩具语义词表，
测试使用新功能名称和随机布局，并不证明未见概念的迁移。

早期第一/二轮环境公开了模拟教师/搜索可靠性特征，属于简化信息条件；这不能
证明 Baby 自行推断了可靠性。新的 cognitive-v1 严格隐藏这些真值，学习者只能
使用观察和所选动作反馈。跨协议结果不能拼接成一条“成长曲线”。

完整重训练复现、历史教师回答重放、参数比对和表格生成见
[复现说明](docs/reproducibility.md)。a5 提供一键检查，所有参考数据随源码归档保留。

查看 [第一轮实验报告](docs/round1-report.md)、[研究差异](docs/research.md)、
[协议与指标](docs/experiment-protocol.md)、[安全与数据](SECURITY.md)。原始记录在
`experiments/`，不能只引用 README 中的概述而省略失败结果。

## 目录

```text
src/axiom/
  core/          公共模型、持续状态、候选列表、显式学习 API
  memory/        SQLite 权威事件记录、JSONL 导入导出
  learning/      线性排序器、LinUCB、小型 PyTorch MLP
  environments/  隐藏真实菜单位置的离线模拟器
  benchmarks/    对照、消融、分布变化与保持测试
  examples/      一次教学演示
  frontier/      独立启用的最小动机模块与兴趣实验
  tools/         可选能力提供者协议
tests/ docs/ experiments/ scripts/
```

详细边界见 [设计](docs/design.md)。当前候选是有限宏动作，不是并行思考模块；
技能版本与结果关联，但默认选择器仅使用最新语义版本。旧版本留作审计，自动组合、
重新激活旧语义版本和通用长期子目标规划尚未实现。Frontier 已有有限的持续问题
状态，尚未接入核心菜单决策器。MIT 许可，GitHub 研究 Alpha 已公开，尚未发布 PyPI。
