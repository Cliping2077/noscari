# Alpha 复现手册

本手册区分安装 smoke、检查点轨迹重放、从初态重新训练，以及新的独立研究。
公开数据的重跑不是新的独立效果验证。原始报告、失败数据和 alpha 1–4 归档均保留。

## 干净环境与实际执行

参考环境：Windows x86-64，Python 3.14.6，CPU Torch 2.14.1、NumPy 2.5.4。
创建全新 venv，`include-system-site-packages=false`，不复制开发依赖。24 个依赖
从 PyPI 分发文件安装；包来自构建的 wheel，明确检查导入来自 site-packages。
操作系统和 Python 解释器仍为同一宿主，因此不是另一台机器/独立实验室复现。

首次安装需要网络；后续测试、训练、评估和重放不需要 API Key 或网络服务。
发布目录不包含虚拟环境、本地数据库、pip 原始报告里的个人目录、缓存或凭据。
依赖哈希和许可证见 [依赖说明](../THIRD_PARTY_NOTICES.md)。

Windows 示例，在解压后的项目目录执行（先保证这些输出目录不存在）：

```powershell
python -m venv work/repro-venv
work/repro-venv/Scripts/python -m pip install -r requirements-executed.txt
work/repro-venv/Scripts/python -m build --wheel --no-isolation
work/repro-venv/Scripts/python -m pip install --no-deps dist/axiom_learning_core-0.1.0a5-py3-none-any.whl
work/repro-venv/Scripts/python -m pip check
work/repro-venv/Scripts/python -m pytest -q --basetemp work/repro-tests
work/repro-venv/Scripts/python -I scripts/installed_smoke.py --output work/repro-smoke
work/repro-venv/Scripts/python -I scripts/reproduce.py --reference experiments --output work/repro-full
work/repro-venv/Scripts/python scripts/summarize_cognitive.py --run work/repro-full/cognitive --output work/repro-tables
```

Linux/macOS 使用 `bin/python`。该精确版本清单对应 Windows 参考环境，不保证
在全部 Python/平台都能解析；其他环境先使用 `.[dev,torch]` 支持范围并记录实际版本。
CPU Torch 可从官方 CPU index 安装，避免不必要的 GPU 依赖。不同 BLAS、Torch、
硬件可能有浮点差异，严格重放失败要保留日志，不能事后放宽阈值并冒充逐值一致。

## 比较范围

`scripts/reproduce.py` 使用当前解释器的已安装包，拒绝直接导入源码；输出目录
已存在会拒绝覆盖。它会重跑：

| 材料 | 重跑与判定 |
|---|---|
| 第一轮 validated | 5 种子全部方法、分格/各种子/置信区间/无泄漏检查；参数审计只排除运行秒数 |
| retention | 全部阶段、旧/新分布、参数和每条测试结果 |
| 第二轮 recovery | 5 种子全部方法、共同失败群体、预测误差、分格与训练参数 |
| Frontier interest | 5 种子、全部方法/阶段、配对结果、主题选择和参数 |
| 两个教学工程演示 | 保存记录/学习/回滚的区别、参数变化与相应评估 |
| 首次教师实验 | 使用公开的历史回答，重新绑定新的请求 ID，重跑验证教学与全部 300 个冻结任务 |
| 教师失败诊断 | 无初始学习、未授权、无回答、错误教学与遍历反事实 |
| cognitive-v1 | 从初态重新训练全部模型，28,800 次评估；对照/分格/种子/参数审计/全部检查点哈希 |
| cognitive 重放 | 重算 4,595 条 GRU 决策、候选、反馈、隐状态和最终恢复状态 |

指标和参数使用固定 `atol=rtol=1e-10`；字典键、数组长度、布尔值必须一致，
NaN 视为失败。认知检查点、经验和轨迹另外比较 SHA-256，实际参考环境下为逐字节
一致。墙钟时间、时间戳、随机事件/请求 UUID 不属于性能量，不要求相同；每项
检查的字段和排除项都在 `comparison.json` 明列。不能拿该文件声称所有文件字节一样。

历史教师回答重放**不会重新创造一次真实提问或独立教师教学**。教师/开发者同源的
限制保留；该重放只验证公开回答与已发布程序能否产生原有参数和行为结果。
恢复演示挑选成功见证的限制也保留，不能拿其一次成功估计泛化性能。

`summarize_cognitive.py` 从 summary、training 和原始 episodes/trace 重新生成报告
表格及固定 seed-0/test-0 的全轨迹，拒绝覆盖已有目录。Markdown 文字解释仍需
人工检查；自动生成数字不等于科学结论自动成立。

## CI 的实际边界

`.github/workflows/tests.yml` 配置 Windows/Linux × Python 3.11/3.14 的认知测试，
以及 Ubuntu/Python 3.11 不装 Torch 的基础接口测试。包含依赖检查、完整 pytest、
源码凭据/隐私扫描、构建和已安装 wheel 的离线小规模闭环。

远程 CI **尚未运行**，本地仅验证了 Windows/Python 3.14.6 的环境；不将 YAML
文件当作四个平台组合已经通过的证据。公开 Release 前需真实运行 CI。用户尚未
授权创建远程仓库，不能为获得绿勾擅自上传。可在获准的私有暂存仓库先运行，
之后再由用户决定公开。

历史归档通过 `scripts/audit_release.py <当前项目> <历史zip> <历史安装包目录>
--output <新报告路径>` 明确输入。工具检查 ZIP/wheel/tar 及嵌套内容，不解压执行。
CI 检查工作树；它不会自动审核未来新增的 Git 历史。当前项目没有 `.git` 或任何
提交，已审查的“历史”指实际存在的 alpha 1–4 归档与安装包，不能声称扫描过提交历史。

所有测试数据公开后，只能用于复现与开发。下一轮确认性实验必须另行冻结新测试
任务，禁止在这里反复调参后仍称其为独立测试。
