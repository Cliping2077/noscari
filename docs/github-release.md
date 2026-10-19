# GitHub 研究 Alpha 发布检查清单 — 0.1.0a5

2026-10-19。**本地发布准备版本，尚未公开。** 未创建远程仓库、推送代码、运行
GitHub Actions 或发布 PyPI。a5 保留 a4 算法与全部历史实验，新增的是复现工具、
发布核验与下一轮设计。原 a4 发布说明仍在其只读归档中。

## 可公开的研究内容

Noscari 是持续学习认知架构研究；Baby 是目前的小型模型实现。可安装包、无需
商业 API 的教学闭环、可训练参数/持续反馈状态、公开基线与负结果、检查点回滚、
成长轨迹、接口和原始日志已具备研究价值。它尚未证明架构优势，也不能接收任意
自然语言教学并从中创造通用执行技能。

GRU 成功率 82.92%，统计＋MLP 83.33%，规则记忆 83.80%；主要效用差区间跨零。
“参数改变”“状态依赖决策”和“自主求助”分别有有限的工程证据，但不能替代
学习收益、可靠能力边界或类人智能的证据。完整结果在 README 和研究报告中。

## 检查与证据

| 项目 | 状态与边界 |
|---|---|
| 冻结研究源码/历史数据 | 对照原 manifest 与 alpha 1–4 归档哈希；a5 不修改学习算法或原数据 |
| 干净依赖环境 | 新 venv、不继承系统包，直接安装最终 wheel 与 24 个固定依赖；同一 Windows/Python 宿主 |
| 全量测试 | 带 Torch：89 passed；独立无 Torch：75 passed、3 个预期 skip 条目（含模块级跳过） |
| 安装后闭环 | 源码之外的 wheel 导入、教学/恢复、保存加载、认知 smoke 训练/评估/轨迹重放 |
| 全部研究复现 | scripts/reproduce.py 重训并比较固定实验；结果与补充审计见 experiments/release-a5/ |
| 数据/参数一致性 | 逐方法、分格、各种子、检查点、完整认知轨迹；不以一次成功或少量 smoke 代替 |
| 依赖与许可证 | pip check 通过；24 个依赖、152 个许可证/署名文件记录；多重许可证没有改标成 MIT |
| 历史凭据/隐私 | 当前材料、4 个历史源码归档及 8 个旧安装包；无 Git 提交存在；扫描与人工来源复核范围公开 |
| 独立复核 | 修复不完整复现假通过、审计字段遗漏及设计因果问题；复核无剩余重要问题 |
| 文档与接入 | README、MIT LICENSE、贡献/安全/测试/复现说明、外部能力边界与失败记录 |
| 下一轮设计 | 信息价值假设、简单与强基线、固定预算/新测试集及停止条件；尚未实施 |
| 远程 CI | **待执行**。已有只读权限 Windows/Linux、Python 3.11/3.14 配置及无 Torch 工作流；本地结果不冒充远程结果 |
| 公开授权 | **待用户确认**；目前没有任何远程发布行为 |

确切检查记录和哈希见 `experiments/release-a5/verification.json`、
`comparison-final.json`、`supplemental-checks.json`、`dependency-inventory.json`、
`history-source-audit.json`、`history-distribution-audit.json`。源码/安装包最终哈希见
交付目录 `noscari-baby-a5-SHA256.json`。

## 公开之前最后的操作顺序

1. 用户审阅当前源码、失败结果、检查清单与研究设计。常规本地修改不再需要设计确认。
2. 若用户授权，先上传到私有暂存仓库，实际运行 CI 矩阵；失败则修复并保留失败记录。
   该步骤当前也尚未授权/执行。它使远程兼容性验证可以发生在公开之前。
3. 将真实 CI 运行链接、最终提交及包哈希交用户确认；获得明确公开授权后才改为公开
   并创建 GitHub Release。授权公开 GitHub 不自动等于授权发布 PyPI。

当前结论是：**本地研究 Alpha 材料具备交付审阅价值；正式公开仍有远程 CI 和用户
授权两道未完成步骤。** 不能将它表述为“所有发布门槛已通过”。

## Release 描述草案

Noscari is an experimental cognitive learning architecture; Baby is its current
small trainable model implementation. This research alpha provides offline
teaching experiments, persistent feedback state, explicit parameter updates,
baselines, checkpoints and complete failure records. The recurrent model changes
context-dependent choices but does not outperform the strong statistical and
rule-memory baselines in the published five-seed menu study. Negative transfer
and the cost/value of asking for help remain open problems. No AGI, consciousness,
human-level development or architecture-superiority claim is made.

该描述只有在真实 CI 与用户公开确认后才能用于正式 Release。实验受固定词表和
菜单原语限制；公开后的测试集只能用于复现，下一轮研究必须另行冻结。
