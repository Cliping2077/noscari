# 固定首会话成长轨迹

固定展示 seed-0 / test-0 的全部 24 个任务，不按表现筛选。权重在整个测试中固定；状态只由实际反馈更新。候选值是估计收益，不是主观想法。

| 任务序号 | 首选 / 实际首动作 | 行动轨迹 | 成功 | 消耗 | 首次决策隐藏状态范数 |
|---|---|---|---|---:|---:|
| 0 | semantic / semantic | semantic | True | 1 | 0.00000 |
| 1 | semantic / semantic | semantic | True | 1 | 1.22066 |
| 2 | semantic / semantic | semantic | True | 1 | 1.59669 |
| 3 | semantic / semantic | semantic | True | 1 | 1.83791 |
| 4 | semantic / semantic | semantic | True | 1 | 1.98170 |
| 5 | semantic / semantic | semantic | True | 1 | 2.07516 |
| 6 | semantic / semantic | semantic | True | 1 | 2.14029 |
| 7 | semantic / semantic | semantic | True | 1 | 2.21155 |
| 8 | semantic / semantic | semantic | True | 1 | 2.28333 |
| 9 | semantic / semantic | semantic | True | 1 | 2.32939 |
| 10 | semantic / semantic | semantic | True | 1 | 2.36452 |
| 11 | semantic / semantic | semantic | True | 1 | 2.39395 |
| 12 | semantic / semantic | semantic | True | 1 | 2.38211 |
| 13 | semantic / semantic | semantic | True | 1 | 2.40299 |
| 14 | semantic / semantic | semantic | True | 1 | 2.40679 |
| 15 | semantic / semantic | semantic | True | 1 | 2.40415 |
| 16 | semantic / semantic | semantic → traverse → traverse → traverse | False | 4 | 2.40598 |
| 17 | semantic / semantic | semantic → traverse → traverse → traverse → traverse → traverse | False | 6 | 1.76713 |
| 18 | semantic / semantic | semantic | True | 1 | 1.18424 |
| 19 | semantic / semantic | semantic | True | 1 | 1.49445 |
| 20 | semantic / semantic | semantic | True | 1 | 1.81325 |
| 21 | semantic / semantic | semantic | True | 1 | 2.03326 |
| 22 | semantic / semantic | semantic | True | 1 | 2.15885 |
| 23 | semantic / semantic | semantic | True | 1 | 2.23820 |

完整候选分数、24 维状态、显式统计和实际反馈在对应 JSONL。运行 `python -m axiom.benchmarks.cognitive replay --output experiments/core/cognitive-v1` 可验证全部种子的正式 GRU 轨迹；重放验证文件已存在时，最后写入会拒绝覆盖，需在独立复制目录中保留/移开既有验证文件后重放。

状态范数变化本身不等于能力进步。请与全局对照、清空状态消融及失败分析共同阅读。
