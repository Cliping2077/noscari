import json
from pathlib import Path
import statistics

import argparse
parser = argparse.ArgumentParser(description='Regenerate cognitive tables and the fixed first trajectory without changing the run')
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
run = args.run
args.output.mkdir(parents=True, exist_ok=False)
summary = json.loads((run/'summary.json').read_text())
training = json.loads((run/'training.json').read_text())
rows = [json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]

def subset(name, regime=None):
    return [r for r in rows if r['policy'] == name and (regime is None or r['regime'] == regime)]

def mean(data, key):
    return statistics.mean(r[key] for r in data)

names = ['traverse','rule_memory','ema_greedy','linear','mlp','context_mlp','gru',
    'gru_untrained','gru_reset_task','gru_no_help','gru_no_replay']
table = ['| 策略 | 成功率 | 平均效用 | 操作数 | 求助次数 | 错误语义尝试/任务 |',
    '|---|---:|---:|---:|---:|---:|']
for name in names:
    m = summary['overall'][name]
    table.append(f"| {name} | {m['success']:.2%} | {m['utility']:.4f} | {m['operations']:.3f} | {m['teacher_calls']:.3f} | {m['wrong_transfer']:.3f} |")
regime_table = ['| 策略 | reliable | unreliable | partial | changing |', '|---|---:|---:|---:|---:|']
for name in ['rule_memory','ema_greedy','context_mlp','gru','gru_reset_task','phase_a_gru','gru_no_replay']:
    regime_table.append('| '+name+' | '+' | '.join(f"{mean(subset(name,r),'success'):.2%}" for r in ['reliable','unreliable','partial','changing'])+' |')
behavior = ['| 策略 / 环境 | 前半段语义首选 | 后半段语义首选 | 前半段效用 | 后半段效用 |', '|---|---:|---:|---:|---:|']
for name in ['gru','context_mlp','gru_reset_task']:
    for regime in ['reliable','unreliable','partial','changing']:
        data = subset(name,regime)
        early, late = ([r for r in data if r['half']==h] for h in ['early','late'])
        behavior.append(f"| {name} / {regime} | {mean(early,'first_semantic'):.2%} | {mean(late,'first_semantic'):.2%} | {mean(early,'utility'):.4f} | {mean(late,'utility'):.4f} |")
parameters = ['| 种子 | A 阶段参数 L2 变化 | B 阶段参数 L2 变化 | B 首/末损失 |', '|---|---:|---:|---:|']
for audit in training['audits']:
    if audit['kind'] == 'gru':
        a,b = audit['phase_a'],audit['phase_b']
        parameters.append(f"| {audit['seed']} | {a['parameter_delta']:.6f} | {b['parameter_delta']:.6f} | {b['losses'][0]:.6f} / {b['losses'][-1]:.6f} |")
output = dict(primary=summary['primary'], table='\n'.join(table), regimes='\n'.join(regime_table),
    behavior='\n'.join(behavior), parameters='\n'.join(parameters),
    training_seconds=training['seconds'], evaluation_seconds=summary['seconds'],
    chosen_training_transitions=sum(a['phase_b']['verified_transitions'] for a in training['audits'] if a['kind']=='gru'),
    executions=summary['test_executions'], gru_brier=summary['overall']['gru']['brier'],
    context_brier=summary['overall']['context_mlp']['brier'],
    gru_exploration_cost=summary['overall']['gru']['exploration_cost'])
(args.output/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))

# An unselected first session, including every outcome; not a best-case example.
trace = [json.loads(line) for line in (run/'seed-0/trace-gru.jsonl').read_text().splitlines()]
first = [r for r in trace if r['session_id']=='test-0']
text = ['# 固定首会话成长轨迹', '',
    '固定展示 seed-0 / test-0 的全部 24 个任务，不按表现筛选。权重在整个测试中固定；状态只由实际反馈更新。候选值是估计收益，不是主观想法。', '',
    '| 任务序号 | 首选 / 实际首动作 | 行动轨迹 | 成功 | 消耗 | 首次决策隐藏状态范数 |', '|---|---|---|---|---:|---:|']
for index in sorted({r['task_index'] for r in first}):
    group = [r for r in first if r['task_index']==index]
    start, end = group[0], group[-1]
    norm = sum(v*v for v in start['hidden_before'])**.5
    text.append(f"| {index} | {start['candidates'][0]['method']} / {start['event']['action']} | {' → '.join(r['event']['action'] for r in group)} | {end['event']['after']['success']} | {end['event']['after']['spent']} | {norm:.5f} |")
text += ['', '完整候选分数、24 维状态、显式统计和实际反馈在对应 JSONL。运行 `python -m axiom.benchmarks.cognitive replay --output experiments/core/cognitive-v1` 可验证全部种子的正式 GRU 轨迹；重放验证文件已存在时，最后写入会拒绝覆盖，需在独立复制目录中保留/移开既有验证文件后重放。', '',
    '状态范数变化本身不等于能力进步。请与全局对照、清空状态消融及失败分析共同阅读。']
(args.output/'cognitive-growth-trace.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
