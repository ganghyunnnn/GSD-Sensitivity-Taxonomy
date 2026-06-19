"""기존 데이터로 routed prompting 근사치 계산."""
import json

with open('annotation/backbone_qwen3vl_8b.json', encoding='utf-8') as f:
    baseline = json.load(f)

bp = baseline['direct']['per_type']
print('=== Baseline (qwen3-vl:8b direct) ===')
for t in ['D', 'M1', 'M2']:
    n = bp[t]['total']
    acc = bp[t]['accuracy']
    print(f'  {t}: N={n}, acc={acc:.1%}, fail={1-acc:.1%}')

with open('annotation/gsd_ablation.json', encoding='utf-8') as f:
    abl = json.load(f)

eval_m1_ids = {r['task_id'] for r in baseline['direct']['results'] if r['gt_type'] == 'M1'}
print(f'\n=== Ablation CoT (eval split M1={len(eval_m1_ids)}) ===')

for cond in ['baseline', 'gsd_value', 'cot', 'few_shot', 'cot_fs']:
    data = abl['qwen3-vl_8b'][cond]
    eval_data = [r for r in data if r['task_id'] in eval_m1_ids]
    if eval_data:
        acc = sum(r['correct'] for r in eval_data) / len(eval_data)
        print(f'  {cond} (N={len(eval_data)}): acc={acc:.1%}, fail={1-acc:.1%}')

# D 결과는 baseline과 동일
# M1 cot가 best
# M2는 현재 모름 - 기존 M2 fail rate 그대로 사용 (conservative)
cot_eval = [r for r in abl['qwen3-vl_8b']['cot'] if r['task_id'] in eval_m1_ids]
m1_routed_acc = sum(r['correct'] for r in cot_eval) / len(cot_eval)
d_acc = bp['D']['accuracy']
m2_acc = bp['M2']['accuracy']  # conservative: same as direct

n_d = bp['D']['total']
n_m1 = bp['M1']['total']
n_m2 = bp['M2']['total']

m_total = n_m1 + n_m2
m_correct = int(m1_routed_acc * n_m1) + int(m2_acc * n_m2)
m_acc = m_correct / m_total
gap = (1 - m_acc) - (1 - d_acc)

print('\n=== Approximate Routed Result (D=direct, M1=cot, M2=direct) ===')
print(f'  D:  fail={1-d_acc:.1%} (unchanged)')
print(f'  M1: fail={1-m1_routed_acc:.1%} (was {1-bp["M1"]["accuracy"]:.1%})')
print(f'  M2: fail={1-m2_acc:.1%} (conservative, unchanged)')
print(f'  Gap (M-D): +{gap*100:.1f}pp (was +{((1-bp["M1"]["accuracy"]*n_m1/m_total - bp["M2"]["accuracy"]*n_m2/m_total) - (1-d_acc))*100:.1f}pp)')

m_fail_direct = 1 - (bp['M1']['correct'] + bp['M2']['correct']) / m_total
gap_direct = m_fail_direct - (1 - d_acc)
print(f'  Direct gap: +{gap_direct*100:.1f}pp')
print(f'  Routed gap: +{gap*100:.1f}pp')
print(f'  Improvement: {(gap_direct-gap)*100:.1f}pp reduction in gap')
