"""Ensemble allocation: minimise sum over estimators of KL_h / KL_h(own optimum); report regret, per-domain, r_eff jitter."""
import json, os, random
from alloc import *
budget = 324
hows = ['total', 'tok', 'excl_top', 'median_row', 'trim_row', 'geo_dom']
tabs = {h: kl_table(h) for h in hows}
opt = {h: solve(tabs[h], budget) for h in hows}
ens = {g: {K: sum(tabs[h][g][K] / opt[h][0] for h in hows) for K in KS} for g in tabs['total']}
cost, s = solve(ens, budget)
print('ensemble regret vs each estimator optimum:', {h: round(total(tabs[h], s) / opt[h][0], 3) for h in hows})
print('ensemble     ', ' '.join('%d%d' % (s[(L, 'GU')], s[(L, 'D')]) for L in range(40)))
bits = sum((2 * s[(L, 'GU')] + s[(L, 'D')]) for L in range(40)) / 120
print('routed bpw', bits)
# per-domain predicted KL vs that domain's own optimum
doms = list(C['0']['GU']['a_by_domain'])
for d in doms:
    t = {(L, g): {K: C[str(L)][g]['a_by_domain'][d] * reff(g, K, L) ** 2 for K in KS} for L in range(40) for g in ('GU', 'D')}
    print('  domain %-13s ensemble/own-opt %.3f   predicted KL %.2e' % (d, total(t, s) / solve(t, budget)[0], total(t, s)))
# r_eff jitter: +-14% multiplicative per sampled layer and K (shared across groups), 50 draws
rng = random.Random(1017); regs = []
orig = json.loads(json.dumps(R['table']))
for i in range(50):
    for g in ('GU', 'D'):
        for K in KS:
            for L in LS:
                R['table'][g][str(K)]['out'][str(L)]['r_eff'] = orig[g][str(K)]['out'][str(L)]['r_eff'] * (1 + rng.uniform(-.14, .14))
    tj = {g: {K: sum(kl_table(h)[g][K] / opt[h][0] for h in hows) for K in KS} for g in tabs['total']}
    regs.append(total(tj, s) / solve(tj, budget)[0])
R['table'] = orig
print('r_eff jitter regret of ensemble alloc: median %.3f max %.3f' % (sorted(regs)[25], max(regs)))
# MoE decode cost estimate at m=4 (dataflow us/call, Phase A table)
us = {(2,2):332,(3,3):482,(4,4):615,(3,2):434,(2,3):379,(3,4):524,(4,3):571,(4,2):524,(2,4):421}
print('MoE us/step m=4: alloc %.0f vs all-2:2 %.0f' % (sum(us[(s[(L,'GU')], s[(L,'D')])] for L in range(40)), 40 * 332))
json.dump({'bpw': bits, 'method': 'ensemble of 6 sensitivity estimators (alloc/ensemble.py)',
           'expert_bits': {str(L): [s[(L, 'GU')], s[(L, 'D')]] for L in range(40)}}, open(os.environ.get('OUT', 'alloc_2.70.json'), 'w'), indent=1)
