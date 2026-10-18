"""Exact (layer, group) K allocation for DSV4.1-1017 routed experts at a bpw budget.
KL_g(K) = a_g * r_eff(g, K, layer)^2  (smooth-mode sensitivity x held-out real-H r_eff, PLAN B-1/B-2/B-6).
r_eff at unsampled layers: linear interpolation between the 8 measured layers.
Budget in units of P_d (D params per layer; GU = 2 P_d): sum(2 K_gu + K_d) == round(bpw * 3 * 40)."""
import json, sys, statistics as st
from pathlib import Path
D = Path('/home/sfxnz/projects/data/dsv41-1017')
import os
C = json.load(open(D / os.environ.get('CURVES', 'sens/curves.json')))['curves']
R = json.load(open(D / 'hess/r_eff.json'))
LS, KS = R['layers'], [2, 3, 4]

def reff(g, K, L):
    v = [R['table'][g][str(K)]['out'][str(x)]['r_eff'] for x in LS]
    if L <= LS[0]: return v[0]
    for i in range(len(LS) - 1):
        if LS[i] <= L <= LS[i + 1]:
            t = (L - LS[i]) / (LS[i + 1] - LS[i]); return v[i] * (1 - t) + v[i + 1] * t
    return v[-1]

def a_est(c, how):
    if how == 'total': return c['a_total']
    if how == 'tok': return c['a_tok']
    if how == 'excl_top': return c['a_total_excl_top_row']
    rows = list(c['a_by_row'].values())
    if how == 'median_row': return st.median(rows)
    if how == 'trim_row':  # mean of rows after dropping the top 2
        s = sorted(rows)[:-2]; return sum(s) / len(s)
    if how == 'geo_dom':
        import math; v = list(c['a_by_domain'].values()); return math.exp(sum(map(math.log, v)) / len(v))
    raise KeyError(how)

def kl_table(how):
    return {(L, g): {K: a_est(C[str(L)][g], how) * reff(g, K, L) ** 2 for K in KS} for L in range(40) for g in ('GU', 'D')}

def solve(kl, budget):
    groups = list(kl)
    w = {('GU'): 2, ('D'): 1}
    INF = float('inf'); best = {0: (0.0, {})}
    for gk in groups:
        nb = {}
        for used, (cost, asg) in best.items():
            for K in KS:
                u = used + w[gk[1]] * K
                if u > budget: continue
                c = cost + kl[gk][K]
                if u not in nb or c < nb[u][0]: nb[u] = (c, {**asg, gk: K})
        best = nb
    return best[budget]

def total(kl, asg): return sum(kl[g][asg[g]] for g in kl)

if __name__ == '__main__':
    bpw = float(sys.argv[1]) if len(sys.argv) > 1 else 2.70
    budget = round(bpw * 120)
    hows = ['total', 'tok', 'excl_top', 'median_row', 'trim_row', 'geo_dom']
    tabs = {h: kl_table(h) for h in hows}
    sols = {h: solve(tabs[h], budget)[1] for h in hows}
    # baselines
    base = {}
    base['uniform_mix_3/2'] = {(L, g): (3 if g == 'GU' else 2) for L in range(40) for g in ('GU', 'D')}  # 2.667
    base['early_K3'] = {(L, g): (3 if L < 28 else 2) for L in range(40) for g in ('GU', 'D')}  # 2.70
    print('budget units', budget, '-> bpw', budget / 120)
    print('cross-eval: rows = optimised under, cols = evaluated under (relative to that col\'s own optimum)')
    print('%-16s' % '', *['%10s' % h for h in hows])
    for h, s in list(sols.items()) + list(base.items()):
        print('%-16s' % h, *['%10.3f' % (total(tabs[e], s) / total(tabs[e], sols[e])) for e in hows])
    for h in hows:
        s = sols[h]
        print(h.ljust(11), ' '.join('%d%d' % (s[(L, 'GU')], s[(L, 'D')]) for L in range(40)))
    json.dump({h: {str(L): [sols[h][(L, 'GU')], sols[h][(L, 'D')]] for L in range(40)} for h in hows},
              open(D / 'alloc/candidates.json', 'w'), indent=0)
