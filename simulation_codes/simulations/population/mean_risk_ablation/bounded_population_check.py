"""Robustness for the Phase-2/3 physical price-slope constraint abs(chi)<=0.45."""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
from population_ablation import aggregate, merge_pred, fixedpoint_partial
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'population_benchmark'))
from benchmark import Benchmark

def run(bound=.45):
    b=Benchmark(K=3,chi_bound=bound)
    q0,d0=b.equilibrium((0,0))
    law=b.law(q0)
    lo=b.fit_one(law,0,0);hi=b.fit_one(law,0,3);other=b.fit_one(law,1,0)
    ab={}
    for m in ('old','new'):
        for v in ('old','new'):
            p=merge_pred(lo if m=='old' else hi,lo if v=='old' else hi)
            q=aggregate(b,[p,other]);ab[m+'_'+v]={'J':b.J(q),'beta':p['beta'],'chi':p['chi'],'variance':p['v']}
    parts={k:{x:v for x,v in fixedpoint_partial(b,aggregate(b,[hi,other]),hi,other,3,0,update_types=ix).items() if x!='path'}
           for k,ix in [('none',()),('upgraded',(0,)),('other',(1,)),('both',(0,1))]}
    qhi,dhi=b.equilibrium((3,0),start=aggregate(b,[hi,other]))
    out={'price_slope_bound':bound,'baseline_J':b.J(q0),'impact_J':ab['new_new']['J'],'final_J':b.J(qhi),
         'rho_jac_baseline':d0['rho_jac'],'rho_jac_high':dhi['rho_jac'],
         'factorial':ab,'retraining_arms':parts}
    (ROOT/'population_bounded_results.json').write_text(json.dumps(out,indent=2))
    print('bounded baseline/impact/final',out['baseline_J'],out['impact_J'],out['final_J'])
    print('factorial',ab);print('retraining',[(k,round(v['J'],5),v['residual']) for k,v in parts.items()])
    return out
if __name__=='__main__':run()
