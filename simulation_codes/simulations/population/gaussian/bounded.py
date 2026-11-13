"""Closed-interval price-slope-constrained population Gaussian learning test."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import root,brentq
from benchmark import Benchmark
from cycles import info_mi
OUT=Path(__file__).resolve().parent

def solve(sd, init_low=None, init_hi=None):
    b=Benchmark(chi_bound=.45)
    b.eco=replace(b.eco,supply_noise_sd=sd); b.sigma_supply=sd
    qlo,_=b.equilibrium((0,0),start=init_low,maxiter=1000)
    qhi,_=b.equilibrium((3,0),start=init_hi,maxiter=1000)
    rad_hi=float(max(abs(np.linalg.eigvals(b.local_jac(qhi,(3,0))))))
    return b,qlo,qhi,rad_hi

def run():
    bo,lo0,hi0,_=solve(.9)
    def rad_at(sd):
        _,lo,hi,rad=solve(sd,init_low=lo0,init_hi=hi0)
        return rad
    critical=brentq(lambda x:rad_at(x)-1.,.63,.66,xtol=1e-10)
    b,qlo,qhi,rho=solve(.62,init_low=lo0,init_hi=hi0)
    impact=b.update(qlo,(3,0))
    q=impact.copy()
    for k in range(3000): q=b.update(q,(3,0))
    sol=root(lambda x:x-b.update(b.update(x,(3,0)),(3,0)),q,tol=1e-11)
    qa=sol.x; qb=b.update(qa,(3,0))
    err=float(max(abs(qa-b.update(qb,(3,0)))))
    rho2=float(max(abs(np.linalg.eigvals(b.local_jac(qb,(3,0))@b.local_jac(qa,(3,0))))))
    _,dqa=b.update(qa,(3,0),True);_,dqb=b.update(qb,(3,0),True)
    out={'price_coefficient_bound':.45,'critical_supply_noise_sd':critical,
         'supply_sd_test':.62,'J_baseline':b.J(qlo),'J_impact':b.J(impact),
         'J_fixed_unstable':b.J(qhi),'rho_fixed_unstable':rho,
         'J_cycle':[b.J(qa),b.J(qb)],'rho_cycle_two_iter':rho2,
         'cycle_residual':err,'conditional_MI_baseline':info_mi(b,qlo)['conditional_MI_nats'],
         'conditional_MI_cycle':[info_mi(b,qa)['conditional_MI_nats'],info_mi(b,qb)['conditional_MI_nats']],
         'fast_radius':[b.law(qlo)['fast_radius'],b.law(qa)['fast_radius'],b.law(qb)['fast_radius']],
         'head_A':[{'chi':r['chi'],'beta':r['beta'],'v':r['v']} for r in dqa['preds']],
         'head_B':[{'chi':r['chi'],'beta':r['beta'],'v':r['v']} for r in dqb['preds']]}
    with open(OUT/'bounded.json','w') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':run()
