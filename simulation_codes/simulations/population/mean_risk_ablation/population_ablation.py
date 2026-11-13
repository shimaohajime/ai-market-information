"""Population Gaussian microfounded factorial ablations.

No sampling/training noise: population OLS/variance from the stationary Lyapunov covariance.
Factorial interventions replace only one component of investor type 0's belief;
these hybrid beliefs are deliberately NOT joint NLL optima.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'population_benchmark'))
from benchmark import Benchmark


def aggregate(b,preds):
    A=0.0; C=np.zeros(b.zdim);Delta=0.0
    for p in preds:
        w=p['pi']/(p['gamma']*p['v'])
        A+=w*p['beta']; C+=w*p['b'];Delta+=w*(b.eco.risk_free_gross-p['chi'])
    return np.r_[A,C,Delta]

def merge_pred(mean_src,var_src):
    p=mean_src.copy();p['b']=mean_src['b'].copy();p['v']=var_src['v'];return p

def fixedpoint_partial(b,q_init,p0,p1,cap0=3,cap1=0,update_types=(0,1),tol=1e-12,maxiter=900):
    q=q_init.copy();hist=[]
    for k in range(maxiter):
        law=b.law(q,verify=False)
        p0new=b.fit_one(law,0,cap0) if 0 in update_types else p0
        p1new=b.fit_one(law,1,cap1) if 1 in update_types else p1
        qq=aggregate(b,[p0new,p1new])
        hist.append(b.J(qq))
        if np.max(np.abs(qq-q))<tol:
            q=qq;break
        q=qq
    return {'J':b.J(q),'q':q.tolist(),'iterations':len(hist),'residual':float(np.max(np.abs(q-aggregate(b,[b.fit_one(b.law(q,verify=False),0,cap0) if 0 in update_types else p0,b.fit_one(b.law(q,verify=False),1,cap1) if 1 in update_types else p1])))),'last_lag1_J_delta':abs(hist[-1]-hist[-2]) if len(hist)>1 else None,'path':hist}

def run():
    b=Benchmark(K=3)
    caps=(0,0)
    q0,d0=b.equilibrium(caps)
    law=b.law(q0)
    lo=b.fit_one(law,0,0);hi=b.fit_one(law,0,3);other=b.fit_one(law,1,0)
    rows={}
    for meanstate in ['old','new']:
        for riskstate in ['old','new']:
            p=merge_pred(lo if meanstate=='old' else hi,lo if riskstate=='old' else hi)
            q=aggregate(b,[p,other]);key=f'mean_{meanstate}_risk_{riskstate}'
            rows[key]={'J':b.J(q),'A':float(q[0]),'Delta':float(q[-1]),'type0_private_loading':float(p['pi']*p['beta']/(p['gamma']*p['v'])),
                'type1_private_loading':float(other['pi']*other['beta']/(other['gamma']*other['v'])),
                'price_coefficient':float(p['chi']),'prediction_var':float(p['v'])}
    J00=rows['mean_old_risk_old']['J'];J10=rows['mean_new_risk_old']['J'];J01=rows['mean_old_risk_new']['J'];J11=rows['mean_new_risk_new']['J']
    shap_mean=((J10-J00)+(J11-J01))/2
    shap_risk=((J01-J00)+(J11-J10))/2
    counter={}
    # Fix type0 at the high-tech policy learned on the old distribution; let type1 adapt or not.
    for label,up in [('none',()),('only_upgraded',(0,)),('only_other',(1,)),('both',(0,1))]:
        d=fixedpoint_partial(b,aggregate(b,[hi,other]),hi,other,3,0,up)
        counter[label]={k:v for k,v in d.items() if k!='path'}
    # Heterogeneity: change relative signal precision while remaining aligned and keeping masses/risk costs constant.
    het=[]
    for ratio in [.3,.6,1.,1.5,2.,3.3333333333,5.,10.]:
        bx=Benchmark(K=3)
        old_noise=bx.temps[0].signal_noise_sd
        bx.temps[0].signal_noise_sd=bx.temps[1].signal_noise_sd*ratio
        try:
            qq,dd=bx.equilibrium((0,0))
            qi=bx.update(qq,(3,0))
            qe,de=bx.equilibrium((3,0),start=qi)
            het.append({'ratio_sd_type0_to_type1':ratio,'baseline_J':bx.J(qq),'impact_pct':100*(bx.J(qi)/bx.J(qq)-1),
                        'equilibrium_pct':100*(bx.J(qe)/bx.J(qq)-1),
                        'rho_base':dd['rho_jac'],'rho_upgrade':de['rho_jac']})
        except Exception as e:het.append({'ratio':ratio,'error':str(e)})
    result={'design':'8d E1 aligned Gaussian CARA population; type0 upgraded K0 to K3; type1 K0; frozen Q0 factorial',
        'baseline_J':b.J(q0),'actual_impact_J':b.J(b.update(q0,(3,0))),'upgraded_fixedpoint_J':b.equilibrium((3,0),start=b.update(q0,(3,0)))[1]['J'],
        'old_v':lo['v'],'high_v':hi['v'],'old_beta':lo['beta'],'high_beta':hi['beta'],
        'factorial':rows,'shapley':{'mean':shap_mean,'risk':shap_risk,'total':J11-J00,'interaction':J11-J10-J01+J00},
        'retraining_arms':counter,'signal_noise_heterogeneity_grid':het}
    (ROOT/'population_results.json').write_text(json.dumps(result,indent=2))
    print('baseline ',b.J(q0),'impact',J11,'long',result['upgraded_fixedpoint_J']);print('factorial',[(k,round(v['J'],6)) for k,v in rows.items()]);print('Shap',result['shapley']);print('retrain',[(k,round(v['J'],6)) for k,v in counter.items()]);print('het',het)

if __name__=='__main__':run()
