"""First-auction and frozen-history comparisons from the saved all-low endpoint.
These are finite-learning candidate-baseline comparisons, not equilibrium claims.
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
import engine as e
from run import CFG,name,mixing_check

def run(seed,n):
    torch.set_num_threads(1)
    stage=e.ROOT/'results'/name(seed,0.)/f'round{n:02d}'
    models={c:e.load_checkpoint(stage/(c+'_deployed.pt'))[0] for c in ['low','high']}
    other=json.loads((stage/'other_deployed.json').read_text())
    out=e.ROOT/'baseline_shocks'/f'n{n:02d}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    fp=e.fingerprint(models,other)
    base,meta=e.simulate(models,other,0.,seed,50000+n,61,128,1024,512)
    y=e.flat(base,'future_dividend');theta=e.flat(base,'theta');supply=e.flat(base,'supply')
    heads,agg0=e.coefficients(models,other,0.)
    public=[]
    for c in ['low','high']:
        z=e.features(models[c],base);co=heads[c]['coef'];public.append(co[0]+z@co[1:-2])
    records=[];generation=[meta];mixing=[]
    for x in [0.,.25,.5,.75,1.]:
        _,agg=e.coefficients(models,other,x)
        K=agg['weights'][0]*public[0]+agg['weights'][1]*public[1]+agg['weights'][2]*other['intercept']
        impact=(K+agg['A']*theta-supply)/agg['delta']
        assert np.max(abs(K+agg['A']*theta-agg['delta']*impact-supply))<1e-10
        p={k:v.copy() for k,v in base.items()}
        p['price'][:,e.WINDOW-1:]=impact.reshape(128,1024)
        if x==0.:assert np.max(abs(impact-e.flat(base,'price')))<1e-6
        rows,market=e.score(models,other,x,p)
        # These datewise counterfactual prices have not been fed into later histories.
        # Their serial correlation is not a stationary deployment statistic.
        market['price_lag1']=None
        for r in rows:records.append(dict(seed=seed,baseline_round=n,x=x,stage='impact',**r,**market))
        frozen,mt=e.simulate(models,other,x,seed,50000+n,61,128,1024,512);generation.append(mt)
        rows,mk=e.score(models,other,x,frozen)
        assert abs(market['J']-mk['J'])<1e-12
        for r in rows:records.append(dict(seed=seed,baseline_round=n,x=x,stage='frozen',**r,**mk))
        mixing.append(dict(x=x,**mixing_check(models,other,x,seed,50000+n)))
    assert e.fingerprint(models,other)==fp
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    e.dump(out/'complete.json',dict(seed=seed,baseline_round=n,baseline_sha256=fp,
        generation=generation,mixing=mixing,
        note='All five adoption shares start from the SAME saved all-low profile. No candidate refitting, no post-shock retraining. The high candidate was last trained on Q_(n-1), like other deployed models. Operational validity is separately assessed in baseline_audit.'))
    print('SHOCKS',seed,n,'complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,default=12);a=p.parse_args()
    for s in CFG['seeds']:run(s,a.round)
