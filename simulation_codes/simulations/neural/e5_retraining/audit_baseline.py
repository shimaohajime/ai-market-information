"""Independent same-law refits: statistical update noise versus systematic drift.

Every replication starts from exactly the same terminal profile, generates all
fresh data first, then fits both neural models and group 2. No audit fit changes
the law that supplies its data. Fits are not selected or averaged for deployment.
"""
import argparse, json, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import engine as e
from run import CFG, name

def audit(seed, n, rep):
    torch.set_num_threads(1)
    stage=e.ROOT/'results'/name(seed,0.)/f'round{n:02d}'
    out=e.ROOT/'baseline_audit'/f'n{n:02d}_seed{seed}_rep{rep}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return str(out)
    models={c:e.load_checkpoint(stage/(c+'_deployed.pt'))[0] for c in ['low','high']}
    other=json.loads((stage/'other_deployed.json').read_text())
    before=e.fingerprint(models,other)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    rid=30000+100*n+rep
    generation=[];paths=[]
    panel,meta=e.simulate(models,other,0.,seed,rid,1,CFG['train_paths'],CFG['train_periods'],CFG['burn'])
    generation.append(meta)
    for j,ix in enumerate(np.array_split(np.arange(CFG['train_paths']),CFG['training_chunks'])):
        p=cache/f'train{j}.npz';np.savez(p,**{k:v[ix] for k,v in panel.items()});paths.append(p)
    del panel
    val,meta=e.simulate(models,other,0.,seed,rid,50,CFG['validation_paths'],CFG['validation_periods'],CFG['burn']);generation.append(meta)
    # Independent from the shared trajectory evaluation bank; common across audit repetitions.
    ev,meta=e.simulate(models,other,0.,seed,40000+n,60,CFG['evaluation_paths'],CFG['evaluation_periods'],CFG['burn']);generation.append(meta)
    generated_finished=time.time()
    old_rows,old_market=e.score(models,other,0.,ev)
    training_started=time.time();new={};details={}
    for c in ['low','high']:
        new[c],details[c]=e.train(models[c],paths,val,seed,rid,out/(c+'_training.csv'),CFG['learning_rate'])
        e.checkpoint(new[c],dict(seed=seed,n=n,rep=rep,**details[c]),out/(c+'_refitted.pt'))
    new_other=e.refit_other(paths);e.dump(out/'other_refitted.json',new_other)
    new_rows,new_market=e.score(new,new_other,0.,ev)
    # market.J and delta use refitted coefficients; price moments still describe OLD Q.
    rows=[]
    for label,rr in [('before',old_rows),('refitted',new_rows)]:
        rows += [dict(version=label,**r) for r in rr]
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    response=[]
    old_heads,_=e.coefficients(models,other,0.);new_heads,_=e.coefficients(new,new_other,0.)
    for c in ['low','high']:
        z0=e.features(models[c],ev);z1=e.features(new[c],ev)
        mu0=e.predict(old_heads[c],z0,ev);mu1=e.predict(new_heads[c],z1,ev)
        h0=(mu0-e.ECO.risk_free_gross*e.flat(ev,'price'))/(e.GAMMA[0]*old_heads[c]['v'])
        h1=(mu1-e.ECO.risk_free_gross*e.flat(ev,'price'))/(e.GAMMA[0]*new_heads[c]['v'])
        response.append(dict(capacity=c,forecast_RMS_change=float(np.sqrt(np.mean((mu1-mu0)**2))),
            holding_RMS_change=float(np.sqrt(np.mean((h1-h0)**2))),
            holding_relative_RMS_change=float(np.sqrt(np.mean((h1-h0)**2)/np.mean(h0**2)))))
    pd.DataFrame(response).to_csv(out/'policy_changes.csv',index=False)
    assert e.fingerprint(models,other)==before
    e.dump(out/'complete.json',dict(seed=seed,round=n,rep=rep,profile_sha256=before,
        J_before=old_market['J'],J_refitted=new_market['J'],delta_before=old_market['delta'],delta_refitted=new_market['delta'],
        generation=generation,generation_finished=generated_finished,training_started=training_started,
        interpretation='One-step refit from identical starting policies on a fixed law; J_refitted is next-profile structural precision, not an evaluated new stationary price law.'))
    shutil.rmtree(cache)
    print('AUDIT',seed,n,rep,'J change pct',100*(new_market['J']/old_market['J']-1),flush=True)
    return str(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,default=12);p.add_argument('--reps',type=int,default=4);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(audit,s,a.round,r) for s in CFG['seeds'] for r in range(a.reps)]
        for f in as_completed(fs):print('DONE',f.result(),flush=True)
