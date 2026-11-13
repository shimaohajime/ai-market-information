"""One additional-data/optimization check on the first x=.5 slow update."""
import json,shutil
import numpy as np
import pandas as pd
import torch
import engine as e

torch.set_num_threads(1);seed=18013;x=.5;root=e.ROOT/'results/seed18013_x050/round00'
out=e.ROOT/'results/training_adequacy';out.mkdir(exist_ok=True)
if not (out/'complete.json').exists():
    frozen={c:e.load_checkpoint(root/(c+'_deployed.pt'))[0] for c in ['low','high']}
    fo=json.loads((root/'other_deployed.json').read_text())
    candidate={c:e.load_checkpoint(root/(c+'_next.pt'))[0] for c in ['low','high']}
    other=json.loads((root/'other_next.json').read_text())
    p,meta=e.simulate(frozen,fo,x,seed,20000,1,512,2048,512)
    cache=e.ROOT/'data/training_adequacy';cache.mkdir(exist_ok=True);paths=[]
    for j,ix in enumerate(np.array_split(np.arange(512),8)):
        path=cache/f'train{j}.npz';np.savez(path,**{k:v[ix] for k,v in p.items()});paths.append(path)
    del p
    val,_=e.simulate(frozen,fo,x,seed,0,50,64,1024,512)
    cal,_=e.simulate(frozen,fo,x,seed,0,51,64,1024,512)
    ev,_=e.simulate(frozen,fo,x,seed,10000,60,128,1024,512)
    rows0,_=e.score(candidate,other,x,ev,cal)
    newer={}
    for c in ['low','high']:
        newer[c],detail=e.train(candidate[c],paths,val,seed,20000,out/(c+'_extra_curve.csv'),lr=.0003)
        e.checkpoint(newer[c],detail,out/(c+'_extra.pt'))
    rows1,_=e.score(newer,other,x,ev,cal)
    rows=[]
    for version,rr in [('primary',rows0),('additional_million',rows1)]:
        for r in rr:rows.append(dict(version=version,**r))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    e.dump(out/'complete.json',dict(seed=seed,x=x,extra_observations=1048576,generation=meta,
        interpretation='Additional optimization on the same frozen Q0; group 2 unchanged; diagnostic is not deployed in the main trajectory.'))
    shutil.rmtree(cache)
d=pd.read_csv(out/'metrics.csv').set_index(['version','capacity'])
for c in ['low','high']:
    a=d.loc[('primary',c)];b=d.loc[('additional_million',c)]
    print(c,'extra MSE improvement',100*(1-b.MSE/a.MSE),'iota change',100*(b.iota/a.iota-1),
          'primary vs independent OLS MSE gap',100*(a.MSE/a.calibrated_MSE-1),flush=True)
