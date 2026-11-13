"""Held-out forecast/demand changes on an immutable round6 anchor law."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
import torch
from common import *

def job(seed,x,mode):
 torch.set_num_threads(1);out=ROOT/'results'/'E9'/tag(seed,x)/mode
 if (out/'functional_check.csv').exists():return 'cached '+str(out)
 if not (out/'round12'/'complete.json').exists():return 'not_ready '+str(out)
 m,o=load_profile(ROOT/'source_e6'/(tag(seed,x)+'_both')/'round06');anchor,meta=e.simulate(m,o,x,seed,94000,60,128,1024,512)
 ysd=e.flat(anchor,'future_dividend').std();p=e.flat(anchor,'price');old={};dscale={};rows=[]
 for n in range(6,13):
  m,o=load_profile(out/f'round{n:02d}')
  for j,c in enumerate(['low','high','other']):
   if c!='other':h=e.prior.base.raw_head(m[c]);mu=e.predict(h,e.features(m[c],anchor),anchor);v=h['v'];gamma=e.GAMMA[0]
   else:mu=o['intercept']+o['beta']*e.flat(anchor,'private_signals',1)+o['chi']*p;v=o['v'];gamma=e.GAMMA[1]
   demand=(mu-e.ECO.risk_free_gross*p)/(gamma*v)
   if n==6:dscale[c]=np.sqrt(np.mean(demand*demand))
   else:
    om,od=old[c];rows.append(dict(seed=seed,x=x,mode=mode,round=n,capacity=c,forecast_RMS_change_pct_payoffSD=100*np.sqrt(np.mean((mu-om)**2))/ysd,demand_RMS_change_pct_initialRMS=100*np.sqrt(np.mean((demand-od)**2))/dscale[c]))
   old[c]=(mu,demand)
 pd.DataFrame(rows).to_csv(out/'functional_check.csv',index=False);e.dump(out/'anchor_generation.json',meta);return tag(seed,x)+' '+mode+' functional checked'
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();jobs=[(s,x,'fresh') for s in SEEDS for x in XS]+[(s,.5,'common_shocks') for s in SEEDS]
 with ProcessPoolExecutor(a.workers) as p:
  for f in as_completed([p.submit(job,*a) for a in jobs]):print(f.result(),flush=True)
