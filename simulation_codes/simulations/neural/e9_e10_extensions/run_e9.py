import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from common import *

def trajectory(seed,x,mode):
 torch.set_num_threads(1);t=time.monotonic();out=ROOT/'results'/'E9'/tag(seed,x)/mode;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return tag(seed,x)+' '+mode+' cached'
 m,o=load_profile(ROOT/'source_e6'/(tag(seed,x)+'_both')/'round06');rows=[]
 for n in range(6,13):
  p=out/f'round{n:02d}';p.mkdir(exist_ok=True)
  if (p/'complete.json').exists():
   rows.append(json.loads((p/'state.json').read_text()))
   if n<12:m,o=load_profile(out/f'round{n+1:02d}')
   continue
  save_profile(p,m,o);fp=e.fingerprint(m,o);row=dict(seed=seed,x=x,mode=mode,round=n,**stats(m,o,x));e.dump(p/'state.json',row);rows.append(row)
  if n<12:
   key=60000+n if mode=='fresh' else 95000
   train,mt=e.simulate(m,o,x,seed,key,1,512,2048,512);val,mv=e.simulate(m,o,x,seed,key,50,64,1024,512)
   new,no,details=refit(m,o,train,val,seed,key,p);save_profile(out/f'round{n+1:02d}',new,no)
   assert e.fingerprint(m,o)==fp;e.dump(p/'generation.json',[mt,mv]);e.dump(p/'training.json',details);m,o=new,no
   del train,val
  e.dump(p/'complete.json',dict(profile_sha256=fp));print(tag(seed,x),mode,'round',n,'complete',flush=True)
 pd.DataFrame(rows).to_csv(out/'states.csv',index=False)
 if mode=='fresh' and x==.5:
  # Four one-step fits independently reset to the SAME terminal profile.
  fp=e.fingerprint(m,o);a=[]
  for rep in range(4):
   p=out/f'audit{rep}';p.mkdir(exist_ok=True)
   if (p/'complete.json').exists():a.append(json.loads((p/'state.json').read_text()));continue
   key=96000+rep;train,mt=e.simulate(m,o,x,seed,key,1,512,2048,512);val,mv=e.simulate(m,o,x,seed,key,50,64,1024,512)
   new,no,details=refit(m,o,train,val,seed,key,p);save_profile(p,new,no);r=dict(seed=seed,x=x,rep=rep,**stats(new,no,x));a.append(r)
   e.dump(p/'state.json',r);e.dump(p/'generation.json',[mt,mv]);e.dump(p/'training.json',details);e.dump(p/'complete.json',dict(inherited_sha256=fp));assert e.fingerprint(m,o)==fp
   del train,val;print(tag(seed,x),'audit',rep,'complete',flush=True)
  pd.DataFrame(a).to_csv(out/'audit.csv',index=False)
 e.dump(out/'complete.json',dict(elapsed_seconds=time.monotonic()-t));return tag(seed,x)+' '+mode+' complete'

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
 jobs=[(s,x,'fresh') for s in SEEDS for x in XS]+[(s,.5,'common_shocks') for s in SEEDS]
 with ProcessPoolExecutor(a.workers) as pool:
  for f in as_completed([pool.submit(trajectory,*j) for j in jobs]):print(f.result(),flush=True)
