import time,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from common import *
from economies import MODES,simulate

def initial_job(seed,mode):
 torch.set_num_threads(1);out=ROOT/'results'/'E10'/mode/f'seed{seed}'/'inherited_fit';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return mode+f' seed{seed} initial cached'
 t=time.monotonic();m,o=load_profile(ROOT/'baselines'/f'seed{seed}');fp=e.fingerprint(m,o)
 train,mt=simulate(mode,m,o,0,seed,101000,1,512,2048);val,mv=simulate(mode,m,o,0,seed,101000,50,64,1024)
 new,no,details=refit(m,o,train,val,seed,101000,out);save_profile(out,new,no);assert e.fingerprint(m,o)==fp
 rows=[]
 for x in XS:rows.append(dict(seed=seed,economy=mode,x=x,**stats(new,no,x)))
 pd.DataFrame(rows).to_csv(out/'impact.csv',index=False);e.dump(out/'generation.json',[mt,mv]);e.dump(out/'training.json',details)
 e.dump(out/'complete.json',dict(inherited_sha256=fp,fitted_sha256=e.fingerprint(new,no),elapsed_seconds=time.monotonic()-t));return mode+f' seed{seed} initial complete'

def measurement_job(seed,mode,x):
 torch.set_num_threads(1);t=time.monotonic();out=ROOT/'results'/'E10'/mode/f'seed{seed}'/f'x{round(x*100):03d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return mode+' '+tag(seed,x)+' cached'
 m,o=load_profile(out.parent/'inherited_fit');fp=e.fingerprint(m,o);gen=[]
 def sim(bank,paths,periods,key=103000,**kw):
  p,meta=simulate(mode,m,o,x,seed,key,bank,paths,periods,**kw);gen.append(meta);return p
 if not (out/'candidate_done.json').exists():
  train=sim(1,512,2048,key=102000);val=sim(50,64,1024,key=102000)
  cand,_,detail=refit(m,o,train,val,seed,102000,out,fit_other=False)
  for c in cand:e.checkpoint(cand[c],{},out/(c+'_candidate.pt'))
  e.dump(out/'training.json',detail);e.dump(out/'training_generation.json',gen.copy());e.dump(out/'candidate_done.json',dict(market_sha256=fp));del train,val
 else:cand={c:e.load_checkpoint(out/(c+'_candidate.pt'))[0] for c in ['low','high']}
 cal=sim(3,128,2048);z={c:e.features(v,cal) for c,v in cand.items()};z['nested']=np.c_[z['low'],z['high']];y=e.flat(cal,'future_dividend');heads={}
 for c in z:
  for hp in [0,1]:heads[f'E3_{c}_{hp}']=fit(design(z[c],cal,price=bool(hp)),y)
  heads[f'E4_ols_{c}']=fit(design(z[c],cal,private=True,price=True),y)
 for c in cand:heads[f'E4_neural_{c}']=e.prior.base.raw_head(cand[c])
 e.dump(out/'heads.json',heads);del cal,z,y
 # Empirical coupling check, separate from training and evaluation streams.
 a=sim(80,16,64,initial_price=-10,return_trace=True);b=sim(80,16,64,initial_price=10,return_trace=True)
 mix=float(np.max(abs(a['trace'][512:]-b['trace'][512:])));assert mix<1e-5;del a,b
 banks=[];pathrows=[];lw_all={};market=[]
 for bank in [20,21,22,23]:
  p=sim(bank,256,1024);z={c:e.features(v,p) for c,v in cand.items()};z['nested']=np.c_[z['low'],z['high']];y=e.flat(p,'future_dividend');price=e.flat(p,'price');mean=e.flat(p,'payoff_mean');nv=e.flat(p,'noise_variance')
  market.append(dict(bank=bank,price_variance=price.var(),payoff_variance=y.var(),noise_variance_mean=nv.mean(),**stats(m,o,x)))
  for c in z:
   mu0=design(z[c],p)@heads[f'E3_{c}_0']['coef'];mu1=design(z[c],p,price=True)@heads[f'E3_{c}_1']['coef'];aa=(y-mu0)**2;bb=(y-mu1)**2
   banks.append(dict(bank=bank,kind='E3',model=c,MSE_no_price=aa.mean(),MSE_price=bb.mean(),gain=(aa-bb).mean()))
   for k,(v,w) in enumerate(zip(aa.reshape(256,1024).mean(1),bb.reshape(256,1024).mean(1))):pathrows.append(dict(bank=bank,path=k,model='E3_'+c,MSE_no_price=v,MSE_price=w,gain=v-w))
  for key,h in heads.items():
   if not key.startswith('E4'):continue
   c=key.split('_')[-1];mu=design(z[c],p,private=True,price=True)@h['coef'];lw=logweight(mu,h['v'],price,mean,nv).reshape(256,1024);lw_all.setdefault(key,[]).append(lw)
   banks.append(dict(bank=bank,kind='E4',model=key,MSE=np.mean((y-mu)**2),v=h['v'],**ce_stats(lw)))
   for k in range(256):pathrows.append(dict(bank=bank,path=k,model=key,logmeanweight=float(logsumexp(lw[k])-np.log(1024))))
  print(mode,tag(seed,x),'bank',bank,flush=True)
 rows=[dict(model=key,**ce_stats(np.concatenate(arr))) for key,arr in lw_all.items()]
 pd.DataFrame(rows).to_csv(out/'CE.csv',index=False);pd.DataFrame(banks).to_csv(out/'banks.csv',index=False);pd.DataFrame(pathrows).to_csv(out/'paths.csv',index=False);pd.DataFrame(market).to_csv(out/'market.csv',index=False)
 e.dump(out/'generation.json',gen);assert e.fingerprint(m,o)==fp;e.dump(out/'complete.json',dict(market_sha256=fp,mixing_error=mix,elapsed_seconds=time.monotonic()-t));return mode+' '+tag(seed,x)+' complete'

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['initial','measure'],required=True);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args()
 with ProcessPoolExecutor(a.workers) as pool:
  jobs=[pool.submit(initial_job,s,mode) for mode in MODES for s in SEEDS] if a.phase=='initial' else [pool.submit(measurement_job,s,mode,x) for mode in MODES for s in SEEDS for x in XS]
  for f in as_completed(jobs):print(f.result(),flush=True)
