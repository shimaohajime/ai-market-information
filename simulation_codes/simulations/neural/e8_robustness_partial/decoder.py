"""Matched-sample affine/RFF ridge diagnostics; model selection uses bank4 only."""
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve
from run_e8 import ROOT,BASE,e,tag,load_market

class Maps:
 def __init__(self,z,p,size):
  self.size=size;self.mu={c:a.mean(0) for c,a in z.items()};self.sd={c:np.maximum(a.std(0),1e-8) for c,a in z.items()};self.pm=p.mean();self.ps=p.std();self.W={};self.phase={}
  for c,seed in [('low',8361),('high',8362),('nested',8363)]:
   for hp in [0,1]:
    rng=np.random.default_rng(seed+hp*100);d=z[c].shape[1]+hp;self.W[c,hp]=rng.normal(size=(d,128))/np.sqrt(d);self.phase[c,hp]=rng.uniform(0,2*np.pi,128)
 def transform(self,z,p,c,hp):
  a=(z[c]-self.mu[c])/self.sd[c];pp=((p-self.pm)/self.ps)[:,None];out=[a]
  if self.size:
   out.append(np.sqrt(2/self.size)*np.cos(a@self.W[c,0][:,:self.size]+self.phase[c,0][:self.size]))
   if c=='nested':
    low=(z['low']-self.mu['low'])/self.sd['low'];out.append(np.sqrt(2/self.size)*np.cos(low@self.W['low',0][:,:self.size]+self.phase['low',0][:self.size]))
  if hp:
   out.append(pp)
   if self.size:
    out.append(np.sqrt(2/self.size)*np.cos(np.c_[a,pp]@self.W[c,1][:,:self.size]+self.phase[c,1][:self.size]))
    if c=='nested':out.append(np.sqrt(2/self.size)*np.cos(np.c_[low,pp]@self.W['low',1][:,:self.size]+self.phase['low',1][:self.size]))
  return np.column_stack(out)

LAMBDAS=[1e-6,1e-5,1e-4,1e-3,1e-2,.1,1.]
def ridge_fit(X,y,V,yv):
 xm=X.mean(0);xs=np.maximum(X.std(0),1e-8);ym=y.mean();a=(X-xm)/xs;b=(V-xm)/xs
 gram=a.T@a/len(a);cross=a.T@(y-ym)/len(a);candidates=[]
 for l in LAMBDAS:
  co=solve(gram+l*np.eye(len(gram)),cross,assume_a='pos');candidates.append((np.mean((ym+b@co-yv)**2),l,co))
 loss,lam,co=min(candidates,key=lambda v:v[0]);return dict(coef=co/xs,intercept=ym-xm@(co/xs),ridge=lam,validation_MSE=loss,features=len(co),fit_observations=len(y))

def job(m):
 torch.set_num_threads(1);t=time.monotonic();name=tag(m);out=ROOT/'results'/name/'decoders';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return name+' decoders cached'
 models,other=load_market(m);cand={c:e.load_checkpoint(BASE/'results'/name/(c+'_candidate.pt'))[0] for c in ['low','high']};gen=[]
 def panel(bank,paths,periods):
  p,meta=e.simulate(models,other,m['x'],m['seed'],91000,bank,paths,periods,512);gen.append(meta);return p
 def extract(p):
  z={c:e.features(v,p) for c,v in cand.items()};z['nested']=np.c_[z['low'],z['high']];return z,e.flat(p,'price'),e.flat(p,'future_dividend')
 # Generate original128 paths first so subset reproduces identical calibration shocks.
 p=panel(3,128,2048);p={k:v[:32] for k,v in p.items()};z,pr,y=extract(p);del p
 p=panel(4,64,1024);zv,pv,yv=extract(p);del p
 maps={};heads={}
 for size in [0,64,128]:
  maps[size]=Maps(z,pr,size)
  for c in ['low','high','nested']:
   for hp in [0,1]:
    X=maps[size].transform(z,pr,c,hp);V=maps[size].transform(zv,pv,c,hp);heads[size,c,hp]=ridge_fit(X,y,V,yv);del X,V
  print(name,'fitted decoder',size,flush=True)
 e.dump(out/'heads.json',{f'{s}_{c}_{hp}':v for (s,c,hp),v in heads.items()});del z,pr,y,zv,pv,yv
 banks=[];paths=[]
 for bank in [20,21,22,23]:
  p=panel(bank,256,1024);z,pr,y=extract(p);del p
  losses={(size,c,hp):np.empty(len(y)) for size in maps for c in ['low','high','nested'] for hp in [0,1]}
  for j in range(0,len(y),4096):
   sl=slice(j,j+4096);zz={c:a[sl] for c,a in z.items()}
   for key,h in heads.items():
    size,c,hp=key;X=maps[size].transform(zz,pr[sl],c,hp);pred=h['intercept']+X@h['coef'];losses[key][sl]=(y[sl]-pred)**2
  for size in maps:
   for c in ['low','high','nested']:
    a=losses[size,c,0];b=losses[size,c,1];banks.append(dict(bank=bank,size=size,model=c,MSE_no_price=a.mean(),MSE_price=b.mean(),gain=(a-b).mean()))
    for k,(aa,bb) in enumerate(zip(a.reshape(256,1024).mean(1),b.reshape(256,1024).mean(1))):paths.append(dict(bank=bank,path=k,size=size,model=c,MSE_no_price=aa,MSE_price=bb,gain=aa-bb))
  print(name,'decoder bank',bank,flush=True)
 pd.DataFrame(banks).to_csv(out/'banks.csv',index=False);pd.DataFrame(paths).to_csv(out/'paths.csv',index=False);e.dump(out/'generation.json',gen)
 e.dump(out/'complete.json',dict(elapsed_seconds=time.monotonic()-t,market_sha256=e.fingerprint(models,other),fit_observations=65536,evaluation_observations=1048576))
 return name+' decoders complete'

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();ms=json.loads((BASE/'markets.json').read_text())
 with ProcessPoolExecutor(a.workers) as p:
  for f in as_completed([p.submit(job,m) for m in ms]):print(f.result(),flush=True)
