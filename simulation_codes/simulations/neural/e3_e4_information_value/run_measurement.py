"""E3/E4 on immutable E6 laws. See PROTOCOL.txt for estimands and budgets."""
import json,time,shutil,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
import engine as e
ROOT=Path(__file__).resolve().parent

def ce_stats(logw,gamma):
    a=np.asarray(logw).ravel();ls=logsumexp(a);w=np.exp(a-ls)
    return dict(CE=float(-(ls-np.log(len(a)))/gamma),ESS=float(1/(w@w)),max_weight=float(w.max()),top_01pct_weight=float(np.sort(w)[-max(1,len(w)//1000):].sum()))

def integrated_logweight(mu,v,price,theta,gamma):
    h=(mu-e.ECO.risk_free_gross*price)/(gamma*v)
    return -gamma*h*(e.LAM*theta-e.ECO.risk_free_gross*price)+.5*gamma**2*h*h*e.SIG2

def fit(X,y):
    co,res,rank,sv=np.linalg.lstsq(X,y,rcond=1e-10)
    return dict(coef=co,v=float(np.mean((y-X@co)**2)),rank=int(rank),columns=X.shape[1],condition=float(sv[0]/sv[-1]))

def design(z,p,private=False,price=False):
    cols=[np.ones((len(z),1)),z]
    if private:cols.append(e.flat(p,'private_signals',0)[:,None])
    if price:cols.append(e.flat(p,'price')[:,None])
    return np.concatenate(cols,axis=1)

def run(market):
    torch.set_num_threads(1);t=time.monotonic();seed=market['seed'];x=market['x'];tag=f'seed{seed}_x{round(x*100):03d}'
    out=ROOT/'results'/tag;out.mkdir(exist_ok=True);cache=ROOT/'data'/tag;cache.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return tag+' cached'
    source=ROOT/market['path'];models={c:e.load_checkpoint(source/(c+'_deployed.pt'))[0] for c in ['low','high']};other=json.loads((source/'other_deployed.json').read_text())
    fp=e.fingerprint(models,other);assert fp==market['profile_sha256'];gen=[]
    def sim(bank,paths,periods):
        panel,meta=e.simulate(models,other,x,seed,91000,bank,paths,periods,512);gen.append(meta);assert meta['profile_sha256']==fp;return panel
    candidate={}
    if not (out/'high_candidate.pt').exists():
        panel=sim(1,512,2048);paths=[]
        for j,ix in enumerate(np.array_split(np.arange(512),8)):
            p=cache/f'train{j}.npz';np.savez(p,**{k:v[ix] for k,v in panel.items()});paths.append(p)
        del panel
        val=sim(2,64,1024);details={}
        for c in ['low','high']:
            # Same pre-adoption initialization across x, distinct from frozen market participants.
            start=e.load_checkpoint(ROOT/'baselines'/f'seed{seed}'/(c+'_deployed.pt'))[0]
            candidate[c],details[c]=e.train(start,paths,val,seed,91000,out/(c+'_training.csv'))
            e.checkpoint(candidate[c],dict(seed=seed,x=x),out/(c+'_candidate.pt'))
        e.dump(out/'training.json',dict(details=details,generation=gen.copy()))
        del val
        for p in paths:p.unlink()
    else:candidate={c:e.load_checkpoint(out/(c+'_candidate.pt'))[0] for c in ['low','high']}
    # Separate independent head-fitting bank. No evaluation-dependent model selection.
    cal=sim(3,128,2048);z={c:e.features(v,cal) for c,v in candidate.items()};z['nested']=np.column_stack((z['low'],z['high']))
    heads={};y=e.flat(cal,'future_dividend')
    for c in z:
        for price in [False,True]:heads[f'E3_{c}_{int(price)}']=fit(design(z[c],cal,price=price),y)
        heads[f'E4_ols_{c}']=fit(design(z[c],cal,private=True,price=True),y)
    for c in candidate:heads[f'E4_neural_{c}']=e.prior.base.raw_head(candidate[c])
    e.dump(out/'heads.json',heads);e.dump(out/'resources.json',{c:v.resource_stats() for c,v in candidate.items()});del cal,z,y
    metrics=[];pathrows=[];lw_all={};bankrows=[]
    for bank in [10,11,12,13]:
        p=sim(bank,256,1024);z={c:e.features(v,p) for c,v in candidate.items()};z['nested']=np.column_stack((z['low'],z['high']))
        y=e.flat(p,'future_dividend');price=e.flat(p,'price');theta=e.flat(p,'theta');se={}
        for c in z:
            for hasp in [False,True]:
                name=f'E3_{c}_{int(hasp)}';mu=design(z[c],p,price=hasp)@np.asarray(heads[name]['coef']);se[name]=(y-mu)**2
            a=se[f'E3_{c}_0'];b=se[f'E3_{c}_1']
            bankrows.append(dict(seed=seed,x=x,bank=bank,kind='E3',model=c,MSE_no_price=a.mean(),MSE_price=b.mean(),gain=(a-b).mean(),relative_gain=1-b.mean()/a.mean()))
        for name,h in heads.items():
            if not name.startswith('E4'):continue
            c=name.split('_')[-1];mu=design(z[c],p,private=True,price=True)@np.asarray(h['coef']);v=h['v'];g=e.GAMMA[0]
            lw=integrated_logweight(mu,v,price,theta,g).reshape(256,1024);lw_all.setdefault(name,[]).append(lw)
            st=ce_stats(lw,g);st.update(seed=seed,x=x,bank=bank,kind='E4',model=name,MSE=float(np.mean((y-mu)**2)),v=v,calibration=float(np.mean((y-mu)**2)/v))
            bankrows.append(st)
            # Independent paths are the block units; keep sums on log scale.
            for k in range(256):pathrows.append(dict(seed=seed,x=x,bank=bank,path=k,model=name,logmeanweight=float(logsumexp(lw[k])-np.log(1024))))
        for c in z:
            a=se[f'E3_{c}_0'].reshape(256,1024).mean(1);b=se[f'E3_{c}_1'].reshape(256,1024).mean(1)
            for k in range(256):pathrows.append(dict(seed=seed,x=x,bank=bank,path=k,model='E3_'+c,MSE_no_price=a[k],MSE_price=b[k],gain=a[k]-b[k]))
        print(tag,'evaluated bank',bank,flush=True)
    for name,arr in lw_all.items():
        a=np.concatenate(arr);st=ce_stats(a,e.GAMMA[0]);st.update(seed=seed,x=x,model=name,CE_first_half=ce_stats(np.concatenate(arr[:2]),e.GAMMA[0])['CE'],CE_last_half=ce_stats(np.concatenate(arr[2:]),e.GAMMA[0])['CE'],observations=a.size)
        metrics.append(st)
    pd.DataFrame(metrics).to_csv(out/'CE.csv',index=False);pd.DataFrame(pathrows).to_csv(out/'paths.csv',index=False);pd.DataFrame(bankrows).to_csv(out/'banks.csv',index=False)
    assert e.fingerprint(models,other)==fp
    e.dump(out/'generation.json',gen);e.dump(out/'complete.json',dict(seed=seed,x=x,market_sha256=fp,elapsed_seconds=time.monotonic()-t,evaluation_observations=4*256*1024))
    shutil.rmtree(cache);return tag+' complete'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=7);args=ap.parse_args()
    markets=json.loads((ROOT/'markets.json').read_text())
    with ProcessPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(run,m) for m in markets]):print(f.result(),flush=True)
