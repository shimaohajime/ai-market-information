"""E6 factorial group retraining, with E7 post-fit mean/risk accounting."""
import argparse,copy,hashlib,json,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import engine as e
CFG=json.loads((e.ROOT/'config.json').read_text())
ARMS={'none':(False,False),'group1':(True,False),'group2':(False,True),'both':(True,True)}

def name(seed,x,arm):return f'seed{seed}_x{round(x*100):03d}_{arm}'

def load_baseline(seed):
    p=e.ROOT/'baselines'/f'seed{seed}'
    return ({c:e.load_checkpoint(p/(c+'_deployed.pt'))[0] for c in ['low','high']},json.loads((p/'other_deployed.json').read_text()))

def group_hashes(models,other):
    def digest(c):
        m=models[c];h=hashlib.sha256()
        for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
        h.update(json.dumps(m.norm.to_dict(),sort_keys=True).encode());return h.hexdigest()
    return dict(low=digest('low'),high=digest('high'),other=hashlib.sha256(json.dumps(other,sort_keys=True).encode()).hexdigest())

def mix_check(models,other,x,seed,n):
    kw=dict(seed=seed,round_index=80000+n,bank=70,paths=16,periods=64,burn=CFG['burn'],return_trace=True)
    a,_=e.simulate(models,other,x,initial_price=-10.,**kw);b,_=e.simulate(models,other,x,initial_price=10.,**kw)
    err=float(np.max(abs(a['trace'][CFG['burn']:]-b['trace'][CFG['burn']:])))
    assert err<1e-5
    return err

def describe(models,other,x):
    h,a=e.coefficients(models,other,x)
    return dict(A=a['A'],J=a['J'],delta=a['delta'],
      low_iota=h['low']['iota'],high_iota=h['high']['iota'],other_iota=other['beta']/(e.GAMMA[1]*other['v']),
      A_low=a['masses'][0]*h['low']['iota'],A_high=a['masses'][1]*h['high']['iota'],
      A_group1=a['masses'][0]*h['low']['iota']+a['masses'][1]*h['high']['iota'],
      A_group2=a['masses'][2]*other['beta']/(e.GAMMA[1]*other['v']))

def e7(old,oo,new,no,x,p):
    """All four swaps use identical inherited histories and shocks. No refitting."""
    hh=[];gg=[]
    for models,other in [(old,oo),(new,no)]:
        h,_=e.coefficients(models,other,x);g=[]
        for c in ['low','high']:
            co=h[c]['coef'];z=e.features(models[c],p);g.append(co[0]+z@co[1:-2])
        g.append(np.full(len(g[0]),other['intercept']));gg.append(np.asarray(g))
        hh.append(dict(beta=np.array([h['low']['beta'],h['high']['beta'],other['beta']]),
            v=np.array([h['low']['v'],h['high']['v'],other['v']]),
            chi=np.array([h['low']['chi'],h['high']['chi'],other['chi']])))
    masses=np.array([e.MASS[0]*(1-x),e.MASS[0]*x,e.MASS[1]])
    gamma=np.array([e.GAMMA[0],e.GAMMA[0],e.GAMMA[1]])
    theta=e.flat(p,'theta');supply=e.flat(p,'supply')
    swaps=[]
    for mi,vi,label in [(0,0,'old'),(1,0,'mean_only'),(0,1,'variance_only'),(1,1,'joint')]:
        w=masses/(gamma*hh[vi]['v']);iota=hh[mi]['beta']/(gamma*hh[vi]['v'])
        A=float(masses@iota);delta=float(w@(e.ECO.risk_free_gross-hh[mi]['chi']));assert delta>0
        K=w@gg[mi];price=(K+A*theta-supply)/delta
        assert np.max(abs(K+A*theta-delta*price-supply))<1e-10
        swaps.append(dict(swap=label,J=A*A/e.ZETA2,A=A,delta=delta,price_mean=float(price.mean()),
            price_variance=float(price.var()),price_rms_change=float(np.sqrt(np.mean((price-e.flat(p,'price'))**2))),
            **{c+'_iota':float(iota[j]) for j,c in enumerate(['low','high','other'])},
            A_group1=float(masses[:2]@iota[:2]),A_group2=float(masses[2]*iota[2])))
    sd=pd.DataFrame(swaps).set_index('swap');assert sd.loc['old','price_rms_change']<1e-6
    parts=[]
    for metric in ['J','A','low_iota','high_iota','other_iota','A_group1','A_group2','price_variance']:
        y=sd[metric];mean=.5*((y.mean_only-y.old)+(y.joint-y.variance_only));risk=.5*((y.variance_only-y.old)+(y.joint-y.mean_only))
        total=y.joint-y.old;assert abs(mean+risk-total)<1e-10
        parts.append(dict(metric=metric,old=y.old,new=y.joint,total=total,mean_contribution=mean,variance_contribution=risk,
            interaction=y.joint-y.mean_only-y.variance_only+y.old))
    return pd.DataFrame(swaps),pd.DataFrame(parts)

def run(seed,x,arm):
    torch.set_num_threads(1);g1,g2=ARMS[arm];out=e.ROOT/'results'/name(seed,x,arm);out.mkdir(exist_ok=True)
    models,other=load_baseline(seed);basefp=e.fingerprint(models,other);start=time.monotonic()
    last=0 if arm=='none' else CFG['slow_updates']
    for n in range(last+1):
        st=out/f'round{n:02d}';st.mkdir(exist_ok=True);done=st/'complete.json'
        if done.exists():
            if n<last:
                models={c:e.load_checkpoint(st/(c+'_next.pt'))[0] for c in ['low','high']};other=json.loads((st/'other_next.json').read_text())
            continue
        before=e.fingerprint(models,other);gh=group_hashes(models,other)
        for c,m in models.items():e.checkpoint(m,dict(seed=seed,x=x,arm=arm,round=n),st/(c+'_deployed.pt'))
        e.dump(st/'other_deployed.json',other)
        mix=mix_check(models,other,x,seed,n)
        cache=e.ROOT/'data'/name(seed,x,arm)/f'round{n:02d}';cache.mkdir(parents=True,exist_ok=True)
        generation=[];paths=[];began=time.time()
        if n<last:
            panel,meta=e.simulate(models,other,x,seed,60000+n,1,CFG['train_paths'],CFG['train_periods'],CFG['burn']);generation.append(meta)
            for j,ix in enumerate(np.array_split(np.arange(CFG['train_paths']),CFG['training_chunks'])):
                p=cache/f'train{j}.npz';np.savez(p,**{k:v[ix] for k,v in panel.items()});paths.append(p)
            del panel
            if g1:
                val,meta=e.simulate(models,other,x,seed,60000+n,50,CFG['validation_paths'],CFG['validation_periods'],CFG['burn']);generation.append(meta)
        # New evaluation stream shared over arms, x and epochs; independent of training.
        ev,meta=e.simulate(models,other,x,seed,70000,60,CFG['evaluation_paths'],CFG['evaluation_periods'],CFG['burn']);generation.append(meta)
        generation_finished=time.time();assert all(z['profile_sha256']==before for z in generation)
        assert e.fingerprint(models,other)==before
        rows,market=e.score(models,other,x,ev);market.update(describe(models,other,x))
        pd.DataFrame(rows).to_csv(st/'deployed_metrics.csv',index=False);e.dump(st/'market.json',market);e.dump(st/'generation.json',generation)
        begun=None;details={}
        if n<last:
            begun=time.time();assert begun>=generation_finished
            new={};no=other.copy()
            for c in ['low','high']:
                if g1:new[c],details[c]=e.train(models[c],paths,val,seed,60000+n,st/(c+'_training.csv'),CFG['learning_rate'])
                else:new[c]=copy.deepcopy(models[c])
            if g2:no=e.refit_other(paths)
            afterhash=group_hashes(new,no)
            if not g1:assert gh['low']==afterhash['low'] and gh['high']==afterhash['high']
            if not g2:assert gh['other']==afterhash['other']
            fitted,_=e.score(new,no,x,ev);pd.DataFrame(fitted).to_csv(st/'refitted_on_old_law.csv',index=False)
            swaps,parts=e7(models,other,new,no,x,ev)
            swaps.to_csv(st/'E7_swaps.csv',index=False);parts.to_csv(st/'E7_decomposition.csv',index=False)
            for c,m in new.items():e.checkpoint(m,dict(seed=seed,x=x,arm=arm,round=n+1),st/(c+'_next.pt'))
            e.dump(st/'other_next.json',no)
            assert e.fingerprint(models,other)==before
            models,other=new,no
        e.dump(done,dict(seed=seed,x=x,arm=arm,round=n,baseline_sha256=basefp,profile_sha256=before,group_hashes=gh,
            generation_started=began,generation_finished=generation_finished,training_started=begun,
            train_observations=CFG['train_paths']*CFG['train_periods'] if n<last else 0,
            updates=details,mixing_error=mix,elapsed_seconds=time.monotonic()-start))
        shutil.rmtree(cache)
        print(name(seed,x,arm),'round',n,'J',round(market['J'],6),'A1',round(market['A_group1'],5),'A2',round(market['A_group2'],5),'seconds',round(time.monotonic()-start,1),flush=True)
    e.dump(out/'complete.json',dict(seed=seed,x=x,arm=arm,slow_updates=last,reported_horizons=CFG['slow_updates'],
       none_note='Frozen law/policy reused at all horizons; only one state simulated.' if arm=='none' else None))
    return name(seed,x,arm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--seed',type=int);p.add_argument('--x',type=float);p.add_argument('--arm',choices=list(ARMS));a=p.parse_args()
    if a.seed is not None:run(a.seed,a.x,a.arm)
    else:
        # Cheap controls first; neural branches then share the same primitive streams.
        jobs=[(s,x,arm) for arm in ['none','group2','group1','both'] for x in CFG['adoption'] for s in CFG['seeds']]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            fs=[pool.submit(run,*j) for j in jobs]
            for f in as_completed(fs):print('DONE',f.result(),flush=True)
