import argparse,json,time,copy,hashlib,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
import engine as e
CFG=json.loads((e.ROOT/'config.json').read_text())

def name(seed,x):return f'seed{seed}_x{int(round(x*100)):03d}'
def save_panel(path,p):np.savez(path,**p)
def load_panel(path):
    with np.load(path) as p:return {k:p[k] for k in p.files}

def mixing_check(models,other,x,seed,n):
    kwargs=dict(seed=seed,round_index=900+n,bank=70,paths=16,periods=64,burn=CFG['burn'],return_trace=True)
    a,_=e.simulate(models,other,x,initial_price=-10.,**kwargs)
    b,_=e.simulate(models,other,x,initial_price=10.,**kwargs)
    diff=np.max(abs(a['trace']-b['trace']),axis=1)
    err=float(diff[CFG['burn']:].max())
    if err>1e-5:raise RuntimeError(f'Fast market did not forget initial prices: {err}')
    return dict(max_retained_price_difference=err,initial_gap=float(diff[0]),gap_at_32=float(diff[31]),gap_at_128=float(diff[127]))

def run(seed,x,max_updates=None):
    torch.set_num_threads(1);updates=CFG['slow_updates'] if max_updates is None else max_updates
    run_id=name(seed,x);out=e.ROOT/'results'/run_id;out.mkdir(exist_ok=True)
    models,other=e.initial(seed);start=time.monotonic()
    for n in range(updates+1):
        stage=out/f'round{n:02d}';stage.mkdir(exist_ok=True)
        completed=stage/'complete.json'
        if completed.exists() and (n==updates or (stage/'low_next.pt').exists()):
            meta=json.loads(completed.read_text());assert meta['x']==x
            if n<updates:
                models={cap:e.load_checkpoint(stage/(cap+'_next.pt'))[0] for cap in ['low','high']}
                other=json.loads((stage/'other_next.json').read_text())
            continue
        # An extended terminal stage is incomplete until its new fit is saved.
        completed.unlink(missing_ok=True)
        (out/'complete.json').unlink(missing_ok=True)
        before=e.fingerprint(models,other)
        for cap,m in models.items():e.checkpoint(m,dict(round=n,seed=seed,x=x),stage/(cap+'_deployed.pt'))
        e.dump(stage/'other_deployed.json',other)
        mix=mixing_check(models,other,x,seed,n)
        cache=e.ROOT/'data'/run_id/f'round{n:02d}';cache.mkdir(parents=True,exist_ok=True)
        paths=[];generation=[];generated_started=time.time()
        # Finish ALL price generation before constructing any optimizer or OLS fit.
        if n<updates:
            train,mt=e.simulate(models,other,x,seed,n,1,CFG['train_paths'],CFG['train_periods'],CFG['burn'])
            generation.append(mt)
            for chunk,ix in enumerate(np.array_split(np.arange(CFG['train_paths']),CFG['training_chunks'])):
                path=cache/f'train_{chunk}.npz';save_panel(path,{k:v[ix] for k,v in train.items()});paths.append(path)
            del train
            val,mv=e.simulate(models,other,x,seed,n,50,CFG['validation_paths'],CFG['validation_periods'],CFG['burn']);generation.append(mv)
        cal,mc=e.simulate(models,other,x,seed,n,51,CFG['calibration_paths'],CFG['calibration_periods'],CFG['burn']);generation.append(mc)
        # Common held-out primitive shocks across updates and adoption rates.
        ev,me=e.simulate(models,other,x,seed,10000,60,CFG['evaluation_paths'],CFG['evaluation_periods'],CFG['burn']);generation.append(me)
        generated_finished=time.time();assert all(g['profile_sha256']==before for g in generation)
        assert e.fingerprint(models,other)==before
        rows,market=e.score(models,other,x,ev,cal)
        pd.DataFrame(rows).to_csv(stage/'deployed_metrics.csv',index=False)
        pp=ev['price'][:,e.WINDOW-1:];path_stats=[]
        for half in [0,1]:
            block=pp[:,half*pp.shape[1]//2:(half+1)*pp.shape[1]//2]
            for j in range(len(block)):path_stats.append(dict(path=j,half=half,mean=float(block[j].mean()),variance=float(block[j].var())))
        pd.DataFrame(path_stats).to_csv(stage/'fast_market_blocks.csv',index=False)
        e.dump(stage/'market.json',market);e.dump(stage/'generation.json',generation)
        update_detail={};training_started=None
        if n<updates:
            training_started=time.time();assert training_started>=generated_finished
            new={}
            for cap in ['low','high']:
                new[cap],update_detail[cap]=e.train(models[cap],paths,val,seed,n,stage/(cap+'_training.csv'),CFG['learning_rate'])
            new_other=e.refit_other(paths)
            # Evaluation on the SAME inherited Q_n isolates learning from deployment.
            fitted,_=e.score(new,new_other,x,ev)
            pd.DataFrame(fitted).to_csv(stage/'refitted_on_old_law.csv',index=False)
            assert e.fingerprint(models,other)==before,'Old market profile was mutated while fitting candidates'
            for cap,m in new.items():e.checkpoint(m,dict(round=n+1,seed=seed,x=x,**update_detail[cap]),stage/(cap+'_next.pt'))
            e.dump(stage/'other_next.json',new_other)
            e.coefficients(new,new_other,x) # Validate new simultaneous deployment.
            models,other=new,new_other
        meta=dict(seed=seed,x=x,round=n,profile_sha256=before,generation_started=generated_started,
                  generation_finished=generated_finished,training_started=training_started,
                  generated_training_observations=CFG['train_paths']*CFG['train_periods'] if n<updates else 0,
                  mixing=mix,updates=update_detail,elapsed_seconds=time.monotonic()-start)
        e.dump(completed,meta);shutil.rmtree(cache)
        lo,hi=rows[0],rows[1]
        print(run_id,'round',n,'J',round(market['J'],5),'MSE gain',round(100*(1-hi['MSE']/lo['MSE']),3),
              'private',round(lo['private_gain_pct'],2),round(hi['private_gain_pct'],2),'seconds',round(time.monotonic()-start,1),flush=True)
    e.dump(out/'complete.json',dict(seed=seed,x=x,slow_updates=updates,elapsed_seconds=time.monotonic()-start))
    return run_id

def summarize():
    rows=[];fits=[];markets=[]
    for seed in CFG['seeds']:
      for x in CFG['adoption']:
        root=e.ROOT/'results'/name(seed,x)
        for n in range(CFG['slow_updates']+1):
          stage=root/f'round{n:02d}'
          if not (stage/'complete.json').exists():continue
          d=pd.read_csv(stage/'deployed_metrics.csv');d['seed']=seed;d['x']=x;d['round']=n;rows.append(d)
          mk=json.loads((stage/'market.json').read_text());markets.append(dict(seed=seed,x=x,round=n,**mk))
          if (stage/'refitted_on_old_law.csv').exists():
            f=pd.read_csv(stage/'refitted_on_old_law.csv');f['seed']=seed;f['x']=x;f['round']=n;fits.append(f)
    pd.concat(rows,ignore_index=True).to_csv(e.ROOT/'results/model_metrics.csv',index=False)
    pd.concat(fits,ignore_index=True).to_csv(e.ROOT/'results/refitting_metrics.csv',index=False)
    pd.DataFrame(markets).to_csv(e.ROOT/'results/market_metrics.csv',index=False)
    print(pd.DataFrame(markets).groupby(['x','round'])[['J','price_variance']].mean().to_string())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=3);a.add_argument('--seed',type=int);a.add_argument('--x',type=float);a.add_argument('--updates',type=int);a.add_argument('--summarize',action='store_true');args=a.parse_args()
    if args.summarize:summarize()
    elif args.seed is not None:run(args.seed,args.x,args.updates)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fs=[pool.submit(run,s,x) for s in CFG['seeds'] for x in CFG['adoption']]
            for f in as_completed(fs):print('DONE',f.result(),flush=True)
        summarize()
