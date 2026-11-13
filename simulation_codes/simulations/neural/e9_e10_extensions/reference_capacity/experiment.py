from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,math,time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_s4',ROOT/'reference/experiment.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from marketlab.learning import ResourceForecaster,prepare_panel,set_seed,gaussian_loss,_init_ols,checkpoint,load_checkpoint,make_windows
CFG=json.loads((ROOT/'config.json').read_text());dump=base.dump

def candidate(name):return next(c for c in CFG['candidates'] if c['name']==name)

def primitives(c):
    old=base.economy_8d('E1_aligned');prior=float(old.target_loading@old.stationary_cov@old.target_loading)
    eco=base.economy_8d('E1_aligned',dimension=c['dimension'])
    a=eco.target_loading*np.sqrt(prior/(eco.target_loading@eco.stationary_cov@eco.target_loading))
    eco=replace(eco,target_loading=a,payoff_loading=base.LAM*a)
    nu=np.array([c['nu0'],.09])
    def bv(A):
        P=1/(1/prior+A*A/base.ZETA2);beta=base.LAM*P/(P+nu)
        return P,beta,base.SIG2+base.LAM*nu*beta
    def fun(A):
        _,b,v=bv(A);return A-np.sum(base.MASS*b/(base.GAMMA*v))
    A=brentq(fun,0,100,xtol=1e-14);P,beta,v=bv(A);w=base.MASS/(base.GAMMA*v)
    k=base.LAM*nu/(P+nu)*A*P/base.ZETA2
    delta=eco.risk_free_gross*np.sum(w)/(1+np.sum(w*k))
    eq=dict(prior=prior,P=P,beta=beta,v=v,chi=k*delta,A=A,delta=delta,J=A*A/base.ZETA2,nu=nu)
    cov=eco.stationary_cov;val,vec=np.linalg.eigh(cov)
    sqrtcov=(vec*np.sqrt(val))@vec.T;invsqrt=(vec*(1/np.sqrt(val)))@vec.T
    u=sqrtcov@a/np.sqrt(prior);target=np.ones(eco.dim)/np.sqrt(eco.dim)
    d=u-target
    H=np.eye(eco.dim) if np.linalg.norm(d)<1e-12 else np.eye(eco.dim)-2*np.outer(d,d)/(d@d)
    W=H@invsqrt
    return eco,eq,W

def panel(c,phase,seed,n,bank=None):
    eco,eq,W=primitives(c)
    root=(310000000 if phase in ['pilot','stream_pilot'] else 510000000)+10000*seed
    ds=root+11 if bank is None else root+100000000+997*bank
    b=400;shock=eco.draw_bank(n,burn=b,seed=ds);f=shock.factor
    theta_old=f@eco.target_loading
    eta=np.random.default_rng(ds+71).normal(0,np.sqrt(eq['prior']),len(f))
    theta=np.sqrt(1-c['rho'])*theta_old+np.sqrt(c['rho'])*eta
    eps=shock.future_dividend-base.LAM*theta_old;payoff=base.LAM*theta+eps
    q=f@W.T if c['public_map']=='balanced' else f/np.sqrt(np.diag(eco.stationary_cov))
    public=q+CFG['public_cubic_strength']*q**3
    public+=CFG['public_noise_sd']*np.random.default_rng(ds+53).standard_normal(q.shape)
    private=theta[:,None]+np.sqrt(eq['nu'])*shock.private_stdnormal[:,:2]
    price=(eq['A']*theta-shock.supply_shock)/eq['delta']
    u=np.column_stack((np.r_[0.,price[:-1]],np.r_[0.,payoff[:-1]],public))
    return dict(public_input=u[b:],price=price[b:],private_signals=private[b:],future_dividend=payoff[b:],
       theta=theta[b:],theta_old=theta_old[b:],eta=eta[b:],factor=f[b:],q=q[b:],
       public=public[b:],supply=shock.supply_shock[b:],seed=ds),eco,eq

def features(model,p):
    w=make_windows((p['public_input'].astype('float32')-model.norm.u_mean)/model.norm.u_sd,model.window);z=[]
    with torch.no_grad():
        for i in range(0,len(w),1024):z.append(model.encode(torch.as_tensor(np.array(w[i:i+1024],copy=True))).numpy().astype(float))
    return np.concatenate(z)

def validation(model,data):
    se=0.;loss=0.;idx=data['val_idx'];model.eval()
    with torch.no_grad():
        for ix in torch.as_tensor(idx).split(1024):
            mu,lv=model(data['windows'][ix],data['s'][ix],data['p'][ix])
            se+=float(((mu-data['y'][ix])**2).sum());loss+=float(gaussian_loss(mu,lv,data['y'][ix]))*len(ix)
    return dict(MSE=se/len(idx)*model.norm.y_sd**2,NLL=loss/len(idx))

def normalized_data(model,p):
    n=model.norm;first=model.window-1
    u=(p['public_input'].astype('float32')-n.u_mean)/n.u_sd
    windows=make_windows(u,model.window)
    return dict(windows=torch.as_tensor(np.array(windows,copy=True)),
       s=torch.as_tensor(((p['private_signals'][first:,0]-n.s_mean)/n.s_sd).astype('float32')),
       p=torch.as_tensor(((p['price'][first:]-n.p_mean)/n.p_sd).astype('float32')),
       y=torch.as_tensor(((p['future_dividend'][first:]-n.y_mean)/n.y_sd).astype('float32')))

def fit_heads(model,p):
    z=features(model,p);first=model.window-1
    X=np.column_stack((np.ones(len(z)),z,p['private_signals'][first:,0],p['price'][first:]))
    coef,v,diag=base.least_squares(X,p['future_dividend'][first:]);beta=coef[-2]
    full=dict(coef=coef,v=v,beta=beta,chi=coef[-1],iota=beta/(base.GAMMA[0]*v),diagnostic=diag)
    cn,vn,dn=base.least_squares(np.delete(X,-2,axis=1),p['future_dividend'][first:])
    no=dict(coef=np.insert(cn,len(cn)-1,0.),v=vn,beta=0.,chi=cn[-1],iota=0.,diagnostic=dn)
    return dict(full=full,no_private=no)

def mse(h,z,p,rows=None):
    first=CFG['context']-1;c=np.asarray(h['coef'])
    error=p['future_dividend'][first:]-(c[0]+z@c[1:-2]+c[-2]*p['private_signals'][first:,0]+c[-1]*p['price'][first:])
    return float(np.mean(error**2 if rows is None else error[rows]**2))

def score(h,z,p,eco,eq):
    M=mse(h,z,p);v=h['v'];c=h['coef'];first=CFG['context']-1
    w=base.MASS/(base.GAMMA*np.array([v,eq['v'][1]]));A=base.MASS[0]*h['iota']+w[1]*eq['beta'][1]
    delta=np.sum(w*(eco.risk_free_gross-np.array([h['chi'],eq['chi'][1]])))
    public=c[0]+z@c[1:-2];price=(w[0]*public+A*p['theta'][first:]-p['supply'][first:])/delta
    dem=w[0]*(public+h['beta']*p['theta'][first:]+(h['chi']-eco.risk_free_gross)*price)
    dem+=w[1]*(eq['beta'][1]*p['theta'][first:]+(eq['chi'][1]-eco.risk_free_gross)*price)
    return dict(MSE=M,NLL=.5*(math.log(2*math.pi*v)+M/v),beta=h['beta'],v=v,iota=h['iota'],chi=h['chi'],
      A=A,J=A*A/base.ZETA2,delta=delta,calibration=M/v,clearing_error=float(np.max(abs(dem-p['supply'][first:]))))

def tag(phase,c,seed,hidden):return f"{phase}_{c['name']}_seed{seed}_h{hidden}"

def train(phase,name,seed,hidden):
    torch.set_num_threads(1);c=candidate(name);job=tag(phase,c,seed,hidden)
    out=ROOT/'results'/job;md=ROOT/'models'/job;out.mkdir(exist_ok=True);md.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return job+' already complete'
    start=time.monotonic();p,eco,eq=panel(c,phase,seed,CFG[phase+'_periods']);data=prepare_panel(p,0,CFG['context']);set_seed(seed+500)
    m=ResourceForecaster(p['public_input'].shape[1],hidden,'gru','A',data['norm'],CFG['context']);_init_ols(m,data)
    opt=torch.optim.Adam(m.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);rng=np.random.default_rng(seed+911)
    a,b=CFG[phase+'_epochs'];curve=[];best=float('inf');steps=0
    for epoch in range(1,a+b+1):
        lr=CFG['lr'] if epoch<=a else CFG['lr']*(CFG['lr_end']/CFG['lr'])**((epoch-a)/b)
        for g in opt.param_groups:g['lr']=lr
        streamed=phase in CFG.get('stream_training_phases',[])
        if streamed:
            fresh,_,_=panel(c,phase,seed,int(.65*CFG[phase+'_periods']),bank=1000+epoch)
            training=normalized_data(m,fresh);order=rng.permutation(len(training['y']))
        else:training=data;order=rng.permutation(data['train_idx'])
        total=0.;m.train()
        for j in range(0,len(order),CFG['batch_size']):
            ix=order[j:j+CFG['batch_size']];opt.zero_grad()
            mu,lv=m(training['windows'][ix],training['s'][ix],training['p'][ix]);loss=gaussian_loss(mu,lv,training['y'][ix])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step();steps+=1;total+=float(loss.detach())*len(ix)
        val=validation(m,data);curve.append(dict(epoch=epoch,updates=steps,lr=lr,validation_MSE=val['MSE'],validation_NLL=val['NLL'],train_online_NLL=total/len(order),stream_bank=1000+epoch if streamed else -1))
        if val['MSE']<best:best=val['MSE'];best_state=copy.deepcopy(m.state_dict());best_epoch=epoch
        pd.DataFrame(curve).to_csv(out/'learning_curve.csv',index=False)
        if epoch in [1,a,a+b]:print(job,'epoch',epoch,'validation MSE',round(val['MSE'],6),'seconds',round(time.monotonic()-start,1),flush=True)
    detail=dict(phase=phase,scenario=name,seed=seed,hidden=hidden,epochs=epoch,updates=steps,train_n=len(data['train_idx']),validation_n=len(data['val_idx']),
      validation_MSE=val['MSE'],validation_best_MSE=best,best_epoch=best_epoch,resources=m.resource_stats(),fresh_training_each_epoch=streamed,optimization_windows=len(order)*epoch)
    checkpoint(m,detail,md/'final.pt');last=copy.deepcopy(m.state_dict());m.load_state_dict(best_state);checkpoint(m,detail,md/'validation_best.pt');m.load_state_dict(last)
    cal,_,_=panel(c,phase,seed,CFG['calibration_periods_'+phase],bank=50);heads=fit_heads(m,cal)
    raw=base.raw_head(m);z=features(m,p);t=np.arange(CFG['context']-1,len(p['price']));rows=(t>=int(.65*len(p['price'])))&(t<int(.82*len(p['price'])))
    fm=mse(heads['full'],z,p,rows);nm=mse(heads['no_private'],z,p,rows)
    detail.update(private_gain_pct=100*(1-fm/nm),calibrated_validation_MSE=fm,
        interior=bool(abs(raw['chi'])<.44 and abs(heads['full']['chi'])<.45),elapsed_seconds=time.monotonic()-start)
    dump(md/'heads.json',dict(raw=raw,calibrated=heads));dump(out/'equilibrium.json',eq);dump(out/'complete.json',detail)
    return job+' DONE '+str(round(detail['elapsed_seconds'],1))+' seconds'

def summarize_pilot(phase='pilot'):
    rows=[];qualified=[]
    for c in CFG['candidates']:
        items=[]
        for seed in CFG['pilot_seeds']:
            files={k:ROOT/'results'/tag(phase,c,seed,c[k])/'complete.json' for k in ['low','high']}
            if not all(f.exists() for f in files.values()):return
            r={k:json.loads(f.read_text()) for k,f in files.items()}
            row=dict(scenario=c['name'],seed=seed,low=r['low']['validation_MSE'],high=r['high']['validation_MSE'],
              MSE_gain_pct=100*(1-r['high']['validation_MSE']/r['low']['validation_MSE']),
              private_low=r['low']['private_gain_pct'],private_high=r['high']['private_gain_pct'],interior=r['low']['interior'] and r['high']['interior'])
            rows.append(row);items.append(row)
        d=pd.DataFrame(items)
        if d.MSE_gain_pct.mean()>=CFG['min_MSE_gain_pct'] and d.MSE_gain_pct.min()>CFG['min_pilot_each_MSE_gain_pct'] and min(d.private_low.mean(),d.private_high.mean())>=CFG['min_private_gain_pct'] and d.interior.all():qualified.append((c,float(d.MSE_gain_pct.mean())))
    preferred=[q for q in qualified if q[0]['low']==4];pool=preferred or qualified
    choice=max(pool,key=lambda x:x[1])[0] if pool else None
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'{phase}_comparisons.csv',index=False)
    dump(ROOT/'SELECTION.json',dict(selected=choice,pilot_phase=phase,config_sha256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),criteria='raw payoff MSE gain and incremental private-information MSE value',rows=rows))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('Selected:',choice,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','stream_pilot','confirmation'],required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.phase in ['pilot','stream_pilot']:cs=CFG['candidates']
    else:
        s=json.loads((ROOT/'SELECTION.json').read_text());assert s['selected'] is not None
        assert s['config_sha256']==hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest();cs=[s['selected']]
    jobs=[(args.phase,c['name'],seed,c[cap]) for c in cs for seed in CFG[args.phase+'_seeds'] for cap in ['low','high']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(train,*j) for j in jobs]
        for f in as_completed(fs):print(f.result(),flush=True)
    if args.phase in ['pilot','stream_pilot']:summarize_pilot(args.phase)
