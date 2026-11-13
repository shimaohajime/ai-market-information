"""Controlled S4 experiment. Forecast fits never receive latent states as inputs."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from marketlab.scenarios import economy_8d
from marketlab.learning import fit_forecaster, checkpoint, make_windows

CFG=json.loads((ROOT/'config.json').read_text())
MASS=np.array([.55,.45]); GAMMA=np.array([.9,1.2]); NU=np.array([1.,.09])
LAM=1.15; SIG2=.43**2; ZETA2=.9**2
START=CFG['context']

def dump(path,obj):
    def default(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(obj,indent=2,default=default,allow_nan=False))

def equilibrium(eco):
    prior=float(eco.target_loading@eco.stationary_cov@eco.target_loading)
    def quantities(A):
        P=1/(1/prior+A*A/ZETA2)
        beta=LAM*P/(P+NU)
        v=SIG2+LAM*NU*beta
        return P,beta,v
    def equation(A):
        _,beta,v=quantities(A)
        return A-np.sum(MASS*beta/(GAMMA*v))
    A=brentq(equation,0,100,xtol=1e-14)
    P,beta,v=quantities(A)
    w=MASS/(GAMMA*v)
    price_y=LAM*NU/(P+NU)*(A*P/ZETA2)
    delta=eco.risk_free_gross*np.sum(w)/(1+np.sum(w*price_y))
    chi=price_y*delta
    assert abs(equation(A))<1e-12
    assert abs(delta-np.sum(w*(eco.risk_free_gross-chi)))<1e-12
    return dict(prior=prior,P=P,beta=beta,v=v,chi=chi,A=A,delta=delta,
                J=A*A/ZETA2,mass=MASS,gamma=GAMMA,nu=NU)

def panel(eco,eq,scenario,n,seed):
    bank=eco.draw_bank(n,burn=CFG['burn'],seed=seed)
    f=bank.factor; theta=f@eco.target_loading
    q=f@eco.public_loading.T
    if scenario=='nonlinear_public':
        var=np.diag(eco.public_loading@eco.stationary_cov@eco.public_loading.T)
        public=bank.public+CFG['public_cubic_strength']*q**3/var
    elif scenario=='linear_public': public=bank.public
    else: raise ValueError(scenario)
    price=(eq['A']*theta-bank.supply_shock)/eq['delta']
    u=np.column_stack((np.r_[0.,price[:-1]],bank.lag_dividend,public))
    s=theta[:,None]+np.sqrt(NU)*bank.private_stdnormal[:,:2]
    b=CFG['burn']
    return dict(public_input=u[b:],price=price[b:],private_signals=s[b:],
                future_dividend=bank.future_dividend[b:],theta=theta[b:],
                supply=bank.supply_shock[b:],factor=f[b:],public=public[b:],seed=seed)

def basis_norm(train):
    x=train['public_input'][:int(.65*len(train['price']))]
    return dict(mean=x.mean(0),sd=np.maximum(x.std(0),1e-6))

def basis_features(p,norm):
    u=(p['public_input']-norm['mean'])/norm['sd']
    # All summaries use the identical finite causal window, ending at t.
    w=CFG['ema_decay']**np.arange(START-1,-1,-1,dtype=float); w/=w.sum()
    win=np.lib.stride_tricks.sliding_window_view(u,START,axis=0)
    smooth=np.einsum('ncl,l->nc',win,w)[1:]
    current=u[START:]
    return smooth,np.column_stack((smooth,current,np.tanh(current)))

def neural_features(model,p):
    u=(p['public_input'].astype('float32')-model.norm.u_mean)/model.norm.u_sd
    win=make_windows(u,model.window)
    pieces=[]
    with torch.no_grad():
        for start in range(1,len(win),1024):
            x=torch.as_tensor(np.array(win[start:start+1024],copy=True),dtype=torch.float32)
            pieces.append(model.encode(x).numpy().astype(float))
    return np.concatenate(pieces)

def least_squares(X,y):
    # An explicit intercept; remaining columns standardized for numerical accuracy.
    mean=X[:,1:].mean(0); sd=X[:,1:].std(0); sd=np.where(sd>1e-12,sd,1.)
    W=np.column_stack((np.ones(len(X)),(X[:,1:]-mean)/sd))
    c,_,rank,singular=np.linalg.lstsq(W,y,rcond=None)
    coef=np.r_[c[0]-np.dot(c[1:]/sd,mean),c[1:]/sd]
    resid=y-X@coef
    return coef,float(np.mean(resid**2)),dict(rank=int(rank),columns=X.shape[1],
        condition=float(singular[0]/singular[-1]),normal_equation_max=float(np.max(abs(W.T@resid/len(y)))))

def refit(z,p,n=None):
    s=p['private_signals'][START:,0]; price=p['price'][START:]
    y=p['future_dividend'][START:]; theta=p['theta'][START:]
    if n is not None: z,s,price,y,theta=z[:n],s[:n],price[:n],y[:n],theta[:n]
    X=np.column_stack((np.ones(len(z)),z,s,price))
    coef,v,lsdiag=least_squares(X,y)
    # Diagnostic regression ONLY. Its output is not used in the fitted payoff head.
    XP=np.column_stack((np.ones(len(z)),z,price))
    _,P,pdiag=least_squares(XP,theta)
    beta=float(coef[-2]); chi=float(coef[-1]); iota=beta/(GAMMA[0]*v)
    tb=LAM*P/(P+NU[0]); tv=SIG2+LAM*NU[0]*tb
    return dict(coef=coef,v=v,beta=beta,chi=chi,iota=iota,
                diagnostic=dict(P=P,beta_theory=tb,v_theory=tv,iota_theory=tb/(GAMMA[0]*tv),
                    beta_error=beta-tb,v_error=v-tv,identity_error=v-SIG2-LAM*NU[0]*beta,
                    iota_error=iota-tb/(GAMMA[0]*tv),ols=lsdiag,projection_ols=pdiag,
                    n=len(y)))

def raw_head(model):
    assert model.head_type=='A'
    norm=model.norm
    z=torch.zeros((1,model.hidden),dtype=torch.float32)
    with torch.no_grad():
        h=model.heads_from_z(z).numpy()[0].astype(float)
        weights=model.public_mean.weight.numpy()[0].astype(float)*norm.y_sd
    return dict(coef=np.r_[h[0],weights,h[1],h[2]],v=float(h[3]),
                beta=float(h[1]),chi=float(h[2]),iota=float(h[1]/(GAMMA[0]*h[3])),
                diagnostic=dict(identity_error=float(h[3]-SIG2-LAM*NU[0]*h[1])))

def score(head,z,p,eq,eco):
    y=p['future_dividend'][START:]; s=p['private_signals'][START:,0]
    price=p['price'][START:]; theta=p['theta'][START:]
    c=np.asarray(head['coef']); public=c[0]+z@c[1:-2]
    mu=public+head['beta']*s+head['chi']*price
    mse=float(np.mean((y-mu)**2)); v=head['v']
    A=MASS[0]*head['iota']+MASS[1]*eq['beta'][1]/(GAMMA[1]*eq['v'][1])
    delta=MASS[0]*(eco.risk_free_gross-head['chi'])/(GAMMA[0]*v)+MASS[1]*(eco.risk_free_gross-eq['chi'][1])/(GAMMA[1]*eq['v'][1])
    if delta<=0: raise RuntimeError(f'Candidate clearing slope nonpositive: {delta}')
    K=MASS[0]*public/(GAMMA[0]*v)
    new_price=(K+A*theta-p['supply'][START:])/delta
    # Independently sum continuum demands at the new price.
    demand=MASS[0]*(public+head['beta']*theta+(head['chi']-eco.risk_free_gross)*new_price)/(GAMMA[0]*v)
    demand+=MASS[1]*(eq['beta'][1]*theta+(eq['chi'][1]-eco.risk_free_gross)*new_price)/(GAMMA[1]*eq['v'][1])
    return dict(MSE=mse,NLL=.5*(math.log(2*math.pi*v)+mse/v),beta=head['beta'],v=v,
        iota=head['iota'],A=A,J=A*A/ZETA2,delta=delta,
        calibration=mse/v,clearing_error=float(np.max(abs(demand-p['supply'][START:]))),
        n=len(y),price_change_rms=float(np.sqrt(np.mean((new_price-price)**2))))

def run(scenario,seed):
    index=CFG['seeds'].index(seed); rootseed=28000000+100000*index
    tag=f'{scenario}_seed{seed}'; out=ROOT/'results'/tag; out.mkdir(exist_ok=True)
    md=ROOT/'models'/tag;md.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        print(tag,'already complete',flush=True);return
    t0=time.monotonic(); eco=economy_8d('E1_aligned');eq=equilibrium(eco)
    dump(out/'inherited_equilibrium.json',eq)
    train=panel(eco,eq,scenario,CFG['train_periods'],rootseed+11)
    cal=panel(eco,eq,scenario,CFG['calibration_periods'],rootseed+23)
    norm=basis_norm(train);dump(md/'basis_normalization.json',norm)
    zl,zh=basis_features(cal,norm)
    methods={'nested_basis':{'low':refit(zl,cal),'high':refit(zh,cal)}}
    prefix=[]
    for n in (2000,10000,60000):
        for label,z in [('low',zl),('high',zh)]:
            h=refit(z,cal,n)
            prefix.append(dict(n=h['diagnostic']['n'],capacity=label,**h))
    dump(out/'basis_sample_size_check.json',prefix)
    models={}; details={}
    for name,hidden in [('low',CFG['hidden_low']),('high',CFG['hidden_high'])]:
        model,detail=fit_forecaster(train,scenario,'gru',hidden,'A',seed=seed+2000,
            type_index=0,window=START,steps=CFG['steps'],batch_size=CFG['batch_size'],
            lr=CFG['lr'],patience=CFG['patience'])
        models[name]=model;details[name]=detail;checkpoint(model,detail,md/f'{name}.pt')
        print(tag,name,'trained',detail['steps_completed'],'steps',round(time.monotonic()-t0,1),'s',flush=True)
    methods['gru_raw']={k:raw_head(m) for k,m in models.items()}
    methods['gru_refit']={k:refit(neural_features(m,cal),cal) for k,m in models.items()}
    dump(md/'fitted_heads.json',methods);dump(out/'training.json',details)
    del cal,zl,zh,train
    rows=[]
    for bankidx in range(CFG['evaluation_banks']):
        ev=panel(eco,eq,scenario,CFG['evaluation_periods'],rootseed+1000+997*bankidx)
        basis=dict(zip(['low','high'],basis_features(ev,norm)))
        neural={k:neural_features(m,ev) for k,m in models.items()}
        for method,heads in methods.items():
            features=basis if method=='nested_basis' else neural
            for capacity,h in heads.items():
                rows.append(dict(scenario=scenario,seed=seed,bank=bankidx,method=method,
                    capacity=capacity,**score(h,features[capacity],ev,eq,eco)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'bank_metrics.csv',index=False)
    dump(out/'complete.json',dict(scenario=scenario,seed=seed,elapsed_seconds=time.monotonic()-t0,
        seeds=dict(training=rootseed+11,calibration=rootseed+23,
            evaluation=[rootseed+1000+997*b for b in range(CFG['evaluation_banks'])])))
    print(tag,'DONE',round(time.monotonic()-t0,1),'seconds',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scenario',choices=CFG['scenarios'])
    ap.add_argument('--seed',type=int);args=ap.parse_args();torch.set_num_threads(1)
    for scenario in ([args.scenario] if args.scenario else CFG['scenarios']):
        for seed in ([args.seed] if args.seed is not None else CFG['seeds']):run(scenario,seed)

if __name__=='__main__':main()
