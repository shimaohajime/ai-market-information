"""Frozen-policy fast market and batched, exactly finite-context GRU inference."""
from pathlib import Path
import sys,json,hashlib,copy,time,math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference_capacity'))
import experiment as prior
from marketlab.learning import load_checkpoint,checkpoint,gaussian_loss,make_windows
CANDIDATE=prior.candidate('balanced8_r20_n25')
ECO,EQ,W=prior.primitives(CANDIDATE)
LAM=prior.base.LAM;SIG2=prior.base.SIG2;ZETA2=prior.base.ZETA2
WINDOW=16;MASS=prior.base.MASS;GAMMA=prior.base.GAMMA;dump=prior.dump

def fingerprint(models,other):
    h=hashlib.sha256()
    for cap in ['low','high']:
        m=models[cap]
        for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
        h.update(json.dumps(m.norm.to_dict(),sort_keys=True).encode())
    h.update(json.dumps(other,sort_keys=True).encode());return h.hexdigest()

def initial(seed):
    models={cap:load_checkpoint(ROOT/'initial'/str(seed)/(cap+'.pt'))[0] for cap in ['low','high']}
    other=dict(intercept=0.,beta=float(EQ['beta'][1]),chi=float(EQ['chi'][1]),v=float(EQ['v'][1]))
    return models,other

def coefficients(models,other,x):
    heads={cap:prior.base.raw_head(m) for cap,m in models.items()}
    masses=np.array([MASS[0]*(1-x),MASS[0]*x,MASS[1]])
    vs=np.array([heads['low']['v'],heads['high']['v'],other['v']])
    bs=np.array([heads['low']['beta'],heads['high']['beta'],other['beta']])
    cs=np.array([heads['low']['chi'],heads['high']['chi'],other['chi']])
    weights=masses/(np.array([GAMMA[0],GAMMA[0],GAMMA[1]])*vs)
    A=float(weights@bs);delta=float(weights@(ECO.risk_free_gross-cs))
    if not np.all(vs>0) or delta<=0:raise ValueError('No admissible clearing slope')
    return heads,dict(weights=weights,A=A,delta=delta,J=A*A/ZETA2,masses=masses)

def draw(seed,round_index,bank,paths,steps):
    """Economy exactly matches the preceding experiment; independent paths."""
    ss=np.random.SeedSequence([20261028,seed,round_index,bank])
    rng=np.random.default_rng(ss);f=rng.multivariate_normal(np.zeros(8),ECO.stationary_cov,size=paths)
    shocks=rng.standard_normal((steps,paths,ECO.G.shape[1]));states=np.empty((steps,paths,8))
    for t in range(steps):f=f@ECO.F.T+shocks[t]@ECO.G.T;states[t]=f
    old=states@ECO.target_loading
    eta=rng.normal(0,np.sqrt(EQ['prior']),old.shape)
    theta=np.sqrt(.8)*old+np.sqrt(.2)*eta
    q=states@W.T;public=q+2*q**3+.2*rng.standard_normal(q.shape)
    y=LAM*theta+.43*rng.standard_normal(theta.shape)
    supply=.9*rng.standard_normal(theta.shape)
    private=theta[:,:,None]+np.sqrt(EQ['nu'])*rng.standard_normal((*theta.shape,2))
    return dict(public=public,theta=theta,y=y,supply=supply,private=private)

def simulate(models,other,x,seed,round_index,bank,paths=128,periods=1024,burn=512,initial_price=0.,shocks=None,return_trace=False):
    """Generate all observations while every deployed coefficient stays fixed.

    Each independent path has its own rolling buffer. A GRU is restarted from
    hidden state zero for every 16-observation window, exactly as in training.
    Current price is solved only after public encoders run, then fed to t+1.
    """
    start=time.monotonic();before=fingerprint(models,other);heads,agg=coefficients(models,other,x)
    total=burn+periods+WINDOW-1
    if shocks is None:shocks=draw(seed,round_index,bank,paths,total)
    assert len(shocks['theta'])==total
    buffer=np.zeros((paths,WINDOW,10),dtype=np.float32)
    price_prev=np.full(paths,initial_price,dtype=float);payoff_prev=np.zeros(paths)
    U=np.empty((paths,periods+WINDOW-1,10),dtype=np.float32)
    P=np.empty((paths,periods+WINDOW-1));trace=[];max_clear=0.
    for m in models.values():m.eval()
    with torch.inference_mode():
        for t in range(total):
            u=np.column_stack((price_prev,payoff_prev,shocks['public'][t])).astype('float32')
            buffer[:,:-1]=buffer[:,1:];buffer[:,-1]=u
            public_mean=np.full(paths,agg['weights'][2]*other['intercept'])
            for j,cap in enumerate(['low','high']):
                if agg['weights'][j]==0:continue
                m=models[cap];z=m.encode(torch.as_tensor((buffer-m.norm.u_mean)/m.norm.u_sd)).numpy()
                co=heads[cap]['coef'];g=co[0]+z@co[1:-2]
                public_mean+=agg['weights'][j]*g
            theta=shocks['theta'][t];supply=shocks['supply'][t]
            p=(public_mean+agg['A']*theta-supply)/agg['delta']
            residual=public_mean+agg['A']*theta-agg['delta']*p-supply
            max_clear=max(max_clear,float(np.max(abs(residual))))
            if not np.isfinite(p).all():raise FloatingPointError('Nonfinite market price')
            if t>=burn:U[:,t-burn]=u;P[:,t-burn]=p
            if return_trace:trace.append(p.copy())
            price_prev=p;payoff_prev=shocks['y'][t]
    assert fingerprint(models,other)==before,'A policy changed during data generation'
    trans=lambda a:np.swapaxes(a[burn:],0,1).copy()
    panel=dict(public_input=U,price=P,private_signals=trans(shocks['private']),future_dividend=trans(shocks['y']),
               theta=trans(shocks['theta']),supply=trans(shocks['supply']))
    meta=dict(seed=seed,round_index=round_index,bank=bank,paths=paths,periods=periods,burn=burn,
        observations=paths*periods,profile_sha256=before,x=x,clearing_error=max_clear,
        elapsed_seconds=time.monotonic()-start,**agg)
    if return_trace:panel['trace']=np.asarray(trace)
    return panel,meta

def normalized(model,p):
    n=model.norm;u=(p['public_input']-n.u_mean)/n.u_sd
    # Windows are formed within paths, never across a path boundary.
    w=np.lib.stride_tricks.sliding_window_view(u,WINDOW,axis=1).transpose(0,1,3,2)
    flat=lambda a:a[:,WINDOW-1:].reshape(-1)
    return dict(windows=torch.as_tensor(np.ascontiguousarray(w.reshape(-1,WINDOW,10))),
        s=torch.as_tensor(((flat(p['private_signals'][:,:,0])-n.s_mean)/n.s_sd).astype('float32')),
        p=torch.as_tensor(((flat(p['price'])-n.p_mean)/n.p_sd).astype('float32')),
        y=torch.as_tensor(((flat(p['future_dividend'])-n.y_mean)/n.y_sd).astype('float32')))

def features(model,p):
    d=normalized(model,p);zs=[]
    with torch.inference_mode():
        for ix in range(0,len(d['y']),2048):zs.append(model.encode(d['windows'][ix:ix+2048]).numpy().astype(float))
    return np.concatenate(zs)

def flat(p,key,type_index=None):
    a=p[key]
    if type_index is not None:a=a[:,:,type_index]
    return a[:,WINDOW-1:].reshape(-1)

def predict(head,z,p,type_index=0):
    co=np.asarray(head['coef']);return co[0]+z@co[1:-2]+co[-2]*flat(p,'private_signals',type_index)+co[-1]*flat(p,'price')

def head_fits(model,p):
    z=features(model,p);y=flat(p,'future_dividend')
    X=np.column_stack((np.ones(len(y)),z,flat(p,'private_signals',0),flat(p,'price')))
    co,v,diagnostic=prior.base.least_squares(X,y)
    no,vno,dno=prior.base.least_squares(np.delete(X,-2,axis=1),y)
    return dict(full=dict(coef=co,v=v,beta=co[-2],chi=co[-1],diagnostic=diagnostic),
                no_private=dict(coef=np.insert(no,len(no)-1,0.),v=vno,diagnostic=dno))

def score(models,other,x,p,cal=None):
    rows=[];heads,agg=coefficients(models,other,x);y=flat(p,'future_dividend')
    for cap,m in models.items():
        z=features(m,p);h=heads[cap];mse=float(np.mean((y-predict(h,z,p))**2))
        r=dict(capacity=cap,MSE=mse,NLL=.5*(np.log(2*np.pi*h['v'])+mse/h['v']),
            beta=h['beta'],v=h['v'],chi=h['chi'],iota=h['iota'],calibration=mse/h['v'],
            identity_error=h['v']-SIG2-LAM*.25*h['beta'])
        if cal is not None:
            hs=head_fits(m,cal);fm=float(np.mean((y-predict(hs['full'],z,p))**2));nm=float(np.mean((y-predict(hs['no_private'],z,p))**2))
            r.update(private_gain_pct=100*(1-fm/nm),calibrated_MSE=fm,
                calibrated_beta=float(hs['full']['beta']),calibrated_v=hs['full']['v'],
                calibrated_iota=float(hs['full']['beta']/(GAMMA[0]*hs['full']['v'])))
        rows.append(r)
    mu=other['intercept']+other['beta']*flat(p,'private_signals',1)+other['chi']*flat(p,'price')
    ms=float(np.mean((y-mu)**2));rows.append(dict(capacity='other',MSE=ms,beta=other['beta'],v=other['v'],chi=other['chi'],iota=other['beta']/(GAMMA[1]*other['v']),calibration=ms/other['v']))
    prices=flat(p,'price');pp=p['price'][:,WINDOW-1:]
    market=dict(A=agg['A'],J=agg['J'],delta=agg['delta'],price_mean=float(prices.mean()),price_variance=float(prices.var()),
        price_lag1=float(np.corrcoef(pp[:,1:].ravel(),pp[:,:-1].ravel())[0,1]),max_abs_price=float(abs(prices).max()))
    return rows,market

def refit_other(paths):
    xx=np.zeros((3,3));xy=np.zeros(3);yy=0.;n=0
    for path in paths:
        with np.load(path) as p:
            y=flat(p,'future_dividend');X=np.column_stack((np.ones(len(y)),flat(p,'private_signals',1),flat(p,'price')))
            xx+=X.T@X;xy+=X.T@y;yy+=y@y;n+=len(y)
    co=np.linalg.solve(xx,xy);v=float((yy-2*co@xy+co@xx@co)/n)
    return dict(intercept=float(co[0]),beta=float(co[1]),chi=float(co[2]),v=v)

def validation(model,p):
    d=normalized(model,p);se=0.;loss=0.;n=len(d['y'])
    with torch.inference_mode():
        for i in range(0,n,2048):
            sl=slice(i,i+2048);mu,lv=model(d['windows'][sl],d['s'][sl],d['p'][sl]);se+=float(((mu-d['y'][sl])**2).sum());loss+=float(gaussian_loss(mu,lv,d['y'][sl]))*len(mu)
    return dict(MSE=se/n*model.norm.y_sd**2,NLL=loss/n)

def train(model,paths,val,seed,round_index,out,lr=.001):
    """All supplied paths were generated before this optimizer is constructed."""
    m=copy.deepcopy(model);opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-5)
    rng=np.random.default_rng(np.random.SeedSequence([seed,round_index,919]));curve=[];steps=0
    # Existing normalization stays fixed. This preserves the initial physical
    # forecast at a warm start and never uses validation or evaluation moments.
    a=len(paths)-3;b=3
    for epoch,path in enumerate(paths,1):
        rate=lr if epoch<=a else lr*(1e-5/lr)**((epoch-a)/b)
        for g in opt.param_groups:g['lr']=rate
        with np.load(path) as p:d=normalized(m,p)
        order=rng.permutation(len(d['y']));m.train()
        for j in range(0,len(order),256):
            ix=order[j:j+256];opt.zero_grad();mu,lv=m(d['windows'][ix],d['s'][ix],d['p'][ix]);loss=gaussian_loss(mu,lv,d['y'][ix])
            if not torch.isfinite(loss):raise FloatingPointError('Training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step();steps+=1
        m.eval();v=validation(m,val);curve.append(dict(epoch=epoch,updates=steps,lr=rate,validation_MSE=v['MSE'],validation_NLL=v['NLL']))
    import pandas as pd
    pd.DataFrame(curve).to_csv(out,index=False)
    return m,dict(epochs=len(paths),updates=steps,optimization_windows=len(order)*len(paths),normalization_frozen=True)
