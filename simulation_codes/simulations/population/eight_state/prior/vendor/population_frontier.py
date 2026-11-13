"""Exact population Gaussian information / bounded public-lag decoder frontiers.
No sampled data, no SGD. Times: U_L=(macro_t, y_{t-1},...,y_{t-L}); y_t=(p_t,D_{t+1},macro_t).
The observer may additionally use current p_t but never next dividend.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
POP = HERE/'vendor'/'population_benchmark'
if not POP.exists(): POP = ROOT/'population_benchmark'
sys.path.insert(0,str(POP))
from benchmark import Benchmark
from dataclasses import replace


def stationary_moments(b, q, maxlag=64):
    law=b.law(q)
    H=law['Yh']; E=law['Ye']; M=law['M']; N=law['N']; P=law['P']
    nobs=H.shape[0];dim=M.shape[0]
    theta=np.zeros(dim);theta[:b.m]=b.eco.target_loading
    ycross=[]; theta_cross=[]
    G0=H@P@H.T+E@E.T
    T0=theta@P@H.T
    ycross.append(G0);theta_cross.append(T0)
    Mk=np.eye(dim)
    for k in range(1,maxlag+1):
        # M^k P H' + M^{k-1} N E'
        pky=Mk@(M@P@H.T+N@E.T)
        ycross.append(H@pky)
        theta_cross.append(theta@pky)
        Mk=M@Mk
    var_theta=float(theta@P@theta)
    return {'auto':np.asarray(ycross), 'theta_cross':np.asarray(theta_cross), 'var_theta':var_theta,'J':b.J(q)}


def conditional_variance(M0, lag, add_price, target="theta"):
    covyy=M0['auto']; cross=M0['theta_cross']; nv=covyy.shape[1]
    # Stack y_t,y_(t-1),...,y_(t-L). y_t includes (p_t,D_(t+1),macro_t).
    ix=[(0,k) for k in range(2,nv)]
    ix += [(j,k) for j in range(1,lag+1) for k in range(nv)]
    if add_price: ix += [(0,0)]
    idx_lags=np.array([i for i,k in ix],int)
    idx_col=np.array([k for i,k in ix],int)
    n=len(ix)
    Sigma=np.empty((n,n))
    for i in range(n):
        for j in range(i+1):
            left=idx_lags[i]; right=idx_lags[j]
            if left<=right:
                vv=covyy[right-left,idx_col[i],idx_col[j]]
            else:
                vv=covyy[left-right,idx_col[j],idx_col[i]]
            Sigma[i,j]=Sigma[j,i]=vv
    if target == "theta":
        v=np.array([cross[ii,cc] for ii,cc in ix]); prior=M0['var_theta']
    elif target == "price":
        v=np.array([covyy[ii,0,cc] for ii,cc in ix]); prior=covyy[0,0,0]
    else: raise ValueError(target)
    try:
        cf=cho_factor(Sigma,lower=True,check_finite=False)
        var=float(prior-v@cho_solve(cf,v,check_finite=False))
    except Exception as e:
        raise ValueError(f'Cov inversion failed L={lag}; eig={np.linalg.eigvalsh(Sigma).min()}') from e
    if var < -1e-8: raise ValueError('invalid conditional var')
    return max(var,0.)


def calculate(maxlag=40, lag_grid=(0,1,2,3,4,6,8,12,16,24,32,40), sd=.9):
    b=Benchmark(K=3)
    b.eco=replace(b.eco,supply_noise_sd=sd);b.sigma_supply=sd
    qlow,statlow=b.equilibrium((0,0))
    impact=b.update(qlow,(3,0))
    high,stathigh=b.equilibrium((3,0),start=impact)
    profiles={'baseline':qlow,'impact':impact,'retrained':high}
    if sd<.721:
        q=impact.copy()
        for _ in range(2500): q=b.update(q,(3,0))
        qa=q;qb=b.update(qa,(3,0));profiles['cycle_a']=qa;profiles['cycle_b']=qb
    rows=[]; meta={'sigma_supply':sd,'lag_grid':list(lag_grid),'oracle_lag':maxlag,'q':{k:v.tolist() for k,v in profiles.items()},'fast_radius':{},'fixed_law_rho':{'baseline':statlow['rho_jac'],'retrained':stathigh['rho_jac']}}
    for name,q in profiles.items():
        moments=stationary_moments(b,q,maxlag)
        meta['fast_radius'][name]=float(b.law(q)['fast_radius'])
        variance_public_inf=conditional_variance(moments,maxlag,False)
        AoverDelta=float(q[0]/q[-1])
        cov_resid_price=AoverDelta*variance_public_inf
        mi_oracle=.5*np.log1p(variance_public_inf*moments['J'])
        for L in lag_grid:
            assert L<=maxlag
            vU=conditional_variance(moments,L,False)
            vP=conditional_variance(moments,L,True)
            mi=.5*np.log(vU/vP)
            var_price_given_public=conditional_variance(moments,L,False,target='price')
            post_inf=max(variance_public_inf - cov_resid_price**2/var_price_given_public,1e-15)
            mi_extract=.5*np.log(variance_public_inf/post_inf)

            rows.append({'regime':name,'lag':int(L),'features_public':int(2+4*L),
                         'features_plus_price':int(3+4*L),
                         'mse_public':vU,'mse_public_price':vP,
                         'conditional_mi_nats':float(mi), 'price_mse_gain':float(1-vP/vU),
                         'fresh_J':moments['J'], 'theta_variance':moments['var_theta'],
                         'innovation_MI_nats':float(mi_extract), 'innovation_MI_oracle_nats':float(mi_oracle),
                         'innovation_recovery_share':float(mi_extract/mi_oracle),
                         'conditional_price_variance':float(var_price_given_public),
                         'innovation_mse':float(post_inf),'innovation_oracle_mse':float(variance_public_inf/(1+variance_public_inf*moments['J']))
                         })
    return rows,meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results'); p.add_argument('--sd',type=float,default=.9);p.add_argument('--max-lag',type=int,default=40)
    z=p.parse_args();ls=[l for l in [0,1,2,3,4,6,8,12,16,24,32,40,64] if l<=z.max_lag];
    if z.max_lag not in ls: ls.append(z.max_lag)
    import pandas as pd
    r,md=calculate(z.max_lag,ls,z.sd)
    d=Path(z.out);d.mkdir(exist_ok=True,parents=True)
    suf=str(z.sd).replace('.','p')
    pd.DataFrame(r).to_csv(d/f'population_frontier_sd{suf}.csv',index=False)
    (d/f'population_frontier_sd{suf}.json').write_text(json.dumps(md,indent=2))
    for rr in r:
        if rr['lag'] in [0,1,3,8,16,40]:print(rr['regime'],rr['lag'],f"MSE={rr['mse_public_price']:.6f}, MI={rr['conditional_mi_nats']:.6f}")
