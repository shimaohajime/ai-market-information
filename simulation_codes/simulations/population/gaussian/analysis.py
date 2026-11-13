"""Population benchmark tests, adoption gradients, robustness, and figures."""
from __future__ import annotations
from dataclasses import replace
from copy import copy
from pathlib import Path
import json
import numpy as np
from scipy.optimize import root
from benchmark import Benchmark

HERE=Path(__file__).resolve().parent

def map_adoption(b:Benchmark,q,w,lo=0,hi=3, price_slope_cap=None, extra=False):
    inherited=b.law(q,verify=False)
    preds=[b.fit_one(inherited,0,lo),b.fit_one(inherited,0,hi),b.fit_one(inherited,1,lo)]
    coeffs=[(1-w)*preds[0]['pi'],w*preds[1]['pi'],preds[2]['pi']]
    A=0.;C=np.zeros(b.zdim);Delta=0.
    for p,mass in zip(preds,coeffs):
        weight=mass/(p['gamma']*p['v'])
        A+=weight*p['beta'];C+=weight*p['b'];Delta+=weight*(b.eco.risk_free_gross-p['chi'])
    q_new=np.r_[A,C,Delta]
    if not Delta>0:raise ValueError('nonpositive slope')
    if extra:return q_new,preds
    return q_new

def steady(b,w,lo=0,hi=3,init=None,maxiter=500,tol=1e-11):
    q=(b.seed() if init is None else np.asarray(init,float)).copy(); traj=[]
    for k in range(maxiter):
        qq=map_adoption(b,q,w,lo,hi)
        traj.append([k,float(b.J(qq)),float(np.max(abs(qq-q)))])
        if np.max(abs(qq-q))<tol: q=qq;break
        q=qq
    resid=np.max(abs(map_adoption(b,q,w,lo,hi)-q))
    if resid>1e-8:raise RuntimeError(f'No converged fixed point at {w=} {resid=}')
    return q,np.asarray(traj)

def derivatives(b,w=.5,lo=0,hi=3):
    q,traj=steady(b,w,lo,hi)
    h=1e-5
    d= len(q)
    M=np.zeros((d,d))
    for j in range(d):
        e=np.zeros(d);e[j]=h*max(1,abs(q[j]))
        M[:,j]=(map_adoption(b,q+e,w,lo,hi)-map_adoption(b,q-e,w,lo,hi))/(2*e[j])
    gp=(map_adoption(b,q,w+h,lo,hi)-map_adoption(b,q,w-h,lo,hi))/(2*h)
    dq=np.linalg.solve(np.eye(d)-M,gp)
    dj_dq=2*q[0]/(b.sigma_supply**2)
    return {'q':q,'rho_jac':max(abs(np.linalg.eigvals(M))),
            'dJ_impact':float(dj_dq*gp[0]),'dJ_longrun':float(dj_dq*dq[0]),
            'dJ_retraining':float(dj_dq*(dq[0]-gp[0])),
            'feedback_multiplier':float(dq[0]/gp[0]),
            'jacobian':M,'g':gp,'slow_mode_eigs':np.linalg.eigvals(M)}

def monte_carlo_check(b,q,n=120000,seed=1234):
    rng=np.random.default_rng(seed)
    law=b.law(q)
    M=law['M'];N=law['N']
    h=rng.multivariate_normal(np.zeros(b.nstate),law['P']);P=[];D=[];Z=[];SS=[]
    # sample 100k exact Gaussian process. Fast filtering, full state simulation.
    for t in range(n+400):
        e=rng.standard_normal(b.ne)
        if t>=400:
            P.append(law['p']@np.r_[h,e]);D.append(law['d']@np.r_[h,e]);Z.append(law['Z']@np.r_[h,e]); SS.append(law['theta']@np.r_[h,e])
        h=M@h+N@e
    P=np.asarray(P);D=np.asarray(D);Z=np.asarray(Z);SS=np.asarray(SS)
    C=law['cov']
    out={
        'analytical_price_variance':float(law['p']@C@law['p']),
        'empirical_price_variance':float(np.var(P)),
        'analytical_dividend_variance':float(law['d']@C@law['d']),
        'empirical_dividend_variance':float(np.var(D)),
        'analytical_corr_price_payoff':float((law['p']@C@law['d'])/np.sqrt((law['p']@C@law['p'])*(law['d']@C@law['d']))),
        'empirical_corr_price_payoff':float(np.corrcoef(P,D)[0,1]),
        'max_z_cov_abs_err':float(np.max(np.abs(np.cov(Z.T)-law['Z']@C@law['Z'].T))),
        'analytical_price_mean':0.0,'empirical_price_mean':float(np.mean(P)),
        'n_simulated':n
    }
    return out

def perturbation_grid(b, lo, hi):
    from dataclasses import replace
    eco0=b.eco;temps0=[(t.gamma,t.signal_noise_sd,t.mass) for t in b.temps]
    diagnostics=[]
    variants=[]
    for par,factors in [('supply_sd',[.8,1.0,1.2]),('payoff_noise_sd',[.85,1.,1.15]),('signal0',[.75,1.,1.25]),('gamma0',[.85,1.,1.15]),('mass0',[.9,1.,1.1])]:
        for fac in factors:
            if par=='supply_sd':b.eco=replace(eco0,supply_noise_sd=eco0.supply_noise_sd*fac);b.sigma_supply=b.eco.supply_noise_sd
            if par=='payoff_noise_sd':b.eco=replace(eco0,payoff_noise_sd=eco0.payoff_noise_sd*fac)
            if par=='signal0':b.temps[0].signal_noise_sd=temps0[0][1]*fac
            if par=='gamma0':b.temps[0].gamma=temps0[0][0]*fac
            if par=='mass0':b.temps[0].mass=temps0[0][2]*fac;b.temps[1].mass=1-b.temps[0].mass
            try:
                q0,_=steady(b,0,lo,hi)
                qim=map_adoption(b,q0,1,lo,hi)
                q1,_=steady(b,1,lo,hi,qim)
                jj=[b.J(q0),b.J(qim),b.J(q1)]
                diag={'param':par,'factor':fac,'baseline_J':jj[0], 'impact_percent':100*(jj[1]/jj[0]-1),'stationary_percent':100*(jj[2]/jj[0]-1)}
                diagnostics.append(diag)
            except Exception as e:
                diagnostics.append({'param':par,'factor':fac,'error':str(e)})
            b.eco=eco0;b.sigma_supply=eco0.supply_noise_sd
            for t,vals in zip(b.temps,temps0):t.gamma,t.signal_noise_sd,t.mass=vals
    return diagnostics

def run():
    b=Benchmark(K=3)
    result={'benchmark':{'latent_dim':b.m,'forecast_features':b.zdim,'history_lags':b.K,
        'exogenous_rho':b.rho_exo,'riskfree':b.eco.risk_free_gross,
        'sigma_supply':b.sigma_supply, 'scenario':'E1_aligned'},'capacities':[]}
    for lo,hi in [(0,3),(1,3),(2,3)]:
        q0,d0=b.equilibrium((lo,lo));qimp,first=b.update(q0,(hi,lo),True)
        q1,d1=b.equilibrium((hi,lo),qimp)
        _,baseline=b.update(q0,(lo,lo),True)
        _,last=b.update(q1,(hi,lo),True)
        rec={'low':lo,'high':hi,'J_baseline':b.J(q0),'J_impact':b.J(qimp),'J_equilibrium':b.J(q1),
             'impact_pct':100*(b.J(qimp)/b.J(q0)-1),
             'long_pct':100*(b.J(q1)/b.J(q0)-1),
             'rho_low':d0['rho_jac'],'rho_high':d1['rho_jac'],'fast_rho_low':d0['fast_rho'],'fast_rho_high':d1['fast_rho'],
             'max_residual_low':d0['residual'],'max_residual_high':d1['residual'],
             'low_iterations':d0['iterations'],'high_iterations':d1['iterations'],
             'baseline_policy':[{k: v for k,v in p.items() if k!='b'} for p in baseline['preds']],
             'impact_policy':[{k: v for k,v in p.items() if k!='b'} for p in first['preds']],
             'equilibrium_policy':[{k: v for k,v in p.items() if k!='b'} for p in last['preds']],
             'path_J':d1['path'][:,1].tolist(),
             'eigenvalues_post':[list(v) for v in d1['eigenvalues']]
             }
        result['capacities'].append(rec)
        print('capacity',lo,'->',hi,'impact',round(rec['impact_pct'],3),'long',round(rec['long_pct'],3),'rho',round(d1['rho_jac'],4),flush=True)
    lo,hi=0,3
    q0,_=steady(b,0,lo,hi)
    adoptions=[]
    for w in np.linspace(0,1,21):
        qw,trajectory=steady(b,w,lo,hi,init=q0)
        qd=map_adoption(b,q0,w,lo,hi)
        adoptions.append({'adoption':float(w),'J_fixedlaw':b.J(qd),'J_longrun':b.J(qw),'rho_fast':b.law(qw)['fast_radius'],'iterations':len(trajectory)})
    result['adoption']=adoptions
    dl=derivatives(b,.5,lo,hi)
    result['local_multiplier']={k:v for k,v in dl.items() if isinstance(v,(float,int))}
    print('derivative at .5',result['local_multiplier'],flush=True)
    variants=perturbation_grid(b,lo,hi)
    result['robustness_oneparam']=variants
    print('robust',len(variants),'all negative?',all(x.get('impact_percent',1)<0 and x.get('stationary_percent',1)<0 for x in variants),flush=True)
    check=monte_carlo_check(b,steady(b,1,lo,hi)[0],n=85000)
    result['independent_mc_check']=check
    print('moments check',check,flush=True)
    # verify perturb derivative h-size
    derivative_checks=[]
    q,d=b.equilibrium((3,0))
    for h in (1e-4,1e-5,1e-6):
        eig=np.linalg.eigvals(b.local_jac(q,(3,0),h))
        derivative_checks.append({'step':h,'radius':float(max(abs(eig)))})
    result['derivative_checks']=derivative_checks
    with open(HERE/'results.json','w') as fh:json.dump(result,fh,indent=2)
    make_plot(result)
    return result

def make_plot(results):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    # 3 standalone plots, no subplot, default colors
    t=results['capacities'][0]
    fig,ax=plt.subplots(figsize=(7.6,4.2))
    ax.plot(np.arange(len(t['path_J']))+1,t['path_J'],marker='o',ms=3,lw=1.5,label='Population-optimal retraining')
    ax.axhline(t['J_baseline'],ls='--',label='Before upgrade')
    ax.axhline(t['J_equilibrium'],ls=':',label='Updated equilibrium')
    ax.set_xlabel('Retraining epoch');ax.set_ylabel('Fresh-price precision J');ax.set_xlim(1,min(40,len(t['path_J'])));
    ax.set_title('Population retraining converges (8-factor aligned economy)');ax.legend(loc='upper right',fontsize=9);fig.tight_layout();fig.savefig(HERE/'population_retraining.png',dpi=180);plt.close(fig)
    dd=results['adoption'];w=np.array([r['adoption'] for r in dd]);q0=results['capacities'][0]['J_baseline']
    fig,ax=plt.subplots(figsize=(7.6,4.2))
    ax.plot(w,np.array([r['J_fixedlaw'] for r in dd])/q0-1,lw=2.2,label='First deployment (inherited distribution)')
    ax.plot(w,np.array([r['J_longrun'] for r in dd])/q0-1,lw=2.2,label='Retraining equilibrium')
    ax.axhline(0,ls='--',lw=1)
    ax.set_xlabel('Adoption share of higher-capacity AI');ax.set_ylabel('Change in fresh-price precision (fraction)');ax.set_title('Population retraining amplifies information crowd-out');ax.legend();fig.tight_layout();fig.savefig(HERE/'population_adoption.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.1))
    cap=results['capacities'];labels=[f"{r['low']} to {r['high']} lags" for r in cap];ii=np.arange(len(labels))
    ax.bar(ii-.18,[r['impact_pct'] for r in cap],.36,label='First deployment')
    ax.bar(ii+.18,[r['long_pct'] for r in cap],.36,label='Long-run equilibrium')
    ax.axhline(0,linewidth=.8)
    ax.set_xticks(ii,labels);ax.set_ylabel('Change in price information (%)');ax.set_title('Upgrades of different magnitudes');ax.legend();fig.tight_layout();fig.savefig(HERE/'population_capacity.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
