"""Red-Queen capacity-value audit, 8D Gaussian E1 microfounded market.

Two policies/laws:
  - Frozen: trade with type/capacity policies learned once under the baseline law
    (but price law is re-cleared and re-stationarized).
  - Retrained: every type/capacity cell population-retrains under current law,
    then clears until a fixed point.
Counterfactual values:
  - New entrant's *optimally trained* capacity premium given the inherited
    frozen or retrained stationary law, using exact CARA ex-ante log variance ratio.
  - Truly frozen candidate policies' actual ex ante CARA certainty equivalents
    under changed law, using an exact bivariate-Gaussian quadratic MGF.

No finite-sample or SGD; the law's covariance is from Lyapunov. All classes
are nested finite public lag capacities (0,1,2,3) plus private signals and p_t.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json, numpy as np
from scipy.optimize import root
from benchmark import Benchmark
from sensitivity import analytic_map_jac

HERE=Path(__file__).resolve().parent

def q_from_preds(b, cells):
    A=0.;C=np.zeros(b.zdim);Delta=0.
    for model,mass in cells:
        if not mass: continue
        w=mass/(model['gamma']*model['v'])
        A+=w*model['beta'];C+=w*model['b'];Delta+=w*(b.eco.risk_free_gross-model['chi'])
    if not Delta>0: raise ValueError('Nonpositive auction price slope')
    return np.r_[A,C,Delta]

def train_cells(b,q,levels=(0,1,2,3)):
    law=b.law(q,False)
    return {(g,k):b.fit_one(law,g,k) for g in (0,1) for k in levels}

def map_shares(b,q,x,cap=((0,3),(0,3)),trained=None):
    preds=train_cells(b,q,levels=tuple(set(cap[0]+cap[1]))) if trained is None else trained
    cells=[]
    for g in (0,1):
        for k,frac in zip(cap[g],(1-x[g],x[g])):
            if frac>0:cells.append((preds[g,k],b.temps[g].mass*frac))
    return q_from_preds(b,cells)

def stationary_population(b,x,cap=((0,3),(0,3)),init=None,tol=5e-12):
    q=np.array(b.seed() if init is None else init,float)
    for it in range(1200):
        nn=map_shares(b,q,x,cap)
        if np.max(np.abs(nn-q))<tol:
            return nn,it+1
        q=nn
    sol=root(lambda u:u-map_shares(b,u,x,cap),q,tol=1e-11)
    if max(abs(sol.fun))>1e-8:raise ValueError('No converged population fixed point')
    return sol.x,it+1

def fitted_value(b,q,g,low,high):
    preds=train_cells(b,q,(low,high))
    vL=preds[g,low]['v'];vH=preds[g,high]['v']
    return float(np.log(vL/vH)/(2*b.temps[g].gamma))

def fitted_values(b,q,group,levels=(0,3)):
    law=b.law(q,False)
    R=law['d']-b.eco.risk_free_gross*law['p']
    V=float(R@law['cov']@R)
    results={}
    for k in levels:
        pm=b.fit_one(law,group,k)
        results[k]={'v':pm['v'],'U':float(np.log(V/pm['v'])/(2*b.temps[group].gamma)),
                    'chi':pm['chi'],'beta':pm['beta'],'partial_R2':None}
    return results

def true_CE_of_frozen_policy(b,q,policy,group):
    """Exact ex-ante CARA CE for out-of-distribution *unchanged* policy.
    Under zero-mean jointly Gaussian fitted excess forecast f and payoff R,
    x=f/(gamma*v_train), CE=(1/(2gamma))log[1+2 Cov(f,R)/v-
             (Var(f)Var(R)-Cov(f,R)^2)/v^2].
    Existence requires positive-definite tilted Gaussian precision.
    """
    law=b.law(q,False)
    cov=law['cov'];th=law['theta'];p=law['p'];d=law['d'];Z=law['Z']
    r=d-b.eco.risk_free_gross*p
    f=policy['b']@Z+policy['beta']*th+(policy['chi']-b.eco.risk_free_gross)*p
    vr=float(r@cov@r)
    vf=float(f@cov@f+policy['beta']**2*b.temps[group].signal_noise_sd**2)
    cr=float(f@cov@r)
    v=policy['v']
    disc=1+2*cr/v+(cr*cr-vr*vf)/(v*v)
    if disc<=0 or 1+cr/v<=0:raise ValueError('Frozen CARA exponential moment diverges')
    return float(np.log(disc)/(2*b.temps[group].gamma))

def frozen_premium(b,q,trained,g,lo=0,hi=3):
    return true_CE_of_frozen_policy(b,q,trained[g,hi],g)-true_CE_of_frozen_policy(b,q,trained[g,lo],g)

def fd_q_gradient(f,q,h=1e-5):
    a=np.zeros_like(q)
    for j in range(len(q)):
        e=np.zeros(len(q));e[j]=h*max(1,abs(q[j]))
        a[j]=(f(q+e)-f(q-e))/(2*e[j])
    return a

def analysis_one_group(b,q,x0,cap=((0,3),(0,0)),h=1e-5):
    # derivative on the regular retraining branch (group 1 held low)
    x=np.array([x0,0.0]);preds=train_cells(b,q,(0,3))
    low=preds[0,0];high=preds[0,3]; pi=b.temps[0].mass
    deriv=np.r_[pi*(high['beta']/(high['gamma']*high['v'])-low['beta']/(low['gamma']*low['v'])),
                 pi*(high['b']/(high['gamma']*high['v'])-low['b']/(low['gamma']*low['v'])),
                 pi*((b.eco.risk_free_gross-high['chi'])/(high['gamma']*high['v'])-
                     (b.eco.risk_free_gross-low['chi'])/(low['gamma']*low['v']))]
    Jq=analytic_map_jac(b,q,adoption=x0,lo=0,hi=3)
    dq=np.linalg.solve(np.eye(len(q))-Jq,deriv)
    data={'rho_learning':float(max(abs(np.linalg.eigvals(Jq)))),
          'dq_jump_norm':float(np.linalg.norm(deriv)),
          'dq_feedback_norm':float(np.linalg.norm(dq-deriv))}
    for g in (0,1):
        grad=fd_q_gradient(lambda qq:fitted_value(b,qq,g,0,3),q,h=h)
        z0=float(grad@deriv);zf=float(grad@(dq-deriv));zt=float(grad@dq)
        data[f'g{g}']={'direct_training_map':z0,'feedback':zf,'total':zt}
    return data

def derivative_profile(b,x=(.5,.5),trained=None,cap=((0,3),(0,3)),h=2e-4):
    method = (lambda xx:map_shares(b,None,xx,cap,trained)) if trained is not None else (lambda xx:stationary_population(b,xx,cap)[0])
    matrix=np.zeros((2,2));rhos=[]
    for j in (0,1):
        xp=list(x); xm=list(x);xp[j]+=h;xm[j]-=h
        qp=method(xp);qm=method(xm)
        for g in (0,1):
            matrix[g,j]=(fitted_value(b,qp,g,0,3)-fitted_value(b,qm,g,0,3))/(2*h)
    return matrix

def shapley_three(b,q0,q1,g=1):
    """Path-symmetric Shapley decomposition across A, public loadings C, and Delta.
    These three blocks interact because lagged prices are endogenous; ignoring
    Delta is not valid when C has coefficients on past nominal prices.
    """
    import itertools,math
    f={}
    for bits in itertools.product((0,1),repeat=3):
        qq=q0.copy()
        if bits[0]: qq[0]=q1[0]
        if bits[1]: qq[1:-1]=q1[1:-1]
        if bits[2]: qq[-1]=q1[-1]
        f[bits]=fitted_value(b,qq,g,0,3)
    contributions=[]
    for j in range(3):
        effect=0.
        for bits in itertools.product((0,1),repeat=3):
            if bits[j]:continue
            sz=sum(bits); weight=math.factorial(sz)*math.factorial(2-sz)/6
            bnew=list(bits);bnew[j]=1
            effect+=weight*(f[tuple(bnew)]-f[bits])
        contributions.append(effect)
    return {'base':f[(0,0,0)],'A_only':f[(1,0,0)],
            'C_only':f[(0,1,0)],'Delta_only':f[(0,0,1)],
            'both_AC_only':f[(1,1,0)],'full':f[(1,1,1)],
            'Shapley_A':contributions[0], 'Shapley_C':contributions[1],
            'Shapley_Delta':contributions[2],
            'sum':sum(contributions),
            'truth_total':f[(1,1,1)]-f[(0,0,0)]}

def run_main():
    b=Benchmark(K=3);cap=((0,3),(0,3));base,_=stationary_population(b,(0,0),cap)
    trained=train_cells(b,base)
    qprev=base.copy();rows=[];solves=[]
    for x0 in np.linspace(0,1,21):
        xf=(float(x0),0.)
        frozen=map_shares(b,None,xf,cap,trained)
        re,itr=stationary_population(b,xf,cap,init=qprev);qprev=re.copy()
        solves.append(itr)
        row={'x0':float(x0),'J_frozen':b.J(frozen),'J_retrained':b.J(re),
             'rho_fast_frozen':b.law(frozen)['fast_radius'],'rho_fast_retrained':b.law(re)['fast_radius'],
             'frozen_clearing_Delta':float(frozen[-1]), 'retrained_Delta':float(re[-1]),
             'iterations':itr}
        for g in (0,1):
            row[f'g{g}_value_frozen_bestresponse']=fitted_value(b,frozen,g,0,3)
            row[f'g{g}_value_retrained_bestresponse']=fitted_value(b,re,g,0,3)
            row[f'g{g}_value_literal_frozen_policy']=frozen_premium(b,frozen,trained,g)
            for k in (0,3):
                info=fitted_values(b,re,g,(k,))[k]
                row[f'g{g}_U{k}_retrained']=info['U'];row[f'g{g}_v{k}_retrained']=info['v']
        rows.append(row)
    at=.5;qi=stationary_population(b,(at,0),cap)[0]
    local=analysis_one_group(b,qi,at)
    x0=rows[0];x5=rows[10];x1=rows[-1]
    q_full=stationary_population(b,(1.,0.),cap)[0]
    decomp=shapley_three(b,base,q_full,1)
    # Exact frozen slope: learned policies fixed Q0; full slopes compared at same point.
    frozen_jac=derivative_profile(b,x=(.5,.5),trained=trained)
    full_jac=derivative_profile(b,x=(.5,.5))
    base_curve=[]
    for x0 in (0,.5,1.):
        qu=stationary_population(b,(x0,0),cap)[0]
        report={f'g{g}':fitted_values(b,qu,g,(0,1,2,3)) for g in (0,1)}
        base_curve.append({'share':x0,'data':report})
    outcome={'design':{'scenario':'E1_aligned','dimension':8,'public_lags_K':3,
                       'group0_mass':b.temps[0].mass,'group1_mass':b.temps[1].mass,
                       'group0_gamma':b.temps[0].gamma,'group1_gamma':b.temps[1].gamma,
                       'group0_private_noise_sd':b.temps[0].signal_noise_sd,
                       'group1_private_noise_sd':b.temps[1].signal_noise_sd,
                       'capacity_menu':[0,1,2,3],
                       'baseline_low_low_J':b.J(base),
                       'baseline_q':base.tolist(),
                       'frozen_is_population_trained_only_at_baseline':True},
             'adoption_curve':rows,'local_at_half_group0':local,
             'full_frozen_Z_at_half_half':frozen_jac.tolist(),
             'full_retrained_Z_at_half_half':full_jac.tolist(),
             'shapley_group1_full_transition':decomp,
             'population_competition_along_levels':base_curve,
             'max_clearing_check':'Analytic competitive clearing by construction',
             'max_fixed_residual':float(max(np.max(abs(stationary_population(b,(r['x0'],0),cap)[0]-map_shares(b,stationary_population(b,(r['x0'],0),cap)[0],(r['x0'],0),cap))) for r in rows[::5]))}
    (HERE/'redqueen_results.json').write_text(json.dumps(outcome,indent=2))
    import csv
    with open(HERE/'adoption_values.csv','w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print('Local',json.dumps(local,indent=2))
    print('2group frozen',frozen_jac,'retrain',full_jac)
    print('Shapley',decomp)
    print('Selected rows')
    for r in rows[::5]:print('x0',r['x0'],'Jf',r['J_frozen'],'Jr',r['J_retrained'],
                              'G1 frozenopt',r['g1_value_frozen_bestresponse'],
                              'G1 retrainopt',r['g1_value_retrained_bestresponse'],
                              'G1 frozenpolicy',r['g1_value_literal_frozen_policy'])
    print('Residual',outcome['max_fixed_residual'])
    make_plots(rows)
    return outcome

def make_plots(rows):
    import matplotlib.pyplot as plt
    x=np.array([r['x0'] for r in rows]);J0=rows[0]['J_retrained']
    fig,ax=plt.subplots(figsize=(7.3,4.3))
    ax.plot(x,[100*(r['J_frozen']/J0-1) for r in rows],marker='s',ms=3,label='Incumbent policies fixed')
    ax.plot(x,[100*(r['J_retrained']/J0-1) for r in rows],marker='o',ms=3,label='All types population-retrain')
    ax.set(xlabel='Group 1 high-capacity adoption rate',ylabel='Price fresh-information change (%)',title='Crowd-out under frozen and retrained policies')
    ax.axhline(0,color='grey',linestyle='--',lw=.8)
    ax.legend(fontsize=9);fig.tight_layout();fig.savefig(HERE/'redqueen_price_precision.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.3))
    ax.plot(x,[r['g1_value_frozen_bestresponse'] for r in rows],marker='s',ms=3,label='Fixed incumbents; marginal entrant retrains')
    ax.plot(x,[r['g1_value_retrained_bestresponse'] for r in rows],marker='o',ms=3,label='All types population-retrain')
    ax.plot(x,[r['g1_value_literal_frozen_policy'] for r in rows],ls=':',label='All policies frozen, actual CARA value')
    ax.set(xlabel='Group 1 high-capacity adoption rate',ylabel='Group 2 upgrade CE (per opportunity)',title='Computational strategic substitution in the baseline market')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'redqueen_private_value.png',dpi=190);plt.close(fig)

if __name__=='__main__':run_main()
