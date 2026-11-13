"""Joint negative and positive informational externalities / capacity investments.

Population Gaussian 8D microfounded CARA market. Full public-history conditioning
approximated with 40 lagged blocks (checked at 32/48). Fixed-incumbent policies
vs all-type population retraining evaluated at each adoption share x.

Metrics:
  I_fresh  = I(theta_t; p_t | m_t, public history 40 lags)
  I_short  = I(theta_t; p_t | m_t)
  I_repack = I_short - I_fresh (chain-rule difference, generally signed)
  W_short  = (1/2)*(Var(theta|m,p)_baseline - Var(theta|m,p)_x)
  V_g      = (1/(2 gamma_g))*log(v_g,K0 / v_g,K3), atomistic, optimal
             capacity-conditioned candidate in each stationary Q.

The "frozen" branch freezes existing training Q(0) predictors, then computes
stationary Q(x) and allows a negligible candidate to optimize its own model.
"""
from __future__ import annotations
import sys, os, csv, json, argparse, time, signal
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.linalg import cho_factor, cho_solve
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vendor'))
from benchmark import Benchmark
from run_redqueen import stationary_population,train_cells,map_shares,fitted_value
from population_frontier import stationary_moments,conditional_variance

ORACLE_L=40

def measurement(b,q,oracle_l=ORACLE_L):
    moments=stationary_moments(b,q,oracle_l)
    vfull=conditional_variance(moments,oracle_l,False)
    vfullp=conditional_variance(moments,oracle_l,True)
    vshort=conditional_variance(moments,0,False)
    vshortp=conditional_variance(moments,0,True)
    fresh=float(.5*np.log(vfull/vfullp))
    short=float(.5*np.log(vshort/vshortp))
    J=b.J(q)
    fresh_analytic=float(.5*np.log1p(J*vfull))
    if abs(fresh-fresh_analytic)>2e-7:
        raise AssertionError(f'Gaussian likelihood mismatch {fresh} vs {fresh_analytic}')
    return {'I_fresh_nats':fresh,'I_short_nats':short,
            'I_repack_nats':short-fresh,'v_theta_public_full':float(vfull),
            'v_theta_short_preprice':float(vshort),
            'v_theta_short_postprice':float(vshortp),
            'J_fresh_precision':J,
            'value_candidate_type2_0to3':fitted_value(b,q,1,0,3),
            'fast_radius':b.law(q,False)['fast_radius'],
            'auction_denominator':float(q[-1])}

def make_b(factors,scenario='aligned'):
    b=Benchmark(K=3)
    # same primitive sampling as prior 463-case frozen/retrained audit
    if scenario == 'aligned':
        b.eco=replace(b.eco,
                      supply_noise_sd=b.eco.supply_noise_sd*factors['supply'],
                      payoff_noise_sd=b.eco.payoff_noise_sd*factors['payoff_noise'],
                      F=b.eco.F*factors['Fmult'])
    elif scenario == 'nonaligned':
        ell=1.15*b.eco.target_loading + factors['macro1']*b.eco.public_loading[0] + factors['macro2']*b.eco.public_loading[1]
        orth=np.zeros(b.m);orth[6]=1.;ell+=factors['macro3']*orth
        b.eco=replace(b.eco,payoff_loading=ell,
                      supply_noise_sd=b.eco.supply_noise_sd*factors['supply'],
                      payoff_noise_sd=b.eco.payoff_noise_sd*factors['payoff_noise'],
                      F=b.eco.F*factors['Fmult'])
    else: raise ValueError(scenario)
    b.sigma_supply=b.eco.supply_noise_sd
    b.temps[0].gamma*=factors['gamma0'];b.temps[1].gamma*=factors['gamma1']
    b.temps[0].signal_noise_sd*=factors['sig0'];b.temps[1].signal_noise_sd*=factors['sig1']
    b.temps[0].mass=factors['pi0'];b.temps[1].mass=1-factors['pi0']
    return b

def adoption_path(b,x_values=np.linspace(0,1,21),oracle_l=ORACLE_L):
    cap=((0,3),(0,3))
    q0,iter0=stationary_population(b,(0,0),cap)
    trained=train_cells(b,q0)
    m0=measurement(b,q0,oracle_l)
    allrows=[]; qlast=q0.copy()
    for x in x_values:
        # policies are trained only at Q(0) for the frozen branch
        qf=map_shares(b,None,(float(x),0.),cap,trained)
        qr,it=stationary_population(b,(float(x),0.),cap,init=qlast)
        qlast=qr.copy()
        for protocol,q in [('frozen',qf),('retrained',qr)]:
            m=measurement(b,q,oracle_l)
            m.update(x=float(x),protocol=protocol, iteration_count=int(it if protocol=='retrained' else 0))
            m.update(dI_fresh=m['I_fresh_nats']-m0['I_fresh_nats'],
                     dI_short=m['I_short_nats']-m0['I_short_nats'],
                     dI_repack=m['I_repack_nats']-m0['I_repack_nats'],
                     downstream_quadratic_value=0.5*(m0['v_theta_short_postprice']-m['v_theta_short_postprice']),
                     dV_candidate=m['value_candidate_type2_0to3']-m0['value_candidate_type2_0to3'])
            allrows.append(m)
    return allrows

def save_csv(rows,path):
    if not rows:return
    with open(path,'w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)

def strict_fixed(b,x,initial,maxiter=450,tol=3e-9):
    q=initial.copy()
    for n in range(maxiter):
        nxt=map_shares(b,q,(float(x),0.))
        if np.max(np.abs(nxt-q))<tol:return nxt
        q=nxt
    raise ValueError('No converged fixed point within 450 iterations')

def analyze_robustness(limit=9999,oracle_l=ORACLE_L,begin=0,only=None):
    """Reproduce the original sweep at x=.25,.75, including its precise root initialization.
    This avoids implicit root continuation near unstable branches.
    """
    allrows=[]; statuses=[];total=0;start=time.time()
    for scenario,file in [('aligned','sweep_results.json'),('nonaligned','sweep_nonaligned_results.json')]:
        if only is not None and scenario!=only:continue
        meta=json.loads((BASE/'vendor'/file).read_text())
        for rec in meta['records'][begin:limit]:
            total+=1; idx=rec['index']; b=make_b(rec['factors'],scenario)
            if scenario=='nonaligned' and total>=285: print('ENTER',total,scenario,idx,flush=True)
            def timeout_handler(signum, frame): raise TimeoutError('exceeded 2s per calibration')
            old_handler=signal.signal(signal.SIGALRM,timeout_handler)
            signal.setitimer(signal.ITIMER_REAL,2.)
            try:
                qb=strict_fixed(b,0,b.seed())
                trained=train_cells(b,qb)
                qf25=map_shares(b,None,(.25,0),trained=trained)
                qf75=map_shares(b,None,(.75,0),trained=trained)
                qr25=strict_fixed(b,.25,qf25)
                qr75=strict_fixed(b,.75,qr25)
                qs={'frozen':(qf25,qf75),'retrained':(qr25,qr75)}
                for protocol,(qa,qz) in qs.items():
                    a=measurement(b,qa,oracle_l);z=measurement(b,qz,oracle_l)
                    d={'scenario':scenario,'index':idx,'protocol':protocol,'x_low':.25,'x_high':.75,
                       'dI_fresh':z['I_fresh_nats']-a['I_fresh_nats'],
                       'dI_short':z['I_short_nats']-a['I_short_nats'],
                       'dI_repack':z['I_repack_nats']-a['I_repack_nats'],
                       'dW_short':.5*(a['v_theta_short_postprice']-z['v_theta_short_postprice']),
                       'dV_type2':z['value_candidate_type2_0to3']-a['value_candidate_type2_0to3'],
                       'crowdout':z['I_fresh_nats']<a['I_fresh_nats']-1e-10,
                       'spillover':z['I_short_nats']>a['I_short_nats']+1e-10,
                       'saturation':z['value_candidate_type2_0to3']<a['value_candidate_type2_0to3']-1e-10}
                    d['all_three']=bool(d['crowdout'] and d['spillover'] and d['saturation'])
                    allrows.append(d)
            except Exception as e:
                statuses.append({'scenario':scenario,'index':idx,'status':'invalid','reason':str(e)})
            finally:
                signal.setitimer(signal.ITIMER_REAL,0.)
                signal.signal(signal.SIGALRM,old_handler)
            if total%50==0:print('Robustness examined',total,'elapsed',round(time.time()-start,1),'s',flush=True)
    return allrows,statuses

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--robust',action='store_true');parser.add_argument('--limit',type=int,default=99999);parser.add_argument('--begin',type=int,default=0);parser.add_argument('--only',default=None);
    p=parser.parse_args()
    b=Benchmark(K=3)
    rows=adoption_path(b)
    save_csv(rows,BASE/'results'/'baseline_adoption_21point.csv')
    print('BASELINE endpoints:')
    for r in rows:
        if r['x'] in [0.,.5,1.]:print('%.2f %-9s J=%8.6f MI_full=%.6f MI_short=%.6f repackage=%.6f V=%.6f welfare=%+.6f'%(r['x'],r['protocol'],r['J_fresh_precision'],r['I_fresh_nats'],r['I_short_nats'],r['I_repack_nats'],r['value_candidate_type2_0to3'],r['downstream_quadratic_value']))
    if p.robust:
        rr,ss=analyze_robustness(p.limit,begin=p.begin,only=p.only)
        save_csv(rr,BASE/'results'/f'robustness_{p.only or "both"}_{p.begin}to{p.limit}.csv')
        (BASE/'results'/f'robustness_status_{p.only or "both"}_{p.begin}to{p.limit}.json').write_text(json.dumps(ss,indent=2))
        for scen in ['aligned','nonaligned']:
            for prot in ['frozen','retrained']:
                sub=[r for r in rr if r['scenario']==scen and r['protocol']==prot]
                print('robust',scen,prot,len(sub),'all3',sum(r['all_three'] for r in sub),'neg-fresh',sum(r['crowdout'] for r in sub),'pos-short',sum(r['spillover'] for r in sub),'neg-private',sum(r['saturation'] for r in sub),flush=True)
        print('invalid',len(ss), 'elapsed',round(time.time()-start if False else 0,1),flush=True)
