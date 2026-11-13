"""Audits the sufficient conditions for the dual-externalities/CARA bridge theorem
in the SAME 8D aligned Gaussian population market with finite-K=3 histories.

Reuse independently stored reproduction package; compute exact covariance-based
moments at the fitted 8d competitive-clearing laws. No Monte Carlo and no long-
history/full-oracle substitutions.
"""
import sys, csv, json, argparse, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'prior'/'src'),str(HERE/'prior'/'vendor')]
from adoption_spillovers import Benchmark,make_b
from run_redqueen import stationary_population,train_cells,map_shares

def cond_var(target, features, sigma):
    if features.shape[0]==0:return float(target@sigma@target)
    sig=features@sigma@features.T
    v=features@sigma@target
    z=cho_solve(cho_factor(sig,check_finite=False),v,check_finite=False)
    return float(target@sigma@target-v@z)

def statistics(b,q,group=1,K=3,checks=True):
    law=b.law(q,verify=False)
    cov=law['cov'];theta=law['theta'];z=law['Z'];price=law['p']
    F=z[:b.npub]
    U=z[:b.npub+b.nobs*K]
    P_L=cond_var(theta,F,cov)
    P_H=cond_var(theta,U,cov)
    L=cond_var(theta,np.vstack((F,price)),cov)
    H=cond_var(theta,np.vstack((U,price)),cov)
    S=.5*np.log(P_L/L); I_H=.5*np.log(P_H/H)
    nu=float(b.temps[group].signal_noise_sd**2)
    gamma=float(b.temps[group].gamma)
    eps2=float(b.eco.payoff_noise_sd**2)
    lam=float(b.eco.payoff_loading@b.eco.target_loading)
    # NOTE lambda assumes payoff_loading EXACTLY parallel to target for aligned case
    if not np.allclose(b.eco.payoff_loading,lam*b.eco.target_loading,atol=1e-11):
        raise AssertionError('Target is not aligned, not theorem-applicable')
    vL=eps2+lam**2*nu*L/(nu+L)
    vH=eps2+lam**2*nu*H/(nu+H)
    V=np.log(vL/vH)/(2*gamma)
    J=b.J(q)
    if checks:
        assert abs(I_H-.5*np.log1p(J*P_H))<1e-7,('high info formula mismatch',q)
        vL_fit=b.fit_one(law,group,0)['v'];vH_fit=b.fit_one(law,group,K)['v']
        assert abs(vL_fit-vL)<1e-8 and abs(vH_fit-vH)<1e-8, (vL_fit,vL,vH_fit,vH)
    return dict(P_L=P_L,P_H=P_H,L=L,H=H,S=S,F=I_H,V=V,
                vL=vL,vH=vH,J=J,gamma=gamma,nu=nu,lam=lam,eps2=eps2)

def calc_one(b,shares=[0,.25,.5,.75,1.]):
    q0,_=stationary_population(b,(0,0)); trained=train_cells(b,q0)
    current=q0.copy(); r={}
    for x in shares:
        qf=map_shares(b,None,(x,0),((0,3),(0,3)),trained)
        qr,it=stationary_population(b,(x,0),((0,3),(0,3)),init=current)
        current=qr.copy()
        for prot,q in [('frozen',qf),('retrained',qr)]:
            r[(prot,x)]=statistics(b,q)
    return r

def flags(before,after):
    d={k:after[k]-before[k] for k in ('P_L','P_H','L','H','S','F','V','J')}
    d.update(positive_spillover=d['S']>1e-10,
             negative_fresh=d['F']< -1e-10,
             strategic_substitution=d['V']< -1e-10,
             bridge_strong=(abs(d['P_L'])<1e-8 and d['P_H']>=-1e-10 and d['S']>1e-10 and d['F'] < -1e-10),
             bridge_weak=(d['L']< -1e-10 and d['H']>1e-10),
             prior_increase=d['P_H']>=-1e-10)
    return d

def run():
    t=time.time()
    aligned=json.loads((HERE/'prior'/'vendor'/'sweep_results.json').read_text())['records']
    rows=[];data=[]
    # Only previously validated aligned fixed-point calibrations (147), not 180 original draws
    for idx,rec in enumerate(aligned):
        b=make_b(rec['factors'],'aligned')
        try: rr=calc_one(b,shares=[0,.25,.5,.75,1.])
        except Exception as ex:
            print('ERR',rec['index'],ex,flush=True);continue
        if idx==0:
            data=rr
        for protocol in ('frozen','retrained'):
            p0=rr[(protocol,0)];p1=rr[(protocol,1)];p25=rr[(protocol,.25)];p75=rr[(protocol,.75)]
            for span,aa,zz in [('0_to_1',p0,p1),('p25_to_p75',p25,p75)]:
                d=flags(aa,zz)
                rows.append({'index':rec['index'],'protocol':protocol,'span':span,**d})
        if idx%30==0: print('aligned',idx,'elapsed',round(time.time()-t,1),flush=True)
    with open(HERE/'aligned_condition_audit.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    for prot in ('frozen','retrained'):
        for sp in ('0_to_1','p25_to_p75'):
            sub=[r for r in rows if r['protocol']==prot and r['span']==sp]
            print('\n',prot,sp,'n',len(sub))
            for flag in ('positive_spillover','negative_fresh','strategic_substitution','bridge_strong','bridge_weak','prior_increase'):
                print(flag,sum(r[flag] for r in sub),'/',len(sub))
            print('PH delta min/median/max',np.min([r['P_H'] for r in sub]),np.median([r['P_H'] for r in sub]),np.max([r['P_H'] for r in sub]))
            print('L delta range',min(r['L'] for r in sub),max(r['L'] for r in sub),'H delta range',min(r['H'] for r in sub),max(r['H'] for r in sub))
            print('Exceptions (bridge strong)', [(r['index'],r['P_H'],r['F'],r['S']) for r in sub if not r['bridge_strong']][:12])
    out={'n_aligned_success':int(len(rows)//4),'elapsed':round(time.time()-t,2),
         'baseline':{p:[{'x':x,**data[(p,x)]} for x in [0,.25,.5,.75,1.]] for p in ['frozen','retrained']}}
    (HERE/'aligned_results.json').write_text(json.dumps(out,indent=2))
    print('elapsed',time.time()-t)

if __name__=='__main__':run()
