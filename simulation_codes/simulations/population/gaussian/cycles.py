"""Analyze exact population flip, attracting 2-cycle and crowd-out relative to baseline."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import root, brentq
from benchmark import Benchmark
from sensitivity import analytic_map_jac
OUT=Path(__file__).resolve().parent


def set_sd(b,sd,eco):
    b.eco=replace(eco,supply_noise_sd=float(sd));b.sigma_supply=float(sd)

def root_fixed(b,caps,init):
    sol=root(lambda x: x-b.update(x,caps),init,tol=1e-12)
    if np.max(abs(sol.fun))>1e-9: raise ValueError(f'no root at {b.sigma_supply}: {np.max(abs(sol.fun))}')
    return sol.x

def info_mi(b,q):
    law=b.law(q)
    cov=law['cov'];z=law['Z'];theta=law['theta'];
    var_theta=float(theta@cov@theta)
    ZCov=z@cov@z.T
    Cz=z@cov@theta
    cond=var_theta-Cz@np.linalg.solve(ZCov,Cz)
    J=b.J(q)
    return {'J':J,'prior_var_theta':var_theta,'conditional_var_theta_full_public':float(cond),
            'conditional_MI_nats':float(.5*np.log1p(cond*J))}

def calculate():
    b=Benchmark(K=3); original=b.eco
    baseeq,_=b.equilibrium((0,0))
    upgradedeq,_=b.equilibrium((3,0),start=baseeq)
    def leading_at_sd(sd):
        set_sd(b,sd,original)
        # same fixed point smoothly continued from known .9 root
        q=root_fixed(b,(3,0),upgradedeq)
        ev=np.linalg.eigvals(analytic_map_jac(b,q,(3,0)))
        lead=ev[np.argmax(abs(ev))]
        return float(lead.real)
    critical=brentq(lambda sd:leading_at_sd(sd)+1.,.715,.73,xtol=1e-12)
    slope=(leading_at_sd(critical+1e-4)-leading_at_sd(critical-1e-4))/2e-4
    print('critical sigma_supply',critical,'d eigen / dsigma',slope,flush=True)
    output={'critical_sd':float(critical),'transversality_dlambda_dsigma':float(slope), 'selected':[]}
    for sd in [.9,.75,.73,.725,.72,.719,.718,.715,.712,.71]:
        set_sd(b,sd,original)
        low=root_fixed(b,(0,0),baseeq)
        high=root_fixed(b,(3,0),upgradedeq)
        imp=b.update(low,(3,0))
        eig_lo=np.linalg.eigvals(analytic_map_jac(b,low,(0,0)))
        eig_hi=np.linalg.eigvals(analytic_map_jac(b,high,(3,0)))
        d={'sd':sd,'base_J':b.J(low),'impact_J':b.J(imp),
           'high_fixed_J':b.J(high),'baseline_rho':float(max(abs(eig_lo))),
           'high_fixed_rho':float(max(abs(eig_hi))),
           'high_fixed_dominant':float(eig_hi[np.argmax(abs(eig_hi))].real),
           'base_info':info_mi(b,low),'impact_info':info_mi(b,imp),
           'high_fixed_info':info_mi(b,high),
           'fast_rho_low':b.law(low)['fast_radius'],
           'fast_rho_high':b.law(high)['fast_radius']}
        if sd<critical:
            q=imp.copy()
            for _ in range(3000): q=b.update(q,(3,0))
            sol=root(lambda xx:xx-b.update(b.update(xx,(3,0)),(3,0)),q,tol=1e-11)
            qa=sol.x; qb=b.update(qa,(3,0))
            erra=float(np.max(abs(qa-b.update(qb,(3,0)))))
            J2=analytic_map_jac(b,qb,(3,0))@analytic_map_jac(b,qa,(3,0))
            rho2=float(max(abs(np.linalg.eigvals(J2))))
            d['cycle']={'J_a':b.J(qa),'J_b':b.J(qb),'mean_J':.5*(b.J(qa)+b.J(qb)),
                'rho_two_iteration':rho2,'residual':erra,'norm_separation':float(max(abs(qa-qb))),
                'qa':qa.tolist(),'qb':qb.tolist(),
                'info_a':info_mi(b,qa),'info_b':info_mi(b,qb),
                'fast_rho_a':b.law(qa)['fast_radius'],'fast_rho_b':b.law(qb)['fast_radius']}
        print('sigma',sd,'baseline',round(d['base_J'],6),'impact',round(d['impact_J'],6),
              'highfixed',round(d['high_fixed_J'],6),'rho',round(d['high_fixed_rho'],5),
              'cycle',(None if 'cycle' not in d else (round(d['cycle']['J_a'],6),round(d['cycle']['J_b'],6),round(d['cycle']['rho_two_iteration'],5))),flush=True)
        output['selected'].append(d)
    with open(OUT/'cycles.json','w') as fh:json.dump(output,fh,indent=2)
    make_figures(output)
    return output

def make_figures(o):
    import matplotlib.pyplot as plt
    D=o['selected'];ds=np.array([d['sd'] for d in D]); fixed=np.array([d['high_fixed_J'] for d in D]); before=np.array([d['base_J'] for d in D]);
    fig,ax=plt.subplots(figsize=(7.8,4.5))
    ax.plot(ds,before,'o--',label='Low-capacity population fixed point')
    ax.plot(ds,fixed,'o-',label='High-capacity population fixed point')
    yy=[d for d in D if 'cycle' in d]
    ax.plot([d['sd'] for d in yy],[d['cycle']['J_a'] for d in yy],'^',markersize=7,label='High-capacity attracting two-cycle')
    ax.plot([d['sd'] for d in yy],[d['cycle']['J_b'] for d in yy],'^',markersize=7)
    ax.axvline(o['critical_sd'],linestyle=':',label=f'Flip threshold ≈ {o["critical_sd"]:.4f}')
    ax.set_xlabel('Supply-noise standard deviation');ax.set_ylabel('Price information precision J');
    ax.set_title('Population-optimal learning: fixed point and two-cycle');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'population_flip.png',dpi=190);plt.close(fig)

    b=Benchmark();from dataclasses import replace
    qlo_seed,_=b.equilibrium((0,0));qhi_seed,_=b.equilibrium((3,0),qlo_seed)
    b.eco=replace(b.eco,supply_noise_sd=.715);b.sigma_supply=.715
    q0=root_fixed(b,(0,0),qlo_seed);qh=root_fixed(b,(3,0),qhi_seed)
    q=b.update(q0,(3,0));vals=[b.J(q)]
    for i in range(140):
        q=b.update(q,(3,0));vals.append(b.J(q))
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    ax.plot(np.arange(1,len(vals)+1),vals,lw=1.2,label='Successive population-best-response epochs')
    ax.axhline(b.J(q0),linestyle='--',label='Low-capacity baseline')
    ax.axhline(b.J(qh),linestyle=':',label='High-capacity unstable fixed point')
    ax.set_xlim(1,140);ax.set_xlabel('Retraining epoch');ax.set_ylabel('Fresh-price precision J')
    ax.set_title('Stable fast markets, nonconvergent optimal retraining')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'population_cycle_path.png',dpi=190);plt.close(fig)

if __name__=='__main__':calculate()
