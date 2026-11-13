"""Independent compatibility test against the Phase-1 competitive auction engine."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from benchmark import Benchmark
from marketlab.policies import InvestorPolicy, HeadOutput
from marketlab.simulation import simulate

class LagHistoryEncoder:
    """Reconstruct pre-auction public features without contemporaneous price/future payoff."""
    def __init__(self,K,K_active,npub):self.K=K;self.K_active=K_active;self.npub=npub;self.reset()
    def reset(self):self.hist=[]
    def step(self,u):
        u=np.asarray(u,float)
        z=np.zeros(self.npub+(self.npub+2)*self.K)
        z[:self.npub]=u[2:]
        for k in range(1,self.K_active+1):
            # block k: public input at t-k+1 gives (p_{t-k}, D_{t-k+1}),
            # public input t-k gives observed macro at t-k.
            source=u if k==1 else (self.hist[-(k-1)] if len(self.hist)>=k-1 else np.zeros_like(u))
            oldmacro=self.hist[-k][2:] if len(self.hist)>=k else np.zeros(self.npub)
            block=np.r_[source[:2],oldmacro]
            offset=self.npub+(k-1)*(self.npub+2)
            z[offset:offset+self.npub+2]=block
        self.hist.append(u.copy())
        if len(self.hist)>self.K:self.hist.pop(0)
        return z

class PopulationHead:
    def __init__(self,b,beta,chi,v):self.b=b;self.beta=beta;self.chi=chi;self.v=v
    def evaluate(self,state):return HeadOutput(float(self.b@state),self.beta,self.chi,self.v)


def check(caps=(3,0), n=26000, chi_bound=None, supply_noise_sd=None):
    b=Benchmark(K=3,chi_bound=chi_bound)
    if supply_noise_sd is not None:
        from dataclasses import replace
        b.eco=replace(b.eco,supply_noise_sd=supply_noise_sd)
        b.sigma_supply=supply_noise_sd
    q,stats=b.equilibrium(caps)
    _,fit=b.update(q,caps,full=True)
    policies=[]
    for i,raw in enumerate(fit['preds']):
        base=b.temps[i]
        policies.append(InvestorPolicy(f'pop_type_{i}',base.mass,base.gamma,base.signal_loading,base.signal_noise_sd,
            LagHistoryEncoder(b.K,caps[i],b.npub),PopulationHead(raw['b'],raw['beta'],raw['chi'],raw['v'])))
    bank=b.eco.draw_bank(n,burn=500,seed=20261016,n_private_streams=2)
    panel=simulate(b.eco,policies,bank)
    law=b.law(q)
    Pvar=float(law['p']@law['cov']@law['p'])
    obsvar=float(np.var(panel.price))
    delta_target=float(law['p']@law['cov']@law['d'])
    obs_cov=float(np.cov(panel.price,panel.future_dividend,bias=True)[0,1])
    out={'n':n,'caps':list(caps), 'max_clearing_residual':float(np.max(abs(panel.demand_residual))),
        'analytic_price_variance':Pvar,'empirical_price_variance':obsvar,
        'price_variance_relative_error':float(obsvar/Pvar-1),
        'analytic_price_payoff_covariance':delta_target,'empirical_price_payoff_covariance':obs_cov,
        'price_payoff_cov_relerr':float(obs_cov/delta_target-1),
        'chi_bound':chi_bound,'supply_noise_sd':b.sigma_supply,
        'stationary_j':b.J(q),'empirical_loading_J':float(np.mean((panel.fresh_factor_loadings@b.eco.target_loading)**2)/b.eco.supply_noise_sd**2),
        'min_clearing_slope':float(np.min(panel.clearing_slope))}
    print(json.dumps(out,indent=2))
    assert out['max_clearing_residual']<1e-11
    assert abs(out['empirical_loading_J']-out['stationary_j'])<1e-9
    assert abs(out['price_variance_relative_error'])<.05
    assert abs(out['price_payoff_cov_relerr'])<.05
    with open(Path(__file__).parent/('independent_engine_check_bound.json' if chi_bound else 'independent_engine_check.json'),'w') as f:json.dump(out,f,indent=2)
    return out
if __name__=='__main__':check()
