"""A competitive price-taking call auction with continuum aggregation.

A representative individual in policy c sees s_ci = L_c f + sigma_xi xi_ci.
The continuum average is E[s_ci|f]=L_c f, NOT an individual noisy signal.
This is why realized clearing depends on underlying f without granting
an individual investor access to f: it is a law-of-large-numbers limit.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
from .economy import Economy, ShockBank
from .policies import InvestorPolicy, HeadOutput


@dataclass
class SimulationResult:
    factor: np.ndarray
    public: np.ndarray
    lag_dividend: np.ndarray
    future_dividend: np.ndarray
    target: np.ndarray
    price: np.ndarray
    public_input: np.ndarray
    supply: np.ndarray
    demand_residual: np.ndarray
    clearing_slope: np.ndarray
    fresh_factor_loadings: np.ndarray
    intercept_aggregate: np.ndarray
    private_signals: np.ndarray
    head_parameters: np.ndarray
    states: list[np.ndarray]
    metadata: dict

    def save(self, path):
        arrays = {k:getattr(self,k) for k in (
            'factor','public','lag_dividend','future_dividend','target','price',
            'public_input','supply','demand_residual','clearing_slope',
            'fresh_factor_loadings','intercept_aggregate','private_signals',
            'head_parameters')}
        for i,s in enumerate(self.states):
            arrays[f'state_{i}']=s
        np.savez_compressed(path, **arrays)


def auction_price(economy: Economy, policies: list[InvestorPolicy],
                  outputs: list[HeadOutput], factors, supply):
    """Exact competitive price: total continuum CARA demand = outstanding supply."""
    num = 0.
    loading = np.zeros(economy.dim)
    denom = 0.
    for policy,h in zip(policies,outputs):
        w = policy.mass/(policy.gamma*h.variance)
        num += w*h.intercept
        loading += w*h.private_coef*policy.signal_loading
        denom += w*(economy.risk_free_gross-h.price_coef)
    if not (np.isfinite(denom) and denom > 1e-10):
        raise ValueError('No unique downward-sloping aggregate demand')
    price = (num+loading@factors-supply)/denom
    demand = sum(policy.mass/(policy.gamma*h.variance)*
        (h.intercept+h.private_coef*(policy.signal_loading@factors)
         -(economy.risk_free_gross-h.price_coef)*price)
        for policy,h in zip(policies,outputs))
    return float(price),float(demand-supply),float(denom),loading,float(num)


def simulate(economy: Economy, policies: list[InvestorPolicy], bank: ShockBank,
             initial_price: float = 0.0) -> SimulationResult:
    if not policies:
        raise ValueError('Need at least one investor type')
    if not np.isclose(sum(c.mass for c in policies),1.0,atol=1e-12):
        raise ValueError('Policy masses must sum to 1')
    if any(len(c.signal_loading)!=economy.dim for c in policies):
        raise ValueError('Private signal loading dimension mismatch')
    if bank.private_stdnormal.shape[1]<len(policies):
        raise ValueError('Not enough private-signal streams')
    cs = copy.deepcopy(policies)
    for c in cs:
        c.encoder.reset()
    n=bank.n; C=len(cs);m=economy.dim
    price=np.empty(n); resid=np.empty(n); denom=np.empty(n)
    loading=np.empty((n,m)); intercept=np.empty(n)
    input_public=np.empty((n,2+economy.n_public))
    private=np.empty((n,C)); params=np.empty((n,C,4))
    states=[[] for _ in range(C)]
    prev_price=float(initial_price)
    for t in range(n):
        # The public input uses PAST price and the payoff paid on a prior claim.
        # Never pass contemporaneous p_t or future D_{t+1} to an encoder.
        u=np.r_[prev_price,bank.lag_dividend[t],bank.public[t]]
        input_public[t]=u
        heads=[]
        for i,c in enumerate(cs):
            z=c.encoder.step(u)
            states[i].append(z)
            h=c.head.evaluate(z)
            heads.append(h)
            params[t,i]=[h.intercept,h.private_coef,h.price_coef,h.variance]
            private[t,i]=c.signal_loading@bank.factor[t]+c.signal_noise_sd*bank.private_stdnormal[t,i]
        supply=economy.supply_mean+bank.supply_shock[t]
        p,clearing_error,delta,vec,a=auction_price(economy,cs,heads,bank.factor[t],supply)
        price[t]=p;resid[t]=clearing_error;denom[t]=delta
        loading[t]=vec;intercept[t]=a
        prev_price=p
    sel=slice(bank.burn,n)
    return SimulationResult(
        factor=bank.factor[sel].copy(), public=bank.public[sel].copy(),
        lag_dividend=bank.lag_dividend[sel].copy(),
        future_dividend=bank.future_dividend[sel].copy(),
        target=bank.factor[sel]@economy.target_loading,price=price[sel],
        public_input=input_public[sel],
        supply=economy.supply_mean+bank.supply_shock[sel],
        demand_residual=resid[sel],clearing_slope=denom[sel],
        fresh_factor_loadings=loading[sel],intercept_aggregate=intercept[sel],
        private_signals=private[sel],head_parameters=params[sel],
        states=[np.asarray(s)[sel] for s in states],
        metadata={'economy':economy.name,'seed':bank.seed,'burn':bank.burn,
                  'n_periods':n-bank.burn,'policy_names':[x.name for x in policies],
                  'timing':'u_t=[p_{t-1},D_t,public_macro_t]; auction p_t; then D_{t+1}'}
    )
