from dataclasses import replace
from scipy.optimize import root
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from benchmark import Benchmark
from sensitivity import analytic_map_jac
from analysis import steady,map_adoption


def test_8d_primitives_match_phase1():
    b=Benchmark(K=3)
    assert b.m==8 and b.npub==2 and b.zdim==14 and b.nstate==20
    assert b.eco.payoff_noise_sd==.43 and b.eco.supply_noise_sd==.9
    assert abs(b.eco.risk_free_gross-1.02)<1e-12
    assert len(b.temps)==2

def test_gaussian_population_ols_normal_equations():
    b=Benchmark();q=b.seed()
    law=b.law(q)
    for k in [0,1,2,3]:
        fit=b.fit_one(law,0,k)
        n=b.npub+b.nobs*k
        reg=np.vstack([law['Z'][:n],law['theta'],law['p']]);
        sigma=reg@law['cov']@reg.T
        sigma[n,n]+=b.temps[0].signal_noise_sd**2
        betahat=np.r_[fit['b'][:n],fit['beta'],fit['chi']]
        normal=sigma@betahat-reg@law['cov']@law['d']
        assert max(abs(normal))<1e-11
        assert fit['v']>0

def test_population_upgrade_improves_inherited_forecast():
    b=Benchmark();q,_=b.equilibrium((0,0))
    low=b.fit_one(b.law(q),0,0)
    high=b.fit_one(b.law(q),0,3)
    assert high['NLL']<low['NLL']
    assert high['v']<low['v']

def test_aligned_first_deployment_and_stationary_crowdout():
    b=Benchmark();low,slo=b.equilibrium((0,0))
    impact=b.update(low,(3,0))
    new,snew=b.equilibrium((3,0),impact)
    assert b.J(impact)<b.J(low) and b.J(new)<b.J(impact)
    assert abs(b.J(low)-2.46950662726)<1e-8
    assert abs(b.J(impact)-2.2625293312)<1e-8
    assert abs(b.J(new)-2.0735141012)<1e-8
    assert snew['rho_jac']<1 and slo['rho_jac']<1
    assert max(slo['fast_rho'],snew['fast_rho'])<1
    assert snew['residual']<1e-9

def test_analytic_jacobian_matches_finite_differences():
    b=Benchmark();q,s=b.equilibrium((3,0))
    J1=analytic_map_jac(b,q,(3,0))
    J2=b.local_jac(q,(3,0))
    assert np.max(abs(J1-J2))<1e-7

def test_impact_equilibrium_multiplier():
    b=Benchmark();q,traj=steady(b,.5,lo=0,hi=3)
    J=analytic_map_jac(b,q,adoption=.5,lo=0,hi=3)
    g=map_adoption(b,q,1,0,3)-map_adoption(b,q,0,0,3)
    der=np.linalg.solve(np.eye(len(q))-J,g)
    scale=2*q[0]/b.sigma_supply**2
    assert scale*g[0]<0 and scale*der[0]<scale*g[0]
    assert abs(scale*g[0]+.1279203176)<1e-7
    assert abs(scale*der[0]+.3666777592)<1e-7

def test_population_two_cycle_and_information_dominance():
    b=Benchmark();b.eco=replace(b.eco,supply_noise_sd=.715);b.sigma_supply=.715
    low,_=b.equilibrium((0,0))
    q=b.update(low,(3,0))
    for i in range(1500):q=b.update(q,(3,0))
    sol=root(lambda z:z-b.update(b.update(z,(3,0)),(3,0)),q,tol=1e-11)
    qa=sol.x;qb=b.update(qa,(3,0))
    assert max(abs(qa-b.update(qb,(3,0))))<1e-9
    assert max(abs(qa-qb))>.1
    jac2=analytic_map_jac(b,qb,(3,0))@analytic_map_jac(b,qa,(3,0))
    assert max(abs(np.linalg.eigvals(jac2)))<1
    assert b.J(qa)<b.J(low) and b.J(qb)<b.J(low)
    assert abs(b.J(qa)-3.1452120789)<1e-7 or abs(b.J(qb)-3.1452120789)<1e-7
    assert max(b.law(qa)['fast_radius'],b.law(qb)['fast_radius'])<1

def test_bounded_price_head_population_cycles():
    b=Benchmark(chi_bound=.45);b.eco=replace(b.eco,supply_noise_sd=.62);b.sigma_supply=.62
    low,slo=b.equilibrium((0,0))
    q=b.update(low,(3,0))
    for i in range(1800):q=b.update(q,(3,0))
    sol=root(lambda z:z-b.update(b.update(z,(3,0)),(3,0)),q,tol=1e-11)
    qa=sol.x;qb=b.update(qa,(3,0))
    assert max(abs(qa-b.update(qb,(3,0))))<1e-8
    assert abs(b.J(qa)-b.J(qb))>.35
    assert b.J(qa)<b.J(low) and b.J(qb)<b.J(low)
    assert max(abs(np.linalg.eigvals(b.local_jac(qb,(3,0))@b.local_jac(qa,(3,0)))))<1

def test_market_price_clearing_exact_by_formula():
    b=Benchmark();q,_=b.equilibrium((3,0));new,detail=b.update(q,(3,0),full=True)
    rng=np.random.default_rng(111)
    f=rng.normal(size=8);z=rng.normal(size=b.zdim);zeta=rng.normal()*b.sigma_supply
    p=(new[0]*(b.eco.target_loading@f)+new[1:-1]@z-zeta)/new[-1]
    demand=0.0
    for pars in detail['preds']:
        avg_s=b.eco.target_loading@f
        x=(pars['b']@z+pars['beta']*avg_s+pars['chi']*p-b.eco.risk_free_gross*p)/(pars['gamma']*pars['v'])
        demand+=pars['pi']*x
    assert abs(demand-zeta)<1e-11
