import numpy as np
from marketlab.witness import (witness_summary, witness_economy,deployed_policies,
                               solve_fixed_point, trained_bundle,theoretical_precision)
from marketlab.simulation import simulate


def test_population_fixed_points_and_independent_formula():
    v=witness_summary()
    assert np.allclose(v['J'],[.28908824277344763,.24448551875319033,.35491227958740407],rtol=1e-9,atol=1e-12)
    assert max(v['fixed_point_errors'])<1e-9
    assert max(v['retraining_jacobian_radius'])<1
    assert v['pct_impact']<-15 and v['pct_retrained']>22


def test_witness_policy_simulator_reproduces_exact_clearing():
    e=witness_economy()
    q0,_,_=solve_fixed_point(0.)
    qstar,_,_=solve_fixed_point(1.,guess=q0)
    bank=e.draw_bank(200,burn=70,seed=77)
    for adoption,trained_q in [(0,q0),(1,q0),(1,qstar)]:
        policies=deployed_policies(trained_q,adoption)
        res=simulate(e,policies,bank)
        q,_=trained_bundle(trained_q,adoption)
        A,B,delta=q
        expected=(A*res.factor[:,0]+B*res.factor[:,1]-res.supply)/delta
        assert np.max(abs(res.price-expected))<2e-12
        assert np.max(abs(res.demand_residual))<1e-10
        assert np.max(abs(res.clearing_slope-delta))<2e-12
        assert np.max(abs(res.fresh_factor_loadings[:,0]-A))<1e-11
        assert np.max(abs(res.fresh_factor_loadings[:,1]))<1e-11
        # The B W component is already-public: it belongs to pre-auction
        # aggregate intercept, not to the private-signal loading.
        assert np.max(abs(res.intercept_aggregate-B*res.public[:,0]))<1e-11
