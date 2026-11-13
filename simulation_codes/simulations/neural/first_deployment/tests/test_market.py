from dataclasses import replace
import numpy as np
import pytest
from marketlab.economy import Economy
from marketlab.scenarios import economy_8d, diagnostic_policies
from marketlab.simulation import simulate, auction_price
from marketlab.policies import HeadOutput
from marketlab.evaluation import coupled_history_check

@pytest.mark.parametrize('scenario',['E1_aligned','E2_nonaligned','E3_state_risk'])
def test_clearing_and_state_dependent_risk(scenario):
    e=economy_8d(scenario)
    b=e.draw_bank(220,burn=100,seed=19)
    r=simulate(e,diagnostic_policies(e,scenario),b)
    assert np.max(abs(r.demand_residual))<1e-10
    assert np.min(r.clearing_slope)>0
    assert np.all(np.isfinite(r.price))
    assert r.factor.shape == (220,8)
    if scenario!='E1_aligned':
        assert np.std(r.head_parameters[:,1,3])>0


def test_current_candidate_price_not_in_encoder_inputs():
    e=economy_8d('E2_nonaligned')
    b=e.draw_bank(20,burn=0,seed=8)
    r=simulate(e,diagnostic_policies(e,'E2_nonaligned'),b)
    assert np.isclose(r.public_input[0,0],0.0)
    assert np.allclose(r.public_input[1:,0],r.price[:-1])
    assert np.allclose(r.public_input[:,1],r.lag_dividend)
    assert np.allclose(r.public_input[:,2:],r.public)


def test_no_lookahead_to_current_payoff():
    e=economy_8d('E2_nonaligned')
    b=e.draw_bank(80,burn=0,seed=15)
    r=simulate(e,diagnostic_policies(e,'E2_nonaligned'),b)
    # A change to D_{18}, paid on a claim sold at t=17, may influence P_18
    # via the lagged paid-dividend input, but cannot change P_17 or earlier.
    modified=b.future_dividend.copy();modified[17]+=20.0
    lag=modified.copy();lag[1:]=modified[:-1];lag[0]=b.lag_dividend[0]
    bb=replace(b,future_dividend=modified,lag_dividend=lag)
    rr=simulate(e,diagnostic_policies(e,'E2_nonaligned'),bb)
    assert np.allclose(r.price[:18],rr.price[:18])
    assert abs(r.price[18]-rr.price[18])>1e-8


def test_coupled_forgetting_and_16d():
    e=economy_8d('E2_nonaligned',dimension=16)
    b=e.draw_bank(300,burn=220,seed=3)
    res=coupled_history_check(e,diagnostic_policies(e,'E2_nonaligned'),b)
    assert res['tail_max_abs_diff']<1e-8


def test_nonpositive_auction_slope_fails():
    e=economy_8d('E1_aligned')
    p=diagnostic_policies(e,'E1_aligned')
    out=[HeadOutput(0.,1.,2.,1.) for _ in p]
    with pytest.raises(ValueError):
        auction_price(e,p,out,np.zeros(e.dim),0.)


def test_seeded_replay_and_variance_head_bounds():
    e=economy_8d('E3_state_risk')
    b=e.draw_bank(90,burn=50,seed=2026)
    ps=diagnostic_policies(e,'E3_state_risk')
    r1=simulate(e,ps,b)
    r2=simulate(e,ps,b)
    assert np.array_equal(r1.price,r2.price)
    assert np.array_equal(r1.head_parameters,r2.head_parameters)
    assert np.min(r1.head_parameters[:,:,3])>0
    assert r1.head_parameters[:,1,3].max()-r1.head_parameters[:,1,3].min()>.005


def test_payoff_alignment_is_structural_not_empirical():
    e1=economy_8d('E1_aligned')
    e2=economy_8d('E2_nonaligned')
    aligned_coefficient=float(e1.payoff_loading@e1.target_loading)
    assert np.allclose(e1.payoff_loading,aligned_coefficient*e1.target_loading)
    # E2's loading is not proportional to the common private target.
    assert np.linalg.norm(e2.payoff_loading-(e2.payoff_loading@e2.target_loading)*e2.target_loading)>0.3
