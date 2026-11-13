from __future__ import annotations
import unittest,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from population_ablation import aggregate,merge_pred,fixedpoint_partial
from bounded_population_check import run as bounded_run
sys.path.insert(0,str(HERE/'vendor'/'population_benchmark'))
from benchmark import Benchmark

class TestPopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=Benchmark(K=3)
        cls.q0,_=cls.b.equilibrium((0,0))
        cls.law=cls.b.law(cls.q0)
        cls.lo=cls.b.fit_one(cls.law,0,0)
        cls.hi=cls.b.fit_one(cls.law,0,3)
        cls.other=cls.b.fit_one(cls.law,1,0)

    def test_true_baseline_is_self_consistent(self):
        self.assertLess(np.max(abs(self.q0-aggregate(self.b,[self.lo,self.other]))),1e-8)

    def test_two_mechanisms_opposite_directions(self):
        lo,hi,o,b=self.lo,self.hi,self.other,self.b
        base=b.J(aggregate(b,[lo,o]));mean=b.J(aggregate(b,[merge_pred(hi,lo),o]))
        risk=b.J(aggregate(b,[merge_pred(lo,hi),o]))
        both=b.J(aggregate(b,[hi,o]))
        self.assertLess(mean,base);self.assertGreater(risk,base);self.assertLess(both,base)

    def test_shapley_adds_exactly(self):
        z=json.loads((HERE/'population_results.json').read_text())
        eff=z['shapley'];self.assertAlmostEqual(eff['mean']+eff['risk'],eff['total'],places=11)

    def test_law_steady_ends_at_economic_market(self):
        p=fixedpoint_partial(self.b,aggregate(self.b,[self.hi,self.other]),self.hi,self.other,3,0,(0,1))
        q_eq,_=self.b.equilibrium((3,0),start=aggregate(self.b,[self.hi,self.other]))
        self.assertLess(np.max(abs(q_eq-np.asarray(p['q']))),1e-7)
        self.assertLess(p['residual'],1e-8)

    def test_heterogeneous_but_equal_signals_still_crowdout(self):
        z=json.loads((HERE/'population_results.json').read_text())
        r=next(t for t in z['signal_noise_heterogeneity_grid'] if t['ratio_sd_type0_to_type1']==1.)
        self.assertLess(r['impact_pct'],0);self.assertLess(r['equilibrium_pct'],0)

class TestNeuralPolicies(unittest.TestCase):
    def test_checkpoints_present(self):
        q=HERE/'vendor'/'phase3_long_validation'/'models'
        self.assertEqual(len(list(q.glob('E3_state_risk*headB_seed11*'))),8)

    def test_factorial_arms_and_clearing(self):
        x=pd.read_csv(HERE/'neural_head_arms.csv',dtype={'arm':str})
        self.assertEqual(len(x),32)
        for (arch,bank),g in x.groupby(['arch','bank']):
            self.assertEqual(set(g.arm),{'00','01','10','11'})
            self.assertLess(float(g.max_clearing_residual.max()),1e-8)
            self.assertLess(float(g.OOS_public_only_MSE.max()-g.OOS_public_only_MSE.min()),1e-8)

    def test_mechanical_shapley_exact(self):
        arms=pd.read_csv(HERE/'neural_head_arms.csv',dtype={'arm':str})
        sums=pd.read_csv(HERE/'neural_factorial_effects.csv')
        for row in sums.itertuples():
            gains=arms[(arms.arch==row.arch)&(arms.bank==row.bank)].set_index('arm').OOS_decoder_gain
            self.assertAlmostEqual(row.mean_effect+row.var_effect,gains['11']-gains['00'],places=10)

    def test_frozen_market_clearing(self):
        f=pd.read_csv(HERE/'frozen_stationary_arms.csv',dtype={'arm':str})
        self.assertEqual(len(f),16)
        self.assertLess(float(f.maximum_clearing_error.max()),1e-8)
        self.assertGreater(float(f.min_demand_slope.min()),0)

    def test_new_joint_forecasts_have_better_inherited_nll(self):
        arms=pd.read_csv(HERE/'neural_head_arms.csv',dtype={'arm':str})
        for arch in ('ssm','gru'):
            means=arms[arms.arch==arch].groupby('arm').inherited_NLL_type0.mean()
            self.assertLess(means['11'],means['00'])

if __name__=='__main__':unittest.main()
