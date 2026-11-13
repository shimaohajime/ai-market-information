"""Reproducibility and invariant tests for the restricted-head ablation."""
import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import calibrated_economy,ARMS,combine_heads

class Audit(unittest.TestCase):
    def test_calibrations(self):
        for c in (.7,1.,1.3):
            x=calibrated_economy(c)
            self.assertEqual(x.dim,8)
            self.assertTrue(np.all(np.abs(np.linalg.eigvals(x.F))<1))
            self.assertGreater(x.supply_noise_sd,0)

    def test_masked_heads(self):
        low=np.array([[2.,.5,.2,1.],[3.,.2,.1,2.]],float)
        high=np.array([[5.,.7,.3,.7],[8.,.9,.4,1.1]],float)
        m=combine_heads('10',low,high)
        r=combine_heads('01',low,high)
        self.assertTrue(np.array_equal(m[:,:3],high[:,:3]))
        self.assertTrue(np.array_equal(m[:,3],low[:,3]))
        self.assertTrue(np.array_equal(r[:,:3],low[:,:3]))
        self.assertTrue(np.array_equal(r[:,3],high[:,3]))
        self.assertTrue(np.array_equal(combine_heads('00',low,high),low))
        self.assertTrue(np.array_equal(combine_heads('11',low,high),high))

    def test_risk_only_gaussian_identity(self):
        # Under extra signal H∈{-1,+1}, Y=H+eps, eps~N(0,sd²), old mu=0.
        # Correct risk-only forecast: Var(Y|H)+(E[Y|H]-mu_old)^2.
        means=np.array([-1.,1.]);var_new=.25;mu_old=0.
        v_risk=var_new+(means-mu_old)**2
        self.assertTrue(np.allclose(v_risk,1.25))
        self.assertTrue(np.allclose(np.mean(v_risk),1.25))

    def test_output_complete_and_cleared(self):
        p=ROOT/'results'/'independently_optimized_arms.csv'
        if not p.exists():self.skipTest('Results not yet available')
        x=pd.read_csv(p,dtype={'arm':str})
        self.assertEqual(len(x),3*2*3*2*4)
        self.assertEqual(set(x.arm),set(ARMS))
        self.assertLess(x.max_clearing_error.max(),1e-8)
        self.assertGreater(x.min_slope.min(),0)
        self.assertTrue(np.isfinite(x.select_dtypes(include=[np.number]).to_numpy()).all())
        self.assertTrue((x.subjective_variance>0).all())
        self.assertEqual(x.groupby(['arch','calibration','seed','bank']).arm.nunique().min(),4)

    def test_reoptimized_not_posthoc_swapped(self):
        p=ROOT/'results'/'swap_vs_reoptimized_effects.csv'
        if not p.exists():self.skipTest('Compare not yet available')
        x=pd.read_csv(p,dtype={'arm':str})
        self.assertEqual(len(x),3*2*3*2)
        self.assertGreater(abs(x.opt_risk_pp-x.swap_risk_pp).max(),1e-4)
        self.assertGreater(abs(x.opt_mean_pp-x.swap_mean_pp).max(),1e-4)

if __name__=='__main__':unittest.main()
