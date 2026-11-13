import unittest,sys,os,csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'prior'/'src'),str(HERE/'prior'/'vendor')]
from audit_theorem2 import statistics
from benchmark import Benchmark
from run_redqueen import stationary_population,train_cells,map_shares
from sensitivity import analytic_map_jac

class TestTheorem2Audit(unittest.TestCase):
  def test_baseline_moments(self):
    b=Benchmark(K=3);q=stationary_population(b,(0,0))[0];r=statistics(b,q)
    self.assertAlmostEqual(r['P_L'],.422094934686,places=8)
    self.assertAlmostEqual(r['S'],.357054528261,places=8)
    self.assertAlmostEqual(r['F'],.178380,places=5)
    self.assertAlmostEqual(r['V'],.0232569702725,places=8)
  def test_aligned_147_strong_weak(self):
    with open(HERE/'aligned_condition_audit.csv') as f: rows=list(csv.DictReader(f))
    for protocol,strong,weak in [('frozen',76,140),('retrained',108,145)]:
      sub=[r for r in rows if r['protocol']==protocol and r['span']=='p25_to_p75']
      self.assertEqual(len(sub),147)
      self.assertEqual(sum(r['bridge_strong']=='True' for r in sub),strong)
      self.assertEqual(sum(r['bridge_weak']=='True' for r in sub),weak)
      self.assertTrue(all(r['strategic_substitution']=='True' for r in sub))
  def test_local_all_strategic_substitutes(self):
    with open(HERE/'aligned_local_derivative_audit.csv') as f: rows=list(csv.DictReader(f))
    self.assertEqual(len(rows),294)
    self.assertTrue(all(float(r['sharp_margin'])>0 for r in rows))
    self.assertTrue(all(float(r['dCE'])<0 for r in rows))
  def test_candidate_prior_finite_K(self):
    b=Benchmark(K=3);q=stationary_population(b,(.5,0))[0]
    r=statistics(b,q)
    self.assertAlmostEqual(r['F'],.5*np.log1p(r['J']*r['P_H']),places=8)
  def test_jac_analytic(self):
    b=Benchmark(K=3);q=stationary_population(b,(.5,0))[0]
    j=analytic_map_jac(b,q,adoption=.5,lo=0,hi=3)
    j_fd=np.zeros_like(j)
    for col in range(len(q)):
      bump=np.zeros(len(q)); bump[col]=1e-5*max(1.,abs(q[col]))
      plus=map_shares(b,q+bump,(.5,0.))
      minus=map_shares(b,q-bump,(.5,0.))
      j_fd[:,col]=(plus-minus)/(2*bump[col])
    self.assertLess(np.max(abs(j-j_fd)),1e-7)
    self.assertTrue(max(abs(np.linalg.eigvals(j)))<1.)

if __name__=='__main__':unittest.main()
