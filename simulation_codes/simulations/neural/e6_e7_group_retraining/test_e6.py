import copy,unittest
import numpy as np
import torch
import engine as e
import run_e6 as r

class FactorialTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.models,cls.other=r.load_baseline(18013)
  cls.panel,_=e.simulate(cls.models,cls.other,.5,18013,98000,3,paths=3,periods=24,burn=32)
 def test_identical_profiles_have_zero_decomposition(self):
  s,p=r.e7(self.models,self.other,self.models,self.other,.5,self.panel)
  self.assertLess(p[['total','mean_contribution','variance_contribution','interaction']].abs().to_numpy().max(),1e-12)
 def test_variance_only_is_assigned_to_variance(self):
  new=copy.deepcopy(self.models)
  with torch.no_grad():
   for m in new.values():m.log_variance.add_(np.log(.8))
  s,p=r.e7(self.models,self.other,new,self.other,.5,self.panel)
  q=p.set_index('metric');self.assertLess(abs(q.loc['J','mean_contribution']),1e-12)
  self.assertGreater(q.loc['J','variance_contribution'],0.)
  for c in ['low','high']:
   v=q.loc[c+'_iota'];self.assertAlmostEqual(v['new']/v['old'],1.25,places=6)
 def test_group2_only_has_zero_group1_structural_change(self):
  other=self.other.copy();other['beta']*=.9;other['v']*=1.1
  s,p=r.e7(self.models,self.other,self.models,other,.5,self.panel)
  q=p.set_index('metric');self.assertEqual(q.loc['A_group1','total'],0.)
  self.assertLess(q.loc['A_group2','total'],0.)
 def test_shapley_and_factorial_identity(self):
  b0,v0,b1,v1,gamma=.6,.3,.4,.25,1.2
  old=b0/(gamma*v0);mean=b1/(gamma*v0);risk=b0/(gamma*v1);new=b1/(gamma*v1)
  m=.5*((mean-old)+(new-risk));v=.5*((risk-old)+(new-mean))
  self.assertAlmostEqual(m+v,new-old)
  # Explicit 2x2 factorial interaction; no additive response is presumed.
  y00,y10,y01,y11=3.4,3.3,3.1,2.9
  interaction=y11-y10-y01+y00
  self.assertAlmostEqual((y10-y00)+(y01-y00)+interaction,y11-y00)

if __name__=='__main__':unittest.main()
