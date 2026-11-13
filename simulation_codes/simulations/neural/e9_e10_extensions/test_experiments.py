import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from common import *
from economies import draw,PSI,S,ell,simulate,MODES

class ExperimentsTests(unittest.TestCase):
 def test_period2_screen_and_convergence(self):
  from analyze import cycle_stats
  alt=np.tile(np.array([1.,1.02]),4)[:7];a=np.c_[alt,np.ones((7,3))];self.assertTrue(cycle_stats(a)['period2_screen'])
  stable=np.ones((7,4));self.assertFalse(cycle_stats(stable)['period2_screen']);self.assertTrue(cycle_stats(stable)['stable_05pct_last3'])
 def test_control_reproduces_original_draws(self):
  a=e.draw(18013,101000,1,8,80);b=draw('control',18013,101000,1,8,80)
  for k in a:np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-14)
 def test_nonaligned_variance_match(self):
  self.assertAlmostEqual(float(PSI@S@PSI),1,places=12);self.assertAlmostEqual(float(PSI@S@ell),0,places=12)
  self.assertAlmostEqual(float(ell@S@ell),float(e.EQ['prior']),places=12)
 def test_common_primitives(self):
  a=draw('control',18013,101000,1,16,100)
  for mode in ['nonaligned','heteroskedastic']:
   b=draw(mode,18013,101000,1,16,100)
   for k in ['theta','public','supply','private']:np.testing.assert_array_equal(a[k],b[k])
 def test_heteroskedastic_bounds(self):
  p=draw('heteroskedastic',18013,101000,1,16,100);v=p['noise_variance']/e.SIG2;self.assertTrue(np.all(v>.4)&np.all(v<1.6))
 def test_state_dependent_noise_integration(self):
  mu=np.array([.1,-.3,1.]);v=.7;p=np.array([.2,.5,-.1]);mean=np.array([.4,-.2,.9]);nv=np.array([.1,.3,.2]);g=e.GAMMA[0];h=(mu-e.ECO.risk_free_gross*p)/(g*v);a,w=hermgauss(80)
  direct=logsumexp(np.log(w/np.sqrt(np.pi))[:,None]-g*h*(mean+np.sqrt(2*nv)*a[:,None]-e.ECO.risk_free_gross*p),axis=0)
  np.testing.assert_allclose(direct,logweight(mu,v,p,mean,nv),atol=1e-12)
 def test_identical_frozen_law_replay(self):
  torch=__import__('torch');torch.set_num_threads(1);m,o=load_profile(ROOT/'baselines'/'seed18013');a,ma=simulate('control',m,o,.5,18013,100000,1,4,16,burn=32);b,mb=simulate('control',m,o,.5,18013,100000,1,4,16,burn=32)
  for k in a:np.testing.assert_array_equal(a[k],b[k])
if __name__=='__main__':unittest.main()
