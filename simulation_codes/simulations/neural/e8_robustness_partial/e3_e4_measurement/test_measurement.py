import unittest
import numpy as np
from scipy.special import logsumexp
from run_measurement import integrated_logweight,ce_stats,fit
import engine as e

class MeasurementTests(unittest.TestCase):
 def test_gaussian_benchmark(self):
  # Correct posterior under jointly Gaussian (theta,z,p), no nonlinear market.
  rng=np.random.default_rng(673);n=1500000;g=.9
  theta=rng.normal(size=n);z=theta+rng.normal(size=n);p=.3*theta+.8*rng.normal(size=n)
  X=np.column_stack((z,p));cov=np.array([[2.,.3],[.3,.73]])
  b=np.linalg.solve(cov,e.LAM*np.array([1.,.3]));mu=X@b;v=e.LAM**2+e.SIG2-b@cov@b
  d=b-np.array([0.,e.ECO.risk_free_gross]);q=d@cov@d
  exact=np.log1p(q/v)/(2*g);actual=ce_stats(integrated_logweight(mu,v,p,theta,g),g)['CE']
  self.assertAlmostEqual(actual,exact,delta=.004)
 def test_integration_by_gauss_hermite(self):
  from numpy.polynomial.hermite import hermgauss
  a,w=hermgauss(80);mu=np.array([.1,-.3,1.]);v=.7;p=np.array([.2,.5,-.1]);theta=np.array([.4,-.2,.9]);g=.9
  h=(mu-e.ECO.risk_free_gross*p)/(g*v)
  direct=logsumexp(np.log(w/np.sqrt(np.pi))[:,None]-g*h*(e.LAM*theta+np.sqrt(2*e.SIG2)*a[:,None]-e.ECO.risk_free_gross*p),axis=0)
  np.testing.assert_allclose(direct,integrated_logweight(mu,v,p,theta,g),atol=1e-12)
 def test_zero_position(self):
  p=np.array([-.1,3.]);self.assertEqual(ce_stats(integrated_logweight(e.ECO.risk_free_gross*p,.5,p,np.ones(2),.9),.9)['CE'],0.)
 def test_log_stability(self):
  st=ce_stats(np.array([1000.,999.]),1.);self.assertTrue(np.isfinite(st['CE']));self.assertGreater(st['ESS'],1)
 def test_e3_identity_and_nested_fit(self):
  rng=np.random.default_rng(55);z=rng.normal(size=(1000,4));extra=rng.normal(size=(1000,3));y=z[:,0]+extra[:,0]+rng.normal(size=1000)
  a=fit(np.c_[np.ones(1000),z],y);b=fit(np.c_[np.ones(1000),z,extra],y);self.assertLess(b['v'],a['v'])
  L0,Lp,H0,Hp=.9,.6,.5,.4;self.assertAlmostEqual((L0-Lp)-(H0-Hp),(L0-H0)-(Lp-Hp))
if __name__=='__main__':unittest.main()
