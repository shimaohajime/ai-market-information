import copy,unittest
import numpy as np
import torch
import engine as e

class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.models,cls.other=e.initial(18013)

    def test_independent_demand_clearing_and_finite_window(self):
        p,meta=e.simulate(self.models,self.other,.5,18013,800,1,paths=2,periods=20,burn=32)
        for i in range(2):
          for t in [15,22,34]:
            total=0.
            for cap,mass in [('low',.275),('high',.275)]:
                a,b,c,v=self.models[cap].infer_from_window(p['public_input'][i,t-15:t+1])
                total+=mass*(a+b*p['theta'][i,t]+(c-e.ECO.risk_free_gross)*p['price'][i,t])/(e.GAMMA[0]*v)
            o=self.other
            total+=.45*(o['intercept']+o['beta']*p['theta'][i,t]+(o['chi']-e.ECO.risk_free_gross)*p['price'][i,t])/(e.GAMMA[1]*o['v'])
            self.assertAlmostEqual(total,p['supply'][i,t],delta=3e-6)

    def test_zero_public_response_recovers_analytic_linear_price(self):
        models=copy.deepcopy(self.models)
        for m in models.values():
            norm=m.norm;b=e.EQ['beta'][0];c=0.;v=e.EQ['v'][0]
            with torch.no_grad():
                m.public_mean.weight.zero_();m.public_mean.bias.fill_((b*norm.s_mean+c*norm.p_mean-norm.y_mean)/norm.y_sd)
                m.raw_beta.fill_(b*norm.s_sd/norm.y_sd);m.raw_chi.fill_(np.arctanh(c/.45));m.log_variance.fill_(np.log(v/norm.y_sd**2))
        other={**self.other,'chi':0.}
        p,_=e.simulate(models,other,.5,18013,801,1,paths=3,periods=30,burn=20)
        delta=e.ECO.risk_free_gross*np.sum(e.MASS/(e.GAMMA*e.EQ['v']))
        expected=(e.EQ['A']*p['theta']-p['supply'])/delta
        np.testing.assert_allclose(p['price'],expected,atol=2e-7,rtol=2e-7)

    def test_future_payoff_and_individual_noise_cannot_change_current_price(self):
        total=16+20+15;shock=e.draw(18013,802,1,2,total);t=24
        p,_=e.simulate(self.models,self.other,.5,18013,802,1,2,20,16,shocks=shock,return_trace=True)
        pert=copy.deepcopy(shock);pert['y'][t]+=10;pert['private']+=100
        q,_=e.simulate(self.models,self.other,.5,18013,802,1,2,20,16,shocks=pert,return_trace=True)
        np.testing.assert_array_equal(p['trace'][:t+1],q['trace'][:t+1])
        self.assertGreater(np.max(abs(p['trace'][t+1:]-q['trace'][t+1:])),1e-6)

    def test_inactive_capacity_does_not_change_prices(self):
        m=copy.deepcopy(self.models)
        with torch.no_grad():m['high'].public_mean.weight.add_(100)
        p,_=e.simulate(self.models,self.other,0.,18013,803,1,2,20,16)
        q,_=e.simulate(m,self.other,0.,18013,803,1,2,20,16)
        np.testing.assert_array_equal(p['price'],q['price'])

    def test_windows_never_cross_paths_and_latents_never_enter_learner(self):
        p,_=e.simulate(self.models,self.other,.5,18013,804,1,3,20,16)
        m=self.models['low'];d=e.normalized(m,p)
        np.testing.assert_allclose(d['windows'][20].numpy(),(p['public_input'][1,:16]-m.norm.u_mean)/m.norm.u_sd,atol=0)
        self.assertEqual(len(d['y']),60)
        z=e.features(m,p);p['theta'][:]=np.nan;p['supply'][:]=np.nan
        np.testing.assert_array_equal(z,e.features(m,p))

    def test_frozen_parameters_and_coupling(self):
        before=e.fingerprint(self.models,self.other)
        a,_=e.simulate(self.models,self.other,.5,18013,805,1,2,20,512,initial_price=-10.)
        b,_=e.simulate(self.models,self.other,.5,18013,805,1,2,20,512,initial_price=10.)
        np.testing.assert_allclose(a['price'],b['price'],atol=1e-5,rtol=0)
        self.assertEqual(before,e.fingerprint(self.models,self.other))

if __name__=='__main__':unittest.main()
