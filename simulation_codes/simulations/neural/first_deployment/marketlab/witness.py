"""Independent covariance-matrix implementation of the market witness.

The old script directly wrote scalar covariances. Here each inherited Gaussian
market law is represented by a measurement matrix and noise covariance. Heads
are population linear projections of payoff onto (private signal, candidate
price, optionally public W), and the fitted variance is the exact conditional
variance. This is an independent formula implementation and allows checking
against legacy report values. Units: contemporaneous 2d IID latent factors.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root
from .economy import Economy
from .policies import InvestorPolicy, SelectPublicEncoder, ZeroEncoder, AffineGaussianHead

WITNESS_PARAMS={'nu1':16.,'nu2':.03,'sig_eps2':.08,'sig_supply2':22.,
                'corr':.93,'lambda_w':-.5,'gamma1':.12,'gamma2':.7,'pi1':.65,'Rf':1.}


def projection_head(profile, params, private_variance: float, see_public_W: bool):
    A,B,delta=np.asarray(profile,float)
    if delta<=0:
        raise ValueError('Auction slope must be positive')
    rho=params['corr'];ell=np.array([1.,params['lambda_w']])
    S=np.array([[1.,rho],[rho,1.]])
    load=[np.array([1.,0.]),np.array([A,B])/delta]
    noise=[private_variance,params['sig_supply2']/delta**2]
    if see_public_W:
        load.append(np.array([0.,1.]));noise.append(0.)
    M=np.stack(load)
    V=M@S@M.T+np.diag(noise)
    cross=M@S@ell
    coef=np.linalg.solve(V,cross)
    variance=float(ell@S@ell+params['sig_eps2']-cross@coef)
    if variance<=0:
        raise ValueError('Bad forecast variance')
    return coef,variance


def trained_bundle(profile,adoption,params=WITNESS_PARAMS):
    pi=params['pi1']
    fits={
      'type1_low':projection_head(profile,params,params['nu1'],False),
      'type1_high':projection_head(profile,params,params['nu1'],True),
      'type2_low':projection_head(profile,params,params['nu2'],False)
    }
    groups=[('type1_low',pi*(1-adoption),params['gamma1']),
            ('type1_high',pi*adoption,params['gamma1']),
            ('type2_low',1-pi,params['gamma2'])]
    agg_A=agg_B=denom=0.
    by_type={}
    for name,mass,gamma in groups:
        coef,v=fits[name]
        bet_s,chi=coef[:2]
        alpha_W=coef[2] if len(coef)==3 else 0.
        w=mass/(gamma*v)
        agg_A+=w*bet_s;agg_B+=w*alpha_W
        denom+=w*(params['Rf']-chi)
        by_type[name]={'beta':float(bet_s),'chi':float(chi),'wcoef':float(alpha_W),
                       'variance':v,'mass':mass,'Acontrib':w*bet_s}
    if denom<=0:raise ValueError('Aggregate demand slope is not positive')
    return np.array([agg_A,agg_B,denom]),by_type


def solve_fixed_point(adoption,params=WITNESS_PARAMS,guess=(2.5,0.,12.)):
    def residual(q):
        try:return trained_bundle(q,adoption,params)[0]-q
        except (ValueError,np.linalg.LinAlgError):return np.ones(3)*1e5
    opt=root(residual,guess,tol=1e-12)
    resid=float(max(abs(residual(opt.x))))
    if resid>1e-9 or opt.x[2]<=0:
        raise RuntimeError(f'Unverified population fixed point, error {resid}')
    h=1e-5
    basis=np.eye(3)*h
    Jac=np.column_stack([(trained_bundle(opt.x+b,adoption,params)[0]-
                         trained_bundle(opt.x-b,adoption,params)[0])/(2*h)
                        for b in basis])
    spectral=float(max(abs(np.linalg.eigvals(Jac))))
    return opt.x,resid,spectral


def theoretical_precision(profile,params=WITNESS_PARAMS):
    """J=A^2/supplyvariance ONLY conditional on publicly observed W."""
    return float(profile[0]**2/params['sig_supply2'])


def witness_economy(params=WITNESS_PARAMS):
    rho=params['corr']
    S=np.array([[1.,rho],[rho,1.]])
    return Economy('2d_iid_witness',np.zeros((2,2)),np.linalg.cholesky(S),
                   payoff_loading=np.array([1.,params['lambda_w']]),
                   target_loading=np.array([1.,0.]),
                   public_loading=np.array([[0.,1.]]),public_noise_sd=np.array([0.]),
                   payoff_noise_sd=np.sqrt(params['sig_eps2']),
                   supply_noise_sd=np.sqrt(params['sig_supply2']),
                   risk_free_gross=params['Rf'])


def deployed_policies(training_profile,adoption,params=WITNESS_PARAMS):
    _,d=trained_bundle(training_profile,adoption,params)
    pi=params['pi1'];theta=np.array([1.,0.]);out=[]
    for name,mass,gamma,private_sd in [
       ('type1_low',pi*(1-adoption),params['gamma1'],np.sqrt(params['nu1'])),
       ('type1_high',pi*adoption,params['gamma1'],np.sqrt(params['nu1'])),
       ('type2_low',1-pi,params['gamma2'],np.sqrt(params['nu2']))]:
        if mass<1e-12:continue
        z_high=(name=='type1_high')
        fit=d[name]
        head=AffineGaussianHead(intercept=0.,private_coef=fit['beta'],
                price_coef=fit['chi'],log_variance=np.log(fit['variance']),
                intercept_weights=np.array([fit['wcoef']]) if z_high else np.zeros(0),
                private_weights=np.zeros(1) if z_high else np.zeros(0),
                price_weights=np.zeros(1) if z_high else np.zeros(0),
                logvar_weights=np.zeros(1) if z_high else np.zeros(0),
                nonlinear_state=False)
        out.append(InvestorPolicy(name,mass,gamma,theta,private_sd,
                 SelectPublicEncoder((2,)) if z_high else ZeroEncoder(),head))
    return out


def witness_summary():
    q0,err0,s0=solve_fixed_point(0.)
    q1,err1,s1=solve_fixed_point(1.,guess=q0)
    qi,d0=trained_bundle(q0,1.)
    qs=(q0,qi,q1)
    return {'q_before':q0.tolist(),'q_impact':qi.tolist(),
            'q_retrained':q1.tolist(),
            'J':[theoretical_precision(q) for q in qs],
            'pct_impact':100*(theoretical_precision(qi)/theoretical_precision(q0)-1),
            'pct_retrained':100*(theoretical_precision(q1)/theoretical_precision(q0)-1),
            'fixed_point_errors':[err0,err1],
            'retraining_jacobian_radius':[s0,s1],
            'A_contributions_impact':{k:v['Acontrib'] for k,v in d0.items()}}
