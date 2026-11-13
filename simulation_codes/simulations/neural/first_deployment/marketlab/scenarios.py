"""Frozen diagnostic scenario registry for Phase 1 (not optimized/trained AI)."""
from __future__ import annotations
import numpy as np
from .economy import Economy
from .policies import InvestorPolicy, ZeroEncoder, RecurrentEncoder, AffineGaussianHead


def economy_8d(name='E1_aligned', dimension=8):
    if dimension not in (8,16):
        raise ValueError('Phase 1 supports the predeclared 8d and 16d environments')
    if name not in ('E1_aligned','E2_nonaligned','E3_state_risk'):
        raise ValueError('Unknown economy scenario')
    rng=np.random.default_rng(20261015+dimension)
    raw=rng.standard_normal((dimension,dimension))
    Q, R=np.linalg.qr(raw)
    Q=Q@np.diag(np.sign(np.diag(R)))
    eig0=[.94,.88,.78,.68,.53,-.43,.32,.15]
    eig=np.array(eig0 + ([-.72,.63,.56,-.38,.29,.22,-.17,.08] if dimension==16 else []))
    F=Q@np.diag(eig)@Q.T
    scales=np.array([.35,.34,.36,.4,.42,.35,.4,.32]+([.3,.32,.35,.38,.31,.33,.34,.29] if dimension==16 else []))
    # G not diagonal in eigenbasis: correlated/shifting factor combinations.
    G=Q@np.diag(scales)
    theta=np.zeros(dimension);theta[0]=1;theta[3]=.25
    theta/=np.linalg.norm(theta)
    macro1=np.zeros(dimension);macro1[1]=1;macro1[4]=.3
    macro2=np.zeros(dimension);macro2[2]=1;macro2[5]=.4
    H=np.stack([macro1,macro2])
    ell=1.15*theta.copy()
    if name in ('E2_nonaligned','E3_state_risk'):
        ell[1]+=-.6;ell[5]+=.42
    vol=None
    if name=='E3_state_risk':
        vol=np.zeros(dimension);vol[2]=.6;vol[6]=-.25
    return Economy(name+f'_{dimension}d',F,G,ell,theta,H,
                   public_noise_sd=np.array([.15,.22]),
                   payoff_noise_sd=.43, supply_noise_sd=.9,
                   risk_free_gross=1.02,volatility_loading=vol)


def diagnostic_policies(economy: Economy, scenario: str):
    """Predetermined Gaussian belief heads; training happens ONLY in Phase 2.

    These parameters are intentionally fixed across E1/E2/E3. They are NOT
    population-optimal estimated beliefs, and should not be used to test the
    information-crowd-out theorem at this stage.
    """
    m=economy.dim;u_dim=2+economy.n_public
    theta=economy.target_loading
    n_types=2 if scenario=='E1_aligned' else 3
    masses=[.55,.45] if n_types==2 else [.4,.4,.2]
    sigmas=[1.0,.3] if n_types==2 else [1.0,.30,.7]
    gammas=[.9,1.2,1.4]
    policies=[]
    for c in range(n_types):
        loading=theta.copy()
        if c==2:
            loading=np.zeros(m); loading[5]=1.
        if c==0:
            encoder=ZeroEncoder()
            head=AffineGaussianHead(intercept=0,private_coef=.38,
                   price_coef=.03,log_variance=np.log(1.3))
        else:
            # Controlled pre-auction recurrence: input [lag price, paid D, macro1,macro2].
            A=np.array([[.26,.04],[.03,.43]])
            B=np.array([[.025,.075,.14,-.06],[-.018,.055,-.08,.11]])
            assert B.shape[1]==u_dim
            encoder=RecurrentEncoder(A,B,nonlinear=True)
            head=AffineGaussianHead(
                intercept=.02 if c==1 else -.02,
                private_coef=.43 if c==1 else .24,
                price_coef=.035,log_variance=np.log(.85 if c==1 else 1.5),
                intercept_weights=np.array([.10,-.06]),
                private_weights=np.array([.055,-.025]),
                price_weights=np.array([.006,-.004]),
                logvar_weights=np.array([.24,-.12])
            )
        policies.append(InvestorPolicy(f'type_{c+1}',masses[c],gammas[c],
                                      loading,sigmas[c],encoder,head))
    return policies
