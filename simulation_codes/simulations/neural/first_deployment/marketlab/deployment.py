"""One-shot auctions on inherited Q0: FROZEN public histories and incumbents.

This does NOT simulate the new stationary law; that is Phase 3. Current
price changes endogenously using the learned price-conditioned demand head.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from .learning import make_windows
from .simulation import auction_price
from .policies import HeadOutput
from .evaluation import ols_predict


def raw_heads_on_inherited_history(model,u,chunk=512):
    L=model.window
    n=len(u)
    # First L-1 periods excluded from all evaluations; padded for book-keeping.
    unorm=(np.asarray(u,dtype=np.float32)-model.norm.u_mean)/model.norm.u_sd
    x=make_windows(unorm,L).copy()
    out=model.physical_heads(x,chunk)
    a=np.full((n,4),np.nan)
    a[L-1:,:]=out
    return a


def frozen_price_panel(economy,policies,inherited,new_type_idx,new_heads):
    """Inherited prices/encoder histories held fixed; only one type changed."""
    n=len(inherited.price);C=len(policies)
    heads=inherited.head_parameters.copy()
    assert heads.shape==(n,C,4)
    replace_idx=np.flatnonzero(np.isfinite(new_heads[:,3]))
    assert len(replace_idx)>n//2
    heads[replace_idx,new_type_idx]=new_heads[replace_idx]
    # For initial burn-in rows, retain inherited head; exclude these from metrics.
    f=inherited.factor;d=economy.dim
    ws=np.stack([p.mass/(p.gamma*heads[:,i,3]) for i,p in enumerate(policies)],axis=1)
    intercept=np.einsum('tc,tc->t',ws,heads[:,:,0])
    loader=np.sum(np.stack([ws[:,i,None]*heads[:,i,1,None]*p.signal_loading[None,:]
                            for i,p in enumerate(policies)],axis=1),axis=1)
    delta=np.einsum('tc,tc->t',ws,economy.risk_free_gross-heads[:,:,2])
    if np.min(delta)<1e-8:raise ValueError('Nonpositive auction slope')
    prices=(intercept+np.einsum('td,td->t',loader,f)-inherited.supply)/delta
    clearing=(intercept+np.einsum('td,td->t',loader,f)-delta*prices-inherited.supply)
    if max(abs(clearing))>1e-8:raise RuntimeError('Market clearing failed')
    info={'price':prices,'loading':loader,'delta':delta,'heads':heads,
          'max_clearing_residual':float(max(abs(clearing))),
          'min_delta':float(min(delta)),
          'updated_beta_over_gamma_v':float(np.mean(heads[replace_idx,new_type_idx,1]/
                        (policies[new_type_idx].gamma*heads[replace_idx,new_type_idx,3]))),
          'fresh_precision_E1':float(np.mean((loader[replace_idx]@economy.target_loading)**2)
                                             /economy.supply_noise_sd**2)
            if len(policies)==2 and all(np.linalg.norm(p.signal_loading-economy.target_loading)<1e-8 for p in policies)
            else None}
    return info


def price_decoder_metrics(inherited,price,start=None):
    """New price value beyond inherited U. No nonlinear-MI claim."""
    n=len(price)
    start=max(65,start or 0)
    y=inherited.target[start:]
    X=inherited.public_input[start:]
    p=price[start:]
    k=int(.60*len(y))
    pred_u=ols_predict(y[:k],X[:k],y[k:],X[k:])
    pred_p=ols_predict(y[:k],np.column_stack([X[:k],p[:k]]),
                       y[k:],np.column_stack([X[k:],p[k:]]))
    mse0=float(np.mean((y[k:]-pred_u)**2))
    mse1=float(np.mean((y[k:]-pred_p)**2))
    return {'OOS_public_MSE':mse0,'OOS_public_plus_price_MSE':mse1,
            'OOS_linear_decoder_gain':float(1-mse1/mse0),
            'gaussian_log_MSE_proxy':float(.5*np.log(mse0/mse1)),
            'evaluation_n':int(len(y)-k),
            'not_exact_conditional_MI':True}


class TorchRollingWindowEncoder:
    """Optionally deploy trained policy as a fully endogenous recurrent market."""
    def __init__(self,model):
        self.model=model
        self.window=model.window
        self.reset()
    def __deepcopy__(self,memo):
        # Keep a shared read-only model; independent input-history buffers.
        return TorchRollingWindowEncoder(self.model)
    def reset(self):
        self.buffer=deque(maxlen=self.window)
    def step(self,u):
        self.buffer.append(np.asarray(u,dtype=np.float32).copy())
        block=np.zeros((self.window,len(u)),dtype=np.float32)
        buf=np.asarray(self.buffer,dtype=np.float32)
        block[-len(buf):,:]=buf
        return self.model.infer_from_window(block)


class PrecomputedTorchHead:
    """Encoder returns already computed physical head coefficients."""
    def evaluate(self,coeff):
        return HeadOutput(*[float(x) for x in coeff])
    def __deepcopy__(self,memo):
        return self
