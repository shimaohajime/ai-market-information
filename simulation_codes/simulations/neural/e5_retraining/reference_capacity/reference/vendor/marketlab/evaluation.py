"""Numerical checks: no claims of analytic stationarity or exact MI for non-Gaussian prices."""
from __future__ import annotations
import numpy as np
from .simulation import simulate


def ols_predict(ytr,xtr,yte,xte,ridge=1e-9):
    xtr = np.column_stack([np.ones(len(xtr)),xtr])
    xte = np.column_stack([np.ones(len(xte)),xte])
    reg = ridge*np.eye(xtr.shape[1]);reg[0,0]=0
    coef=np.linalg.solve(xtr.T@xtr+reg,xtr.T@ytr)
    return xte@coef


def heldout_linear_price_value(result, train_fraction=.6):
    """Incremental OOS linear decoder value, not general conditional mutual information.

    Public features are exactly u_t=(past price, paid dividend, macro released
    BEFORE this auction). Price p_t is an additional input only for the augmented
    *observer*, not for the pre-auction encoder.
    """
    y=result.target
    u=result.public_input
    up=np.column_stack([u,result.price])
    k=int(len(y)*train_fraction)
    b=ols_predict(y[:k],u[:k],y[k:],u[k:])
    a=ols_predict(y[:k],up[:k],y[k:],up[k:])
    mse_b=float(np.mean((y[k:]-b)**2))
    mse_a=float(np.mean((y[k:]-a)**2))
    return {'mse_public':mse_b,'mse_public_price':mse_a,
            'reduction_fraction':1-mse_a/mse_b,
            'gaussian_log_mse_proxy':float(.5*np.log(mse_b/mse_a)),
            'sample_train':k,'sample_test':len(y)-k,
            'interpretation':'Out-of-sample linear predictive improvement; not exact MI unless Gaussian assumptions hold.'}


def coupled_history_check(economy,policies,bank,initial_prices=(0.0,10.0)):
    """Pathwise numerical forgetting under identical exogenous shocks; NOT a proof."""
    left=simulate(economy,policies,bank,initial_price=initial_prices[0])
    right=simulate(economy,policies,bank,initial_price=initial_prices[1])
    diff=np.abs(left.price-right.price)
    return {'initial_price_difference':abs(initial_prices[0]-initial_prices[1]),
            'max_abs_price_diff_after_burn':float(np.max(diff)),
            'mean_abs_price_diff_after_burn':float(np.mean(diff)),
            'tail_max_abs_diff':float(np.max(diff[-min(200,len(diff)):])),
            'interpretation':'Coupling/forgetting on a common shock path, not an analytic stability certificate.'}


def simple_stationarity_diagnostics(result):
    """Compare first and second halves; descriptive evidence only."""
    p=result.price;k=len(p)//2
    ma,mb=float(p[:k].mean()),float(p[k:].mean())
    va,vb=float(p[:k].var()),float(p[k:].var())
    def ac1(q):
        return float(np.corrcoef(q[1:],q[:-1])[0,1])
    return {'first_half_mean':ma,'second_half_mean':mb,
            'first_half_variance':va,'second_half_variance':vb,
            'lag1_acf_first':ac1(p[:k]),'lag1_acf_second':ac1(p[k:]),
            'interpretation':'Descriptive half-sample diagnostics, not a stationarity proof.'}


def subjective_forecast_diagnostics(result):
    """Realized NLL and MSE of deployed *subjective* Gaussian beliefs.

    These are not minimized or calibrated for 8d Phase-1 diagnostic policies.
    """
    out={}
    for j,name in enumerate(result.metadata['policy_names']):
        a,b,chi,v=result.head_parameters[:,j,:].T
        mu=a+b*result.private_signals[:,j]+chi*result.price
        residual=result.future_dividend-mu
        out[name]={'payoff_mse':float(np.mean(residual**2)),
                   'gaussian_nll':float(.5*np.mean(np.log(2*np.pi*v)+residual**2/v)),
                   'mean_subjective_variance':float(np.mean(v)),
                   'std_subjective_variance':float(np.std(v))}
    return out
