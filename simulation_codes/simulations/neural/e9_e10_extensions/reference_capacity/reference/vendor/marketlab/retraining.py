"""Two-clock microfounded market experiment: fixed-policy stationary law -> all-type re-estimation.

Designed for finite-epoch controlled experiments, NOT a population equilibrium solver.
Each Q is represented by a simulated stationary candidate under FROZEN learned policies.
The shock bank is common across stages/arms to reduce Monte Carlo noise, with a
separate independent evaluation bank. Price histories are endogenous in every Q.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math
import numpy as np
import torch
from .policies import InvestorPolicy
from .simulation import simulate
from .learning import fit_forecaster, make_windows, checkpoint
from .deployment import TorchRollingWindowEncoder,PrecomputedTorchHead,price_decoder_metrics,raw_heads_on_inherited_history,frozen_price_panel
from .fast_inference import FastFiniteContextEncoder


def model_policy(template:InvestorPolicy,model, name=None)->InvestorPolicy:
    """Replace one continuum-type's belief and state encoder, retaining primitives."""
    return InvestorPolicy(name or template.name,template.mass,template.gamma,
        template.signal_loading.copy(),template.signal_noise_sd,
        FastFiniteContextEncoder(model),PrecomputedTorchHead())


def fit_all_types(panel, templates, architecture, hiddens, heads, seed, training_steps=105,
                  batch_size=96,window=32, label=''):
    """Every continuum investor type optimizes a subjective Gaussian predictive NLL.
    Type c MUST use private_signals[:,c]; critical to avoid treating every type as c=0.
    """
    models=[]; details=[]
    for c,template in enumerate(templates):
        fitseed=seed+119*c
        model,d=fit_forecaster(vars(panel) if not isinstance(panel,dict) else panel,
            f'{label}:type{c}',architecture,int(hiddens[c]),heads[c],seed=fitseed,
            type_index=c,window=window,steps=training_steps,batch_size=batch_size,
            patience=3)
        models.append(model);details.append(d)
    return models,details


def stationary_panel(economy, templates, models, bank):
    policies=[model_policy(templates[c],models[c]) for c in range(len(models))]
    out=simulate(economy,policies,bank)
    if not np.all(np.isfinite(out.price)) or not np.all(np.isfinite(out.head_parameters)):
        raise ValueError('Non-finite endogenous market path')
    if np.max(np.abs(out.demand_residual))>1e-7 or np.min(out.clearing_slope)<=0:
        raise ValueError('Competitive clearing check failed')
    return out


def holdout_nll(model,panel,type_index,window=None,batch_size=400):
    """Score trained Gaussian beliefs on independent endogenous market-law sample.
    Always use frozen training normalizations (not the evaluation-panel norms).
    """
    norm=model.norm
    u=np.asarray(panel.public_input,dtype=np.float32)
    window=window or model.window
    U=make_windows((u-norm.u_mean)/norm.u_sd,window).copy()
    t=np.arange(window-1,len(u))
    s=(panel.private_signals[t,type_index]-norm.s_mean)/norm.s_sd
    p=(panel.price[t]-norm.p_mean)/norm.p_sd
    y=(panel.future_dividend[t]-norm.y_mean)/norm.y_sd
    preds=[];logvs=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(t),batch_size):
            sl=slice(i,i+batch_size)
            mu,logv=model(torch.from_numpy(U[sl]),torch.as_tensor(s[sl],dtype=torch.float32),
                     torch.as_tensor(p[sl],dtype=torch.float32))
            preds.append(mu.numpy());logvs.append(logv.numpy())
    mu=np.concatenate(preds); logv=np.concatenate(logvs)
    residual=y-mu
    nll=float(.5*np.mean(np.log(2*np.pi)+logv+residual**2*np.exp(-logv))+np.log(norm.y_sd))
    mse=float(np.mean(residual**2)*norm.y_sd**2)
    return {'NLL':nll,'MSE':mse,'evaluated_n':len(t),
        'variance_mean':float(np.mean(np.exp(logv))*norm.y_sd**2)}


def mean_fresh_precision(economy,templates,panel):
    if not all(np.allclose(p.signal_loading,economy.target_loading) for p in templates):return None
    A=panel.fresh_factor_loadings@economy.target_loading
    return float(np.mean(A**2)/economy.supply_noise_sd**2)


def panel_summary(economy,templates,panel,models=None,eval_law=False):
    scores={'mean_price':float(np.mean(panel.price)),
        'sd_price':float(np.std(panel.price)),
        'mean_risk_inverse':float(np.mean(1/np.maximum(panel.head_parameters[:,:,3],1e-12))),
        'max_clearing_abs':float(np.max(abs(panel.demand_residual))),
        'min_aggregate_demand_slope':float(np.min(panel.clearing_slope)),
        'fresh_precision_E1':mean_fresh_precision(economy,templates,panel),
        'beta_over_gamma_v':[(float(np.mean(panel.head_parameters[:,i,1]/(templates[i].gamma*panel.head_parameters[:,i,3]))))
            for i in range(len(templates))]}
    if eval_law:
        dec=price_decoder_metrics(panel,panel.price,start=65)
        scores['OOS_linear_decoder_gain']=dec['OOS_linear_decoder_gain']
        scores['OOS_public_MSE']=dec['OOS_public_MSE']
        scores['OOS_public_plus_price_MSE']=dec['OOS_public_plus_price_MSE']
    if models is not None:
        scores['agent_OOS']=[holdout_nll(m,panel,c) for c,m in enumerate(models)]
    return scores


def history_drift(ref,new):
    assert len(ref.price)==len(new.price)
    denom=max(float(np.std(ref.price)),1e-9)
    return {'price_RMSE_scaled':float(np.sqrt(np.mean((ref.price-new.price)**2))/denom),
        'price_mean_shift':float(np.mean(new.price-ref.price)),
        'absolute_price_loading_shift':float(np.mean(np.linalg.norm(new.fresh_factor_loadings-ref.fresh_factor_loadings,axis=1)))}
