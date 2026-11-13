"""Phase 2 unit & integration checks, including microfounded deployment."""
import numpy as np
import torch
from marketlab.scenarios import economy_8d,diagnostic_policies
from marketlab.simulation import simulate
from marketlab.learning import (Normalization,ResourceForecaster,fit_forecaster,
                                make_windows,checkpoint,load_checkpoint)
from marketlab.deployment import (raw_heads_on_inherited_history,frozen_price_panel,
                                  TorchRollingWindowEncoder,PrecomputedTorchHead)
from marketlab.policies import InvestorPolicy


def test_ssm_is_linear_and_nested_by_zero_padding():
    norm=Normalization(np.zeros(4),np.ones(4),0.,1.,0.,1.,0.,1.)
    small=ResourceForecaster(4,3,'ssm','A',norm,window=10)
    big=ResourceForecaster(4,7,'ssm','A',norm,window=10)
    with torch.no_grad():
        big.raw_decay[:3]=small.raw_decay
        big.input_projection.zero_()
        big.input_projection[:3]=small.input_projection
        big.public_mean.weight.zero_()
        big.public_mean.weight[0,:3]=small.public_mean.weight
        big.public_mean.bias.copy_(small.public_mean.bias)
        big.raw_beta.copy_(small.raw_beta)
        big.raw_chi.copy_(small.raw_chi)
        big.log_variance.copy_(small.log_variance)
    x=torch.randn(7,10,4);s=torch.randn(7);p=torch.randn(7)
    assert torch.allclose(small(x,s,p)[0],big(x,s,p)[0],atol=1e-6)


def test_head_coefficients_reconstruct_training_mean_and_variance():
    n=Normalization(np.array([.3,-.1]),np.array([2.,3.]),1.,2.,-.4,1.5,.7,.8)
    for architecture in ['ssm','gru']:
        for head in ['A','B']:
            m=ResourceForecaster(2,4,architecture,head,n,window=8)
            u=torch.randn(12,8,2);s=torch.randn(12);p=torch.randn(12)
            y,lv=m(u,s,p)
            z=m.encode(u);h=m.heads_from_z(z)
            s_real=s*n.s_sd+n.s_mean;p_real=p*n.p_sd+n.p_mean
            y_real=h[:,0]+h[:,1]*s_real+h[:,2]*p_real
            assert torch.allclose(y_real,y*n.y_sd+n.y_mean,atol=1e-5)
            assert torch.allclose(h[:,3],torch.exp(lv)*n.y_sd**2,atol=1e-5)
            assert torch.all(h[:,2]<.45)


def test_training_and_frozen_auction(tmp_path):
    e=economy_8d('E1_aligned')
    p=diagnostic_policies(e,'E1_aligned')
    b=e.draw_bank(310,burn=80,seed=202)
    r=simulate(e,p,b)
    panel={'public_input':r.public_input,'price':r.price,'future_dividend':r.future_dividend,
           'private_signals':r.private_signals}
    m,detail=fit_forecaster(panel,'E1_aligned','ssm',4,'B',seed=21,window=16,
                            steps=60,batch_size=24)
    assert np.isfinite(detail['scores']['test']['NLL'])
    assert detail['scores']['test']['variance_mean']>0
    file=tmp_path/'m.pt';checkpoint(m,detail,file)
    m2,_=load_checkpoint(file)
    hh=raw_heads_on_inherited_history(m2,r.public_input)
    deployed=frozen_price_panel(e,p,r,0,hh)
    assert deployed['min_delta']>0
    assert deployed['max_clearing_residual']<1e-8
    assert np.isfinite(deployed['fresh_precision_E1'])
    # Dynamic stepwise auction with trained head also clears by the same engine.
    p[0].encoder=TorchRollingWindowEncoder(m2)
    p[0].head=PrecomputedTorchHead()
    dynamic=simulate(e,p,b)
    assert np.max(np.abs(dynamic.demand_residual))<1e-8
    assert np.min(dynamic.clearing_slope)>0


def test_future_dividend_cannot_change_frozen_price_or_inputs():
    e=economy_8d('E1_aligned')
    p=diagnostic_policies(e,'E1_aligned')
    b=e.draw_bank(80,burn=80,seed=4)
    r=simulate(e,p,b)
    # One-shot frozen-law auctions depend only on f, preauction history, supply and heads.
    from marketlab.deployment import frozen_price_panel
    heads=r.head_parameters[:,0,:].copy()
    old=frozen_price_panel(e,p,r,0,heads)
    assert np.allclose(old['price'],r.price,atol=1e-12)
    # Altering only future dividends cannot affect same-period frozen clearing.
    from dataclasses import replace
    rr=replace(r,future_dividend=r.future_dividend+999.)
    new=frozen_price_panel(e,p,rr,0,heads)
    assert np.allclose(old['price'],new['price'],atol=1e-12)


def test_gru_hypothesis_class_nested_by_padding():
    """GRU(4) can be embedded exactly inside GRU(9), even though SGD need not find it."""
    norm=Normalization(np.zeros(3),np.ones(3),0.,1.,0.,1.,0.,1.)
    small=ResourceForecaster(3,4,'gru','A',norm,window=12)
    large=ResourceForecaster(3,9,'gru','A',norm,window=12)
    with torch.no_grad():
        large.gru.weight_ih_l0.zero_();large.gru.weight_hh_l0.zero_()
        large.gru.bias_ih_l0.zero_();large.gru.bias_hh_l0.zero_()
        for gate in range(3):
            i0=gate*4;i1=gate*9
            large.gru.weight_ih_l0[i1:i1+4,:]=small.gru.weight_ih_l0[i0:i0+4,:]
            large.gru.weight_hh_l0[i1:i1+4,:4]=small.gru.weight_hh_l0[i0:i0+4,:]
            large.gru.bias_ih_l0[i1:i1+4]=small.gru.bias_ih_l0[i0:i0+4]
            large.gru.bias_hh_l0[i1:i1+4]=small.gru.bias_hh_l0[i0:i0+4]
        large.public_mean.weight.zero_()
        large.public_mean.weight[0,:4]=small.public_mean.weight
        large.public_mean.bias.copy_(small.public_mean.bias)
        large.raw_beta.copy_(small.raw_beta)
        large.raw_chi.copy_(small.raw_chi)
        large.log_variance.copy_(small.log_variance)
    u=torch.randn(7,12,3);s=torch.randn(7);p=torch.randn(7)
    mu1,lv1=small(u,s,p);mu2,lv2=large(u,s,p)
    assert torch.allclose(mu1,mu2,atol=1e-6)
    assert torch.allclose(lv1,lv2,atol=1e-6)


def test_rolling_window_encoder_matches_frozen_batch_inference():
    norm=Normalization(np.array([0.,.2]),np.array([1.,.8]),0.,1.,0.,1.,0.,1.)
    for arch in ['ssm','gru']:
        model=ResourceForecaster(2,4,arch,'B',norm,window=12)
        U=np.random.default_rng(2026).normal(size=(70,2)).astype('float32')
        enc=TorchRollingWindowEncoder(model)
        online=np.vstack([enc.step(u) for u in U])[11:]
        batch=raw_heads_on_inherited_history(model,U)[11:]
        assert np.allclose(online,batch,atol=1e-6)
