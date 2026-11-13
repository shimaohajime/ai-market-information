#!/usr/bin/env python3
"""Run Phase 2: fitted SSM/GRU Gaussian heads and one-shot inherited-law auctions.

No dynamic learning feedback here: Phase 3 evaluates stationary retrained markets.
"""
from __future__ import annotations
import argparse
import json
import platform
import csv
from pathlib import Path
import numpy as np
import scipy
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from marketlab.scenarios import economy_8d,diagnostic_policies
from marketlab.simulation import simulate,auction_price
from marketlab.learning import fit_forecaster,checkpoint,load_checkpoint
from marketlab.deployment import raw_heads_on_inherited_history,frozen_price_panel,price_decoder_metrics
from marketlab.policies import HeadOutput

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
MODELS=ROOT/'models'
MODELS.mkdir(parents=True,exist_ok=True)


def round_floats(v):
    if isinstance(v,(np.floating,np.integer)):return v.item()
    if isinstance(v,dict):return {k:round_floats(x) for k,x in v.items()}
    if isinstance(v,list):return [round_floats(x) for x in v]
    return v


def score_baseline(economy,policies,baseline):
    if len(policies)==2 and all(np.allclose(p.signal_loading,economy.target_loading) for p in policies):
        precision=float(np.mean((baseline.fresh_factor_loadings@economy.target_loading)**2)/economy.supply_noise_sd**2)
    else:precision=None
    return {'decoder':price_decoder_metrics(baseline,baseline.price),
            'fresh_precision_E1':precision,
            'max_abs_clearing_error':float(max(abs(baseline.demand_residual)))}


def true_price_microfoundation_check(e,p,r,h,idx):
    """Directly recompute price using the Phase-1 demand-schedule clearing routine."""
    for t in idx:
        outputs=[HeadOutput(*[float(v) for v in h['heads'][t,i]]) for i in range(len(p))]
        price,res,delta,loading,_=auction_price(e,p,outputs,r.factor[t],r.supply[t])
        assert abs(price-h['price'][t])<1e-9
        assert abs(res)<1e-8
        assert abs(delta-h['delta'][t])<1e-9
        assert np.linalg.norm(loading-h['loading'][t])<1e-8


def config_grid(quick=False):
    seeds=[11] if quick else [11,23,37]
    configs=[]
    for scenario in ('E1_aligned','E2_nonaligned'):
        for architecture in ('ssm','gru'):
            for hidden in (4,12):
                for seed in seeds:
                    configs.append((scenario,architecture,hidden,'A',seed))
    # B-head tests are supplementary: allow both state-dependent variance
    # and state-dependent private-signal/price coefficients.
    for scenario in ('E2_nonaligned','E3_state_risk'):
        for architecture in ('ssm','gru'):
            for head in (('B',) if scenario=='E2_nonaligned' else ('A','B')):
                for seed in ([11] if quick else [11,23]):
                    configs.append((scenario,architecture,12,head,seed))
    return configs


def make_figures(rows,base):
    a=[r for r in rows if r['head_type']=='A' and r['scenario'] in ('E1_aligned','E2_nonaligned')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.0),sharey=False)
    for ax,sc in zip(axes,('E1_aligned','E2_nonaligned')):
        for arch in ('ssm','gru'):
            for h in (4,12):
                ys=[r['heldout']['NLL'] for r in a if r['scenario']==sc and r['architecture']==arch and r['hidden']==h]
                x={('ssm',4):0,('ssm',12):1,('gru',4):3,('gru',12):4}[(arch,h)]
                if ys:
                    ax.scatter([x]*len(ys),ys,alpha=.4,s=24)
                    ax.scatter([x],[np.mean(ys)],marker='_',s=220,linewidth=3)
        ax.set_xticks([0,1,3,4],['SSM4','SSM12','GRU4','GRU12'])
        ax.set_ylabel('Held-out Gaussian NLL (lower is better)')
        ax.set_title(sc)
    fig.suptitle('Capacity and forecasting performance on inherited market data')
    fig.tight_layout();fig.savefig(OUT/'phase2_forecasting_NLL.png',dpi=165);plt.close(fig)

    b=[r for r in a if r['scenario']=='E1_aligned' and r['impact']['fresh_precision_E1'] is not None]
    fig,ax=plt.subplots(figsize=(7.7,4.3))
    # The inherited policy is not a fitted low-capacity comparator, so do not
    # shrink the axis or imply it is the theoretical low-capacity baseline.
    labels=['SSM4','SSM12','GRU4','GRU12']
    for i,(arch,h) in enumerate((('ssm',4),('ssm',12),('gru',4),('gru',12))):
        ys=[r['impact']['fresh_precision_E1'] for r in b if (r['architecture'],r['hidden'])==(arch,h)]
        if ys:
            ax.scatter([i]*len(ys),ys,alpha=.5,s=30)
            ax.scatter([i],[np.mean(ys)],marker='_',s=250,linewidth=3)
    ax.set_xticks(range(4),labels)
    ax.set_ylabel('Conditional fresh-price likelihood precision $E[A_t^2]/\\sigma_\\zeta^2$')
    ax.set_title('E1: one-shot counterfactual auctions, fixed inherited histories')
    ax.set_ylim(.235,.30)
    fig.tight_layout();fig.savefig(OUT/'phase2_E1_fresh_precision.png',dpi=165);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.2))
    for arch,marker in [('ssm','o'),('gru','s')]:
        subset=[r for r in a if r['architecture']==arch]
        for h in (4,12):
            v=[r for r in subset if r['hidden']==h]
            for sc in ('E1_aligned','E2_nonaligned'):
                g=[r for r in v if r['scenario']==sc]
                if g:
                    ax.scatter([r['heldout']['NLL'] for r in g],
                               [100*r['impact']['decoder']['OOS_linear_decoder_gain'] for r in g],
                               marker=marker,s=28+2*h,alpha=.65,label=f'{sc} {arch}{h}')
    ax.set_xlabel('Held-out dividend predictive NLL (lower = better)')
    ax.set_ylabel('Incremental held-out LINEAR price decoder gain (%)')
    ax.set_title('Predictive accuracy vs. public price information (non-MI proxy)')
    ax.legend(fontsize=6,loc='best',ncol=2)
    fig.tight_layout();fig.savefig(OUT/'phase2_prediction_vs_price.png',dpi=165);plt.close(fig)

    # E3 heteroskedastic pilot: A vs B within architecture.
    c=[r for r in rows if r['scenario']=='E3_state_risk']
    if c:
        fig,ax=plt.subplots(figsize=(6.7,4.0))
        for i,(arch,head) in enumerate((('ssm','A'),('ssm','B'),('gru','A'),('gru','B'))):
            y=[r['heldout']['NLL'] for r in c if r['architecture']==arch and r['head_type']==head]
            if y:ax.scatter([i]*len(y),y,s=40,alpha=.7);ax.scatter(i,np.mean(y),s=220,marker='_',linewidth=3)
        ax.set_xticks(range(4),['SSM-A','SSM-B','GRU-A','GRU-B'])
        ax.set_ylabel('Held-out NLL (lower = better)')
        ax.set_title('E3: state-dependent mean/variance pilot')
        fig.tight_layout();fig.savefig(OUT/'phase2_E3_head_comparison.png',dpi=165);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true',help='Small smoke pilot (1 seed, 150 optimization steps)')
    ap.add_argument('--no-retrain',action='store_true',help='Reuse existing checkpoints')
    opts=ap.parse_args()
    configs=config_grid(opts.quick)
    baselines={}; economies={}; policies={};bdata={}
    for scen in {c[0] for c in configs}:
        e=economy_8d(scen)
        pol=diagnostic_policies(e,scen)
        bank=e.draw_bank(periods=6000 if not opts.quick else 1500,burn=450,seed=20261119)
        r=simulate(e,pol,bank)
        economies[scen]=e;policies[scen]=pol;baselines[scen]=r
        bdata[scen]=score_baseline(e,pol,r)
        print('FROZEN LAW',scen,'baseline-price MSE gain',round(bdata[scen]['decoder']['OOS_linear_decoder_gain'],5),flush=True)
    rows=[]
    for counter,(scen,arch,hidden,head,seed) in enumerate(configs,1):
        basename=f'{scen}_{arch}{hidden}_head{head}_seed{seed}'
        ckpt=MODELS/f'{basename}.pt'
        panel=np.load(OUT/f'panel_{scen}_8d.npz')
        if opts.no_retrain and ckpt.exists():
            model,fit=load_checkpoint(ckpt)
        else:
            model,fit=fit_forecaster(panel,scen,arch,hidden,head,seed,
                                 window=48,steps=(150 if opts.quick else 420),
                                 batch_size=96,patience=4)
            checkpoint(model,fit,ckpt)
        r=baselines[scen]
        h=raw_heads_on_inherited_history(model,r.public_input)
        deployed=frozen_price_panel(economies[scen],policies[scen],r,0,h)
        true_price_microfoundation_check(economies[scen],policies[scen],r,deployed,
                                        [50,77,len(r.price)//2,len(r.price)-1])
        decoder=price_decoder_metrics(r,deployed['price'])
        impact={k:v for k,v in deployed.items() if k not in ('price','loading','delta','heads')}
        impact['decoder']=decoder
        row={'scenario':scen,'architecture':arch,'hidden':hidden,'head_type':head,'seed':seed,
             'checkpoint':str(ckpt.relative_to(ROOT)),'fit_steps':fit['steps_completed'],
             'heldout':fit['scores']['test'],'training':fit['scores']['train'],
             'resources':fit['resources'],'impact':impact}
        rows.append(row)
        print(f'{counter:2}/{len(configs)} {basename}: NLL={row["heldout"]["NLL"]:.5f} '
              f'gain={100*decoder["OOS_linear_decoder_gain"]:.3f}% '
              f'precision={impact["fresh_precision_E1"]} '
              f'delta={impact["min_delta"]:.3f}',flush=True)
    record={'protocol':'Phase2-v1','comparison':'frozen-inherited-law one-shot counterfactuals, NOT stationary post-deployment',
            'architectures':['diagonal stable linear state-space','finite-context GRU'],
            'candidate_type':0,'training_benchmark':'Phase 1 exogenously specified diagnostic market',
            'dataset':'same market law Q0 for both capacities; chronology-split Q0 training; independent shock bank for auction evaluation',
            'torch_version':torch.__version__,'python_version':platform.python_version(),'scipy_version':scipy.__version__,
            'test_law_seed':20261119,'training_seeds':[11] if opts.quick else [11,23,37],
            'head_types':{'A':'constant private/price coefficients and variance, nonlinear public intercept allowed',
                           'B':'state-dependent private/price coefficients and variance'},
            'limitations':['incumbent Q0 policies and other types are diagnostic, not population-optimal',
                           'independent training runs need not preserve realized monotonicity despite nested function families',
                           'E1 fresh likelihood precision is not mutual information',
                           'E2/E3 decoder measures heldout OOS linear MSE only',
                           'one-shot frozen inherited public histories are not the counterfactual dynamic stationary law',
                           'computational resource accounting is approximate MACs and allocated states, not hardware profiler FLOPs',
                           'no transformer or repeated retraining in Phase 2 pilot'],
            'inherited_baselines':bdata,'models':rows}
    (OUT/'phase2_results.json').write_text(json.dumps(round_floats(record),indent=2),encoding='utf8')
    with (OUT/'phase2_models.csv').open('w',newline='',encoding='utf8') as f:
        fields=['scenario','architecture','hidden','head_type','seed','fit_steps','NLL','MSE',
                'variance_calibration_ratio','updated_private_response','fresh_precision_E1',
                'OOS_linear_decoder_gain','clearing_error','min_delta','params','MACs']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in rows:
            writer.writerow({'scenario':r['scenario'],'architecture':r['architecture'],
              'hidden':r['hidden'],'head_type':r['head_type'],'seed':r['seed'],'fit_steps':r['fit_steps'],
              'NLL':r['heldout']['NLL'],'MSE':r['heldout']['MSE'],
              'variance_calibration_ratio':r['heldout']['variance_calibration_ratio'],
              'updated_private_response':r['impact']['updated_beta_over_gamma_v'],
              'fresh_precision_E1':r['impact']['fresh_precision_E1'],
              'OOS_linear_decoder_gain':r['impact']['decoder']['OOS_linear_decoder_gain'],
              'clearing_error':r['impact']['max_clearing_residual'],
              'min_delta':r['impact']['min_delta'],
              'params':r['resources']['params'],'MACs':r['resources']['forward_macs_approx']})
    make_figures(rows,bdata)
    print('DONE',len(rows),'experiments. Results in',OUT,flush=True)

if __name__=='__main__':main()
