#!/usr/bin/env python3
"""Independent law-seed and numerical fixed-policy stability checks.
This is a descriptive sensitivity check, NOT a proof of training convergence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from marketlab.scenarios import economy_8d,diagnostic_policies
from marketlab.simulation import simulate
from marketlab.learning import load_checkpoint
from marketlab.deployment import (raw_heads_on_inherited_history,frozen_price_panel,
                                  TorchRollingWindowEncoder,PrecomputedTorchHead)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SEEDS=[11,23,37]


def block_mean_ci(x,block=64,reps=500,seed=732):
    rng=np.random.default_rng(seed)
    x=np.asarray(x,float)
    n=len(x)
    k=int(np.ceil(n/block))
    starts=rng.integers(0,n-block+1,size=(reps,k))
    sampled=np.concatenate([starts[:,:,None]+np.arange(block)[None,None,:]],axis=2)[:,:n]
    means=np.mean(x[sampled],axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]


def main():
    results={'protocol':'Phase2-v1-validation','secondary_law_seed':20261120,
             'moving_block_size':64,'bootstrap_reps':500}
    for scen in ('E1_aligned','E2_nonaligned','E3_state_risk'):
        e=economy_8d(scen)
        p=diagnostic_policies(e,scen)
        bank=e.draw_bank(6000,burn=450,seed=20261120)
        inherited=simulate(e,p,bank)
        if scen=='E1_aligned':
            paired=[]
            for arch in ('ssm','gru'):
                for seed in SEEDS:
                    vals=[]
                    for h in (4,12):
                        m,_=load_checkpoint(ROOT/'models'/f'{scen}_{arch}{h}_headA_seed{seed}.pt')
                        hh=raw_heads_on_inherited_history(m,inherited.public_input)
                        vals.append(frozen_price_panel(e,p,inherited,0,hh))
                    start=48
                    small=(vals[0]['loading'][start:]@e.target_loading)**2/e.supply_noise_sd**2
                    large=(vals[1]['loading'][start:]@e.target_loading)**2/e.supply_noise_sd**2
                    diff=large-small
                    paired.append({'architecture':arch,'seed':seed,
                                   'mean_low':float(small.mean()),'mean_high':float(large.mean()),
                                   'higher_minus_lower_precision':float(diff.mean()),
                                   'fixed_models_block_bootstrap_95CI':block_mean_ci(diff)})
            results['E1_paired_fresh_precision']=paired
        # We deploy selected representative learned models in fully recursive markets.
        # Still no retraining: only the market price path becomes endogenous.
        m,_=load_checkpoint(ROOT/'models'/f'{scen}_{"gru" if scen=="E3_state_risk" else "ssm"}12_head{"B" if scen=="E3_state_risk" else "A"}_seed11.pt')
        p[0].encoder=TorchRollingWindowEncoder(m)
        p[0].head=PrecomputedTorchHead()
        shortbank=e.draw_bank(650,burn=450,seed=20261121)
        a=simulate(e,p,shortbank,initial_price=0.)
        b=simulate(e,p,shortbank,initial_price=10.)
        results[scen+'_dynamic_fixed_policy_check']={
            'max_demand_residual':float(np.max(abs(a.demand_residual))),
            'min_clearing_slope':float(np.min(a.clearing_slope)),
            'tail_coupling_max_difference':float(np.max(abs(a.price[-100:]-b.price[-100:]))),
            'head_variance_mean':float(a.head_parameters[:,0,3].mean()),
            'head_variance_sd':float(a.head_parameters[:,0,3].std()),
            'n_after_burn':650,
            'not_a_stationarity_proof':True}
        print(scen,'dynamic fixed policy clearing',results[scen+'_dynamic_fixed_policy_check'],flush=True)
    (OUT/'phase2_validation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print('Phase 2 secondary validation complete')

if __name__=='__main__':main()
