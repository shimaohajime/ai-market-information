"""Summaries and numerical audits for complete slow-fast trajectories."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
import torch,scipy
import engine as e
from run import CFG,name,summarize

def analyze():
    torch.set_num_threads(1);summarize()
    d=pd.read_csv(e.ROOT/'results/model_metrics.csv');m=pd.read_csv(e.ROOT/'results/market_metrics.csv')
    f=pd.read_csv(e.ROOT/'results/refitting_metrics.csv');pairs=[];improvements=[];checks=[]
    assert len(m)==len(CFG['seeds'])*len(CFG['adoption'])*(CFG['slow_updates']+1)
    for (seed,x,n),g in d.groupby(['seed','x','round']):
        g=g.set_index('capacity');lo,hi=g.loc['low'],g.loc['high'];other=g.loc['other']
        mk=m[(m.seed==seed)&(m.x==x)&(m['round']==n)].iloc[0]
        al=.55*(1-x)*lo.iota;ah=.55*x*hi.iota;ao=.45*other.iota
        assert abs((al+ah+ao)**2/.81-mk.J)<1e-11
        pairs.append(dict(seed=seed,x=x,round=n,MSE_gain_pct=100*(1-hi.MSE/lo.MSE),
            iota_high_minus_low_pct=100*(hi.iota/lo.iota-1),private_low_pct=lo.private_gain_pct,
            private_high_pct=hi.private_gain_pct,A_low=al,A_high=ah,A_group1=al+ah,A_other=ao,J=mk.J))
        stage=e.ROOT/'results'/name(seed,x)/f'round{int(n):02d}'
        meta=json.loads((stage/'complete.json').read_text());gen=json.loads((stage/'generation.json').read_text())
        models={cap:e.load_checkpoint(stage/(cap+'_deployed.pt'))[0] for cap in ['low','high']}
        ot=json.loads((stage/'other_deployed.json').read_text());fp=e.fingerprint(models,ot)
        assert meta['profile_sha256']==fp
        for b in gen:assert b['profile_sha256']==fp and b['x']==x
        expected_banks=[1,50,51,60] if n<CFG['slow_updates'] else [51,60]
        assert [b['bank'] for b in gen]==expected_banks
        if n<CFG['slow_updates']:
            assert meta['training_started']>=meta['generation_finished']>meta['generation_started']
            assert meta['generated_training_observations']==1048576
            assert all(v['optimization_windows']==1048576 and v['updates']==4096 for v in meta['updates'].values())
            newer=f[(f.seed==seed)&(f.x==x)&(f['round']==n)].set_index('capacity')
            for cap in ['low','high','other']:
                improvements.append(dict(seed=seed,x=x,round=n,capacity=cap,
                    MSE_before=g.loc[cap,'MSE'],MSE_after=newer.loc[cap,'MSE'],
                    same_law_MSE_improvement_pct=100*(1-newer.loc[cap,'MSE']/g.loc[cap,'MSE'])))
            nx=e.ROOT/'results'/name(seed,x)/f'round{int(n)+1:02d}'
            for cap in ['low','high']:
                a=e.load_checkpoint(stage/(cap+'_next.pt'))[0];b=e.load_checkpoint(nx/(cap+'_deployed.pt'))[0]
                for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],atol=0,rtol=0)
                assert a.norm.to_dict()==b.norm.to_dict()
            assert json.loads((stage/'other_next.json').read_text())==json.loads((nx/'other_deployed.json').read_text())
        blocks=pd.read_csv(stage/'fast_market_blocks.csv')
        z={}
        for metric in ['mean','variance']:
            b=blocks.pivot(index='path',columns='half',values=metric);delta=b[1]-b[0]
            z[metric+'_split_z']=float(delta.mean()/(delta.std(ddof=1)/np.sqrt(len(delta))))
        checks.append(dict(seed=seed,x=x,round=n,clearing_error=max(b['clearing_error'] for b in gen),
            initial_price_coupling_error=meta['mixing']['max_retained_price_difference'],**z))
    pairs=pd.DataFrame(pairs);pairs.to_csv(e.ROOT/'results/paired_metrics.csv',index=False)
    adoption=[]
    for (seed,n),g in pairs.groupby(['seed','round']):
        g=g.set_index('x');baseline=g.loc[0.]
        for x,row in g.iterrows():
            adoption.append(dict(seed=seed,round=n,x=x,J_vs_zero_adoption_pct=100*(row.J/baseline.J-1),
                capacity_MSE_gap_difference_pp=row.MSE_gain_pct-baseline.MSE_gain_pct))
    pd.DataFrame(adoption).to_csv(e.ROOT/'results/adoption_comparisons.csv',index=False)
    pd.DataFrame(improvements).to_csv(e.ROOT/'results/same_law_refit_improvements.csv',index=False)
    checks=pd.DataFrame(checks);checks.to_csv(e.ROOT/'results/fast_checks.csv',index=False)
    trajectory=pairs.groupby(['x','round']).mean(numeric_only=True).drop(columns='seed');trajectory.to_csv(e.ROOT/'results/mean_trajectories.csv')
    endpoints=[];convergence=[]
    for (seed,x),g in pairs.groupby(['seed','x']):
        g=g.sort_values('round');a,b=g.iloc[0],g.iloc[-1]
        endpoints.append(dict(seed=seed,x=x,J_initial=a.J,J_final=b.J,J_change_pct=100*(b.J/a.J-1),
            MSE_gain_initial=a.MSE_gain_pct,MSE_gain_final=b.MSE_gain_pct,
            private_low_final=b.private_low_pct,private_high_final=b.private_high_pct,
            A_group1_change=b.A_group1-a.A_group1,A_other_change=b.A_other-a.A_other))
        deltas=[]
        for n in range(CFG['slow_updates']-2,CFG['slow_updates']+1):
            row=dict(round=n,J=float(abs(g.set_index('round').loc[n,'J']/g.set_index('round').loc[n-1,'J']-1)))
            for cap in ['low','high','other']:
                q=d[(d.seed==seed)&(d.x==x)&(d.capacity==cap)].set_index('round')
                for metric in ['MSE','beta','v']:
                    row[cap+'_'+metric]=float(abs(q.loc[n,metric]/q.loc[n-1,metric]-1))
            deltas.append(row)
        maximum=max(v for r in deltas for k,v in r.items() if k!='round')
        convergence.append(dict(seed=seed,x=x,max_relative_change_last_three=maximum,
                                within_tolerance=maximum<CFG['convergence_relative_tolerance'],details=deltas))
    endpoints=pd.DataFrame(endpoints);endpoints.to_csv(e.ROOT/'results/endpoints.csv',index=False)
    ref=json.loads((e.ROOT/'source_hashes.json').read_text())
    for name0,h in ref.items():assert hashlib.sha256((e.ROOT/name0).read_bytes()).hexdigest()==h,name0
    assert checks.clearing_error.max()<1e-10 and checks.initial_price_coupling_error.max()<1e-5
    audit=dict(passed=True,complete_trajectories=9,slow_updates_per_trajectory=CFG['slow_updates'],
       complete_deployed_states=len(m),reference_files_unchanged=len(ref),
       max_clearing_error=float(checks.clearing_error.max()),max_initial_price_coupling_error=float(checks.initial_price_coupling_error.max()),
       max_abs_split_mean_z=float(abs(checks.mean_split_z).max()),max_abs_split_variance_z=float(abs(checks.variance_split_z).max()),
       max_neural_abs_price_slope=float(d[d.capacity!='other'].chi.abs().max()),
       minimum_clearing_slope=float(m.delta.min()),all_generations_precede_learning=True,
       exact_checkpoint_continuity=True,convergence=convergence,
       versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,scipy=scipy.__version__))
    e.dump(e.ROOT/'VALIDATION.json',audit)
    print(endpoints.groupby('x').mean(numeric_only=True).drop(columns='seed').to_string())
    print('Fast checks:',audit['max_clearing_error'],audit['max_initial_price_coupling_error'])
    print('Descriptive settling criterion:',sum(x['within_tolerance'] for x in convergence),'/ 9')

if __name__=='__main__':analyze()
