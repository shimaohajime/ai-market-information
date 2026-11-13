"""Reproducible E5 endpoint, uncertainty and baseline eligibility summary."""
import json, shutil, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import engine as e
from run import CFG,name

def interval(values):
    v=np.asarray(values,dtype=float);n=len(v);mean=float(v.mean())
    sd=float(v.std(ddof=1)) if n>1 else float('nan')
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
    return dict(n=n,mean=mean,sd=sd,lower=mean-half,upper=mean+half)

def main(n=12):
    out=e.ROOT/'e5_summary';out.mkdir(exist_ok=True)
    d=pd.read_csv(e.ROOT/'results/model_metrics.csv')
    mk=pd.read_csv(e.ROOT/'results/market_metrics.csv')
    pa=pd.read_csv(e.ROOT/'results/paired_metrics.csv')
    # Preserve all individual paired results, rather than treating observations as seeds.
    endpoints=pa[pa['round']==n].copy();base=endpoints[endpoints.x==0].set_index('seed')['J']
    endpoints['J_vs_zero_pct']=[100*(r.J/base.loc[r.seed]-1) for r in endpoints.itertuples()]
    endpoints.to_csv(out/'endpoints.csv',index=False)
    summary=[]
    for x,g in endpoints.groupby('x'):
        for metric in ['J','J_vs_zero_pct','MSE_gain_pct','iota_high_minus_low_pct','private_low_pct','private_high_pct']:
            summary.append(dict(x=x,metric=metric,**interval(g[metric])))
    pd.DataFrame(summary).to_csv(out/'endpoint_intervals.csv',index=False)
    # Predeclared late blocks: shifts are descriptive, not 3-independent-round CIs.
    drifts=[]
    for seed in CFG['seeds']:
        s=mk[(mk.seed==seed)&(mk.x==0)].set_index('round')
        for metric in ['J','price_variance','price_lag1']:
            a=s.loc[n-5:n-3,metric].mean();b=s.loc[n-2:n,metric].mean()
            drifts.append(dict(seed=seed,metric=metric,earlier=a,later=b,change_pct=100*(b/a-1)))
        for c in ['low','high','other']:
            s=d[(d.seed==seed)&(d.x==0)&(d.capacity==c)].set_index('round')
            for metric in ['iota','MSE','beta','v','chi']:
                a=s.loc[n-5:n-3,metric].mean();b=s.loc[n-2:n,metric].mean()
                drifts.append(dict(seed=seed,metric=c+'_'+metric,earlier=a,later=b,change_pct=100*(b/a-1)))
    drift=pd.DataFrame(drifts);drift.to_csv(out/'late_block_drifts.csv',index=False)
    audits=[];policy=[]
    for p in sorted((e.ROOT/'baseline_audit').glob(f'n{n:02d}_seed*_rep*/complete.json')):
        m=json.loads(p.read_text());assert m['training_started']>=m['generation_finished']
        assert all(z['profile_sha256']==m['profile_sha256'] for z in m['generation'])
        s=d[(d.seed==m['seed'])&(d.x==0)&(d['round']==n)].set_index('capacity')
        v=pd.read_csv(p.parent/'metrics.csv').set_index(['version','capacity'])
        a=dict(seed=m['seed'],rep=m['rep'],metric='J',before=m['J_before'],after=m['J_refitted'])
        a['change_pct']=100*(a['after']/a['before']-1);audits.append(a)
        for c in ['low','high','other']:
            for metric in ['iota','beta','v','chi','MSE']:
                a=dict(seed=m['seed'],rep=m['rep'],metric=c+'_'+metric,
                    before=float(v.loc[('before',c),metric]),after=float(v.loc[('refitted',c),metric]))
                a['change_pct']=100*(a['after']/a['before']-1);audits.append(a)
        pol=pd.read_csv(p.parent/'policy_changes.csv');pol['seed']=m['seed'];pol['rep']=m['rep'];policy.append(pol)
    a=pd.DataFrame(audits);assert len(a)==3*4*16
    a.to_csv(out/'same_law_audit_replicates.csv',index=False)
    pd.concat(policy).to_csv(out/'same_law_policy_changes.csv',index=False)
    within=[]
    for (seed,metric),g in a.groupby(['seed','metric']):within.append(dict(seed=seed,metric=metric,**interval(g.change_pct)))
    within=pd.DataFrame(within);within.to_csv(out/'same_law_within_seed.csv',index=False)
    across=[]
    for metric,g in within.groupby('metric'):across.append(dict(metric=metric,**interval(g['mean'])))
    across=pd.DataFrame(across);across.to_csv(out/'same_law_across_seeds.csv',index=False)
    active=['J','low_iota','other_iota']
    gate1=bool((drift[drift.metric.isin(active)].change_pct.abs()<1).all())
    gate2=bool((within[within.metric.isin(active)]['mean'].abs()<1).all())
    aa=across[across.metric.isin(active)]
    gate3=bool(((aa.lower>-1)&(aa.upper<1)).all())
    validation=json.loads((e.ROOT/'VALIDATION.json').read_text())
    baseline=[]
    for seed in CFG['seeds']:
        src=e.ROOT/'results'/name(seed,0.)/f'round{n:02d}'
        dest=e.ROOT/'baselines'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        for filename in ['low_deployed.pt','high_deployed.pt','other_deployed.json','market.json','deployed_metrics.csv','complete.json']:
            shutil.copy2(src/filename,dest/filename)
        meta=json.loads((src/'complete.json').read_text())
        baseline.append(dict(seed=seed,round=n,x=0,profile_sha256=meta['profile_sha256'],
            path=str(dest.relative_to(e.ROOT)),candidate_high_fitted_law_round=n-1))
    status=dict(round=n,strict_original_passes=sum(z['within_tolerance'] for z in validation['convergence']),
        strict_original_total=len(validation['convergence']),late_block_band_pass=gate1,
        fixed_law_seed_mean_band_pass=gate2,fixed_law_across_seed_CI_band_pass=gate3,
        operational_mechanism_baseline_supported=gate1 and gate2 and gate3,
        practical_band_percent=1,baseline_profiles=baseline,
        caveat='Finite-learning operational diagnostic only. No population optimality, uniqueness, exact retraining fixed point, or convergence proof. Failure retains a candidate checkpoint, not a validated baseline.',
        continued_states=len(mk),audit_fits=12)
    e.dump(out/'BASELINE_STATUS.json',status)
    print(json.dumps(status,indent=2));print(across[across.metric.isin(active)].to_string(index=False))

if __name__=='__main__':main()
