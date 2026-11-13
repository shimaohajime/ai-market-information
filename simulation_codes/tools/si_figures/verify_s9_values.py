"""Independent, low-cost audit of archived E3/E4 summaries, path-level CE, and manuscript numbers.
Does not train neural networks or regenerate simulated paths.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parent
S=ROOT/'recovered/e3_e4_measurement'
SUM=S/'summary';R=pd.read_csv(SUM/'seed_results.csv');INTER=pd.read_csv(SUM/'intervals.csv')
tests=0
max_errors={'e3_identity':0.,'ce_path_reconstruction':0.,'seed_csv_reconstruction':0.,'intervals':0.,'market_clear':0.}

def check(expr,label):
    global tests
    if not expr:raise AssertionError(label)
    tests+=1

for _,row in R.iterrows():
    seed=int(row.seed); x=float(row.x); name=f'seed{seed}_x{int(round(x*100)):03d}'
    p=S/'results'/name
    paths=pd.read_csv(p/'paths.csv');banks=pd.read_csv(p/'banks.csv');ce=pd.read_csv(p/'CE.csv').set_index('model')
    complete=json.loads((p/'complete.json').read_text())
    check(complete['evaluation_observations']==1048576,f'{name} evaluation count')
    check(set(paths.bank)=={10,11,12,13},f'{name} bank set')
    for cap in ['low','high','nested']:
        v=banks[(banks.kind=='E3')&(banks.model==cap)]
        check(len(v)==4,f'{name} {cap} bank count')
        errs=[v.MSE_no_price.mean()-row[f'E3_{cap}_MSE_no_price'],v.MSE_price.mean()-row[f'E3_{cap}_MSE_price'],v.gain.mean()-row[f'E3_{cap}_gain']]
        max_errors['seed_csv_reconstruction']=max(max_errors['seed_csv_reconstruction'],*map(abs,errs))
        check(max(map(abs,errs))<1e-12,f'{name} E3 {cap} means')
        gain=float(v.MSE_no_price.mean()-v.MSE_price.mean())
        max_errors['e3_identity']=max(max_errors['e3_identity'],abs(gain-row[f'E3_{cap}_gain']))
        check(abs(gain-row[f'E3_{cap}_gain'])<1e-12,f'{name} E3 {cap} gain')
    check(abs(row.E3_T_high - (row.E3_low_gain-row.E3_high_gain))<1e-12,'E3 high T')
    check(abs(row.E3_T_nested - (row.E3_low_gain-row.E3_nested_gain))<1e-12,'E3 nested T')
    for model in ['E4_neural_low','E4_neural_high','E4_ols_low','E4_ols_high','E4_ols_nested']:
        pp=paths[paths.model==model]
        check(len(pp)==1024,f'{name} E4 {model} paths')
        check(len(pp.groupby(['bank','path']))==1024,f'{name} E4 {model} uniqueness')
        new_ce=-(logsumexp(pp.logmeanweight.values)-np.log(len(pp)))/0.9
        delta=abs(new_ce-row[model+'_CE'])
        max_errors['ce_path_reconstruction']=max(max_errors['ce_path_reconstruction'],delta)
        check(delta<1e-12,f'{name} E4 {model} true CE reconstruction')
        check(abs(new_ce-ce.loc[model,'CE'])<1e-12,f'{name} E4 {model} CE file')
    for label,cap in [('neural','high'),('ols','high'),('ols','nested')]:
        prefix='E4_'+label
        v=row[prefix+'_'+cap+'_CE']-row[prefix+'_low_CE']
        check(abs(v-row[prefix+'_'+cap+'_upgrade'])<1e-12,f'{name} upgrade decomposition')
    train=json.loads((p/'training.json').read_text())
    check(all(a['updates']==4096 and a['optimization_windows']==1048576 for a in train['details'].values()),'candidate budget')
    ge=json.loads((p/'generation.json').read_text())
    check(len(ge)==7,f'{name} train + val + cal + 4 eval banks')
    max_errors['market_clear']=max(max_errors['market_clear'],max(q['clearing_error'] for q in ge))

for metric in ['E3_low_gain','E3_high_gain','E3_T_high','E3_T_nested','E4_neural_high_upgrade','E4_ols_high_upgrade','E4_ols_nested_upgrade']:
    for x in [0.,.5,1.]:
        d=R[R.x==x].set_index('seed')[metric]
        y=INTER[(INTER.x==x)&(INTER.metric==metric)&(INTER.contrast=='level')].iloc[0]
        mean=d.mean();hw=t.ppf(.975,2)*d.std(ddof=1)/np.sqrt(3)
        err=max(abs(mean-y['mean']),abs(mean-hw-y.ci_low),abs(mean+hw-y.ci_high))
        max_errors['intervals']=max(max_errors['intervals'],err)
        check(err<2e-12,f'{metric} {x} intervals')
        if x>0:
            a=d-R[R.x==0].set_index('seed')[metric]
            y=INTER[(INTER.x==x)&(INTER.metric==metric)&(INTER.contrast=='change_from_x0')].iloc[0]
            mean=a.mean();hw=t.ppf(.975,2)*a.std(ddof=1)/np.sqrt(3)
            check(max(abs(mean-y['mean']),abs(mean-hw-y.ci_low),abs(mean+hw-y.ci_high))<2e-12,'paired change CI')

# original 202-file provenance manifest
manifest=json.loads((S/'MANIFEST_SHA256.json').read_text());missing=[];changed=[]
for relative,expected in manifest.items():
    f=S/relative
    if not f.exists():missing.append(relative)
    elif hashlib.sha256(f.read_bytes()).hexdigest()!=expected:changed.append(relative)
font_exclusions=[a for a in missing if a.startswith('fonts/')]
check(not (set(missing)-set(font_exclusions)),'source manifest complete, except omitted font assets')
check(not changed,'source manifest SHA256 consistent')

# exact claims displayed in SI
get=lambda x,m:float(INTER[(INTER.x==x)&(INTER.metric==m)&(INTER.contrast=='level')]['mean'].iloc[0])
checks={
 'low_price_pct':100*(get(1,'E3_low_gain')/get(0,'E3_low_gain')-1),
 'high_price_pct':100*(get(1,'E3_high_gain')/get(0,'E3_high_gain')-1),
 'gross_upgrade_pct':100*(get(1,'E4_neural_high_upgrade')/get(0,'E4_neural_high_upgrade')-1),
 'price_gap_additive_change':get(1,'E3_T_high')-get(0,'E3_T_high'),
 'ce_upgrade_additive_change':get(1,'E4_neural_high_upgrade')-get(0,'E4_neural_high_upgrade'),
}
for key,value,target in [('low_price_pct',checks['low_price_pct'],12.4),('high_price_pct',checks['high_price_pct'],-6.8),('gross_upgrade_pct',checks['gross_upgrade_pct'],-44.2)]:check(abs(value-target)<.07,key)
assert max_errors['market_clear']<2e-14
out=dict(tests_passed=tests,source_manifest_entries=len(manifest),source_files_verified=len(manifest)-len(missing),font_assets_omitted_from_shareable_package=font_exclusions,source_missing_other=[a for a in missing if not a.startswith('fonts/')],source_changed=changed,max_errors=max_errors,highlighted_estimates=checks)
(ROOT/'checks/S9_NUMERICAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
