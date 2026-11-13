from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DIR=Path(__file__).resolve().parent;F=DIR/'figures';F.mkdir(exist_ok=True)

p=json.loads((DIR/'population_results.json').read_text())
x=['Old mean\nOld risk','New mean\nOld risk','Old mean\nNew risk','New mean\nNew risk']
keys=['mean_old_risk_old','mean_new_risk_old','mean_old_risk_new','mean_new_risk_new']
y=[p['factorial'][k]['J'] for k in keys]
fig,ax=plt.subplots(figsize=(8,4.3))
ax.bar(x,y)
ax.axhline(y[0],ls='--',lw=1,label='Pre-adoption benchmark')
for i,z in enumerate(y):ax.text(i,z+.045,f'{z:.3f}',ha='center',fontsize=9)
ax.set_ylim(0,3.25);ax.set_ylabel('Incremental fresh-price precision J')
ax.set_title('Competing mean and risk channels (population Gaussian economy)')
ax.legend(fontsize=9);fig.tight_layout();fig.savefig(F/'population_mean_variance.png',dpi=185);plt.close(fig)

labels=['No retraining','Only upgraded type','Only other type','Both types']
y=[p['retraining_arms'][k]['J'] for k in ('none','only_upgraded','only_other','both')]
fig,ax=plt.subplots(figsize=(8,4.2));ax.bar(labels,y)
ax.axhline(p['baseline_J'],ls='--',lw=1,label='Pre-adoption equilibrium')
for i,z in enumerate(y):ax.text(i,z+.04,f'{z:.3f}',ha='center',fontsize=9)
ax.set_ylim(0,3.);ax.set_ylabel('Incremental fresh-price precision J')
ax.set_title('Who amplifies crowd-out through retraining?');ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(F/'population_retraining_roles.png',dpi=185);plt.close(fig)

het=p['signal_noise_heterogeneity_grid'];rr=np.array([r['ratio_sd_type0_to_type1'] for r in het]);d=np.array([r['impact_pct'] for r in het]);l=np.array([r['equilibrium_pct'] for r in het])
fig,ax=plt.subplots(figsize=(7.6,4.3));ax.plot(rr,d,'o-',label='Initial deployment');ax.plot(rr,l,'s-',label='After all-type retraining')
ax.axhline(0,linewidth=1,linestyle='--');ax.set_xlabel('Ratio of private-signal noise SDs (upgraded / other)');ax.set_ylabel('Change in J vs matching baseline (%)')
ax.set_title('Information heterogeneity changes retraining amplification')
ax.legend();fig.tight_layout();fig.savefig(F/'population_heterogeneity.png',dpi=185);plt.close(fig)

f=pd.read_csv(DIR/'neural_factorial_effects.csv')
fig,ax=plt.subplots(figsize=(8.1,4.5));inds=np.arange(3)
for i,a in enumerate(('ssm','gru')):
    part=f[f.arch==a]
    mean=np.array([part['mean_only_effect'].mean(),part['var_only_effect'].mean(),part['joint_effect'].mean()])*100
    sd=np.array([part['mean_only_effect'].std(),part['var_only_effect'].std(),part['joint_effect'].std()])*100
    ax.bar(inds-.18+i*.36,mean,.36,yerr=sd,capsize=4,label=a.upper())
ax.axhline(0,lw=1);ax.set_xticks(inds,['Mean update','Variance update','Both'])
ax.set_ylabel('Change in OOS linear decoder gain (percentage points)')
ax.set_title('E3: State-dependent learned heads, frozen inherited histories')
ax.legend();ax.text(.02,.02,'Error bars: SD across four independent shock-bank pairs',transform=ax.transAxes,fontsize=8)
fig.tight_layout();fig.savefig(F/'neural_factorial_E3.png',dpi=185);plt.close(fig)

q=pd.read_csv(DIR/'frozen_stationary_arms.csv',dtype={'arm':str})
fig,ax=plt.subplots(figsize=(7.8,4.3));ind=np.arange(3)
for i,a in enumerate(('ssm','gru')):
    part=q[q.arch==a].pivot(index='bank',columns='arm',values='OOS_decoder_gain')
    vals=[(part['10']-part['00']).mean(),(part['01']-part['00']).mean(),(part['11']-part['00']).mean()]
    sd=[(part['10']-part['00']).std(),(part['01']-part['00']).std(),(part['11']-part['00']).std()]
    ax.bar(ind-.18+i*.36,np.array(vals)*100,.36,yerr=np.array(sd)*100,capsize=4,label=a.upper())
ax.axhline(0,lw=1);ax.set_xticks(ind,['Mean update','Variance update','Both'])
ax.set_ylabel('Change in OOS linear decoder gain (percentage points)')
ax.set_title('E3: Market histories adjust endogenously, no further training');ax.legend()
ax.text(.02,.02,'Error bars: SD across two independent shock-bank pairs',transform=ax.transAxes,fontsize=8)
fig.tight_layout();fig.savefig(F/'neural_frozen_E3.png',dpi=185);plt.close(fig)
print('figures',list(F.glob('*.png')))
