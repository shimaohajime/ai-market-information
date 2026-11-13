"""Create independent-training-cell summaries and manuscript-ready diagnostic figures."""
from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
A=pd.read_csv(R/'results'/'paired_effects.csv')
B=pd.read_csv(R/'results'/'swap_vs_reoptimized_effects.csv')
K=['calibration','arch','seed']
x=A.groupby(K,as_index=False).mean(numeric_only=True)
x.to_csv(R/'results'/'training_cell_summary.csv',index=False)
z=B.groupby(K,as_index=False).mean(numeric_only=True)

# Figure 1: paired decoder effects from separately optimized restricted models.
fig,ax=plt.subplots(figsize=(8.1,4.5))
columns=['mean_effect_pp','variance_effect_pp','joint_effect_pp'];label=['Mean only','Risk only','Mean + risk']
rng=np.random.default_rng(100)
for i,(name,coll) in enumerate(zip(label,columns)):
    for a,arch in enumerate(('ssm','gru')):
        vals=x[x.arch==arch][coll].to_numpy()
        xpos=i*2.45+a*.8
        ax.scatter(xpos+rng.uniform(-.115,.115,len(vals)),vals,s=30,alpha=.65)
        ax.plot([xpos-.19,xpos+.19],[vals.mean(),vals.mean()],lw=3,color='black')
ax.axhline(0,color='gray',lw=.8)
ax.set_ylabel('Incremental linear decoder gain vs low-capacity baseline (pp)')
ax.set_xticks([i*2.45+j*.8 for i in range(3) for j in range(2)], [f'{t}\n{a.upper()}' for t in label for a in ('ssm','gru')], fontsize=8)
ax.set_title('Task 3.8: independently reoptimized constrained predictors')
ax.text(.015,.02,'Dots: 3 risk calibrations × 3 training seeds per architecture; 2 shock-bank tests averaged.',
        transform=ax.transAxes,fontsize=8)
# Markers directly labelled on x axis; no ambiguous legend.
fig.tight_layout();fig.savefig(R/'figures'/'independent_decoder_effects.png',dpi=190);plt.close(fig)

# Figure 2: not an externally meaningful effect size, NLL gains in each cell.
fig,ax=plt.subplots(figsize=(8.1,4.4))
for i,col in enumerate(['NLL_mean_gain','NLL_var_gain','NLL_joint_gain']):
    for j,arch in enumerate(('ssm','gru')):
        v=x[x.arch==arch][col].to_numpy()
        xp=i*2.4+j*.85
        ax.scatter(xp+rng.uniform(-.11,.11,len(v)),v,s=28,alpha=.65)
        ax.plot([xp-.20,xp+.20],[v.mean(),v.mean()],lw=3,color='black')
ax.axhline(0,color='gray',lw=.8)
ax.set_xticks([i*2.4+j*.85 for i in range(3) for j in range(2)], [f'{t}\n{a.upper()}' for t in ['Mean only','Risk only','Mean + risk'] for a in ('ssm','gru')], fontsize=8)
ax.set_ylabel('Improvement in held-out Gaussian NLL (nats / observation)')
ax.set_title('Statistical gains do not determine the sign of market information effects')
ax.text(.015,.02,'18 training cells total. One dot per architecture × calibration × training seed.',
        transform=ax.transAxes,fontsize=8)
fig.tight_layout();fig.savefig(R/'figures'/'independent_NLL_improvement.png',dpi=190);plt.close(fig)

# Figure 3: reoptimized vs post hoc swap paired arm effects.
fig,ax=plt.subplots(figsize=(7.8,4.6))
y=z.copy()
scales=['opt_mean_pp','swap_mean_pp','opt_risk_pp','swap_risk_pp']
for j,(nm,spec) in enumerate([('SSM mean',('ssm','opt_mean_pp','swap_mean_pp')),
      ('GRU mean',('gru','opt_mean_pp','swap_mean_pp')),
      ('SSM risk',('ssm','opt_risk_pp','swap_risk_pp')),
      ('GRU risk',('gru','opt_risk_pp','swap_risk_pp'))]):
    arch,op,sw=spec
    q=z[z.arch==arch]
    for i,row in enumerate(q.itertuples(index=False)):
        a=getattr(row,op);b=getattr(row,sw)
        ax.plot([j-.13,j+.13],[a,b],color='gray',alpha=.37,lw=.7)
        ax.scatter([j-.13],[a],s=15,alpha=.7)
        ax.scatter([j+.13],[b],s=15,marker='x',alpha=.7)
ax.axhline(0,lw=.8,color='gray')
ax.set_xticks(range(4),['SSM mean','GRU mean','SSM risk','GRU risk'])
ax.set_ylabel('Incremental linear decoder gain vs low baseline (pp)')
ax.set_title('Reoptimizing a constrained head differs from swapping trained outputs')
ax.text(.01,.01,'Left dot: separately optimized. Right cross: post-training swap. One pair per training cell.',
        fontsize=8,transform=ax.transAxes)
fig.tight_layout();fig.savefig(R/'figures'/'independent_vs_swapped.png',dpi=190);plt.close(fig)

cols=['mean_effect_pp','variance_effect_pp','joint_effect_pp','NLL_mean_gain','NLL_var_gain','NLL_joint_gain']
top={arch:{k:{'mean':float(g[k].mean()),'sd_across_cells':float(g[k].std()),'positive_cells':int((g[k]>0).sum()),'total_cells':int(len(g))} for k in cols}
     for arch,g in x.groupby('arch')}
for arch,g in z.groupby('arch'):
    top[arch]['swap_comparison']={k:float(g[k].mean()) for k in ['opt_mean_pp','swap_mean_pp','opt_risk_pp','swap_risk_pp','joint_pp']}
extra={'by_architecture':top,'training_cells':int(len(x)),'evaluation_pairs':int(len(A)),
       'calibrations':sorted(map(float,x.calibration.unique())),'seeds':sorted(map(int,x.seed.unique())),
       'experimental_cautions':['Diagnostic incumbent Q0, not a learning equilibrium',
       'Separate trainings are finite-optimization outcomes, not certified global minimizers',
       'E3 nonaligned payoff and heteroskedasticity; OOS linear decoder, not true MI',
       'Only two independent eval pairs per training cell; correlated scenarios and small number of seeds']}
(R/'results'/'analysis_summary.json').write_text(json.dumps(extra,indent=2))
print(json.dumps(extra,indent=2))
