"""Rebuild S8 vector plots from archived primary experiment tables.
This script never simulates market outcomes, retrains models, or changes numerical estimates.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'figures'; OUT.mkdir(exist_ok=True)
PH=BASE/'recovered'/'phase2_market_lab'/'results'/'phase2_models.csv'
E6=BASE/'recovered'/'e6_group_retraining'/'summary'/'E6_intervals.csv'

# (1) Within-architecture training-seed paired changes in Gaussian NLL and fresh-signal J
z=pd.read_csv(PH)
z=z[(z.scenario=='E1_aligned') & (z.head_type=='A')]
fig, axes=plt.subplots(1,2,figsize=(9.7,3.5),sharex=True)
for ax,(variable,name,unit) in zip(axes,[('NLL','Held-out negative log score','High minus low'),('fresh_precision_E1', 'Fresh-signal likelihood precision', 'High minus low')]):
    for pos,arch in enumerate(['ssm','gru']):
        rows=z[z.architecture==arch].pivot(index='seed',columns='hidden',values=variable).sort_index()
        diff=rows[12]-rows[4]
        offsets=np.array([-0.095,0,0.095]);
        ax.scatter(np.full(len(diff),pos)+offsets,diff.values,s=28,marker='o',facecolor='#b8c0cc',edgecolor='#536575',linewidth=.65,zorder=3)
        ax.scatter(pos,np.mean(diff),s=68,marker='D',c='#154f64',edgecolor='white',linewidth=.7,zorder=4)
    ax.axhline(0,color='#677181',lw=.8,linestyle='--')
    ax.set_xticks([0,1],['Linear SSM','GRU'])
    ax.set_xlim(-.42,1.42)
    ax.set_ylabel(unit+' (width 12 - width 4)')
    ax.set_title(name,fontsize=10,pad=7)
    ax.tick_params(axis='both',labelsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.12)
fig.tight_layout(w_pad=2.4)
fig.savefig(OUT/'S8_first_deployment.pdf',bbox_inches='tight')
fig.savefig(OUT/'S8_first_deployment.png',bbox_inches='tight',dpi=180)
plt.close(fig)

# (2) E6 factorial, paired comparisons from the same nonconverged all-low candidate.
r=pd.read_csv(E6)
components=[('direct','Initial adoption'),('group1','Group 1 refit'),('group2','Group 2 refit'),('interaction','Interaction'),('total','Total')]
fig,axes=plt.subplots(1,2,figsize=(10.3,3.65),sharey=True)
for ax,x in zip(axes,[.5,1.0]):
    a=r[(r.x==x) & (r['round']==6)&(r.metric=='J') & (r.units=='percent_of_pre_adoption_J')]
    arr=[]
    for key,_ in components:
        rr=a[a.component==key].iloc[0]
        arr.append((rr['mean'],rr['lower'],rr['upper']))
    vals=np.array(arr);xx=np.arange(len(vals));
    cols=['#84919f','#4084a4','#c27c4f','#70aaa3','#45415c']
    ax.bar(xx,vals[:,0],color=cols,width=.65,edgecolor='white',linewidth=.4)
    ax.errorbar(xx,vals[:,0],yerr=[vals[:,0]-vals[:,1], vals[:,2]-vals[:,0]],fmt='none',ecolor='#26323b',capsize=3,linewidth=.9)
    ax.axhline(0,color='#505965',lw=.8)
    ax.set_xticks(xx,[name.replace(' ','\n',1) for _,name in components])
    ax.set_title(f'Group-1 adoption: {int(x*100)}%',fontsize=10,pad=7)
    ax.tick_params(axis='x',labelsize=8)
    ax.tick_params(axis='y',labelsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.11)
axes[0].set_ylabel('Change in J (% of pre-adoption J)')
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT/'S8_group_retraining.pdf',bbox_inches='tight')
fig.savefig(OUT/'S8_group_retraining.png',bbox_inches='tight',dpi=180)
plt.close(fig)

print('saved figures:',*[p.name for p in OUT.iterdir()])
