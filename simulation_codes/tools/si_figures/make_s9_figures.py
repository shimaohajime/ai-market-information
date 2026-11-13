"""Rebuild S9 figures from archived E3/E4 seed-level summary (no retraining)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'recovered/e3_e4_measurement/summary/intervals.csv'
OUT = ROOT/'figures'
OUT.mkdir(exist_ok=True)
r=pd.read_csv(SOURCE)

def curve(metric):
    d=r[(r.metric==metric)&(r.contrast=='level')].sort_values('x')
    assert list(d.x)==[0.0,0.5,1.0]
    return d.x.to_numpy()*100,d['mean'].to_numpy(),(d['mean']-d['ci_low']).to_numpy(),(d['ci_high']-d['mean']).to_numpy()

fig,ax=plt.subplots(1,2,figsize=(10.0,3.4))
for metric,label,offset,style in [('E3_low_gain','Smaller network',-2.1,'o-'),('E3_high_gain','Larger network',2.1,'s--')]:
    xx,y,lo,hi=curve(metric)
    ax[0].errorbar(xx+offset,y,yerr=np.row_stack([lo,hi]),fmt=style,linewidth=1.55,markersize=5.3,capsize=3.1,label=label)
ax[0].set_ylabel('Reduction in held-out dividend MSE')
ax[0].set_title('(A) Value of seeing price')
ax[0].legend(fontsize=9,loc='center right',frameon=False)
xx,y,lo,hi=curve('E3_T_high')
ax[1].errorbar(xx,y,yerr=np.row_stack([lo,hi]),fmt='D-',linewidth=1.5,markersize=5,capsize=3.3)
ax[1].set_title('(B) How much price narrows the gap')
ax[1].set_ylabel('Smaller minus larger price gain')
for a in ax:
    a.set_xticks([0,50,100],['0','50','100'])
    a.set_xlabel('Group-1 high-capacity adoption (%)')
    a.grid(axis='y',alpha=.15)
    a.spines[['top','right']].set_visible(False)
    a.tick_params(labelsize=9)
fig.tight_layout(w_pad=2.4)
fig.savefig(OUT/'S9_price_predictive_value.pdf',bbox_inches='tight')
fig.savefig(OUT/'S9_price_predictive_value.png',bbox_inches='tight',dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.6,3.65))
xx,y,lo,hi=curve('E4_neural_high_upgrade')
ax.errorbar(xx,y,yerr=np.row_stack([lo,hi]),fmt='o-',lw=1.8,markersize=6,capsize=3,label='Neural forecasting heads')
xx,z,lo,hi=curve('E4_ols_nested_upgrade')
ax.plot(xx,z,linestyle='--',marker='s',markersize=4.5,linewidth=1.1,alpha=.9,label='Both encoders, recalibrated affine head')
ax.set_xticks([0,50,100],['0','50','100']);ax.set_xlabel('Group-1 high-capacity adoption (%)')
ax.set_ylabel('High minus low certainty equivalent')
ax.set_title('Gross value of the capacity upgrade')
ax.legend(fontsize=8.5,loc='upper right',frameon=False)
ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT/'S9_upgrade_value.pdf',bbox_inches='tight')
fig.savefig(OUT/'S9_upgrade_value.png',bbox_inches='tight',dpi=180)
plt.close(fig)
print('S9 figures generated from:',SOURCE)
