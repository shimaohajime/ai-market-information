from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
src=Path(__file__).resolve().parent/'data'/'E9_states.csv'
df=pd.read_csv(src)
fig, ax=plt.subplots(1,2,figsize=(9.2,3.45),sharey=True)
for i,(mode,title) in enumerate([('fresh','A  Fresh training samples'),('common_shocks','B  Reused primitive shocks')]):
    a=df[(df.x==.5)&(df['mode']==mode)&(df['round']>=6)].copy()
    for seed,g in a.groupby('seed'):
        g=g.sort_values('round')
        b=float(g.loc[g['round']==6,'J'].iloc[0]);v=100*(g.J.to_numpy()/b-1)
        ax[i].plot(g['round'],v,marker='o',markersize=3,linewidth=1.25,label=f'Seed {seed}')
    ax[i].axhline(0,color='0.45',linewidth=.8,linestyle='--')
    ax[i].set_title(title,loc='left',fontsize=11,pad=8)
    ax[i].set_xlim(6,12);ax[i].set_xticks([6,7,8,9,10,11,12])
    ax[i].set_xlabel('Learning update',fontsize=9)
    ax[i].tick_params(axis='both',labelsize=8)
    ax[i].grid(alpha=.17,linewidth=.5)
ax[0].set_ylabel('Change in fresh-price precision\nrelative to update 6 (%)',fontsize=9)
ax[1].legend(loc='upper right',frameon=False,fontsize=7)
fig.tight_layout(w_pad=1.5)
out=Path(__file__).resolve().parent/'figures'/'S10_learning_dynamics.pdf';fig.savefig(out,bbox_inches='tight')
fig.savefig(out.with_suffix('.png'),dpi=180,bbox_inches='tight');plt.close(fig)
print(out)
