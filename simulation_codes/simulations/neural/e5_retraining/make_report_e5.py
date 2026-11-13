from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("E5Sans", str(Path(__file__).resolve().parent/"fonts/DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("E5Sans-Bold", str(Path(__file__).resolve().parent/"fonts/DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("E5Sans",normal="E5Sans",bold="E5Sans-Bold",italic="E5Sans",boldItalic="E5Sans-Bold")

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
S=ROOT/'e5_summary'
status=json.loads((S/'BASELINE_STATUS.json').read_text())
validation=json.loads((ROOT/'VALIDATION.json').read_text())
m=pd.read_csv(ROOT/'results/market_metrics.csv');d=pd.read_csv(ROOT/'results/model_metrics.csv')
ep=pd.read_csv(S/'endpoints.csv');iv=pd.read_csv(S/'same_law_across_seeds.csv').set_index('metric')
within=pd.read_csv(S/'same_law_within_seed.csv');drift=pd.read_csv(S/'late_block_drifts.csv')
audit=pd.read_csv(S/'same_law_audit_replicates.csv')
shock=pd.concat([pd.read_csv(p) for p in (ROOT/'baseline_shocks').glob('*/metrics.csv')],ignore_index=True)
shock.to_csv(S/'baseline_shock_metrics.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
cols=['#245b86','#bc7226','#793d78']
fig,axes=plt.subplots(1,2,figsize=(10.3,3.6),gridspec_kw={'width_ratios':[1.25,1]})
for x,c in zip([0,.5,1],cols):
 g=m[m.x==x]
 for seed,z in g.groupby('seed'):axes[0].plot(z['round'],z.J,color=c,alpha=.2,lw=1)
 a=g.groupby('round').J.mean();axes[0].plot(a.index,a.values,'o-',color=c,label=f'x = {x:g}',lw=1.8,ms=3)
axes[0].axvline(6,color='.65',ls='--',lw=1);axes[0].set(xlabel='Simultaneous retraining updates',ylabel='Fresh-signal precision J',title='A. Original paths continued to update 12');axes[0].legend(frameon=False)
for cap,c in zip(['low','other'],cols):
 g=d[(d.x==0)&(d.capacity==cap)]
 for seed,z in g.groupby('seed'):
  q=z.set_index('round').iota;axes[1].plot(q.index,100*(q/q.loc[6]-1),color=c,alpha=.2)
 a=g.pivot(index='round',columns='seed',values='iota');a=100*(a/a.loc[6]-1)
 axes[1].plot(a.index,a.mean(axis=1),'o-',color=c,label=cap,lw=1.8,ms=3)
axes[1].axhline(0,color='.7',lw=1);axes[1].set(xlabel='Simultaneous retraining updates',ylabel='Response change from update 6 (%)',title='B. Active responses in the all-low market');axes[1].legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/'E5_retraining_paths.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10.3,3.0),sharey=False)
for ax,metric,title in zip(axes,['J','low_iota','other_iota'],['Price precision J','Low-model response','Group-2 response']):
 for j,seed in enumerate(sorted(audit.seed.unique())):
  v=audit[(audit.seed==seed)&(audit.metric==metric)].change_pct.values
  ax.scatter(np.full(len(v),j)+np.linspace(-.1,.1,len(v)),v,color=cols[j],s=20,alpha=.7)
  ax.plot([j-.17,j+.17],[v.mean()]*2,color='black',lw=2)
 ax.axhline(0,color='.4',lw=1);ax.axhline(1,color='.7',ls='--');ax.axhline(-1,color='.7',ls='--')
 ax.set(xticks=range(3),xticklabels=['18013','19031','20047'],xlabel='Initial seed',ylabel='One-step change (%)',title=title)
fig.tight_layout();fig.savefig(OUT/'E5_fixed_law_refits.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.3,3.4))
sh=shock[shock.capacity=='low']
for stage,ls,col in [('impact','--',cols[1]),('frozen','-',cols[0])]:
 a=sh[sh.stage==stage].groupby('x').mean(numeric_only=True)
 axes[0].plot(a.index,a.J,'o'+ls,color=col,label=stage)
 axes[1].plot(a.index,a.price_variance,'o'+ls,color=col,label=stage)
axes[0].set(xlabel='High-model share within group 1',ylabel='J',title='A. Same coefficients imply the same J')
axes[1].set(xlabel='High-model share within group 1',ylabel='Price variance',title='B. Endogenous histories can change moments')
for ax in axes:ax.legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/'E5_baseline_adoption.png');plt.close(fig)
styles=getSampleStyleSheet()
for style in styles.byName.values():style.fontName='E5Sans-Bold' if 'Heading' in style.name or style.name=='Title' else 'E5Sans'
styles.add(ParagraphStyle(name='BodyE5',fontName='E5Sans',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#243545')))
styles.add(ParagraphStyle(name='SmallE5',fontName='E5Sans',fontSize=8.5,leading=11,spaceAfter=6))
styles['Title'].fontSize=23;styles['Title'].leading=28;styles['Title'].textColor=colors.HexColor('#173e5e')
styles['Heading1'].fontSize=15;styles['Heading1'].leading=19;styles['Heading1'].textColor=colors.HexColor('#173e5e')
styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=15
story=[]
def p(s,small=False):story.append(Paragraph(s,styles['SmallE5' if small else 'BodyE5']))
def h(s):story.append(Paragraph(s,styles['Heading1']))
def tab(rows,widths):
 cells=[[Paragraph(str(v),styles['SmallE5']) for v in row] for row in rows]
 t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#6a879b')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dde3')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,10))
def img(name,height):story.append(Image(str(OUT/name),width=490,height=height));story.append(Spacer(1,7))
# Page 1
story.append(Paragraph('E5: Continued retraining<br/>and an all-low baseline',styles['Title']))
p('The Opposing Informational Effects of AI Trading | Experiment report | 29 October 2026',True)
confirmed=status['operational_mechanism_baseline_supported']
p('<b>Outcome.</b> Nine trajectories were continued from six to twelve simultaneous updates. The all-low endpoint '+('passes the predeclared practical mechanism-stability diagnostic and is retained as a provisional operational baseline.' if confirmed else 'does not pass every predeclared practical mechanism-stability diagnostic. It is retained as a reproducible candidate, not a validated retraining equilibrium.'))
p(f"The original strict settling test passes in {status['strict_original_passes']} of {status['strict_original_total']} trajectories. Twelve independent same-law refits distinguish update variability from average remaining drift. Neither diagnostic establishes population optimality or convergence.")
p('Mean J changes by only 0.027%, 0.034% and 0.019% between updates 6 and 12 at x=0, .5 and 1, respectively. These endpoint averages summarize a stable range but do not erase the intervening fluctuations.')
rows=[['Group-1 adoption','J at update 12','J vs x=0','High-model MSE gain','High-model response vs low']]
for x,g in ep.groupby('x'):
 rows.append([f'{100*x:.0f}%',f'{g.J.mean():.4f}',f'{g.J_vs_zero_pct.mean():+.2f}%',f'{g.MSE_gain_pct.mean():.2f}%',f'{g.iota_high_minus_low_pct.mean():+.2f}%'])
tab(rows,[88,85,85,112,120])
p('Entries are means over the three inherited initial seeds. Adoption is measured within group 1 (55% of investors). These are continued original paths, not shocks from the new baseline. The high and low predictors are evaluated on the same market within each comparison.',True)
img('E5_retraining_paths.png',171)
p('Thin lines show individual seeds; thick lines show means. The dashed vertical line marks the previous stopping point. The response plot uses each seed\'s own update-6 value as its reference.',True)
story.append(PageBreak())
# Page 2
h('1. Is the all-low market sufficiently settled?')
p('Each audit repetition starts from exactly the same update-12 profile. It generates a fresh bank of 1,048,576 training observations under that frozen market, then refits both GRUs and group 2. Audit fits are never used to generate their own training observations. This measures one-step update noise and average learning drift under a common law.')
img('E5_fixed_law_refits.png',143)
rows=[['Quantity','Mean update (%)','95% interval (%)','Within +/-1%?']]
for key,label in [('J','J'),('low_iota','Low-model response'),('other_iota','Group-2 response')]:
 r=iv.loc[key];rows.append([label,f'{r["mean"]:+.3f}',f'[{r.lower:+.3f}, {r.upper:+.3f}]','Yes' if r.lower>-1 and r.upper<1 else 'No'])
tab(rows,[155,105,145,85])
p('Each dot is one independent-data refit; black bars are seed means. Aggregate intervals use the three seed means as independent units (Student t, 2 degrees of freedom), not the millions of observations or the twelve refits as twelve independent initial conditions.',True)
rows=[['Predeclared check','Result']]
for key,label in [('late_block_band_pass','Rounds 7-9 vs 10-12: every seed\'s J and active responses change by less than 1%'),('fixed_law_seed_mean_band_pass','Same-law mean update: every seed\'s J and active responses change by less than 1%'),('fixed_law_across_seed_CI_band_pass','Across-seed 95% intervals for those mean updates lie inside +/-1%')]:rows.append([label,'Pass' if status[key] else 'Fail'])
tab(rows,[420,70])
p('The 1% band is an additional practical diagnostic for J and the two active trading responses. It does not replace the original .5% rule for successive J, MSE, beta and variance changes. It also does not certify the stability of every parameter or every downstream statistic.')
story.append(PageBreak())
# Page 3
h('2. Adoption from a common saved baseline')
p('For each seed, all five adoption shares use the same saved all-low update-12 profile. The first-auction comparison retains inherited histories and recomputes clearing. The frozen comparison recursively generates new price histories without changing any model parameter. These comparisons remain candidate-baseline diagnostics if the stability tests fail.')
img('E5_baseline_adoption.png',162)
rows=[['Adoption','J: impact = frozen','Impact price variance','Frozen price variance']]
for x in sorted(sh.x.unique()):
 a=sh[(sh.x==x)&(sh.stage=='impact')].mean(numeric_only=True);b=sh[(sh.x==x)&(sh.stage=='frozen')].mean(numeric_only=True)
 rows.append([f'{x*100:.0f}%',f'{a.J:.4f}',f'{a.price_variance:.4f}',f'{b.price_variance:.4f}'])
tab(rows,[85,140,130,135])
p('In this calibration, history adjustment is quantitatively small: the largest price-variance difference between the frozen and first-auction stages is about 0.023% across the 15 seed/adoption pairs. The equality of J across stages is a Model-A identity: the response coefficients remain fixed. Price variance and prediction error can change as price histories adjust. The high candidate at update 12 was most recently fitted under the update-11 law, like the other deployed models. No separate same-law optimality claim is made.')
p('<b>Remaining E5 work.</b> These new adoption branches have not been retrained after the shock. Their starting profiles and shock evaluator are saved. The separately continued x=.5 and x=1 paths cannot be relabeled as post-shock paths from the mature all-low market. E3/E4 information-sharing and utility measurements are also outside this experiment.')
story.append(PageBreak())
# Page 4
h('3. Implementation, validation and reproducibility')
rows=[['Item','Specification']]
for a,b in [('Economy','Eight public coordinates; nonlinear noisy cubic releases; 20% fresh component in fundamental variance; unchanged private-signal and supply-noise primitives.'),('Models','Group 1: GRU width 4/32, common 16-date windows, constant affine Gaussian heads. Group 2: own private signal and current price; no public encoder.'),('Slow update','Freeze every model; finish all market-data generation; refit both GRUs and group 2; deploy simultaneously. Inactive candidate models are also fitted.'),('Data per update','Training 1,048,576; validation 65,536; head calibration 65,536; evaluation 131,072. Burn-in: 512. Windows never cross independent paths.'),('Learning','One pass through eight training chunks; Adam; inherited normalizations frozen; optimizer reset per slow update; same learning-rate schedule as the original experiment.'),('Continuation','3 initial seeds x 3 adoption shares x 13 deployed states = 117 states. The first 7 states per path originate in the earlier experiment; terminal round 6 is reopened only to fit its next update.'),('Noise audit','3 baseline seeds x 4 fresh-data one-step refits. Independent audit evaluation bank; same-law MSE and policy changes retained.'),('Fast-market diagnostics','Initial prices -10 and +10 are coupled under common shocks; price moments are compared across retained path halves. These are empirical diagnostics, not an existence proof.')]:rows.append([a,b])
tab(rows,[108,382])
p(f"<b>Validation.</b> Six engine tests pass. All {validation['complete_deployed_states']} deployed states pass the generation-before-learning and exact checkpoint-continuity audit. Maximum clearing residual: {validation['max_clearing_error']:.2e}; maximum retained initial-price coupling error: {validation['max_initial_price_coupling_error']:.2e}; minimum clearing slope: {validation['minimum_clearing_slope']:.4f}. Reference source and inherited initial-model hashes remain unchanged.")
p('The archive contains source code, configurations, every continued checkpoint, all audit fits, the three candidate baseline profiles, CSV summaries, status JSON, plots and this report. E5_PROTOCOL.txt records the diagnostics; BASELINE_STATUS.json records their outcomes. run.py resumes completed stages. Re-run analyze.py, summarize_e5.py and make_report_e5.py to reconstruct the summaries and report.',True)
p('Scope: finite-sample, finite-optimization evidence under three inherited initial seeds. No claim of a population retraining fixed point, global stability, equilibrium uniqueness, mutual-information change, or welfare improvement. Accepted manuscript sections are unchanged.',True)

def footer(c,doc):
 c.setFont('E5Sans',8);c.setFillColor(colors.HexColor('#6d7b85'));c.drawString(48,29,'AI-market-Science | E5 continuation | 29 October 2026');c.drawRightString(547,29,str(doc.page))
SimpleDocTemplate(str(OUT/'E5_Retraining_and_Baseline_20261029.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=46,title='E5: Continued retraining and an all-low baseline',author='AI-market-Science').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'E5_Retraining_and_Baseline_20261029.pdf')
