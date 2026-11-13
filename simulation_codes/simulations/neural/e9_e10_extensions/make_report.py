from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
R=Path(__file__).resolve().parent;D=R/'deliverables';D.mkdir(exist_ok=True);S=R/'summary'
for n,f in [('Sans','DejaVuSans.ttf'),('SansB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(n,str(R/'fonts'/f)))
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansB',italic='Sans',boldItalic='SansB')
st=pd.read_csv(S/'E9_states.csv');dg=pd.read_csv(S/'E9_diagnostics.csv');ns=pd.read_csv(S/'E9_fixed_law_audit.csv');fn=pd.read_csv(S/'E9_functional_changes.csv');ni=pd.read_csv(S/'E9_intervals.csv');v=pd.read_csv(S/'E10_seed_results.csv');lev=pd.read_csv(S/'E10_levels.csv');ch=pd.read_csv(S/'E10_changes.csv');ces=pd.read_csv(S/'E10_CE_diagnostics.csv');audit=json.loads((S/'VALIDATION.json').read_text())
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
fig,axs=plt.subplots(1,3,figsize=(10,3.1))
for ax,x in zip(axs,[0,.5,1]):
 a=st[(st.x==x)&(st['mode']=='fresh')].copy()
 for s,g in a.groupby('seed'):
  baseline=g[g['round']==6].J.iloc[0];a.loc[g.index,'pct']=100*(g.J/baseline-1);ax.plot(g['round'],100*(g.J/baseline-1),color='#7196b4',alpha=.65,lw=.8)
 z=a.groupby('round').pct.mean();ax.plot(z.index,z.values,'o-',ms=3,lw=1.7,color='#235b85',label='Fresh draws: mean')
 if x==.5:
  b=st[(st.x==x)&(st['mode']=='common_shocks')&(st['round']>=6)].copy()
  for s,g in b.groupby('seed'):b.loc[g.index,'pct']=100*(g.J/g[g['round']==6].J.iloc[0]-1)
  z=b.groupby('round').pct.mean();ax.plot(z.index,z.values,'s--',ms=3,lw=1.5,color='#b57b31',label='Common draws: mean');ax.legend(fontsize=7,frameon=False)
 ax.axvline(6,color='.5',ls=':',lw=.8);ax.axhline(0,color='.7',lw=.6);ax.set(title=f'Group-1 adoption {100*x:.0f}%',xlabel='Joint retraining update')
axs[0].set_ylabel('J relative to update 6 (%)');fig.tight_layout();fig.savefig(D/'E9_trajectories.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(10,3.1));names={'control':'Matched control','nonaligned':'Nonaligned payoff','heteroskedastic':'Variable noise variance'};cols={'control':'#286a9c','nonaligned':'#be7d32','heteroskedastic':'#508965'}
for ax,metric,title in zip(axs,['J_impact_pct','E3_T_high','E4_neural_high_upgrade'],['First-deployment J change','Price-induced prediction-gap narrowing','Gross capacity premium']):
 for mode in names:
  a=lev[(lev.economy==mode)&(lev.metric==metric)].sort_values('x');ax.errorbar(a.x*100,a['mean'],yerr=[a['mean']-a.ci_low,a.ci_high-a['mean']],marker='o',ms=3,lw=1.1,capsize=2,label=names[mode],color=cols[mode])
 ax.axhline(0,color='.65',lw=.6);ax.set_title(title,fontsize=9);ax.set_xlabel('Group-1 adoption (%)')
axs[0].set_ylabel('% of own-economy x=0 J');axs[1].set_ylabel('Dividend MSE');axs[2].set_ylabel('Certainty-equivalent difference');axs[0].legend(fontsize=6.5,frameon=False);fig.tight_layout();fig.savefig(D/'E10_results.png');plt.close(fig)
styles=getSampleStyleSheet()
for n,font,size,lead in [('BodyX','Sans',9,12.6),('TitleX','SansB',21,25),('HeadingX','SansB',12,16),('SmallX','Sans',7.8,10.5)]:styles.add(ParagraphStyle(name=n,fontName=font,fontSize=size,leading=lead,spaceAfter=8))
story=[]
def p(s,sty='BodyX'):story.append(Paragraph(s,styles[sty]))
def tbl(rows,widths):
 t=Table([[Paragraph(str(x),styles['SmallX']) for x in row] for row in rows],colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#738591')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,8))
def est(df,metric,dec=5,**kw):
 a=df[df.metric==metric]
 for k,x in kw.items():a=a[a[k]==x]
 r=a.iloc[0];return f"{r['mean']:.{dec}f} [{r.ci_low:.{dec}f}, {r.ci_high:.{dec}f}]"
def foot(c,d):
 c.setFont('Sans',8);c.setFillColor(colors.HexColor('#647382'));c.drawString(42,27,'AI Trading | E9 dynamics and E10 extensions | 29 October 2026');c.drawRightString(570,27,str(d.page))
p('E9: dynamics, cycles, and update noise','TitleX')
p('Extend all nine E6 both-group markets from six to twelve updates.','HeadingX')
p('At 50% adoption, three additional trajectories repeat the same primitive shocks and optimizer order, recomputing prices after each policy update. Four independent one-step refits from each fresh-draw terminal profile measure update noise and drift. The short horizon supports diagnostics, not a proof of equilibrium or absence of longer cycles.')
story.append(Image(str(D/'E9_trajectories.png'),width=526,height=163));story.append(Spacer(1,8))
rows=[['Scheme / adoption','J change, updates 6 to 12 (%) [95% interval]','Last 3 updates <0.5%','Period-2 screen']]
for mode,x in [('fresh',0),('fresh',.5),('fresh',1),('common_shocks',.5)]:
 a=dg[(dg['mode']==mode)&(dg.x==x)];label=('Fresh' if mode=='fresh' else 'Common')+f' / {100*x:.0f}%';rows.append([label,est(ni,'J_total_extension_pct',3,mode=mode,x=x),f'{int(a.stable_05pct_last3.sum())}/3',f'{int(a.period2_screen.sum())}/3'])
tbl(rows,[110,220,100,96])
p('Intervals are descriptive t(df2) intervals over the three original seeds. The 0.5% screen requires all final three changes in J and all three iotas to stay below the threshold. Period-2 requires repeated alternation, small two-step return distances, and a non-negligible one-step amplitude; full continuous diagnostics are retained. At endpoints, the all-rule screen includes a zero-mass candidate; active-rule checks are also reported in the CSV.','SmallX')
p('What changes when fresh sampling noise is removed','HeadingX')
f=dg[dg['mode']=='fresh'];c=dg[dg['mode']=='common_shocks']
p(f'Fresh-draw runs pass the final-update screen in {int(f.stable_05pct_last3.sum())}/9 cases; common-draw runs in {int(c.stable_05pct_last3.sum())}/3. The period-2 screen flags {int(dg.period2_screen.sum())}/12 trajectories. A common-draw fixed point would be a fixed point of this finite-sample procedure, not of the population retraining operator.')
p('The price statistic J is computed analytically from fitted demand coefficients. Its movements therefore reflect changes in learned policies, not Monte Carlo noise in estimating J. Fresh training data can nevertheless make those fitted coefficients fluctuate. Agreement of late-block averages is weaker than convergence of the full policy or market distribution.')
story.append(PageBreak())
p('E9: fixed-law and functional checks','TitleX')
p('Each audit refit starts from the same saved update 12 profile at 50% adoption. It is not the next step of an evolving trajectory. The table reports mean one-step changes and 95% t(df3) intervals across four independent refits, in percent of the inherited metric.')
rows=[['Seed','Metric','Mean change (%)','95% interval','Contained in +/-1%']]
for _,r in ns.iterrows():rows.append([int(r.seed),r.metric,f'{r.mean_pct:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]','Yes' if r.within_1pct else 'No'])
tbl(rows,[65,95,105,151,110])
p('Operational drift criterion','HeadingX')
p(f'{int(ns.within_1pct.sum())}/12 metric-by-seed intervals lie entirely inside +/-1%. All three other-group iota intervals lie below zero, indicating residual directional adjustment in these fitted rules. None of the three profiles passes all four tolerance checks. Neither population equilibrium nor merely zero-mean update noise is established.')
p('Changes in actual forecasts and orders','HeadingX')
p('Every policy is evaluated on the same held-out update 6 histories, signals and prices. Values below are averages over the final three updates and three seeds at 50% adoption. Forecast percentages use payoff standard deviation; order percentages use initial order RMS. This checks whether stable aggregate coefficients conceal changes in the implemented rules.','SmallX')
rows=[['Rule','Fresh forecast','Common forecast','Fresh orders','Common orders']]
for cap in ['low','high','other']:
 g=fn[(fn.x==.5)&(fn['round']>=10)&(fn.capacity==cap)];a=g[g['mode']=='fresh'];b=g[g['mode']=='common_shocks'];rows.append([cap,f'{a.forecast_RMS_change_pct_payoffSD.mean():.4f}%',f'{b.forecast_RMS_change_pct_payoffSD.mean():.4f}%',f'{a.demand_RMS_change_pct_initialRMS.mean():.4f}%',f'{b.demand_RMS_change_pct_initialRMS.mean():.4f}%'])
tbl(rows,[66,115,115,115,115])
p('The functional check concerns one anchor law. It does not show equality on all possible histories. Audit noise scales and complete per-update diagnostics are included in the accompanying CSV files.','SmallX')
story.append(PageBreak())
p('E10: relaxing dividend assumptions','TitleX')
p('A matched control separates the extension from a change in experiment timing.','HeadingX')
p('For each economy, low/high and group2 predictors are first fitted on the same inherited all-low law. Their profiles are then held fixed as adoption changes. After history adjustment, negligible low/high candidates are refitted on each resulting law for E3/E4. There is no additional joint market retraining in E10, so its levels are not the E6 six-update outcomes.')
story.append(Image(str(D/'E10_results.png'),width=526,height=163));story.append(Spacer(1,10))
rows=[['Economy','J impact at 100% (%)','E3 T change, 0% to 100%','E4 premium change, 0% to 100%']]
for mode in names:rows.append([names[mode],est(ch,'J_impact_pct',3,economy=mode,x=1),est(ch,'E3_T_high',5,economy=mode,x=1),est(ch,'E4_neural_high_upgrade',5,economy=mode,x=1)])
tbl(rows,[110,138,139,139])
p('Bracketed ranges are descriptive 95% intervals across three paired initial seeds. J concerns the private-signal target theta. Under a nonaligned payoff, J does not measure precision about the entire dividend. E3 directly targets dividends and E4 uses their actual simulated payoff distribution.','SmallX')
p('Adoption levels and gross capacity value','HeadingX')
rows=[['Economy','E4 premium at 0%','E4 premium at 50%','E4 premium at 100%']]
for mode in names:rows.append([names[mode]]+[f'{v[(v.economy==mode)&(v.x==x)].E4_neural_high_upgrade.mean():.5f}' for x in [0,.5,1]])
tbl(rows,[160,122,122,122])
p('All three seeds retain the direction of each endpoint effect in each economy. Under nonalignment, however, the J decline ranges from 2.90% to 7.27%, and its mean interval includes zero. Its magnitude is therefore imprecisely estimated. E3 and E4 retain their respective positive and negative adoption-change intervals in this small design; this is finite-learning evidence, not a general sign theorem.')
p('E3 uses public-only dividend predictors with/without price. E4 compares negligible candidates using the same price and own signal. OLS-recalibrated and nested-encoder checks are retained separately.','SmallX')
story.append(PageBreak())
p('Design details, validation, and limits','TitleX')
p('Two changes to the dividend law','HeadingX')
p('Nonaligned payoff: 20% of the fundamental dividend variance is assigned to a unit-variance latent factor orthogonal to the original target. The total unconditional fundamental variance is matched to control. Private signals continue to concern theta. This relaxes the alignment restriction without merely inflating the total payoff variance.')
p('Variable noise variance: the payoff mean remains lambda times theta, while the conditional noise-variance multiplier is 1 + 0.6 tanh(public0 / SD(public0)). It lies between 0.4 and 1.6 and has unconditional mean 1 by symmetry. The current variance is determined by released public information. The fitted Model A variance remains constant, so this deliberately tests a misspecified risk forecast; it is not a Model B experiment.')
p('Learning and evaluation','HeadingX')
p('Each refit uses 1,048,576 training windows and 4,096 Adam steps per capacity, with the inherited normalization and 16-period context. E10 uses an independent 262,144-observation head-fitting bank and four evaluation banks totaling 1,048,576 observations per frozen law. Primitive draws are paired across economies and adoption levels. No parameters or specifications are selected using test outcomes.')
p('Utility calculation and tails','HeadingX')
p('Current Gaussian dividend noise is integrated analytically using the correct mode-specific conditional mean and variance. All remaining expectations use held-out paths and log-sum-exp without clipping. In the nonaligned case, the extra latent factor enters scoring only; it is never a forecast input.')
primary=ces[ces.model.str.contains('neural')]
p(f'The smallest primary exponential-weight ESS is {primary.ESS.min():,.0f} of 1,048,576; the largest single normalized weight is {primary.max_weight.max():.4%}; the largest top-0.1% weight share is {primary.top_01pct_weight.max():.2%}. These are concentration diagnostics, not time-series effective sample sizes. Conditional CE uncertainty uses 500 paired, bank-stratified path bootstraps.')
p('Audit and interpretation','HeadingX')
p(f'Seven tests validate the matched draws, covariance construction, heteroskedastic bounds, deterministic replay, utility integration, and synthetic period-2/convergence screens. All saved-profile, data-separation, and learning-budget checks pass. Maximum clearing residuals are {audit["E9"]["max_clearing_error"]:.2e} for E9 and {audit["E10"]["max_clearing_error"]:.2e} for E10. Frozen-law coupling checks pass at the retained histories.')
p('The results remain conditional on three initial seeds, one economic calibration, Model A, and finite learning horizons. Relaxing sufficient assumptions does not imply that every underlying mechanism must fail; conversely, signs that survive these two extensions do not establish a general nonlinear theorem. State-dependent demand heads, other economic calibrations, longer cycles, and population retraining equilibria remain outside this experiment.')
SimpleDocTemplate(str(D/'E9_E10_Dynamics_and_Extensions_20261029.pdf'),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=44,title='E9 Dynamics and E10 Extensions',author='AI Trading Research').build(story,onFirstPage=foot,onLaterPages=foot)
print(D/'E9_E10_Dynamics_and_Extensions_20261029.pdf')
