from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont('E6Sans',str(ROOT/'fonts/DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('E6Sans-Bold',str(ROOT/'fonts/DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('E6Sans',normal='E6Sans',bold='E6Sans-Bold',italic='E6Sans',boldItalic='E6Sans-Bold')
S=ROOT/'summary';state=pd.read_csv(S/'states.csv');eff=pd.read_csv(S/'E6_effects.csv');fac=pd.read_csv(S/'E6_factorial.csv');ints=pd.read_csv(S/'E6_intervals.csv');e7=pd.read_csv(S/'E7_adoption_adjusted.csv');audit=json.loads((S/'VALIDATION.json').read_text())
F=fac[(fac.metric=='J')&(fac['round']==6)].groupby('x').mean(numeric_only=True)
E=eff[(eff.metric=='J')&(eff['round']==6)].groupby(['arm','x']).mean(numeric_only=True)
labels={'none':'No updating','group1':'Group 1 only','group2':'Group 2 only','both':'Both groups'}
cols={'none':'#78818b','group1':'#2375a3','group2':'#b96427','both':'#723c81'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':170})
fig,axes=plt.subplots(1,3,figsize=(10.3,3.5),sharey=False)
for ax,x in zip(axes,[0,.5,1]):
 for arm in labels:
  g=eff[(eff.metric=='J')&(eff.x==x)&(eff.arm==arm)]
  metric='raw_retraining_pct_baseJ' if x==0 else 'extra_retraining_pct_baseJ'
  for seed,z in g.groupby('seed'):ax.plot(z['round'],z[metric],color=cols[arm],alpha=.18,lw=1)
  a=g.groupby('round')[metric].mean();ax.plot(a.index,a.values,'o-',color=cols[arm],ms=3,lw=1.7,label=labels[arm])
 ax.axhline(0,color='.65',lw=.7);ax.set(xlabel='Retraining updates',title=f'x = {x:g}')
axes[0].set_ylabel('Change in J (% of pre-adoption J)')
axes[1].legend(fontsize=7.5,frameon=False,loc='best');fig.tight_layout();fig.savefig(OUT/'E6_group_trajectories.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.3,3.3),sharey=True)
components=['direct','group1','group2','interaction','total'];names=['Direct\nadoption','Group 1\nalone','Group 2\nalone','Interaction','Total']
for ax,x in zip(axes,[.5,1]):
 q=ints[(ints.metric=='J')&(ints.x==x)&(ints['round']==6)&(ints.units=='percent_of_pre_adoption_J')].set_index('component')
 vals=[q.loc[k,'mean'] for k in components];lo=[q.loc[k,'mean']-q.loc[k,'lower'] for k in components];hi=[q.loc[k,'upper']-q.loc[k,'mean'] for k in components]
 ax.bar(range(5),vals,color=[cols['none'],cols['group1'],cols['group2'],'#49a19b',cols['both']],width=.65)
 ax.errorbar(range(5),vals,yerr=[lo,hi],fmt='none',color='#2d3440',capsize=3,lw=1)
 ax.axhline(0,color='.4',lw=.8);ax.set(xticks=range(5),xticklabels=names,title=f'Group-1 adoption: {x*100:.0f}%')
axes[0].set_ylabel('Effect on J (% of pre-adoption J)');fig.tight_layout();fig.savefig(OUT/'E6_factorial_effects.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.3,3.3),sharey=False)
for ax,x in zip(axes,[.5,1]):
 q=e7[(e7.x==x)&(e7['round']==6)].groupby('arm').mean(numeric_only=True)
 cats=['Initial\nadoption','Group 1\nretraining','Group 2\nretraining','Both\nretraining']
 means=[q.loc['both','adoption_mean_pct']]+[q.loc[a,'extra_retraining_mean_pct'] for a in ['group1','group2','both']]
 risks=[q.loc['both','adoption_variance_pct']]+[q.loc[a,'extra_retraining_variance_pct'] for a in ['group1','group2','both']]
 xx=np.arange(4);ax.bar(xx-.17,means,width=.32,color='#315d83',label='Mean / private-signal coefficient');ax.bar(xx+.17,risks,width=.32,color='#d39b42',label='Forecast variance')
 ax.plot(xx,np.array(means)+risks,'ko',ms=4,label='Net effect');ax.axhline(0,color='.5',lw=.7)
 ax.set(xticks=xx,xticklabels=cats,title=f'Group-1 adoption: {x*100:.0f}%')
axes[0].set_ylabel('Contribution to J (% of pre-adoption J)');axes[1].legend(fontsize=7.5,frameon=False);fig.tight_layout();fig.savefig(OUT/'E7_mean_risk.png');plt.close(fig)
styles=getSampleStyleSheet()
for z in styles.byName.values():z.fontName='E6Sans-Bold' if 'Heading' in z.name or z.name=='Title' else 'E6Sans'
styles.add(ParagraphStyle(name='BodyE6',fontName='E6Sans',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#223746')))
styles.add(ParagraphStyle(name='SmallE6',fontName='E6Sans',fontSize=8.3,leading=11,spaceAfter=6))
styles['Title'].fontSize=22;styles['Title'].leading=27;styles['Title'].textColor=colors.HexColor('#153e5c')
styles['Heading1'].fontSize=15;styles['Heading1'].leading=19;styles['Heading1'].textColor=colors.HexColor('#153e5c')
story=[]
def p(s,small=False):story.append(Paragraph(s,styles['SmallE6' if small else 'BodyE6']))
def h(s):story.append(Paragraph(s,styles['Heading1']))
def table(rows,widths):
 t=Table([[Paragraph(str(c),styles['SmallE6']) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#68869b')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d7dfe4')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,9))
def image(name,height):story.append(Image(str(OUT/name),width=490,height=height));story.append(Spacer(1,7))
# Page 1
story.append(Paragraph('E6-E7: Who drives retraining,<br/>and through which channel?',styles['Title']))
p('The Opposing Informational Effects of AI Trading | Experiment report | 29 October 2026',True)
p('We compare four updating rules from the same all-low E5 checkpoint: nobody, group 1, group 2, or both groups update. Group 1 uses the low/high GRUs; group 2 predicts from its own private signal and current price. All comparisons use three inherited initial seeds, adoption shares 0, .5 and 1, and six slow updates.')
p(f"At full group-1 adoption, group-2-only retraining changes J by an additional <b>{F.loc[1,'group2_pct_baseJ']:+.2f}%</b> of pre-adoption J after subtracting ordinary x=0 adaptation. The corresponding group-1-only effect is <b>{F.loc[1,'group1_pct_baseJ']:+.2f}%</b>. Their interaction is <b>{F.loc[1,'interaction_pct_baseJ']:+.2f}%</b>; it cannot be silently assigned to one group.")
rows=[['Updating rule','Total adoption effect: x=.5','Total adoption effect: x=1']]
for arm in labels:rows.append([labels[arm],f"{E.loc[(arm,.5),'total_adoption_pct_baseJ']:+.2f}%",f"{E.loc[(arm,1.),'total_adoption_pct_baseJ']:+.2f}%"])
table(rows,[210,140,140])
p('Total adoption effect compares x with x=0 under the SAME updating rule and horizon. Every percentage uses the seed-specific frozen all-low J as denominator. No updating is the direct adoption effect; the other rows include additional retraining.',True)
image('E6_group_trajectories.png',167)
p('Thin lines are individual seeds; thick lines are means. The left panel shows raw baseline adaptation. The other panels subtract that same-arm x=0 adaptation. E5 checkpoints remain candidate baselines, not certified retraining equilibria. Group masses and private-signal precision differ, so the ranking is conditional on this calibration.',True)
story.append(PageBreak())
# Page 2
h('1. Separating the two groups and their interaction')
p('For an updating rule a, define R(a,x,n) = J(a,x,n) - J(none,x,0). The primary retraining contrast is T(a,x,n) = R(a,x,n) - R(a,0,n). It removes ordinary residual learning at the imperfect starting baseline. Comparisons are paired by initial seed, primitive shock stream, optimization stream and update horizon.')
image('E6_factorial_effects.png',157)
rows=[['Contribution at update 6','x=.5: mean [95% interval]','x=1: mean [95% interval]']]
for key,label in [('direct','Direct adoption'),('group1','Group 1 alone: extra retraining'),('group2','Group 2 alone: extra retraining'),('interaction','Additional group interaction'),('both','Both: extra retraining'),('total','Direct + both-group retraining')]:
 row=[label]
 for x in [.5,1.]:
  q=ints[(ints.metric=='J')&(ints.x==x)&(ints['round']==6)&(ints.units=='percent_of_pre_adoption_J')&(ints.component==key)].iloc[0]
  row.append(f'{q["mean"]:+.2f} [{q.lower:+.2f}, {q.upper:+.2f}]')
 rows.append(row)
table(rows,[200,145,145])
p('All entries are percentages of pre-adoption J. Intervals use Student t over the three seed-level contrasts (2 degrees of freedom); they are descriptive, unadjusted intervals from a small inherited cohort. Dates and correlated arms are not treated as independent initial seeds.',True)
p('Because J is proportional to A squared, part of its interaction is purely algebraic. We separately audit interaction in A: at the first update it is exactly zero; later it can reflect different learning responses. The mechanical portion is only 0.011 percentage points at x=.5 and approximately zero at x=1; most measured interaction comes from nonadditivity in learned A. The archived interaction_channels.csv separates these two terms. The accounting identity is exact: both-group extra retraining = group-1-only extra retraining + group-2-only extra retraining + interaction. Adding direct adoption gives the total adoption effect. A frozen group retains identical parameters and normalization; its realized forecasts can still change with price history.')
p('At x=.5, the group-1-only decline is almost exactly canceled by interaction when both groups learn. At x=1, adding group-1 updating when group 2 already updates offsets 1.28% of pre-adoption J (95% interval: 0.93 to 1.64%). Thus the marginal effect of one group depends on whether the other learns.',True)
story.append(PageBreak())
# Page 3
h('2. E7: forecast mean versus forecast variance')
p('On the same inherited histories and economic shocks, we clear four fitted profiles: old mean/old variance, new mean/old variance, old mean/new variance, and both new. We average the two possible orders of replacement to allocate their interaction symmetrically. The mean and variance contributions therefore sum exactly to the observed coefficient-induced change.')
image('E7_mean_risk.png',157)
rows=[['Comparison','Adoption','Mean channel','Variance channel','Net J effect']]
for x in [.5,1.]:
 q=e7[(e7.x==x)&(e7['round']==6)].groupby('arm').mean(numeric_only=True)
 z=q.loc['both'];rows.append(['Initial adoption',f'{x*100:.0f}%',f'{z.adoption_mean_pct:+.2f}',f'{z.adoption_variance_pct:+.2f}',f'{z.adoption_mean_pct+z.adoption_variance_pct:+.2f}'])
 for arm in ['group1','group2','both']:
  z=q.loc[arm];rows.append([labels[arm]+': extra retraining',f'{x*100:.0f}%',f'{z.extra_retraining_mean_pct:+.2f}',f'{z.extra_retraining_variance_pct:+.2f}',f'{z.extra_retraining_mean_pct+z.extra_retraining_variance_pct:+.2f}'])
table(rows,[190,55,85,85,75])
p('Entries are percentages of pre-adoption J. Retraining rows subtract the corresponding x=0 mean/risk contributions. Initial adoption changes only the upgrading cell from the low to the high model. Those initial contributions are distinct from subsequent learning contributions.',True)
p('For J, the mean channel acts through the private-signal coefficient beta; the variance channel changes beta/(gamma v). The complete mean-model swap also includes the encoder, intercept and price coefficient when measuring counterfactual prices. These are post-fit mechanical diagnostics, not separately optimized models or counterfactual retraining equilibria.')
p('Per-update contributions telescope for structural J, A and the private-signal loadings. Price-variance swaps are evaluated on the old law and are retained only as one-auction diagnostics; they do not telescope into changes in stationary price variance.',True)
story.append(PageBreak())
# Page 4
h('3. Protocol, validation and scope')
rows=[['Item','Specification']]
for a,b in [('Starting point','Same E5 all-low update-12 low/high models and group-2 head for every arm and adoption share within a seed. Initial parameters are identical across arms.'),('Economic controls','Group masses .55/.45; private-signal noise variances .25/.09; risk aversions .9/1.2. Payoff and supply processes are fixed. Adoption refers only to the high-model fraction within group 1.'),('Slow-fast separation','Each arm generates its own recursive prices with all parameters fixed. All data generation finishes before permitted groups refit. Updated groups deploy simultaneously.'),('Per-update budget','1,048,576 training observations; 65,536 neural validation observations; 131,072 held-out evaluation observations; 512-date burn-in; 16-date context; unchanged E5 learning schedule.'),('Common random numbers','Training: seed, 60000+round, bank 1; validation: bank 50. Evaluation: seed, 70000, bank 60. Neither arm nor adoption enters the primitive or optimizer stream key.'),('Fixed group enforcement','Group 1 freezes both GRUs, or group 2 freezes its entire head and variance, exactly as assigned. Inactive candidate GRUs refit when group 1 is assigned to update.'),('State counts','198 explicitly simulated states: 9 unchanged-policy states plus 27 updating branches x 7 states. There are 252 reported arm/horizon states because 54 unchanged-policy horizons reuse the same law.'),('Reproducibility','All checkpoints, data-generation fingerprints, fit histories, coefficients, four-way swaps, paired contrasts, intervals and plots are archived. Training panels can be regenerated from their keys.')]:rows.append([a,b])
table(rows,[112,378])
p(f"<b>Validation passed.</b> Ten tests cover the engine and new decomposition. All {audit['simulated_states']} states pass fingerprint and chronology checks. First-update fitted blocks match exactly across corresponding arms; disabled groups remain bitwise unchanged; saved next models equal the next deployed models. Maximum clearing residual: {audit['maximum_clearing_residual']:.2e}. Maximum retained initial-price coupling error: {audit['maximum_initial_price_coupling_error']:.2e}. Minimum clearing slope: {audit['minimum_clearing_slope']:.4f}.")
p('Every factorial, paired-control, Shapley and structural telescoping identity is checked numerically. The initial high candidate was last trained under the preceding E5 law, not refitted to certify the starting law. Results concern finite six-update adjustment from that shared profile. They do not establish an equilibrium, global stability, information-sharing benefits, computation value, or aggregate welfare.',True)
p('Run run_e6.py, e7_adoption.py, analyze_e6.py and make_report.py to reproduce. PROTOCOL.txt gives the full estimands and timing. Price-usefulness measurements and learning-dynamics extensions are implemented in their corresponding experiment directories.',True)
def footer(c,doc):
 c.setFont('E6Sans',8);c.setFillColor(colors.HexColor('#6c7c88'));c.drawString(48,29,'AI-market-Science | E6-E7 | 29 October 2026');c.drawRightString(547,29,str(doc.page))
SimpleDocTemplate(str(OUT/'E6_E7_Group_Retraining_20261029.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=46,title='E6-E7: Group retraining and mean-risk channels',author='AI-market-Science').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'E6_E7_Group_Retraining_20261029.pdf')
