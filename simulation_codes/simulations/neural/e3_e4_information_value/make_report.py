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
R=Path(__file__).resolve().parent;D=R/'deliverables';D.mkdir(exist_ok=True)
for name,f in [('Sans','DejaVuSans.ttf'),('SansB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(R/'fonts'/f)))
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansB',italic='Sans',boldItalic='SansB')
v=pd.read_csv(R/'summary/seed_results.csv');i=pd.read_csv(R/'summary/intervals.csv');cd=pd.read_csv(R/'summary/CE_diagnostics.csv');eu=pd.read_csv(R/'summary/CE_evaluation_uncertainty.csv');avg=v.groupby('x').mean(numeric_only=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
fig,axs=plt.subplots(1,2,figsize=(10,3.4))
for ax,series in zip(axs,[[('E3_low_gain','Low encoder','#286a9c'),('E3_high_gain','High encoder','#cc8538'),('E3_nested_gain','Nested encoders','#518864')],[('E4_neural_high_upgrade','Neural high - low','#286a9c'),('E4_ols_high_upgrade','OLS high - low','#cc8538'),('E4_ols_nested_upgrade','OLS nested - low','#518864')]]):
 for m,label,c in series:
  q=i[(i.metric==m)&(i.contrast=='level')].sort_values('x');ax.errorbar(q.x*100,q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],label=label,color=c,marker='o',capsize=3,lw=1.5)
 ax.axhline(0,color='.5',lw=.7);ax.set_xlabel('Adoption in frozen market (%)');ax.legend(frameon=False,fontsize=8)
axs[0].set_ylabel('Held-out dividend MSE reduction');axs[0].set_title('E3: incremental predictive value of price')
axs[1].set_ylabel('Gross certainty-equivalent upgrade');axs[1].set_title('E4: price-taking capacity value')
fig.tight_layout();fig.savefig(D/'E3_E4_results.png');plt.close(fig)
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyX',fontName='Sans',fontSize=9.1,leading=13,spaceAfter=8));styles.add(ParagraphStyle(name='TitleX',fontName='SansB',fontSize=22,leading=26,spaceAfter=15));styles.add(ParagraphStyle(name='HeadingX',fontName='SansB',fontSize=13,leading=17,spaceAfter=10));styles.add(ParagraphStyle(name='SmallX',fontName='Sans',fontSize=8,leading=11,spaceAfter=7))
story=[]
def p(s,sty='BodyX'):story.append(Paragraph(s,styles[sty]))
def table(rows,widths=None):
 rows=[[Paragraph(str(x),styles['SmallX']) for x in r] for r in rows];t=Table(rows,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#6e8596')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,10))
def estimate(metric,x,contrast='level'):
 q=i[(i.metric==metric)&(i.x==x)&(i.contrast==contrast)].iloc[0];return f"{q['mean']:.5f} [{q.ci_low:.5f}, {q.ci_high:.5f}]"
def footer(c,d):
 c.setFont('Sans',8);c.setFillColor(colors.HexColor('#647382'));c.drawString(42,27,'AI Trading | E3 / E4 | 29 October 2026');c.drawRightString(570,27,str(d.page))
p('E3 / E4: prediction and private value','TitleX')
p('Nine frozen E6 markets; independently refitted, negligible price-taking candidates.','HeadingX')
p('Three initial training seeds, adoption 0%, 50%, and 100%; market participants are fixed at the both-group E6 endpoint after six updates. All quantities below are finite-learning comparisons, not certified equilibrium or unrestricted information results.')
story.append(Image(str(D/'E3_E4_results.png'),width=526,height=179));story.append(Spacer(1,12))
table([['Primary estimate','0% adoption','50% adoption','100% adoption'],['E3 low price gain']+[f'{avg.loc[x,"E3_low_gain"]:.5f}' for x in [0,.5,1]],['E3 high price gain']+[f'{avg.loc[x,"E3_high_gain"]:.5f}' for x in [0,.5,1]],['E3 low minus high gain']+[f'{avg.loc[x,"E3_T_high"]:.5f}' for x in [0,.5,1]],['E4 neural CE: high - low']+[f'{avg.loc[x,"E4_neural_high_upgrade"]:.5f}' for x in [0,.5,1]]],[190,112,112,112])
p('Plot intervals are descriptive 95% Student-t intervals across only three paired initial seeds. The frozen laws differ across adoption levels; within each law, both capacities face exactly the same prices and private-signal realizations. Values are gross of computation costs.','SmallX')
p('Main result: price helps the low-capacity observer more; the upgrade premium falls','HeadingX')
p(f'From 0% to 100% adoption, the price-induced narrowing of the low/high prediction gap rises from {avg.loc[0,"E3_T_high"]:.5f} to {avg.loc[1,"E3_T_high"]:.5f}. The neural high-minus-low CE remains positive but falls by {100*(1-avg.loc[1,"E4_neural_high_upgrade"]/avg.loc[0,"E4_neural_high_upgrade"]):.1f}%. Both directions hold in all three seeds and in the nested auxiliary comparison.')
p('E3 measures how much current price improves a specified affine dividend predictor on held-out data. A larger low-encoder gain means that price narrows the low/high prediction gap. It does not isolate purely public information from private information aggregated into price. E4 measures the true ex-ante exponential utility of the fitted trading rule, evaluated under the simulated law. A positive high-minus-low difference favors the fitted high-capacity candidate at that market law.')
story.append(PageBreak())
p('E3: price-sharing measurement','TitleX')
p('Primary public information is zL or zH, without the observer\'s own private signal. For each fixed encoder, separate affine prediction heads are fitted with and without current price using an independent calibration bank. The target is realized next dividend, not the latent state.')
table([['Adoption','Low gain S','High gain D','Gap narrowing T = S - D']]+[[f'{x*100:.0f}%',estimate('E3_low_gain',x),estimate('E3_high_gain',x),estimate('E3_T_high',x)] for x in [0,.5,1]],[55,157,157,157])
p('Bracketed ranges: 95% t intervals over three seeds. Units are absolute dividend MSE. The exact accounting identity is T = (MSE_L,pre - MSE_H,pre) - (MSE_L,post - MSE_H,post). Relative gains have different denominators and are not used in this identity.','SmallX')
p('Adoption effect on gap narrowing','HeadingX')
table([['Change from 0%','Primary high encoder','Nested (low + high) encoder']]+[[f'{x*100:.0f}%',estimate('E3_T_high',x,'change_from_x0'),estimate('E3_T_nested',x,'change_from_x0')] for x in [.5,1]],[95,215,216])
p('The nested diagnostic explicitly contains both implemented representations. It requires both encoders and a new affine head; it is not the stand-alone width32 technology. Price may reduce uncertainty about both previously processed public information and private fundamentals. Neither a positive T nor its change alone proves a Gaussian information identity or a causal public-information channel.')
p('Estimation limits','HeadingX')
p('Only affine decoding is tested here. The representation includes nonlinear public encoders, but this loss reduction is not Bayes risk over all measurable decoders and is not mutual information. Negative out-of-sample gains would be retained. No observer input contains the current latent fundamental, current payoff innovation, or future payoff.')
p('Same-law retraining and evaluation','HeadingX')
p('Each market uses fresh candidate learning data. Candidates share a seed-specific E5 pre-adoption initialization across adoption levels, so starting from different E6 participant models cannot create an extra candidate learning-history effect. Both capacities receive 1,048,576 training windows and 4,096 optimizer updates. Width32 uses more parameters and computation. Calibration uses 262,144 observations; evaluation uses four independent banks totaling 1,048,576 observations per market.')
story.append(PageBreak())
p('E4: utility, calibration, and tails','TitleX')
p('Primary rules use the neural mean and variance fitted by Gaussian prediction likelihood. With h = (mu - Rf p)/(gamma v), CE = -log E[exp(-gamma h(D - Rf p))]/gamma. The independent current Gaussian dividend innovation is integrated analytically; all remaining expectations use held-out market paths. No normal-posterior variance formula is imposed on the nonlinear market.')
table([['Adoption','Neural high - low','OLS high - low','OLS nested - low']]+[[f'{x*100:.0f}%',estimate('E4_neural_high_upgrade',x),estimate('E4_ols_high_upgrade',x),estimate('E4_ols_nested_upgrade',x)] for x in [0,.5,1]],[55,157,157,157])
p('OLS variants refit an affine mean and constant residual variance on the same independent calibration bank. They are secondary diagnostics with additional fitting. Neither nesting nor prediction training guarantees better true utility under misspecification.','SmallX')
table([['Neural upgrade change from 0%','Paired seed estimate [95% interval]']]+[[f'{x*100:.0f}% adoption',estimate('E4_neural_high_upgrade',x,'change_from_x0')] for x in [.5,1]],[220,306])
primary=cd[cd.model.str.contains('neural')]
p('Numerical and sampling diagnostics','HeadingX')
p(f'Across the 18 primary capacity/market evaluations, the smallest exponential-weight effective sample size is {primary.ESS.min():,.0f} of 1,048,576. The largest single normalized weight is {primary.max_weight.max():.3%}; the largest top-0.1% weight share is {primary.top_01pct_weight.max():.2%}. ESS is a weight-concentration diagnostic, not a time-series effective sample size.')
p('Within-market uncertainty is estimated by 1,000 paired path bootstraps stratified across four evaluation banks; it is conditional on the fitted models. Independent-bank and first-half/last-half CE comparisons are retained in the result tables. These checks cannot prove finite exponential moments or rule out rarer tail losses beyond the simulated sample.')
p('Validation and scope','HeadingX')
p('Five measurement tests pass: Gaussian benchmark, analytic dividend-noise integration against Gauss-Hermite quadrature, zero-position CE, log-sum-exp stability, and E3 accounting/nested fitting. All nine market profiles remain unchanged; bank separation, sample budgets, and market clearing are audited. E3/E4 results do not establish retraining equilibrium, aggregate welfare, or net technology adoption after computing costs.')
SimpleDocTemplate(str(D/'E3_E4_Measurement_20261029.pdf'),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=44).build(story,onFirstPage=footer,onLaterPages=footer)
print(D/'E3_E4_Measurement_20261029.pdf')
