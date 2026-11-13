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
for n,f in [('Sans','DejaVuSans.ttf'),('SansB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(n,str(R/'e3_e4_measurement/fonts'/f)))
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansB',italic='Sans',boldItalic='SansB')
l=pd.read_csv(S/'learning_levels.csv');lc=pd.read_csv(S/'learning_changes.csv');d=pd.read_csv(S/'decoder_levels.csv');dc=pd.read_csv(S/'decoder_changes.csv');seeds=pd.read_csv(S/'learning_seed_changes.csv');ds=pd.read_csv(S/'decoder_seed_changes.csv');ce=pd.read_csv(S/'CE_diagnostics.csv');audit=json.loads((S/'VALIDATION.json').read_text())
variants=['original_1x','original_2x','original_4x','newrep1_1x','newrep2_1x'];names={'original_1x':'Original 1x','original_2x':'Continued 2x','original_4x':'Continued 4x','newrep1_1x':'Fresh refit 1','newrep2_1x':'Fresh refit 2'}
cols=['#286a9c','#508965','#b5802c','#925e96','#717c89']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
fig,axs=plt.subplots(1,2,figsize=(10,3.5))
for ax,metric in zip(axs,['E3_T_high','E4_neural_high_upgrade']):
 for v,c in zip(variants,cols):
  q=l[(l.variant==v)&(l.metric==metric)].sort_values('x');ax.plot(q.x*100,q['mean'],'o-',color=c,ms=4,lw=1.3,label=names[v])
 ax.set_xlabel('Adoption in frozen market (%)');ax.axhline(0,color='.65',lw=.7)
axs[0].set(title='E3: price-induced prediction-gap narrowing',ylabel='Absolute dividend MSE');axs[1].set(title='E4: gross high-minus-low CE',ylabel='Payoff-equivalent value');axs[0].legend(frameon=False,fontsize=7);fig.tight_layout();fig.savefig(D/'E8_learning.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.3))
for size,c,label in [(0,cols[0],'Matched affine ridge'),(64,cols[1],'RFF 64'),(128,cols[2],'RFF 128')]:
 for ax,metric in zip(axs,['E3_T_high','E3_T_nested']):
  q=d[(d['size']==size)&(d.metric==metric)].sort_values('x');ax.errorbar(q.x*100,q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],marker='o',color=c,label=label,capsize=3,lw=1.3)
for ax in axs:ax.axhline(0,color='.65',lw=.7);ax.set_xlabel('Adoption in frozen market (%)')
axs[0].set(title='Low vs high encoder',ylabel='Price-induced MSE-gap narrowing');axs[1].set_title('Low vs explicitly nested encoders');axs[0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(D/'E8_decoders.png');plt.close(fig)
styles=getSampleStyleSheet()
for n,font,size,lead in [('BodyX','Sans',9.1,13),('TitleX','SansB',21,25),('HeadingX','SansB',12,16),('SmallX','Sans',8,11)]:styles.add(ParagraphStyle(name=n,fontName=font,fontSize=size,leading=lead,spaceAfter=8))
story=[]
def p(s,sty='BodyX'):story.append(Paragraph(s,styles[sty]))
def tbl(rows,widths):
 t=Table([[Paragraph(str(x),styles['SmallX']) for x in r] for r in rows],colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#738591')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
def est(df,metric,**filters):
 q=df[df.metric==metric]
 for k,v in filters.items():q=q[q[k]==v]
 r=q.iloc[0];return f"{r['mean']:.5f} [{r.ci_low:.5f}, {r.ci_high:.5f}]"
def foot(c,d):
 c.setFont('Sans',8);c.setFillColor(colors.HexColor('#647382'));c.drawString(42,27,'AI Trading | E8 learning and measurement robustness | 29 October 2026');c.drawRightString(570,27,str(d.page))
p('E8: learning and measurement robustness','TitleX')
p('Fixed market laws; changed learning procedures and prediction decoders.','HeadingX')
p('The nine E6 endpoint markets remain unchanged. Adoption denotes the high-capacity share within group 1, which is 55% of the population. Five candidate-learning conditions are evaluated on four fresh common shock banks. The two extra refits change training data and optimizer shuffling; they do not create additional independent market seeds. Three decoder specifications test whether E3 depends on a linear readout.')
story.append(Image(str(D/'E8_learning.png'),width=526,height=184));story.append(Spacer(1,8))
tbl([['Learning condition','E3 T: change from 0% to 100%','E4 CE upgrade: change from 0% to 100%']]+[[names[v],est(lc,'E3_T_high',variant=v,x=1),est(lc,'E4_neural_high_upgrade',variant=v,x=1)] for v in variants],[130,198,198])
p('T = price gain for the low encoder minus price gain for the high encoder. Brackets are descriptive 95% Student-t intervals across the same three market seeds, df = 2; refits and individual dates are not counted as independent seeds.','SmallX')
ss=seeds[seeds.x==1];good3=int((ss.E3_T_high>0).sum());good4=int((ss.E4_neural_high_upgrade<0).sum())
p('Sign stability and effect size','HeadingX')
p(f'Across the 15 seed-by-learning-condition endpoint contrasts, E3 T increases in {good3}/15 and the E4 upgrade falls in {good4}/15. These counts describe correlated robustness comparisons, not 15 independent replications. The curves and tables retain all prespecified variants, including changes in magnitude.')
p('The multipliers count new training after the common E5 warm start, excluding inherited training. The 2x/4x comparisons continue prediction training on new data from the same law. Each additional block restarts the original Adam schedule. Thus the exercise tests a practical finite-learning procedure, not global optimality or a single smoothly annealed optimizer trajectory.')
story.append(PageBreak())
p('E3: sensitivity to the prediction decoder','TitleX')
p('The original 1x encoders are held fixed. The matched affine-ridge control and random Fourier feature (RFF) decoders use exactly the same 65,536 calibration observations, selected as the first 32 original calibration paths before evaluation. Ridge penalties are selected using a separate validation bank. Full-sample OLS remains the primary benchmark on page 1.')
story.append(Image(str(D/'E8_decoders.png'),width=526,height=174));story.append(Spacer(1,10))
tbl([['Decoder','Low vs high: change in T','Low vs nested: change in T']]+[[label,est(dc,'E3_T_high',size=s,x=1),est(dc,'E3_T_nested',size=s,x=1)] for s,label in [(0,'Affine ridge'),(64,'RFF64'),(128,'RFF128')]],[110,208,208])
p('Every price-augmented decoder retains all public-only features. The nested decoder retains the low decoder\'s raw and nonlinear features plus joint low/high features; it requires both encoders and additional computation. The high-only encoder is a separate primary comparison.','SmallX')
dss=ds[ds.x==1]
p(f'The high-only comparison has a positive adoption change in T in {int((dss.E3_T_high>0).sum())}/9 seed-by-decoder cases; the explicitly nested comparison in {int((dss.E3_T_nested>0).sum())}/9. These are finite prediction-loss results, not mutual information or a decomposition isolating the public-information channel.')
p('What this check can establish','HeadingX')
p('A sign that survives richer supervised decoding is less likely to be specific to an affine readout. A change in size is informative: price value is conditional on the implemented decoder and its finite data budget. Neither 64 nor 128 random features approximates all measurable predictors, so agreement cannot establish an unrestricted information theorem.')
p('No test-set tuning','HeadingX')
p('The RFF maps have fixed seeds. Ridge strengths are selected from 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, and 1 using validation only. All prespecified feature sizes are reported. The low observer receives no own private signal; latent fundamentals and future dividends never enter its features. The fitted heads and selected penalties are retained for audit.')
story.append(PageBreak())
p('E4: magnitude, diagnostics, and limits','TitleX')
tbl([['Learning condition','Upgrade at 0% adoption','Upgrade at 100% adoption']]+[[names[v],est(l,'E4_neural_high_upgrade',variant=v,x=0),est(l,'E4_neural_high_upgrade',variant=v,x=1)] for v in variants],[130,198,198])
p('Primary E4 uses neural means and variances trained for prediction. The trading rule is h=(mu-Rf p)/(gamma v); its ex-ante certainty equivalent is evaluated under the true simulated payoff law, with the independent Gaussian dividend innovation integrated analytically. Values exclude computation costs. OLS-recalibrated and nested-candidate checks are included in the accompanying result tables.')
primary=ce[ce.model.str.contains('neural')]
p('Numerical reliability','HeadingX')
p(f'Across the 90 primary capacity-by-market-by-learning evaluations, the smallest exponential-weight ESS is {primary.ESS.min():,.0f} of 1,048,576. The largest single weight is {primary.max_weight.max():.4%}; the largest top-0.1% weight share is {primary.top_01pct_weight.max():.2%}. These are concentration measures, not effective sample sizes correcting serial correlation. Conditional CE uncertainty uses 500 paired path bootstraps stratified by evaluation bank.')
p('Audit','HeadingX')
p(f'The run contains {audit["learning_evaluations"]} learning-condition evaluations and {audit["decoder_evaluations"]} decoder-size evaluations across nine immutable markets. All simulation fingerprints and training/evaluation bank separation checks pass. Maximum market-clearing residual: {audit["max_clearing_error"]:.2e}. Four new nesting/decoder tests and five inherited measurement tests pass. Archive contents are checksum verified.')
p('Scope of the evidence','HeadingX')
p('This E8 exercise addresses learning-data variation, optimization schedules, computational budgets, decoder flexibility, and fresh evaluation samples. The economic calibration, market-generation architecture, three original market seeds, and six-update E6 endpoints remain fixed. Therefore it does not show that the market-level crowd-out result survives changing supply noise, private-signal quality, risk aversion, or population composition. It also does not certify convergence of retraining or rule out rare utility-tail events outside the simulated sample.')
p('The evidence should be described as finite-learning numerical coexistence of price-induced prediction-gap narrowing and a declining gross capacity premium, conditional on these laws. E9 should examine the learning dynamics; broader economic and model-class changes require separate experiments.')
SimpleDocTemplate(str(D/'E8_Learning_and_Measurement_Robustness_20261029.pdf'),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=44,title='E8: Learning and Measurement Robustness',author='AI Trading Research').build(story,onFirstPage=foot,onLaterPages=foot)
print(D/'E8_Learning_and_Measurement_Robustness_20261029.pdf')
