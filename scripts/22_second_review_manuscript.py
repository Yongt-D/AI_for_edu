"""Second-review presentation built from frozen predictions and course-stratified audit."""
from pathlib import Path
import runpy,re,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'src')
from edu_shift.revision_experiments import wilson
runpy.run_path('scripts/19_peer_manuscript.py')
out=Path('outputs/revision_v4');tables=out/'tables';tables.mkdir(exist_ok=True);fig=Path('figures/revision_v4');fig.mkdir(exist_ok=True)
names={'iid':'IID','temporal_1':'Temporal-1','temporal_2':'Temporal-2','course_1':'Course-1','course_2':'Course-2'}
models={'logistic':'Logistic','catboost':'CatBoost'}
def table(frame,name):
 frame.to_csv(tables/(name+'.csv'),index=False)
 esc=lambda v:str(v).replace('_',r'\_').replace('%',r'\%')
 lines=[r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\toprule',' & '.join(map(esc,frame.columns))+r'\\',r'\midrule']
 for row in frame.itertuples(index=False,name=None):
  lines.append(' & '.join('--' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else esc(v) for v in row)+r'\\')
 (tables/(name+'.tex')).write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}']),encoding='utf-8')
m=pd.read_csv('outputs/revision_v3/metrics.csv');primary=((m.model=='logistic')&(m.class_weight=='balanced'))|((m.model=='catboost')&(m.class_weight=='unweighted'))
z=m[primary&(m.scope=='full')&(m.calibration=='isotonic')&(m.method!='probability')]
for model in models:
 rows=[]
 for setting in names:
  for method in ['MCP','EQ','EIQ-batch']:
   g=z[(z.model==model)&(z.setting==setting)&(z.method==method)]
   rows.append({'Setting':names[setting],'Method':method,'Cov.':g.observed_coverage.mean(),'Fail':g.fail_coverage.mean(),'Review':1-g.automatic_coverage.mean(),'Error':g.automatic_error_rate.mean(),'Missed':g.automatic_false_negative_fraction.mean()})
 table(pd.DataFrame(rows),'operational_'+model)
ci=pd.read_csv(out/'target_intervals.csv');cm=pd.read_csv(out/'course_metrics.csv')
for model in models:
 rows=[]
 for setting in ['temporal_1','temporal_2']:
  for course in ['Accountancy','Global economics']:
   for method in ['MCP','EQ','EIQ-batch']:
    g=cm[(cm.model==model)&(cm.setting==setting)&(cm.course==course)&(cm.method==method)&(cm.calibration=='isotonic')]
    rows.append({'Setting':names[setting],'Course':'Acc.' if course=='Accountancy' else 'Econ.','Method':method,'N':int(g.n_test.iloc[0]),'Fail N':int(g.n_fail.iloc[0]),'Cov.':g.observed_coverage.mean(),'Fail':g.fail_coverage.mean(),'Review':g.review_fraction.mean(),'Error':g.automatic_error_rate.mean(),'Missed':g.automatic_false_negative_fraction.mean()})
 table(pd.DataFrame(rows),'course_'+model)
 rows=[]
 for setting in names:
  for method in ['MCP','EQ','EIQ-batch']:
   g=ci[(ci.model==model)&(ci.setting==setting)&(ci.course=='pooled')&(ci.method==method)].set_index('metric')
   fmt=lambda v:f"{100*g.loc[v,'estimate']:.1f} [{100*g.loc[v,'lower']:.1f}, {100*g.loc[v,'upper']:.1f}]"
   rows.append({'Setting':names[setting],'Method':method,'N / Fail':f"{int(g.n_test.iloc[0])} / {int(g.n_fail.iloc[0])}",'Fail [CI], %':fmt('fail_coverage'),'Error [CI], %':fmt('singleton_error')})
 table(pd.DataFrame(rows),'intervals_'+model)
external=pd.read_csv('outputs/revision_v2/oulad/metrics.csv');rows=[]
for setting in ['temporal_B','temporal_J','course_DDD']:
 for model in models:
  g=external[(external.setting==setting)&(external.model==model)&(external.calibration=='isotonic')]
  r=g[(g.kind=='conformal')&(g.method=='mondrian_cp')&(g.alpha==.1)].iloc[0]
  prob=g[g.kind=='probability'].iloc[0]
  rows.append({'Setting':{'temporal_B':'B','temporal_J':'J','course_DDD':'DDD'}[setting],'Model':models[model],'AUC':prob.roc_auc,'Cov.':r.observed_coverage,'Fail':r.fail_coverage,'Review':1-r.automatic_coverage,'Error':r.automatic_error_rate,'Missed':r.automatic_false_negative_fraction})
table(pd.DataFrame(rows),'oulad_main')

p=pd.read_parquet('outputs/revision_v3/predictions.parquet');pri=((p.model=='logistic')&(p.class_weight=='balanced'))|((p.model=='catboost')&(p.class_weight=='unweighted'))
p=p[pri&(p.scope=='full')&(p.method=='probability')&(p.seed==42)]
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#2364aa','#d97c11','#23885b']
def panel(ax,setting,model):
 for color,cal in zip(colors,['none','platt','isotonic']):
  g=p[(p.setting==setting)&(p.model==model)&(p.calibration==cal)].copy();g['bin']=np.minimum((g.probability*10).astype(int),9)
  xs=[];ys=[];lo=[];hi=[]
  for _,v in g.groupby('bin'):
   mean=v.risk.mean();lower,upper=wilson(v.risk.sum(),len(v));xs.append(v.probability.mean());ys.append(mean);lo.append(mean-lower);hi.append(upper-mean)
  ax.errorbar(xs,ys,yerr=np.maximum(0,[lo,hi]),fmt='.-',color=color,lw=1,capsize=2,label=cal)
 ax.plot([0,1],[0,1],'--',color='grey',lw=.8);ax.set(xlim=(0,1),ylim=(0,1),title=names[setting]+' / '+models[model],xlabel='Predicted failure probability',ylabel='Observed failure fraction')
figure,axes=plt.subplots(1,2,figsize=(7,3.2))
for ax,model in zip(axes,models):panel(ax,'temporal_2',model)
axes[0].legend(fontsize=8);figure.tight_layout();figure.savefig(fig/'calibration_temporal2.pdf',bbox_inches='tight');plt.close(figure)
figure,axes=plt.subplots(5,2,figsize=(7,11.5))
for row,setting in enumerate(names):
 for col,model in enumerate(models):panel(axes[row,col],setting,model)
axes[0,0].legend(fontsize=8);figure.tight_layout();figure.savefig(fig/'calibration_all.pdf',bbox_inches='tight');plt.close(figure)
figure,axes=plt.subplots(1,2,figsize=(7,3.5))
for ax,model in zip(axes,models):
 values=[]
 for setting in ['temporal_1','temporal_2']:
  for course in ['Accountancy','Global economics']:
   r=ci[(ci.setting==setting)&(ci.course==course)&(ci.model==model)&(ci.method=='MCP')&(ci.metric=='fail_coverage')].iloc[0];values.append(r)
 ax.errorbar(range(4),[v.estimate for v in values],yerr=[[v.estimate-v.lower for v in values],[v.upper-v.estimate for v in values]],fmt='o',capsize=4,color=colors[0]);ax.axhline(.9,ls='--',lw=.8,color='grey')
 ax.set(xticks=range(4),xticklabels=['T1 / Acc.','T1 / Econ.','T2 / Acc.','T2 / Econ.'],ylim=(0,1.04),title=models[model],ylabel='Failure-class coverage');ax.tick_params(axis='x',rotation=30)
figure.tight_layout();figure.savefig(fig/'course_intervals.pdf',bbox_inches='tight');plt.close(figure)


print('Course tables and calibration/interval figures generated.')
