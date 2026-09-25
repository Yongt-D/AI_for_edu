"""Generate post-review tables, figures and manuscript from frozen experiment outputs."""
from pathlib import Path
import runpy,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('outputs/revision_v3');tables=out/'tables';fig=Path('figures/revision_v3');fig.mkdir(exist_ok=True)
names={'iid':'IID','temporal_1':'Temporal-1','temporal_2':'Temporal-2','course_1':'Course-1','course_2':'Course-2'}
models={'logistic':'Logistic','catboost':'CatBoost'}
def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%')
def table(frame,name):
 frame.to_csv(tables/(name+'.csv'),index=False)
 lines=[r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\toprule',' & '.join(map(esc,frame.columns))+r'\\',r'\midrule']
 for row in frame.itertuples(index=False,name=None):
  lines.append(' & '.join('--' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else esc(v) for v in row)+r'\\')
 (tables/(name+'.tex')).write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}']),encoding='utf-8')
m=pd.read_csv(out/'metrics.csv');s=pd.read_csv(tables/'primary_seed_summary.csv')
primary=((m.model=='logistic')&(m.class_weight=='balanced'))|((m.model=='catboost')&(m.class_weight=='unweighted'))
z=m[primary&(m.scope=='full')&(m.calibration=='isotonic')&(m.method!='probability')]
for model in models:
 rows=[]
 for setting in names:
  for method in ['MCP','EQ','EIQ-batch']:
   r=s[(s.model==model)&(s.setting==setting)&(s.method==method)].iloc[0]
   rng=lambda metric:f"{r[metric+'_mean']:.3f} [{r[metric+'_min']:.3f}, {r[metric+'_max']:.3f}]"
   rows.append({'Setting':names[setting],'Method':method,'Coverage [range]':rng('observed_coverage'),'Fail [range]':rng('fail_coverage'),'Auto':r.automatic_coverage_mean,'Error':r.automatic_error_rate_mean})
 table(pd.DataFrame(rows),'primary_'+model)
rows=[]
for setting in names:
 for model in models:
  g=m[primary&(m.scope=='full')&(m.calibration=='none')&(m.method=='probability')&(m.setting==setting)&(m.model==model)]
  rows.append({'Setting':names[setting],'Model':models[model],'AUC':g.roc_auc.mean(),'AUC min':g.roc_auc.min(),'AUC max':g.roc_auc.max(),'Brier':g.brier.mean(),'ECE':g.ece.mean()})
table(pd.DataFrame(rows),'predictive_seeds')
part=pd.read_csv('outputs/revision_v2/partitions.csv');part=part[(part.seed==42)&(part.partition=='conformal_score')]
q=part.groupby('setting').risk.agg(['count','sum']).reindex(names);table(pd.DataFrame({'Setting':[names[v] for v in q.index],'Pass':q['count'].to_numpy()-q['sum'].to_numpy(),'Fail':q['sum'].to_numpy(),'Total':q['count'].to_numpy()}),'calibration_counts')
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
figure,axes=plt.subplots(2,3,figsize=(7.2,5.8))
colors=['#2364aa','#d97c11','#23885b']
for row,model in enumerate(models):
 for col,(metric,label) in enumerate([('observed_coverage','Marginal coverage'),('fail_coverage','Failure-class coverage'),('automatic_error_rate','Singleton error')]):
  ax=axes[row,col]
  for k,method in enumerate(['MCP','EQ','EIQ-batch']):
   for j,setting in enumerate(names):
    vals=z[(z.model==model)&(z.method==method)&(z.setting==setting)].sort_values('seed')[metric].to_numpy()
    x=j+(k-1)*.24;ax.scatter(x+np.linspace(-.04,.04,len(vals)),vals,s=16,alpha=.7,color=colors[k],label=method if j==0 else None)
    ax.plot([x-.07,x+.07],[vals.mean()]*2,color=colors[k],lw=2)
  if col<2:ax.axhline(.9,color='black',ls='--',lw=.8)
  ax.set_xticks(range(5),names.values(),rotation=35,ha='right');ax.set_title(models[model]+'\n'+label);ax.set_ylim((.45,1.025) if col<2 else (0,.6))
axes[0,0].legend(fontsize=7,loc='lower left');figure.tight_layout();figure.savefig(fig/'figure1_seed_reliability.pdf',bbox_inches='tight');figure.savefig(fig/'figure1_seed_reliability.png',dpi=160,bbox_inches='tight');plt.close(figure)

ad=pd.read_csv(tables/'adaptation_contrasts.csv');rows=[]
for setting in names:
 for model in models:
  g=ad[(ad.setting==setting)&(ad.model==model)].set_index('metric')
  rows.append({'Setting':names[setting],'Model':models[model],'Batch cov.':g.loc['observed_coverage','batch'],'Indep. cov.':g.loc['observed_coverage','independent'],'Batch err.':g.loc['automatic_error_rate','batch'],'Indep. err.':g.loc['automatic_error_rate','independent']})
table(pd.DataFrame(rows),'adaptation_display')
cw=pd.read_csv(tables/'class_weight_contrasts.csv');rows=[]
for setting in names:
 for model in models:
  g=cw[(cw.setting==setting)&(cw.model==model)&(cw.method=='MCP')].set_index('metric')
  rows.append({'Setting':names[setting],'Model':models[model],'Unw. fail':g.loc['fail_coverage','unweighted'],'Bal. fail':g.loc['fail_coverage','balanced'],'Unw. error':g.loc['automatic_error_rate','unweighted'],'Bal. error':g.loc['automatic_error_rate','balanced']})
table(pd.DataFrame(rows),'weight_display')
zs=pd.read_csv(tables/'zero_activity_summary.csv');rows=[]
for setting in names:
 for model in models:
  for method in ['MCP','EIQ-batch']:
   r=zs[(zs.setting==setting)&(zs.model==model)&(zs.method==method)&(zs.zero_activity==1)].iloc[0]
   rows.append({'Setting':names[setting],'Model':models[model],'Method':method,'N':int(r.n_test),'Fail N':int(r.n_fail),'Cov.':r.observed_coverage,'Auto':r.automatic_coverage,'Error':r.automatic_error_rate})
table(pd.DataFrame(rows),'zero_display')
cal=pd.read_csv(tables/'calibration_summary.csv')
for model in models:
 rows=[]
 for setting in names:
  for transform in ['none','platt','isotonic']:
   g=cal[(cal.model==model)&(cal.setting==setting)&(cal.calibration==transform)].set_index('method')
   rows.append({'Setting':names[setting],'Cal.':transform,'Brier':g.loc['probability','brier'],'Slope':g.loc['probability','calibration_slope'],'Intercept':g.loc['probability','calibration_intercept'],'MCP fail':g.loc['MCP','fail_coverage'],'EIQ fail':g.loc['EIQ-batch','fail_coverage']})
 table(pd.DataFrame(rows),'calibration_'+model)
cl=pd.read_csv(tables/'oulad_cluster_intervals.csv');rows=[]
for (setting,model,method),g in cl.groupby(['setting','model','method']):
 g=g.set_index('metric');fmt=lambda key:f"{g.loc[key,'estimate']:.3f} [{g.loc[key,'lower']:.3f}, {g.loc[key,'upper']:.3f}]"
 rows.append({'Setting':setting,'Model':models[model],'Method':{'mondrian_cp':'MCP','empirical_mondrian':'EQ','weighted_cap10':'EIQ-batch'}[method],'Coverage [CI]':fmt('coverage'),'Error [CI]':fmt('automatic_error')})
table(pd.DataFrame(rows),'cluster_display')

# Regenerate the previous complete draft, then apply an explicit reproducible revision.

print('Factorial tables and seed-reliability figure generated.')
