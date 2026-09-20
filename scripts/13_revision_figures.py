"""Generate all manuscript tables and figures from corrected saved predictions."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'src')
from edu_shift.revision_experiments import wilson
out=Path('outputs/revision_v2');fig=Path('figures/revision_v2');fig.mkdir(parents=True,exist_ok=True);tables=out/'tables';tables.mkdir(exist_ok=True)
m=pd.read_csv(out/'metrics.csv');p=pd.read_parquet(out/'predictions.parquet');ci=pd.read_csv(out/'auc_intervals.csv')
order=['iid','temporal_1','temporal_2','course_1','course_2'];names=dict(zip(order,['IID','Temporal-1','Temporal-2','Course-1','Course-2']))
methods=['mondrian_cp','empirical_mondrian','weighted_cap10'];labels={'mondrian_cp':'Mondrian','empirical_mondrian':'Unit-weight empirical','weighted_cap10':'Weighted empirical'}
colours=['#2364aa','#ef8a17','#239b7a'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
def save(figure,name):
    figure.savefig(fig/(name+'.pdf'));figure.savefig(fig/(name+'.png'),dpi=180);plt.close(figure)
def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(frame,name,formats=None):
    frame.to_csv(tables/(name+'.csv'),index=False)
    lines=[r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\toprule',' & '.join(esc(c) for c in frame.columns)+r'\\',r'\midrule']
    for row in frame.itertuples(index=False,name=None):
        vals=[]
        for v in row:
            vals.append('--' if pd.isna(v) else (f'{v:.3f}' if isinstance(v,(float,np.floating)) else esc(v)))
        lines.append(' & '.join(vals)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(tables/(name+'.tex')).write_text('\n'.join(lines),encoding='utf-8')

stats=pd.read_csv('data/processed_v2/dataset_statistics_week4.csv');ds=stats[['course','year','n_enrolments','failure_rate','n_zero_early_activity']].copy();ds.columns=['Course','Academic year','N','Failure rate','Zero activity'];table(ds,'dataset')
base=m[(m.kind=='probability')&(m.seed==42)&(m.calibration=='none')].merge(ci,on=['setting','model','roc_auc'])
base['setting']=pd.Categorical(base.setting,order);base=base.sort_values(['setting','model'])
pred=base[['setting','model','roc_auc','pr_auc','brier','ece']].copy();pred['setting']=pred.setting.map(names);pred['model']=pred.model.map({'logistic':'Logistic','catboost':'CatBoost'});pred.columns=['Setting','Model','AUC','AP','Brier','ECE'];table(pred,'predictive')
z=m[(m.kind=='conformal')&(m.seed==42)&(m.calibration=='isotonic')&(m.alpha==.1)&m.method.isin(methods)].copy()
z['setting']=pd.Categorical(z.setting,order);z=z.sort_values(['setting','model','method'])
main=z[(z.model=='logistic')&z.method.isin(['mondrian_cp','weighted_cap10'])][['setting','method','observed_coverage','pass_coverage','fail_coverage','automatic_coverage','automatic_error_rate']].copy()
main['setting']=main.setting.map(names);main['method']=main.method.map({'mondrian_cp':'MCP','weighted_cap10':'IW-M'})
main.columns=['Setting','Method','Coverage','Pass','Fail','Auto','Error'];table(main,'conformal_main')
# Supplement tables are compact and grouped by scientific purpose.
full=z[['setting','model','method','observed_coverage','average_set_size','automatic_coverage','automatic_error_rate']].copy();full['setting']=full.setting.map(names);full['method']=full.method.map({'mondrian_cp':'MCP','empirical_mondrian':'EMP','weighted_cap10':'IW-M'});full.columns=['Setting','Model','Method','Coverage','Size','Auto','Error'];table(full,'conformal_both')
part=pd.read_csv(out/'partitions.csv');counts=part[part.seed==42].groupby(['setting','partition']).risk.agg(N='size',Failures='sum').reset_index();table(counts,'partition_counts')
weights=z[(z.model=='logistic')&(z.method=='weighted_cap10')][['setting','domain_auc','effective_sample_size','ess_class_0','ess_class_1']].copy();weights['setting']=weights.setting.map(names);weights.columns=['Setting','Domain AUC','ESS','Pass ESS','Fail ESS'];table(weights,'weight_diagnostics')
ext=pd.read_csv(out/'oulad/metrics.csv');ext=ext[(ext.kind=='conformal')&(ext.alpha==.1)&(ext.calibration=='isotonic')&ext.method.isin(['mondrian_cp','weighted_cap10'])][['setting','model','method','observed_coverage','pass_coverage','fail_coverage','automatic_coverage','automatic_error_rate']].copy();ext['method']=ext.method.map({'mondrian_cp':'MCP','weighted_cap10':'IW-M'});ext.columns=['Setting','Model','Method','Coverage','Success','Non-success','Auto','Error'];table(ext,'external')
boot=pd.read_csv(out/'paired_bootstrap.csv');boot=boot[(boot.model=='logistic')&(boot.comparison=='weighted_cap10-minus-empirical_mondrian')&boot.metric.isin(['coverage','automatic_error'])][['setting','metric','estimate','ci_lower','ci_upper']].copy();table(boot,'bootstrap')
sens=pd.read_csv(out/'seed_sensitivity.csv');sens=sens[(sens.model=='logistic')&sens.method.isin(['mondrian_cp','weighted_cap10'])][['setting','method','observed_coverage_mean','observed_coverage_min','observed_coverage_max','automatic_error_rate_mean']].copy();sens['method']=sens.method.map({'mondrian_cp':'MCP','weighted_cap10':'IW-M'});sens.columns=['Setting','Method','Mean coverage','Min','Max','Mean error'];table(sens,'seed_sensitivity')
bud=pd.read_csv(out/'budgets.csv');bud=bud[(bud.model=='logistic')&(bud.calibration=='isotonic')&(bud.matched_method=='weighted_cap10')][['setting','policy','automatic_coverage','automatic_error_rate','automatic_false_negative_fraction']].copy();bud.columns=['Setting','Policy','Auto','Error','Missed failure fraction'];table(bud,'matched_budget')

figure,axes=plt.subplots(1,2,figsize=(9,3))
for i,model in enumerate(['logistic','catboost']):
    b=base[base.model==model].set_index('setting').loc[order];x=np.arange(5)+(i-.5)*.15
    axes[0].errorbar(x,b.roc_auc,yerr=[b.roc_auc-b.ci_lower,b.ci_upper-b.roc_auc],fmt='o',capsize=3,color=colours[i],label=model.title())
axes[0].set(xticks=range(5),xticklabels=[names[v] for v in order],ylabel='ROC-AUC',ylim=(.5,.8),title='a  Discrimination (95% bootstrap intervals)');axes[0].tick_params(axis='x',rotation=25);axes[0].legend(frameon=False)
q=z.drop_duplicates('setting').set_index('setting').loc[order];axes[1].bar(range(5),q.domain_auc,color=colours[0]);axes[1].axhline(.5,ls='--',color='grey');axes[1].set(xticks=range(5),xticklabels=[names[v] for v in order],ylabel='Domain ROC-AUC',ylim=(0,1),title='b  Source-target feature differences');axes[1].tick_params(axis='x',rotation=25);figure.tight_layout();save(figure,'figure1_performance_shift')

figure,axes=plt.subplots(2,3,figsize=(9,5.6),sharex=True,sharey=True)
for row,model in enumerate(['logistic','catboost']):
    for col,setting in enumerate(['iid','temporal_1','course_1']):
        ax=axes[row,col]
        for i,calibration in enumerate(['none','platt','isotonic']):
            v=p[(p.kind=='probability')&(p.seed==42)&(p.model==model)&(p.setting==setting)&(p.calibration==calibration)]
            v=v.assign(bin=np.minimum((v.probability*10).astype(int),9));xs=[];ys=[];lower=[];upper=[];ns=[]
            for _,g in v.groupby('bin'):
                mean=g.risk.mean();lo,hi=wilson(g.risk.sum(),len(g));xs.append(g.probability.mean());ys.append(mean);lower.append(mean-lo);upper.append(hi-mean);ns.append(len(g))
            ax.errorbar(xs,ys,yerr=np.maximum(0,[lower,upper]),fmt='.-',color=colours[i],lw=1,capsize=2,label=calibration)
        ax.plot([0,1],[0,1],'--',color='grey',lw=.8);ax.set(title=f'{names[setting]} / {model.title()}',xlim=(0,1),ylim=(0,1))
        if row==1:ax.set_xlabel('Mean predicted probability')
        if col==0:ax.set_ylabel('Observed failure fraction')
axes[0,0].legend(frameon=False,fontsize=8);figure.tight_layout();save(figure,'figure2_calibration')

figure,axes=plt.subplots(1,3,figsize=(10,3.3))
for i,method in enumerate(methods):
    q=z[(z.model=='logistic')&(z.method==method)].set_index('setting').loc[order];x=np.arange(5)+(i-1)*.17
    for j,metric in enumerate(['observed_coverage','fail_coverage','automatic_error_rate']):
        ax=axes[j];prefix=['coverage','fail','automatic_error'][j]
        ax.errorbar(x,q[metric],yerr=np.maximum(0,[q[metric]-q[prefix+'_ci_lower'],q[prefix+'_ci_upper']-q[metric]]),fmt='o',color=colours[i],capsize=2,label=labels[method])
for j,ax in enumerate(axes):
    ax.set(xticks=range(5),xticklabels=[names[v] for v in order],ylim=(.45,1.03) if j<2 else (0,.6),title=['a  Marginal set coverage','b  Failure-class set coverage','c  Error among singleton predictions'][j]);ax.tick_params(axis='x',rotation=35)
    if j<2:ax.axhline(.9,color='grey',ls='--',lw=.8)
handles,lab=axes[0].get_legend_handles_labels();figure.legend(handles,lab,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.06),frameon=False);figure.tight_layout();save(figure,'figure3_reliability')

rc=pd.read_csv(out/'curves.csv');figure,axes=plt.subplots(1,2,figsize=(8,3.3))
for ax,model in zip(axes,['logistic','catboost']):
    for setting in order:
        q=rc[(rc.model==model)&(rc.calibration=='isotonic')&(rc.setting==setting)];ax.plot(q.automatic_coverage,q.selective_error,label=names[setting])
    ax.set(xlabel='Retained prediction fraction',ylabel='Expected error among retained predictions',xlim=(0,1),ylim=(0,.65),title=model.title());ax.legend(fontsize=7,frameon=False)
figure.tight_layout();save(figure,'figureS1_risk_coverage')

cp=pd.read_csv(out/'checkpoint_robustness.csv');figure,axes=plt.subplots(1,2,figsize=(8,3.3))
for ax,metric in zip(axes,['roc_auc','ece']):
    for setting in order:
        q=cp[(cp.model=='logistic')&(cp.setting==setting)].sort_values('week');ax.plot(q.week,q[metric],'o-',label=names[setting])
    ax.set(xlabel='Completed teaching weeks',ylabel=metric.upper(),xticks=[2,4,6,8]);ax.legend(frameon=False,fontsize=7)
figure.tight_layout();save(figure,'figureS2_checkpoints')

figure,axes=plt.subplots(1,2,figsize=(8,3.3))
for ax,model in zip(axes,['logistic','catboost']):
    for i,method in enumerate(['mondrian_cp','weighted_cap5','weighted_cap10','weighted_cap20','weighted_p99']):
        q=m[(m.seed==42)&(m.model==model)&(m.calibration=='isotonic')&(m.alpha==.1)&(m.method==method)].set_index('setting').loc[order]
        ax.plot(range(5),q.observed_coverage,'o-',label=method.replace('weighted_',''))
    ax.set(xticks=range(5),xticklabels=[names[v] for v in order],ylabel='Marginal coverage',title=model.title(),ylim=(.45,1.02));ax.tick_params(axis='x',rotation=25);ax.axhline(.9,color='grey',ls='--');ax.legend(fontsize=7,frameon=False)
figure.tight_layout();save(figure,'figureS3_clipping')
print('Tables and figures generated',flush=True)
