"""Second-review stratification and conditional target bootstrap; no new fitting."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
sys.path.insert(0,'src')
from edu_shift.revision_experiments import set_metrics
out=Path('outputs/revision_v4');out.mkdir(exist_ok=True)
protocol={'scope':'Post-review analyses of frozen predictions; no new training or target-label tuning',
 'seeds':[42,7,19,73,101],'bootstrap_replicates':2000,'bootstrap_seed':20260920,
 'bootstrap_unit':'KU Leuven enrolment; same resampling counts for all five seed predictions and methods within a cohort',
 'estimand':'Arithmetic mean of five fitted-model performance metrics on the target distribution represented by the observed cohort',
 'limitations':'Conditional on fitted models, source partitions and estimated weights; enrolment exchangeability approximation; no cross-environment person linkage or institutional generalisation',
 'course_analysis':'Stratify predictions of the pooled-source models; not course-specific model retraining'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
p=pd.read_parquet('outputs/revision_v3/predictions.parquet')
f=pd.read_parquet('data/processed_v2/features_week4.parquet')
primary=((p.model=='logistic')&(p.class_weight=='balanced'))|((p.model=='catboost')&(p.class_weight=='unweighted'))
p=p[primary&(p.scope=='full')&(p.method!='probability')].merge(f[['entity_id','course']],on='entity_id',validate='many_to_one')
rows=[]
for key,g in p[p.setting.isin(['temporal_1','temporal_2'])].groupby(['setting','model','calibration','method','course','seed']):
 r=set_metrics([json.loads(v) for v in g.conformal_set],g.risk.to_numpy(),.1)
 rows.append(dict(zip(['setting','model','calibration','method','course','seed'],key),**r,review_fraction=1-r['automatic_coverage']))
pd.DataFrame(rows).to_csv(out/'course_metrics.csv',index=False)
rows=[];rng=np.random.default_rng(20260920)
metrics=['coverage','fail_coverage','automatic_coverage','review_fraction','singleton_error','missed_failure']
for setting,gg in p[p.calibration=='isotonic'].groupby('setting'):
 cohorts=[('pooled',gg)]+([(course,v) for course,v in gg.groupby('course')] if setting.startswith('temporal') else [])
 for course,cohort in cohorts:
  ids=np.array(sorted(cohort.entity_id.unique()));n=len(ids)
  counts=rng.multinomial(n,np.full(n,1/n),size=2000).astype(float)
  for (model,method),g in cohort.groupby(['model','method']):
   covered=[];automatic=[];errors=[];missed=[]
   for seed in protocol['seeds']:
    v=g[g.seed==seed].set_index('entity_id').loc[ids]
    assert len(v)==n and v.index.is_unique
    y=v.risk.to_numpy(int);sets=[json.loads(s) for s in v.conformal_set]
    auto=np.array([len(s)==1 for s in sets]);pred=np.array([s[0] if len(s)==1 else -1 for s in sets])
    covered.append(np.array([yy in s for yy,s in zip(y,sets)]));automatic.append(auto);errors.append(auto&(pred!=y));missed.append(auto&(pred==0)&(y==1))
   cov=np.array(covered,dtype=float).T;auto=np.array(automatic,dtype=float).T;err=np.array(errors,dtype=float).T;fn=np.array(missed,dtype=float).T
   fail=(y==1).astype(float);nf=fail.sum();bs_fail=counts@fail
   bs_cov=counts@cov/n;bs_auto=counts@auto;bs_error=counts@err
   def divide(a,b):return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
   boots=[bs_cov.mean(1),divide(counts@(cov*fail[:,None]),bs_fail[:,None]).mean(1),(bs_auto/n).mean(1),1-(bs_auto/n).mean(1),np.nanmean(divide(bs_error,bs_auto),axis=1),divide(counts@fn,bs_fail[:,None]).mean(1)]
   points=[cov.mean(),(cov*fail[:,None]).sum(0).mean()/nf,auto.mean(),1-auto.mean(),np.nanmean(divide(err.sum(0),auto.sum(0))),fn.sum(0).mean()/nf]
   for metric,value,boot in zip(metrics,points,boots):
    rows.append(dict(setting=setting,course=course,model=model,method=method,metric=metric,estimate=value,lower=np.nanquantile(boot,.025),upper=np.nanquantile(boot,.975),n_test=n,n_fail=int(nf),n_automatic_mean=auto.sum(0).mean(),n_automatic_min=int(auto.sum(0).min()),n_automatic_max=int(auto.sum(0).max()),finite_replicates=int(np.isfinite(boot).sum())))
  print(setting,course,'conditional intervals completed',flush=True)
pd.DataFrame(rows).to_csv(out/'target_intervals.csv',index=False)
paths=[Path(__file__),Path('outputs/revision_v3/predictions.parquet'),Path('data/processed_v2/features_week4.parquet')]
(out/'input_sha256.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},indent=2))
