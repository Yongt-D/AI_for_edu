"""Validate course decomposition, conditional estimates and generated manuscript claims."""
from pathlib import Path
import json,re,sys
import numpy as np,pandas as pd
sys.path.insert(0,'src')
from edu_shift.revision_experiments import set_metrics
out=Path('outputs/revision_v4');p=pd.read_parquet('outputs/revision_v3/predictions.parquet')
f=pd.read_parquet('data/processed_v2/features_week4.parquet')
primary=((p.model=='logistic')&(p.class_weight=='balanced'))|((p.model=='catboost')&(p.class_weight=='unweighted'))
p=p[primary&(p.scope=='full')&(p.method!='probability')].merge(f[['entity_id','course']],on='entity_id',validate='many_to_one')
c=pd.read_csv(out/'course_metrics.csv');lookup=c.set_index(['setting','model','calibration','method','course','seed']);checks=[]
for key,g in p[p.setting.isin(['temporal_1','temporal_2'])].groupby(['setting','model','calibration','method','course','seed']):
 actual=set_metrics([json.loads(s) for s in g.conformal_set],g.risk.to_numpy(),.1);row=lookup.loc[key]
 for metric,value in actual.items():assert np.isclose(value,row[metric],equal_nan=True),(key,metric)
checks.append(f'All {len(lookup)} course/model/calibration/method/seed groups independently recomputed from saved predictions')
m=pd.read_csv('outputs/revision_v3/metrics.csv');pri=((m.model=='logistic')&(m.class_weight=='balanced'))|((m.model=='catboost')&(m.class_weight=='unweighted'))
m=m[pri&(m.scope=='full')&(m.calibration=='isotonic')&(m.method!='probability')]
ci=pd.read_csv(out/'target_intervals.csv');mapping={'coverage':'observed_coverage','fail_coverage':'fail_coverage','automatic_coverage':'automatic_coverage','singleton_error':'automatic_error_rate','missed_failure':'automatic_false_negative_fraction'}
for r in ci.itertuples():
 g=m[(m.setting==r.setting)&(m.model==r.model)&(m.method==r.method)] if r.course=='pooled' else c[(c.setting==r.setting)&(c.course==r.course)&(c.model==r.model)&(c.method==r.method)&(c.calibration=='isotonic')]
 expected=1-g.automatic_coverage.mean() if r.metric=='review_fraction' else g[mapping[r.metric]].mean()
 assert np.isclose(expected,r.estimate,equal_nan=True)
 assert 0<=r.lower<=r.upper<=1 and r.finite_replicates==2000
 assert r.n_test==g.n_test.iloc[0] and r.n_fail==g.n_fail.iloc[0]
checks.append(f'All {len(ci)} conditional interval point estimates and denominators match source results; 2,000 finite replicates per interval')
for (setting,model,seed,method),g in c[c.calibration=='isotonic'].groupby(['setting','model','seed','method']):
 row=m[(m.setting==setting)&(m.model==model)&(m.seed==seed)&(m.method==method)].iloc[0]
 for metric,weight in [('observed_coverage','n_test'),('fail_coverage','n_fail'),('automatic_false_negative_fraction','n_fail')]:
  assert np.isclose(np.average(g[metric],weights=g[weight]),row[metric])
checks.append('Course-stratified coverage and missed-failure numerators reconstruct every pooled temporal estimate')
(out/'validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
