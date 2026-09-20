"""Check post-review predictions, primary reproducibility and evaluation separation."""
from pathlib import Path
import sys,json,re
import numpy as np,pandas as pd
sys.path.insert(0,'src')
from edu_shift.revision_experiments import set_metrics
from edu_shift.metrics import evaluate
out=Path('outputs/revision_v3');m=pd.read_csv(out/'metrics.csv');p=pd.read_parquet(out/'predictions.parquet')
keys=['setting','seed','model','class_weight','calibration','scope','method'];lookup=m.set_index(keys);checks=[]
for key,g in p.groupby(keys):
 assert not g.entity_id.duplicated().any()
 row=lookup.loc[key];assert isinstance(row,pd.Series)
 if key[-1]=='probability':values=evaluate(g.risk.to_numpy(),g.probability.to_numpy())
 else:values=set_metrics([json.loads(v) for v in g.conformal_set],g.risk.to_numpy(),.1)
 for metric,value in values.items():
  assert np.isclose(value,row[metric],equal_nan=True),(key,metric,value,row[metric])
checks.append('All saved post-review set/probability metrics independently recomputed from prediction rows')
old=pd.read_csv('outputs/revision_v2/metrics.csv');primary=((m.model=='logistic')&(m.class_weight=='balanced'))|((m.model=='catboost')&(m.class_weight=='unweighted'))
new=m[primary&(m.scope=='full')&(m.calibration=='isotonic')&(m.method!='probability')].copy()
mapping={'MCP':'mondrian_cp','EQ':'empirical_mondrian','EIQ-batch':'weighted_cap10'}
new['method']=new.method.map(mapping)
joined=new.merge(old[(old.kind=='conformal')&(old.calibration=='isotonic')&(old.alpha==.1)],on=['setting','model','seed','calibration','method'],suffixes=('_new','_old'),validate='one_to_one')
assert len(joined)==150
for col in ['observed_coverage','fail_coverage','automatic_error_rate','automatic_coverage']:
 assert np.allclose(joined[col+'_new'],joined[col+'_old'],equal_nan=True),col
checks.append('All 150 inherited primary method/model/seed results reproduced')
a=pd.read_csv(out/'adaptation_partitions.csv');part=pd.read_csv('outputs/revision_v2/partitions.csv')
for setting,g in a.groupby('setting'):
 assert not g.entity_id.duplicated().any()
 target=set(part[(part.setting==setting)&(part.seed==42)&(part.partition=='test')].entity_id)
 assert set(g.entity_id)==target
 held=set(g[g.role=='evaluation'].entity_id)
 for _,v in p[(p.setting==setting)&(p.scope=='heldout')].groupby(keys):assert set(v.entity_id)==held
 for seed,v in part[part.setting==setting].groupby('seed'):
  assert not set(v[v.partition!='test'].entity_id)&target
checks.append('Five label-independent adaptation/evaluation partitions disjoint; identical held-out subjects in every paired comparison')
for model in ['logistic','catboost']:
 v=p[(((p.model=='logistic')&(p.class_weight=='balanced'))|((p.model=='catboost')&(p.class_weight=='unweighted')))&(p.model==model)&(p.setting=='temporal_1')&(p.scope=='heldout')&(p.calibration=='isotonic')]
 w=v[v.method=='EIQ-batch'].set_index(['entity_id','seed']).sort_index();i=v[v.method=='EIQ-independent'].set_index(['entity_id','seed']).sort_index()
 assert w.conformal_set.equals(i.conformal_set)
checks.append('Reported Temporal-1 identical batch/independent sets verified across both models and all seeds')
cluster=pd.read_csv(out/'tables/oulad_cluster_intervals.csv');external=pd.read_csv('outputs/revision_v2/oulad/metrics.csv')
for row in cluster.itertuples():
 original=external[(external.kind=='conformal')&(external.alpha==.1)&(external.calibration=='isotonic')&(external.setting==row.setting)&(external.model==row.model)&(external.method==row.method)].iloc[0]
 metric={'coverage':'observed_coverage','automatic_error':'automatic_error_rate'}.get(row.metric,row.metric)
 assert np.isclose(row.estimate,original[metric],equal_nan=True)
 assert 0<=row.lower<=row.upper<=1
checks.append('All OULAD cluster-bootstrap point estimates match original target performance; interval bounds are valid')
# Quantify ECE sensitivity without retraining or selecting bins by results.
rows=[]
for key,g in p[(p.scope=='full')&(p.method=='probability')].groupby(['setting','seed','model','class_weight','calibration']):
 for n in [5,10,15]:
  prob=g.probability.to_numpy();y=g.risk.to_numpy();idx=np.minimum((prob*n).astype(int),n-1);value=0
  for k in range(n):
   mask=idx==k
   if mask.any():value+=mask.mean()*abs(prob[mask].mean()-y[mask].mean())
  rows.append(dict(zip(['setting','seed','model','class_weight','calibration'],key),bins=n,ece=value))
pd.DataFrame(rows).to_csv(out/'tables/ece_binning_sensitivity.csv',index=False)
(out/'validation.json').write_text(json.dumps({'checks':checks,'groups':len(lookup),'primary_reproduction_groups':len(joined)},indent=2))
print(json.dumps(checks,indent=2))
