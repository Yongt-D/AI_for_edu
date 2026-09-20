"""Post-review summaries; seed variability and person-clustered sampling are distinct."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,'src')
from edu_shift.revision_experiments import set_metrics
out=Path('outputs/revision_v3');tables=out/'tables';tables.mkdir(exist_ok=True)
m=pd.read_csv(out/'metrics.csv');p=pd.read_parquet(out/'predictions.parquet')
primary=((m.model=='logistic')&(m.class_weight=='balanced'))|((m.model=='catboost')&(m.class_weight=='unweighted'))
z=m[primary&(m.calibration=='isotonic')&(m.scope=='full')&(m.method!='probability')]
fields=['observed_coverage','pass_coverage','fail_coverage','automatic_coverage','automatic_error_rate']
summary=z.groupby(['setting','model','method'])[fields].agg(['mean','min','max','std']);summary.columns=['_'.join(c) for c in summary.columns]
summary.reset_index().to_csv(tables/'primary_seed_summary.csv',index=False)
rows=[]
for (setting,model),g in z.groupby(['setting','model']):
 a=g[g.method=='EIQ-batch'].set_index('seed');b=g[g.method=='EQ'].set_index('seed')
 for metric in fields:
  v=a[metric]-b[metric];rows.append(dict(setting=setting,model=model,metric=metric,mean=v.mean(),minimum=v.min(),maximum=v.max(),positive=int((v>1e-12).sum()),negative=int((v< -1e-12).sum()),zero=int((abs(v)<=1e-12).sum())))
pd.DataFrame(rows).to_csv(tables/'paired_seed_effects.csv',index=False)

# Paired same-seed contrasts; no inference pretending seeds are independent cohorts.
rows=[]
for (setting,model,method),g in m[(m.calibration=='isotonic')&(m.scope=='full')&(m.method!='probability')].groupby(['setting','model','method']):
 a=g[g.class_weight=='balanced'].set_index('seed');b=g[g.class_weight=='unweighted'].set_index('seed')
 for metric in fields:
  d=a[metric]-b[metric];rows.append(dict(setting=setting,model=model,method=method,metric=metric,balanced=a[metric].mean(),unweighted=b[metric].mean(),delta=d.mean(),minimum=d.min(),maximum=d.max()))
pd.DataFrame(rows).to_csv(tables/'class_weight_contrasts.csv',index=False)
rows=[]
for (setting,model),g in m[primary&(m.scope=='heldout')&(m.calibration=='isotonic')].groupby(['setting','model']):
 a=g[g.method=='EIQ-independent'].set_index('seed');b=g[g.method=='EIQ-batch'].set_index('seed')
 for metric in fields:
  d=a[metric]-b[metric];rows.append(dict(setting=setting,model=model,metric=metric,independent=a[metric].mean(),batch=b[metric].mean(),delta=d.mean(),minimum=d.min(),maximum=d.max()))
pd.DataFrame(rows).to_csv(tables/'adaptation_contrasts.csv',index=False)
m[primary&(m.scope=='full')].groupby(['setting','model','calibration','method'])[[c for c in ['observed_coverage','fail_coverage','automatic_error_rate','brier','ece','calibration_slope','calibration_intercept'] if c in m]].mean().reset_index().to_csv(tables/'calibration_summary.csv',index=False)

pp=p[(((p.model=='logistic')&(p.class_weight=='balanced'))|((p.model=='catboost')&(p.class_weight=='unweighted')))&(p.calibration=='isotonic')&(p.scope=='full')&(p.method!='probability')]
rows=[]
for keys,g in pp.groupby(['setting','model','method','seed','zero_activity']):
 sets=[json.loads(v) for v in g.conformal_set];r=set_metrics(sets,g.risk.to_numpy(),.1)
 rows.append(dict(zip(['setting','model','method','seed','zero_activity'],keys),**r,failure_rate=g.risk.mean()))
sub=pd.DataFrame(rows);sub.to_csv(out/'subgroup_metrics.csv',index=False)
sub.groupby(['setting','model','method','zero_activity'])[['n_test','n_fail','failure_rate']+fields+['average_set_size','empty_set_rate','ambiguity_rate','n_automatic']].mean().reset_index().to_csv(tables/'zero_activity_summary.csv',index=False)

# Cluster-bootstrap target persons, preserving all enrolments and paired predictions.
ep=pd.read_parquet('outputs/revision_v2/oulad/predictions.parquet')
ep=ep[(ep.kind=='conformal')&(ep.calibration=='isotonic')&(ep.alpha==.1)&ep.method.isin(['mondrian_cp','empirical_mondrian','weighted_cap10'])]
rows=[];deltas=[];counts=[];rng=np.random.default_rng(20260916)
for (setting,model),g in ep.groupby(['setting','model']):
 methods={name:v.sort_values('entity_id').reset_index(drop=True) for name,v in g.groupby('method')}
 base=methods['mondrian_cp'];codes,persons=pd.factorize(base.student_id);n=len(persons);ns=np.bincount(codes)
 draws=rng.integers(0,n,size=(2000,n));boot={};point={}
 counts.append(dict(setting=setting,model=model,enrolments=len(base),persons=n,repeated_persons=int((ns>1).sum()),maximum_enrolments=int(ns.max())))
 for name,v in methods.items():
  assert v.entity_id.equals(base.entity_id)
  y=v.risk.to_numpy(int);sets=[json.loads(s) for s in v.conformal_set];cover=np.array([yy in s for yy,s in zip(y,sets)]);auto=np.array([len(s)==1 for s in sets]);pred=np.array([s[0] if len(s)==1 else -1 for s in sets]);err=auto&(pred!=y)
  def ratio(a,b):
   aa=np.bincount(codes,weights=a,minlength=n);bb=np.bincount(codes,weights=b,minlength=n)
   den=bb[draws].sum(axis=1);return a.sum()/b.sum() if b.sum() else np.nan,np.divide(aa[draws].sum(axis=1),den,out=np.full(2000,np.nan),where=den>0)
  point[name]={};boot[name]={}
  for metric,a,b in [('coverage',cover,np.ones(len(y))),('fail_coverage',cover&(y==1),y==1),('automatic_coverage',auto,np.ones(len(y))),('automatic_error',err,auto)]:
   value,boots=ratio(a,b);point[name][metric]=value;boot[name][metric]=boots
   rows.append(dict(setting=setting,model=model,method=name,metric=metric,estimate=value,lower=np.nanquantile(boots,.025),upper=np.nanquantile(boots,.975),persons=n,enrolments=len(y)))
 for baseline in ['mondrian_cp','empirical_mondrian']:
  for metric in boot[baseline]:
   d=boot['weighted_cap10'][metric]-boot[baseline][metric]
   deltas.append(dict(setting=setting,model=model,baseline=baseline,metric=metric,estimate=point['weighted_cap10'][metric]-point[baseline][metric],lower=np.nanquantile(d,.025),upper=np.nanquantile(d,.975)))
pd.DataFrame(rows).to_csv(tables/'oulad_cluster_intervals.csv',index=False)
pd.DataFrame(deltas).to_csv(tables/'oulad_cluster_differences.csv',index=False)
pd.DataFrame(counts).to_csv(tables/'oulad_cluster_counts.csv',index=False)
print('Five-seed contrasts, subgroups and 2,000 person-cluster bootstrap replicates saved.')
