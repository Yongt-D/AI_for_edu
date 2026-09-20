import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'src')
out=Path('outputs/revision_v2');d=pd.read_parquet(out/'predictions.parquet');m=pd.read_csv(out/'metrics.csv');rng=np.random.default_rng(42);rows=[];auc_rows=[]
iterations=2000
for (setting,model),g in d[(d.seed==42)&(d.calibration=='isotonic')&(d.kind=='conformal')&(d.alpha==.1)].groupby(['setting','model']):
    methods={name:v.set_index('entity_id').sort_index() for name,v in g.groupby('method')}
    a=methods['weighted_cap10'];ids=a.index;y=a.risk.to_numpy(int);n=len(a)
    # Shared resampling indices preserve the method pairing.
    ix=rng.integers(0,n,size=(iterations,n))
    def values(frame):
        sets=[json.loads(v) for v in frame.loc[ids].conformal_set];cover=np.array([v in s for v,s in zip(y,sets)]);auto=np.array([len(s)==1 for s in sets]);pred=np.array([s[0] if len(s)==1 else -1 for s in sets]);err=(pred!=y)&auto
        point=np.array([cover.mean(),max(0,.9-cover.mean()),abs(cover.mean()-.9),auto.mean(),err.sum()/auto.sum() if auto.sum() else np.nan])
        covers=cover[ix].mean(axis=1);autos=auto[ix].sum(axis=1)
        boot=np.c_[covers,np.maximum(0,.9-covers),abs(covers-.9),autos/n,np.divide(err[ix].sum(axis=1),autos,out=np.full(iterations,np.nan),where=autos>0)]
        return point,boot
    ap,ab=values(a)
    for baseline in ('mondrian_cp','empirical_mondrian'):
        bp,bb=values(methods[baseline]);delta=ab-bb
        for k,metric in enumerate(['coverage','undercoverage','absolute_coverage_gap','automatic_coverage','automatic_error']):
            rows.append({'setting':setting,'model':model,'comparison':'weighted_cap10-minus-'+baseline,'metric':metric,'estimate':ap[k]-bp[k],'ci_lower':np.nanquantile(delta[:,k],.025),'ci_upper':np.nanquantile(delta[:,k],.975),'n_entities':n,'iterations':iterations})
for (setting,model),g in d[(d.seed==42)&(d.calibration=='none')&(d.kind=='probability')].groupby(['setting','model']):
    y=g.risk.to_numpy(int);p=g.probability.to_numpy();vals=[]
    for _ in range(iterations):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))==2:vals.append(roc_auc_score(y[ix],p[ix]))
    auc_rows.append({'setting':setting,'model':model,'roc_auc':roc_auc_score(y,p),'ci_lower':np.quantile(vals,.025),'ci_upper':np.quantile(vals,.975)})
pd.DataFrame(rows).to_csv(out/'paired_bootstrap.csv',index=False);pd.DataFrame(auc_rows).to_csv(out/'auc_intervals.csv',index=False)
z=m[(m.kind=='conformal')&(m.alpha==.1)&(m.calibration=='isotonic')&m.method.isin(['mondrian_cp','empirical_mondrian','weighted_cap10'])]
cols=['observed_coverage','pass_coverage','fail_coverage','automatic_coverage','automatic_error_rate','undercoverage']
summary=z.groupby(['setting','model','method'])[cols].agg(['mean','min','max']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'seed_sensitivity.csv',index=False)
print('Bootstrap and seed summaries completed',flush=True)
