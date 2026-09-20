"""Reproducible reliability experiments, with fixed test identities across seeds."""
import json,platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from .metrics import evaluate
from .calibration import ProbabilityCalibrator,calibration_slope_intercept
from .conformal import split_conformal,weighted_mondrian,conformal_metrics
from .selective import conformal_selective_metrics,confidence_risk_coverage,retention_weights
from .shift import fit_domain_classifier,density_ratio,weight_summary

EXCLUDE={'entity_id','student_id','risk','course','course_page','year','checkpoint_week','course_start','first_event_time','last_event_time'}

def predictors(f):return [c for c in f if c not in EXCLUDE and pd.api.types.is_numeric_dtype(f[c])]

def make_model(name,seed):
    if name=='logistic':return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed))
    return CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,thread_count=4,random_seed=seed,verbose=False,allow_writing_files=False)

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.959963984540054;p=k/n;d=1+z*z/n;centre=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(centre-half),float(centre+half)

def set_metrics(sets,y,alpha):
    y=np.asarray(y,int);covered=np.array([v in s for v,s in zip(y,sets)]);auto=np.array([len(s)==1 for s in sets]);pred=np.array([s[0] if len(s)==1 else -1 for s in sets])
    r=conformal_metrics(sets,y,alpha);r.update(conformal_selective_metrics(sets,y))
    r.update(undercoverage=max(0,1-alpha-covered.mean()),overcoverage=max(0,covered.mean()-(1-alpha)),n_test=len(y),n_pass=int((y==0).sum()),n_fail=int((y==1).sum()))
    for label,mask,values in [('coverage',np.ones(len(y),bool),covered),('pass',y==0,covered),('fail',y==1,covered),('automatic_error',auto,pred!=y)]:
        lo,hi=wilson(int(values[mask].sum()),int(mask.sum()));r[label+'_ci_lower']=lo;r[label+'_ci_upper']=hi
    r['automatic_false_negative_fraction']=float(((pred==0)&(y==1)&auto).sum()/max(1,(y==1).sum()))
    r['automatic_false_positive_fraction']=float(((pred==1)&(y==0)&auto).sum()/max(1,(y==0).sum()))
    return r

def budget_metrics(p,y,fraction,method):
    y=np.asarray(y,int);pred=np.asarray(p)>=.5
    w=retention_weights(p,fraction) if method=='confidence' else np.full(len(p),fraction)
    return {'policy':method,'automatic_coverage':fraction,'automatic_error_rate':float(np.dot(w,pred!=y)/w.sum()) if w.sum() else np.nan,'automatic_false_negative_fraction':float(np.dot(w,(pred==0)&(y==1))/max(1,(y==1).sum())),'automatic_false_positive_fraction':float(np.dot(w,(pred==1)&(y==0))/max(1,(y==0).sum())),'expected_n_automatic':float(w.sum())}

def run_setting(f,base_split,setting,out,config,seeds=None):
    print('Starting',setting,flush=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    seeds=seeds or config['seeds'];eligible=f[f.risk.isin([0,1])].copy().set_index('entity_id',drop=False)
    split=base_split.set_index('entity_id').split
    test_ids=split[split=='test'].index;source_ids=split[split.isin(['train','calibration'])].index
    te=eligible.loc[test_ids];source=eligible.loc[source_ids]
    metrics=[];predictions=[];weights=[];manifests=[];budgets=[];curves=[]
    for seed in seeds:
        if seed==42:
            tr=eligible.loc[split[split=='train'].index];cal=eligible.loc[split[split=='calibration'].index]
        else:
            tr,cal=train_test_split(source,test_size=.25,random_state=seed,stratify=source.risk)
        if config.get('group_person',False):
            ids=cal.student_id.drop_duplicates().to_numpy();fit_ids,score_ids=train_test_split(ids,test_size=.5,random_state=seed)
            cf=cal[cal.student_id.isin(fit_ids)];cc=cal[cal.student_id.isin(score_ids)]
        else:cf,cc=train_test_split(cal,test_size=.5,random_state=seed,stratify=cal.risk)
        cols=[c for c in predictors(f) if tr[c].nunique(dropna=True)>1]
        for label,part in [('train',tr),('calibrator_fit',cf),('conformal_score',cc),('test',te)]:
            assert part.index.is_unique
            manifest=part[['entity_id','risk']].copy();manifest['partition']=label;manifest['setting']=setting;manifest['seed']=seed;manifests.append(manifest.reset_index(drop=True))
        print('Fitting domain',setting,seed,len(cols),flush=True)
        domain,auc,ps,pt=fit_domain_classifier(pd.concat([tr,cf])[cols].to_numpy(),te[cols].to_numpy(),seed)
        wraw=density_ratio(domain,cc[cols].to_numpy(),ps,pt,max_weight=np.inf)[1]
        caps=[('cap5',5),('cap10',10),('cap20',20),('p99',float(np.quantile(wraw,.99)))] if seed==42 else [('cap10',10)]
        for name in config['models']:
            print('Fitting',name,flush=True)
            model=make_model(name,seed);model.fit(tr[cols],tr.risk)
            rawcf=model.predict_proba(cf[cols])[:,1];rawcc=model.predict_proba(cc[cols])[:,1];rawte=model.predict_proba(te[cols])[:,1]
            for calibration in (config['calibration_methods'] if seed==42 else ['isotonic']):
                cali=ProbabilityCalibrator(calibration).fit(rawcf,cf.risk)
                pc=cali.predict(rawcc);pt=cali.predict(rawte);y=te.risk.to_numpy(int)
                common={'setting':setting,'model':name,'seed':seed,'calibration':calibration}
                r=evaluate(y,pt);r['calibration_slope'],r['calibration_intercept']=calibration_slope_intercept(y,pt)
                r.update(common,kind='probability',n_test=len(te),n_train=len(tr),n_calibrator_fit=len(cf),n_conformal_score=len(cc),n_features=len(cols),features='|'.join(cols));metrics.append(r)
                pred=te[['entity_id','student_id','risk','course','year']].copy().reset_index(drop=True);pred['probability']=pt
                prob=pred.assign(**common,kind='probability',method='probability');predictions.append(prob)
                cov,risk,aurc=confidence_risk_coverage(pt,y)
                if seed==42:
                    curves.append(pd.DataFrame({'automatic_coverage':cov,'selective_error':risk,'aurc':aurc,**common}))
                    for fraction in (.2,.4,.6,.8):
                        for policy in ('confidence','random'):budgets.append(dict(budget_metrics(pt,y,fraction,policy),**common,matched_method='fixed_budget'))
                alphas=[1-v for v in config['nominal_coverages']] if seed==42 else [.1]
                for alpha in alphas:
                    methods={}
                    for method,mondrian in [('split_cp',False),('mondrian_cp',True)]:methods[method]=split_conformal(pc,cc.risk.to_numpy(int),pt,alpha,mondrian)[0]
                    if calibration=='isotonic':
                        methods['empirical_mondrian']=weighted_mondrian(pc,cc.risk.to_numpy(int),np.ones(len(cc)),pt,alpha)[0]
                        for capname,cap in caps:
                            w=np.minimum(wraw,cap);methods['weighted_'+capname]=weighted_mondrian(pc,cc.risk.to_numpy(int),w,pt,alpha)[0]
                    for method,sets in methods.items():
                        r=set_metrics(sets,y,alpha);r.update(common,kind='conformal',method=method,alpha=round(alpha,6),domain_auc=auc)
                        if method.startswith('weighted_'):
                            cap=dict(caps)[method[9:]];w=np.minimum(wraw,cap)
                            r.update(weight_summary(w,wraw,w.sum()**2/(w*w).sum()),weight_cap=cap)
                            for label in (0,1):
                                wc=w[cc.risk.to_numpy()==label];r[f'ess_class_{label}']=wc.sum()**2/(wc*wc).sum() if len(wc) else 0
                        metrics.append(r)
                        # Preserve all primary-seed sets; other seeds need only nominal 90%.
                        predictions.append(pred.assign(**common,kind='conformal',method=method,alpha=round(alpha,6),conformal_set=[json.dumps(s) for s in sets]))
                        if seed==42 and calibration=='isotonic' and abs(alpha-.1)<1e-8:
                            fraction=r['automatic_coverage']
                            for policy in ('confidence','random'):budgets.append(dict(budget_metrics(pt,y,fraction,policy),**common,matched_method=method))
                            budgets.append({**common,'policy':'singleton','matched_method':method,**{k:r[k] for k in ['automatic_coverage','automatic_error_rate','automatic_false_negative_fraction','automatic_false_positive_fraction']},'expected_n_automatic':r['n_automatic']})
            print(setting,seed,name,'completed',flush=True)
        weights.append(cc[['entity_id','risk']].reset_index(drop=True).assign(setting=setting,seed=seed,raw_weight=wraw))
    pd.DataFrame(metrics).to_csv(out/f'{setting}_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/f'{setting}_predictions.parquet',index=False)
    pd.concat(manifests,ignore_index=True).to_csv(out/f'{setting}_partitions.csv',index=False)
    pd.concat(weights,ignore_index=True).to_csv(out/f'{setting}_weights.csv',index=False)
    if curves:pd.concat(curves,ignore_index=True).to_csv(out/f'{setting}_curves.csv',index=False)
    pd.DataFrame(budgets).to_csv(out/f'{setting}_budgets.csv',index=False)

def combine(out):
    out=Path(out)
    for suffix in ('metrics','partitions','weights','curves','budgets'):
        paths=list((out/'runs').glob(f'*_{suffix}.csv'))
        if paths:pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(out/f'{suffix}.csv',index=False)
    paths=list((out/'runs').glob('*_predictions.parquet'))
    if paths:pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
