"""Post-review sensitivity analysis. Freeze protocol before running; no target-label tuning."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0,'src')
from edu_shift.revision_experiments import make_model,predictors,set_metrics
from edu_shift.calibration import ProbabilityCalibrator,calibration_slope_intercept
from edu_shift.conformal import split_conformal,weighted_mondrian
from edu_shift.shift import fit_domain_classifier,density_ratio
from edu_shift.metrics import evaluate

OUT=Path('outputs/revision_v3');OUT.mkdir(exist_ok=True,parents=True)
protocol={'status':'Post-peer-review exploratory extension, fixed before new fitting',
 'seeds':[42,7,19,73,101],'models':['logistic','catboost'],
 'class_weight':['unweighted','balanced'],'calibration':['none','platt','isotonic'],
 'target_split':'50% adaptation / 50% evaluation, random_state=20260916, no label stratification',
 'primary':'Inherited Logistic balanced / CatBoost unweighted; isotonic; full target batch; all five seeds equally reported',
 'alpha':.1,'weight_cap':10,'cluster_bootstrap_replicates':2000}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
f=pd.read_parquet('data/processed_v2/features_week4.parquet').set_index('entity_id',drop=False)
parts=pd.read_csv('outputs/revision_v2/partitions.csv');allrows=[];allpred=[];adaptparts=[]
for setting in parts.setting.unique():
    metrics=[];preds=[]
    ids=parts[(parts.setting==setting)&(parts.seed==42)&(parts.partition=='test')].entity_id.to_numpy()
    adapt_ids,eval_ids=train_test_split(ids,test_size=.5,random_state=20260916)
    assert not set(adapt_ids)&set(eval_ids)
    adaptparts.append(pd.DataFrame({'entity_id':np.r_[adapt_ids,eval_ids],'role':['adaptation']*len(adapt_ids)+['evaluation']*len(eval_ids),'setting':setting}))
    for seed in protocol['seeds']:
        part=parts[(parts.setting==setting)&(parts.seed==seed)]
        tr,cf,cc,te=[f.loc[part[part.partition==k].entity_id] for k in ['train','calibrator_fit','conformal_score','test']]
        cols=[c for c in predictors(f) if tr[c].nunique()>1]
        weights={}
        for mode,target in [('batch',te),('independent',f.loc[adapt_ids])]:
            model,auc,ps,pt=fit_domain_classifier(pd.concat([tr,cf])[cols].to_numpy(),target[cols].to_numpy(),seed)
            weights[mode]=density_ratio(model,cc[cols].to_numpy(),ps,pt,max_weight=10)[0]
        for name in protocol['models']:
            for weighting in protocol['class_weight']:
                model=make_model(name,seed)
                if name=='logistic':model.set_params(logisticregression__class_weight='balanced' if weighting=='balanced' else None)
                else:model.set_params(auto_class_weights='Balanced' if weighting=='balanced' else 'None')
                model.fit(tr[cols],tr.risk)
                rawcf=model.predict_proba(cf[cols])[:,1];rawcc=model.predict_proba(cc[cols])[:,1];rawte=model.predict_proba(te[cols])[:,1]
                for calibration in protocol['calibration']:
                    cal=ProbabilityCalibrator(calibration).fit(rawcf,cf.risk)
                    pc=cal.predict(rawcc);prob=cal.predict(rawte)
                    common=dict(setting=setting,seed=seed,model=name,class_weight=weighting,calibration=calibration)
                    # All controls evaluated on exactly the same subjects per comparison.
                    for scope,mask in [('full',np.ones(len(te),bool)),('heldout',te.index.isin(eval_ids))]:
                        y=te.risk.to_numpy()[mask];p=prob[mask]
                        pr=evaluate(y,p);pr['calibration_slope'],pr['calibration_intercept']=calibration_slope_intercept(y,p)
                        metrics.append(dict(pr,**common,scope=scope,method='probability'))
                        base=te.iloc[np.flatnonzero(mask)][['entity_id','student_id','risk','zero_activity']].reset_index(drop=True).assign(probability=p,**common,scope=scope)
                        preds.append(base.assign(method='probability',conformal_set=''))
                        methods={'MCP':split_conformal(pc,cc.risk.to_numpy(),p,.1,True)[0],
                            'EQ':weighted_mondrian(pc,cc.risk.to_numpy(),np.ones(len(cc)),p,.1)[0],
                            'EIQ-batch':weighted_mondrian(pc,cc.risk.to_numpy(),weights['batch'],p,.1)[0]}
                        if scope=='heldout':methods['EIQ-independent']=weighted_mondrian(pc,cc.risk.to_numpy(),weights['independent'],p,.1)[0]
                        for method,sets in methods.items():
                            metrics.append(dict(set_metrics(sets,y,.1),**common,scope=scope,method=method))
                            preds.append(base.assign(method=method,conformal_set=[json.dumps(v) for v in sets]))
                print(setting,seed,name,weighting,'complete',flush=True)
    frame=pd.DataFrame(metrics);pred=pd.concat(preds,ignore_index=True)
    frame.to_csv(OUT/f'{setting}_metrics.csv',index=False);pred.to_parquet(OUT/f'{setting}_predictions.parquet',index=False)
    allrows.append(frame);allpred.append(pred)
pd.concat(allrows,ignore_index=True).to_csv(OUT/'metrics.csv',index=False)
pd.concat(allpred,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
pd.concat(adaptparts,ignore_index=True).to_csv(OUT/'adaptation_partitions.csv',index=False)
paths=[Path(__file__),Path('outputs/revision_v2/partitions.csv'),Path('data/processed_v2/features_week4.parquet')]+list(Path('src/edu_shift').glob('*.py'))
(OUT/'input_code_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
