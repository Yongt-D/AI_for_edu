import sys,json,hashlib
from pathlib import Path
import pandas as pd
sys.path.insert(0,'src')
from edu_shift.revision_experiments import run_setting,combine,make_model,predictors
from edu_shift.splits import make_splits
from edu_shift.metrics import evaluate
out=Path('outputs/revision_v2');out.mkdir(parents=True,exist_ok=True)
config=json.loads(Path('configs/revision_v2.json').read_text());(out/'config.json').write_text(json.dumps(config,indent=2))
proc=Path('data/processed_v2');f=pd.read_parquet(proc/'features_week4.parquet')
for setting in ('iid','temporal_1','temporal_2','course_1','course_2'):
    run_setting(f,make_splits(f,setting),setting,out/'runs',config)
combine(out)
rows=[]
for week in (2,4,6,8):
    fw=pd.read_parquet(proc/f'features_week{week}.parquet')
    for setting in ('iid','temporal_1','temporal_2','course_1','course_2'):
        split=make_splits(f,setting);m=fw.merge(split,on='entity_id');tr=m[m.split=='train'];te=m[m.split=='test']
        cols=[c for c in predictors(fw) if tr[c].nunique()>1]
        for name in config['models']:
            model=make_model(name,42);model.fit(tr[cols],tr.risk);p=model.predict_proba(te[cols])[:,1]
            rows.append(dict(evaluate(te.risk.to_numpy(),p),week=week,setting=setting,model=name,n_test=len(te)))
    print('Checkpoint',week,'completed',flush=True)
pd.DataFrame(rows).to_csv(out/'checkpoint_robustness.csv',index=False)
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src/edu_shift').glob('*.py'),Path('configs/revision_v2.json'),*proc.glob('features_week*.parquet')]}
(out/'input_code_sha256.json').write_text(json.dumps(hashes,indent=2))
