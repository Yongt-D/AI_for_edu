import sys,json
from pathlib import Path
sys.path.insert(0,'src')
from edu_shift.revised_data import load_bundle,enrolment_statistics
from edu_shift.features import build_features
from edu_shift.splits import make_splits

out=Path('data/processed_v2');out.mkdir(parents=True,exist_ok=True)
events,roster,posts,audit=load_bundle()
roster.to_parquet(out/'enrolments.parquet',index=False)
(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
for week in (2,4,6,8):
    f=build_features(events,week,roster,posts)
    f.to_parquet(out/f'features_week{week}.parquet',index=False)
    enrolment_statistics(roster,events,f).to_csv(out/f'dataset_statistics_week{week}.csv',index=False)
    if week==4:
        sd=out/'splits';sd.mkdir(exist_ok=True)
        for setting in ('iid','temporal_1','temporal_2','course_1','course_2'):
            make_splits(f,setting).to_csv(sd/f'{setting}_week4.csv',index=False)
    print('Prepared week',week,len(f),'enrolments',flush=True)
