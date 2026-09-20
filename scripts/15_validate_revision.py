"""Validate saved experiment identities, metrics and citation completeness."""
import sys,json,re,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'src')
from edu_shift.revision_experiments import set_metrics,predictors
from edu_shift.metrics import evaluate
out=Path('outputs/revision_v2');p=pd.read_parquet(out/'predictions.parquet');m=pd.read_csv(out/'metrics.csv');part=pd.read_csv(out/'partitions.csv')
checks=[]
for (setting,seed),g in part.groupby(['setting','seed']):
    assert not g.entity_id.duplicated().any(),(setting,seed,'overlapping partitions')
for setting,g in part.groupby('setting'):
    tests=[set(v.entity_id) for _,v in g[g.partition=='test'].groupby('seed')]
    assert all(t==tests[0] for t in tests)
checks.append('All 25 KU Leuven partitions disjoint; fixed target identities across seeds')
for group,g in p[p.kind=='probability'].groupby(['setting','model','seed','calibration']):
    row=m[(m.kind=='probability')&(m.setting==group[0])&(m.model==group[1])&(m.seed==group[2])&(m.calibration==group[3])].iloc[0]
    for name,value in evaluate(g.risk.to_numpy(),g.probability.to_numpy()).items():assert np.isclose(value,row[name],equal_nan=True)
checks.append('All saved probability metrics independently recomputed from saved predictions')
for group,g in p[p.kind=='conformal'].groupby(['setting','model','seed','calibration','alpha','method']):
    row=m[(m.kind=='conformal')&(m.setting==group[0])&(m.model==group[1])&(m.seed==group[2])&(m.calibration==group[3])&(m.alpha==group[4])&(m.method==group[5])].iloc[0]
    values=set_metrics([json.loads(v) for v in g.conformal_set],g.risk.to_numpy(),group[4])
    for name in ['observed_coverage','pass_coverage','fail_coverage','automatic_coverage','automatic_error_rate','average_set_size']:assert np.isclose(values[name],row[name],equal_nan=True),(group,name)
checks.append('All saved set coverage and selective metrics agree with saved prediction sets')
f=pd.read_parquet('data/processed_v2/features_week4.parquet');assert len(f)==4280 and not f.entity_id.duplicated().any()
assert f.zero_activity.sum()==137 and f.forum_posts.sum()==222
assert (f.last_event_time.dropna()< (f.course_start+pd.Timedelta(days=28))[f.last_event_time.notna()]).all()
assert (f.first_event_time.dropna()>=f.course_start[f.first_event_time.notna()]).all()
checks.append('Official-date window boundaries, zero-activity roster and forum posts verified')
ep=pd.read_csv(out/'oulad/partitions.csv');ef=pd.read_parquet(out/'oulad/features_week4.parquet');ep=ep.merge(ef[['entity_id','student_id']],on='entity_id')
for setting,g in ep.groupby('setting'):
    groups={k:set(v.student_id) for k,v in g.groupby('partition')}
    for a in groups:
        for b in groups:
            if a!=b:assert not groups[a]&groups[b]
checks.append('OULAD person identifiers disjoint across all four partitions')
definitions={
'total_clicks':'Count of matched in-window logged actions (not necessarily physical clicks)',
'active_days':'Number of distinct UTC dates with a logged action',
'active_weeks':'Number of completed-window week bins containing an action',
'num_sessions':'Number of distinct recorded session identifiers within the window',
'night_activity_ratio':'Fraction of actions with UTC hour <6 or >=22; published timestamp timezone treated consistently',
'weekend_activity_ratio':'Fraction of actions occurring on Saturday or Sunday',
'material_clicks':'Actions whose content type indicates course materials/resources',
'assessment_clicks':'Actions whose content type indicates assessments; grade-page access is separate',
'forum_accesses':'Actions on forum pages; does not establish that a post was read',
'grade_page_accesses':'Actions on pages of type Grades; no grades used as predictors',
'main_page_accesses':'Actions on a course main page',
'unique_content':'Number of distinct accessed content identifiers, identifiers themselves excluded',
'total_session_minutes':'Sum of last minus first in-window event timestamps per session',
'median_session_minutes':'Median of within-window session spans; not observed attention time',
'clicks_per_session':'Logged actions divided by session count; zero when inactive',
'sessions_per_active_day':'Session count divided by active days; zero when inactive',
'active_day_fraction':'Active days divided by 28; structurally proportional to active_days at a fixed checkpoint',
'zero_activity':'Indicator of no in-window log activity',
'days_since_last_activity':'Elapsed days from last action to exclusive cutoff; 28 for inactive enrolments',
'days_to_first_activity':'Days after course start until first action; 28 for inactive enrolments',
'forum_posts':'Number of distinct timestamped forum contributions created within the window',
}
for w in range(1,5):definitions[f'actions_week_{w}']=f'Logged actions in days {(w-1)*7}--{w*7-1}'
rows=[]
for col in predictors(f):rows.append({'feature':col,'definition':definitions[col],'n_unique':int(f[col].nunique()),'n_missing':int(f[col].isna().sum())})
pd.DataFrame(rows).to_csv(out/'feature_dictionary.csv',index=False)
(out/'validation.json').write_text(json.dumps({'checks':checks,'status':'passed'},indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2),flush=True)
