"""Independent OULAD protocol replication; not transfer of a KU Leuven fitted model."""
import sys,json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0,'src')
from edu_shift.revision_experiments import run_setting,combine
raw=Path('data/raw/oulad');out=Path('outputs/revision_v2/oulad');out.mkdir(parents=True,exist_ok=True)
keys=['code_module','code_presentation','id_student']
roster=pd.read_csv(raw/'studentInfo.csv')[keys+['final_result']].merge(pd.read_csv(raw/'studentRegistration.csv'),on=keys,validate='one_to_one')
roster=roster[roster.date_registration.notna()&(roster.date_registration<28)&(roster.date_unregistration.isna()|(roster.date_unregistration>=28))].copy()
roster['risk']=roster.final_result.isin(['Fail','Withdrawn']).astype(int)
roster['entity_id']=roster[keys].astype(str).agg('::'.join,axis=1)
roster['student_id']=roster.id_student.astype(str);roster['course']=roster.code_module;roster['year']=roster.code_presentation
parts=[]
for chunk in pd.read_csv(raw/'studentVle.csv',chunksize=500000):
    chunk=chunk[(chunk.date>=0)&(chunk.date<28)]
    if len(chunk):parts.append(chunk)
d=pd.concat(parts,ignore_index=True);d['week']=d.date//7+1
agg=d.groupby(keys).agg(total_clicks=('sum_click','sum'),active_days=('date','nunique'),active_weeks=('week','nunique'),unique_content=('id_site','nunique'),last_day=('date','max'),first_day=('date','min')).reset_index()
f=roster.merge(agg,on=keys,how='left',validate='one_to_one')
for col in ['total_clicks','active_days','active_weeks','unique_content']:f[col]=f[col].fillna(0)
f['days_since_last_activity']=28-f.last_day.fillna(0);f['days_to_first_activity']=f.first_day.fillna(28)
f['clicks_per_active_day']=f.total_clicks/f.active_days.clip(lower=1);f['zero_activity']=(f.total_clicks==0).astype(int)
for week in (1,2,3,4):
    w=d[d.week==week].groupby(keys).sum_click.sum().rename(f'actions_week_{week}').reset_index()
    f=f.merge(w,on=keys,how='left');f[f'actions_week_{week}']=f[f'actions_week_{week}'].fillna(0)
metadata=f[['entity_id','code_module','code_presentation','id_student','final_result']].copy()
f=f.drop(columns=keys+['final_result','date_registration','date_unregistration','last_day','first_day'])
f.to_parquet(out/'features_week4.parquet',index=False)
metadata.to_csv(out/'outcome_metadata.csv',index=False)
config=json.loads(Path('configs/revision_v2.json').read_text());config['seeds']=[42];config['group_person']=True
stats=[]
for setting in ('temporal_B','temporal_J','course_DDD'):
    if setting.startswith('temporal'):
        suffix=setting[-1];source=f[f.year=='2013'+suffix];target=f[f.year=='2014'+suffix]
        modules=set(source.course)&set(target.course);source=source[source.course.isin(modules)];target=target[target.course.isin(modules)]
    else:source=f[f.course!='DDD'];target=f[f.course=='DDD']
    before=len(source);source=source[~source.student_id.isin(target.student_id)]
    # A person may have multiple enrolments. Assign all source records for that
    # person to one partition; the revised runner is used with the fixed seed.
    persons=source.student_id.drop_duplicates().to_numpy();train_ids,cal_ids=train_test_split(persons,test_size=.25,random_state=42)
    split=pd.concat([source[['entity_id']].assign(split=np.where(source.student_id.isin(train_ids),'train','calibration')),target[['entity_id']].assign(split='test')],ignore_index=True)
    stats.append({'setting':setting,'n_source':len(source),'n_test':len(target),'source_overlap_rows_excluded':before-len(source),'source_non_success_rate':source.risk.mean(),'test_non_success_rate':target.risk.mean(),'test_unique_persons':target.student_id.nunique()})
    run_setting(f,split,setting,out/'runs',config,seeds=[42])
combine(out);pd.DataFrame(stats).to_csv(out/'cohort_statistics.csv',index=False)
(out/'protocol.json').write_text(json.dumps({'checkpoint':'day 0 <= date < 28; enrolled at day 28','outcome':'Fail or Withdrawn versus Pass or Distinction','scope':'Independent protocol replication; neither same endpoint as first-attempt failure nor cross-institution model transfer','features':'VLE click aggregates only; no assessments, demographics, final-result or registration dates as predictors','source_person_overlap_excluded':True,'source':'https://doi.org/10.6084/m9.figshare.5081998.v1'},indent=2))
