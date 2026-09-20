import numpy as np
import pandas as pd

def checkpoint_events(df, week, origin=None):
    if week <= 0: raise ValueError('week must be positive')
    d=df.dropna(subset=['timestamp']).copy()
    if origin is None:
        origin=d.course_start if 'course_start' in d else (d.groupby(['course','year']).timestamp.transform('min') if {'course','year'} <= set(d.columns) else d.timestamp.min())
    elapsed=(d.timestamp-origin).dt.total_seconds()/86400
    d['_week']=np.floor(elapsed/7).astype(int)+1
    return d[(elapsed>=0)&(elapsed<7*week)].copy()

def build_features(df,week=4,roster=None,posts=None):
    df=df.copy();key='entity_id' if 'entity_id' in df else 'student_id'
    if 'risk' not in df and 'outcome' in df:df['risk']=pd.to_numeric(df.outcome,errors='coerce')
    if roster is None:
        meta=[c for c in [key,'student_id','course','year','risk','course_start'] if c in df]
        roster=df[list(dict.fromkeys(meta))].drop_duplicates(key)
    d=checkpoint_events(df,week);d['_day']=d.timestamp.dt.floor('D')
    if 'session_id' not in d:d['session_id']=d['_day'].astype(str)
    d['session_id']=d.session_id.fillna(d['_day'].astype(str)).astype(str)
    d['_night']=((d.timestamp.dt.hour<6)|(d.timestamp.dt.hour>=22)).astype(int)
    d['_weekend']=(d.timestamp.dt.dayofweek>=5).astype(int)
    et=d.event_type.astype(str).str.lower() if 'event_type' in d else pd.Series('',index=d.index)
    types={'material_clicks':'material|resource','assessment_clicks':'assessment|quiz|exam','forum_accesses':'forum','grade_page_accesses':'grade','main_page_accesses':'main page'}
    for name,pattern in types.items():d[name]=et.str.contains(pattern,regex=True).astype(int)
    agg={'total_clicks':(key,'size'),'active_days':('_day','nunique'),'active_weeks':('_week','nunique'),'num_sessions':('session_id','nunique'),'night_activity_ratio':('_night','mean'),'weekend_activity_ratio':('_weekend','mean'),'first_event_time':('timestamp','min'),'last_event_time':('timestamp','max')}
    for name in types:agg[name]=(name,'sum')
    if 'content' in d:agg['unique_content']=('content','nunique')
    f=d.groupby(key).agg(**agg).reset_index()
    sessions=d.groupby([key,'session_id']).timestamp.agg(['min','max'])
    sessions['minutes']=(sessions['max']-sessions['min']).dt.total_seconds()/60
    f=f.merge(sessions.groupby(key).minutes.agg(total_session_minutes='sum',median_session_minutes='median'),on=key,how='left')
    out=roster.merge(f,on=key,how='left',validate='one_to_one')
    numeric=out.select_dtypes('number').columns.difference(['risk','student_id'])
    out[numeric]=out[numeric].fillna(0)
    out['clicks_per_session']=out.total_clicks/out.num_sessions.clip(lower=1)
    out['sessions_per_active_day']=out.num_sessions/out.active_days.clip(lower=1)
    out['active_day_fraction']=out.active_days/(7*week)
    out['zero_activity']=(out.total_clicks==0).astype(int)
    if 'course_start' in out:
        cutoff=out.course_start+pd.to_timedelta(week*7,unit='D')
        out['days_since_last_activity']=((cutoff-out.last_event_time).dt.total_seconds()/86400).fillna(week*7)
        out['days_to_first_activity']=((out.first_event_time-out.course_start).dt.total_seconds()/86400).fillna(week*7)
    for w in range(1,week+1):out[f'actions_week_{w}']=out[key].map(d[d._week==w].groupby(key).size()).fillna(0).astype(int)
    if posts is not None:out['forum_posts']=out[key].map(checkpoint_events(posts,week).groupby(key).size()).fillna(0).astype(int)
    out['checkpoint_week']=week
    return out
