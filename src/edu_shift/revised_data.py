"""KU Leuven ingestion with enrolment roster and original course-page joins."""
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import pandas as pd

YEARS={'1819':'2018-2019','1920':'2019-2020','2021':'2020-2021'}

def read_xlsx(path):
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            shared=[''.join(n.text or '' for n in si.findall('.//m:t',ns)) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        rows=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
            values={}
            for c in row.findall('m:c',ns):
                col=''.join(x for x in c.attrib['r'] if x.isalpha()); index=0
                for char in col:index=index*26+ord(char)-64
                v=c.find('m:v',ns); value=v.text if v is not None else None
                if c.attrib.get('t')=='s' and value is not None:value=shared[int(value)]
                if c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//m:t',ns))
                values[index-1]=value
            rows.append(values)
        width=max(max(r,default=-1) for r in rows)+1
        vals=[[r.get(i) for i in range(width)] for r in rows]
    return pd.DataFrame(vals[1:],columns=vals[0])

def normalise_keys(frame):
    frame=frame.rename(columns={'USER_ID':'student_id','COURSE_ID':'course_page','CONTENT_ID':'content','SESSION_ID':'session_id','TIMESTAMP':'timestamp'})
    for col in ('student_id','course_page','content','session_id'):
        if col in frame:frame[col]=frame[col].astype('string').str.strip()
    return frame

def load_bundle(root='data/raw'):
    root=Path(root);info=json.loads((root/'course_info.json').read_text())
    rosters=[];events=[];posts=[];audit=[]
    for path in sorted(root.rglob('*_log_activity.csv')):
        if '__MACOSX' in path.parts or path.name.startswith('._'):continue
        prefix=path.name.split('_')[0];year=YEARS[prefix]
        roster=normalise_keys(read_xlsx(path.with_name(prefix+'_course_participation.xlsx')))
        roster=roster[['student_id','course_page','PASSED_FIRST_ATTEMPT']].dropna(subset=['student_id','course_page'])
        roster_duplicates=int(roster.duplicated().sum());roster=roster.drop_duplicates()
        if roster.duplicated(['student_id','course_page']).any():raise ValueError('Duplicate enrolment key')
        roster['course']=roster.course_page.str.replace(r'^Global economics [12]$','Global economics',regex=True)
        roster['year']=year;roster['risk']=1-pd.to_numeric(roster.PASSED_FIRST_ATTEMPT,errors='coerce')
        roster['entity_id']=roster.student_id+'::'+roster.course_page+'::'+year
        starts={name:pd.Timestamp(year=v['course_start'][0],month=v['course_start'][1],day=v['course_start'][2],tz='UTC') for name,v in info[prefix].items()}
        roster['course_start']=roster.course.map(starts)
        if roster.course_start.isna().any():raise ValueError('Missing official course start')
        roster=roster.drop(columns='PASSED_FIRST_ATTEMPT')
        raw=normalise_keys(pd.read_csv(path,encoding='latin1',dtype=str));count=len(raw)
        log=raw.merge(roster,on=['student_id','course_page'],how='inner',validate='many_to_one');unmatched=count-len(log)
        log['timestamp']=pd.to_datetime(log.timestamp,errors='coerce',utc=True)
        content=normalise_keys(read_xlsx(path.with_name(prefix+'_course_content.xlsx')))[['course_page','content','CONTENT_TYPE']].drop_duplicates()
        if content.duplicated(['course_page','content']).any():raise ValueError('Conflicting content types')
        log=log.merge(content,on=['course_page','content'],how='left',validate='many_to_one')
        log['event_type']=log.CONTENT_TYPE.fillna('Unknown')
        duplicates=int(log.duplicated(['entity_id','timestamp','content','session_id']).sum())
        log=log.drop_duplicates(['entity_id','timestamp','content','session_id'])
        post=normalise_keys(read_xlsx(path.with_name(prefix+'_df_contribution.xlsx')))
        post=post.merge(roster,on=['student_id','course_page'],how='inner',validate='many_to_one')
        serial=pd.to_numeric(post.DTCREATED,errors='coerce')
        post['timestamp']=pd.to_datetime(serial,unit='D',origin='1899-12-30',utc=True)
        post=post.drop_duplicates(['entity_id','POST_ID'])
        audit.append({'year':year,'duplicate_enrolments_removed':roster_duplicates,'raw_log_rows':count,'unmatched_enrolment_log_rows':unmatched,'duplicate_log_rows_removed':duplicates,'retained_log_rows':len(log),'unknown_content_rows':int(log.CONTENT_TYPE.isna().sum()),'enrolments':len(roster),'valid_outcome':int(roster.risk.isin([0,1]).sum())})
        rosters.append(roster);events.append(log);posts.append(post[['entity_id','timestamp','course_start']])
        print('Loaded',year,len(log),flush=True)
    return pd.concat(events,ignore_index=True),pd.concat(rosters,ignore_index=True),pd.concat(posts,ignore_index=True),audit

def enrolment_statistics(roster,events,features):
    r=roster.merge(features[['entity_id','total_clicks']],on='entity_id',how='left',validate='one_to_one')
    r['valid_outcome']=r.risk.isin([0,1]);r['zero_early_activity']=r.total_clicks.fillna(0).eq(0)
    r['valid_risk']=r.risk.where(r.valid_outcome)
    result=r.groupby(['course','year']).agg(n_enrolments=('entity_id','size'),n_eligible=('valid_outcome','sum'),failure_rate=('valid_risk','mean'),n_zero_early_activity=('zero_early_activity','sum')).reset_index()
    result['n_missing_outcome']=result.n_enrolments-result.n_eligible
    ev=events.groupby(['course','year']).size().rename('n_events').reset_index()
    return result.merge(ev,on=['course','year'],how='left')
