from pathlib import Path
import json, re
import pandas as pd

ALIASES = {
 'student_id':['student_id','studentid','id_student','userid','user_id','person_id'],
 'course':['course','course_id','module','module_presentation','course_name'],
 'year':['year','academic_year','academic year','year_group','cohort'],
 'timestamp':['timestamp','datetime','date','event_time','time','activity_time'],
 'event_type':['event_type','activity_type','action','type','event','activity'],
 'content':['content_id','content','resource','item','page'],
 'outcome':['passed_first_attempt','passed first attempt','PASSED_FIRST_ATTEMPT','first_attempt_pass','pass','label','risk']}

def _norm(x): return re.sub(r'[^a-z0-9]+','_',str(x).lower()).strip('_')
def load_events(path, column_map=None):
    p=Path(path)
    # KU Leuven distribution stores outcome labels in *_course_participation.xlsx
    # and events in *_log_activity.csv; parse this pair explicitly so companion
    # workbooks are not accidentally treated as event tables.
    ku=sorted(x for x in p.rglob('*_log_activity.csv') if not x.name.startswith('._') and '__MACOSX' not in x.parts) if p.is_dir() else []
    if ku:
        return load_kuleuven(ku)
    files=[p] if p.is_file() else sorted(x for x in p.rglob('*') if x.suffix.lower() in {'.csv','.xlsx','.xls','.parquet'})
    if not files: raise FileNotFoundError(f'No tabular files found under {p}')
    frames=[]
    for f in files:
        if f.suffix.lower()=='.csv': frames.append(pd.read_csv(f))
        elif f.suffix.lower() in {'.xlsx','.xls'}: frames.append(pd.read_excel(f))
        else: frames.append(pd.read_parquet(f))
    df=pd.concat(frames,ignore_index=True)
    cmap=column_map or {}
    lookup={_norm(c):c for c in df.columns}
    rename={}
    for canonical, aliases in ALIASES.items():
        val=cmap.get(canonical)
        if val and val in df.columns: rename[val]=canonical; continue
        for a in aliases:
            if _norm(a) in lookup: rename[lookup[_norm(a)]]=canonical; break
    df=df.rename(columns=rename)
    missing=[x for x in ['student_id','timestamp','outcome'] if x not in df]
    if missing: raise ValueError(f'Missing required columns {missing}; found {list(df.columns)}')
    if 'course' not in df: df['course']='unknown'
    if 'year' not in df: df['year']='unknown'
    df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce',utc=True)
    df['risk']=_risk_label(df['outcome'])
    df['entity_id']=(df['student_id'].astype(str)+'::'+df['course'].astype(str)+'::'+df['year'].astype(str))
    return df

def load_kuleuven(log_files):
    """Load the public KU Leuven event/participation file pairs."""
    frames=[]
    for log in log_files:
        stem=log.name.split('_')[0]; part=log.with_name(f'{stem}_course_participation.xlsx')
        events=pd.read_csv(log, encoding='latin1')
        events=events.rename(columns={'USER_ID':'student_id','COURSE_ID':'course','CONTENT_ID':'content','SESSION_ID':'session_id','TIMESTAMP':'timestamp'})
        events['course']=events['course'].astype(str).str.replace(r'^Global economics [12]$','Global economics',regex=True)
        events['student_id']=events['student_id'].astype(str)
        events['year']={'1819':'2018-2019','1920':'2019-2020','2021':'2020-2021'}.get(stem,stem)
        if part.exists():
            try:
                rows=_read_xlsx_rows(part); head=[str(x) for x in rows[0]]; lab=pd.DataFrame(rows[1:],columns=head)
                lab=lab.rename(columns={'USER_ID':'student_id','COURSE_ID':'course'})[['student_id','course','PASSED_FIRST_ATTEMPT']]
                lab['course']=lab['course'].astype(str).str.replace(r'^Global economics [12]$','Global economics',regex=True)
                lab['student_id']=lab['student_id'].astype(str)
                events=events.merge(lab,on=['student_id','course'],how='left')
            except Exception as exc: raise RuntimeError(f'Could not read {part}: {exc}') from exc
        frames.append(events)
    df=pd.concat(frames,ignore_index=True)
    # USER_ID values can recur across course/year environments; use a scoped key.
    df['entity_id']=(df['student_id'].astype(str)+'::'+df['course'].astype(str)+'::'+df['year'].astype(str))
    df['timestamp']=pd.to_datetime(df.timestamp,errors='coerce',utc=True); df['risk']=1-pd.to_numeric(df['PASSED_FIRST_ATTEMPT'],errors='coerce')
    df['event_type']='click'; return df

def _read_xlsx_rows(path):
    """Small dependency-free reader for the simple single-sheet label workbooks."""
    from zipfile import ZipFile
    from xml.etree import ElementTree as ET
    with ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml')); ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            shared=[''.join(t.text or '' for t in si.findall('.//m:t',ns)) for si in root.findall('m:si',ns)]
        root=ET.fromstring(z.read('xl/worksheets/sheet1.xml')); ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}; rows=[]
        for rr in root.findall('.//m:row',ns):
            vals=['']*12
            for c in rr.findall('m:c',ns):
                v=c.find('m:v',ns); val='' if v is None else v.text
                if c.attrib.get('t')=='s' and val is not None: val=shared[int(val)]
                if c.attrib.get('t')=='inlineStr': val=''.join(t.text or '' for t in c.findall('.//m:t',ns))
                ref=c.attrib.get('r','A1'); col=''.join(x for x in ref if x.isalpha()); idx=0
                for ch in col: idx=idx*26+ord(ch.upper())-64
                vals[idx-1]=val
            rows.append(vals[:8])
        return rows

def _risk_label(s):
    if pd.api.types.is_numeric_dtype(s):
        vals=set(s.dropna().astype(float).unique())
        if vals <= {0.,1.}: return s.astype('float').astype('Int64') .astype(float)
    t=s.astype(str).str.lower().str.strip()
    # risk=1 means failure; pass=1 means pass and is inverted
    return t.map(lambda x: 1.0 if x in {'fail','failed','failure','failing','risk','at_risk','0'} else (0.0 if x in {'pass','passed','success','1'} else float('nan')))

def audit_events(df):
    groups=df.groupby(['course','year'],dropna=False)
    stats=groups.agg(n_events=('student_id','size'),n_students=('student_id','nunique'),failure_rate=('risk','mean')).reset_index()
    stats['duplicate_events']=df.duplicated(['student_id','timestamp','event_type','content'],keep=False).groupby([df.course,df.year]).sum().values if 'event_type' in df else 0
    return {'n_rows':len(df),'n_students':int(df.student_id.nunique()),'missing':df.isna().sum().to_dict(),'duplicate_rows':int(df.duplicated().sum()),'timestamp_min':str(df.timestamp.min()),'timestamp_max':str(df.timestamp.max()),'groups':stats.to_dict('records')}

def save_json(obj,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
