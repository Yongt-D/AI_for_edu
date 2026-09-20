import numpy as np
from sklearn.model_selection import train_test_split

def make_splits(features, setting='iid', seed=42):
    f=features.copy(); key='entity_id' if 'entity_id' in f else 'student_id'; info=f.drop_duplicates(key).set_index(key); info=info[info.risk.notna() & info.risk.isin([0,1])]; entities=info.index.astype(str).to_numpy()
    if setting=='iid':
        strat=info.loc[entities].risk if info.loc[entities].risk.value_counts().min()>=2 and len(entities)>=6 else None
        tr,rest=train_test_split(entities,test_size=.4,random_state=seed,stratify=strat)
        if len(rest)<2: ca,te=rest,rest[:0]
        else:
            strat2=info.loc[rest].risk if len(rest)>=4 and info.loc[rest].risk.value_counts().min()>=2 else None
            ca,te=train_test_split(rest,test_size=.5,random_state=seed,stratify=strat2)
    else:
        years=sorted(f.year.astype(str).unique()); courses=sorted(f.course.astype(str).unique())
        if setting=='temporal_1': train,test=years[:1],years[1:2]; masktr=f.year.astype(str).isin(train); maskte=f.year.astype(str).isin(test)
        elif setting=='temporal_2': train,test=years[:2],years[2:3]; masktr=f.year.astype(str).isin(train); maskte=f.year.astype(str).isin(test)
        elif setting in {'course_1','course_2'}:
            train,test=([courses[0]],[courses[1]]) if setting=='course_1' else ([courses[1]],[courses[0]]); masktr=f.course.astype(str).isin(train); maskte=f.course.astype(str).isin(test)
        else: raise ValueError(setting)
        tr_all=np.intersect1d(f.loc[masktr,key].astype(str).unique(), info.index.astype(str)); te=np.intersect1d(f.loc[maskte,key].astype(str).unique(), info.index.astype(str))
        ca, tr=train_test_split(tr_all,test_size=.75,random_state=seed,stratify=info.loc[tr_all].risk)
    return _frame(f,key,tr,ca,te)

def _frame(f,key,tr,ca,te):
    ids=f[key].astype(str); return f[[key]].assign(split=np.where(ids.isin(tr),'train',np.where(ids.isin(ca),'calibration',np.where(ids.isin(te),'test','excluded')))).drop_duplicates()
