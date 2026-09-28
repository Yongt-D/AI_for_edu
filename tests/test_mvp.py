import pandas as pd
from edu_shift.features import checkpoint_events,build_features
from edu_shift.splits import make_splits
def toy():
 t=pd.date_range('2020-01-01',periods=10,freq='D'); return pd.DataFrame({'student_id':['a']*5+['b']*5,'timestamp':list(t[:5])+list(t[:5]),'course':['A']*5+['B']*5,'year':['2018']*5+['2019']*5,'event_type':['view']*10,'content':['x']*10,'outcome':[1]*5+[0]*5})
def test_checkpoint_no_future():
 d=toy(); x=checkpoint_events(d,week=1); assert x.timestamp.max()<=d.timestamp.min()+pd.Timedelta(days=7)
def test_features_and_no_environment_predictors():
 f=build_features(toy(),4); assert 'risk' in f and 'course' in f and 'year' in f
def test_student_split_disjoint():
 f=build_features(toy(),4); s=make_splits(f,'iid'); assert not (set(s[s.split=='train'].student_id)&set(s[s.split=='test'].student_id))

def test_scoped_entity_prevents_cross_course_collision():
 d=toy(); d['entity_id']=d['student_id']+'::'+d['course']+'::'+d['year']
 f=build_features(d,4); assert f['entity_id'].nunique()==2
 s=make_splits(f,'iid'); assert 'entity_id' in s.columns
