import numpy as np
import pandas as pd
from edu_shift.features import checkpoint_events,build_features
from edu_shift.metrics import ece
from edu_shift.conformal import weighted_mondrian,split_conformal
from edu_shift.selective import confidence_risk_coverage,retention_weights

def fixture():
    origin=pd.Timestamp('2020-01-01',tz='UTC')
    roster=pd.DataFrame({'entity_id':['a','b'],'student_id':['a','b'],'course':'A','year':'2020','risk':[0,1],'course_start':origin})
    events=pd.DataFrame({'entity_id':['a']*5,'timestamp':[origin+pd.Timedelta(days=d) for d in (-1,0,22,27,28)],'course_start':origin,'session_id':['x','s1','s2','s3','s4'],'event_type':['Course Material']*5,'content':'x'})
    return events,roster

def test_complete_four_weeks_and_exclusion_of_prestart_and_cutoff():
    events,_=fixture();x=checkpoint_events(events,4)
    assert len(x)==3
    assert x._week.tolist()==[1,4,4]

def test_zero_activity_and_true_sessions_and_no_future():
    events,roster=fixture();f=build_features(events,4,roster).set_index('entity_id')
    assert f.loc['b','total_clicks']==0 and f.loc['b','risk']==1
    assert f.loc['a','num_sessions']==3 and f.loc['a','material_clicks']==3
    altered=events.copy();altered.loc[4,'event_type']='Forum'
    pd.testing.assert_frame_equal(build_features(altered,4,roster),build_features(events,4,roster))

def test_ece_includes_certain_errors_and_zero():
    assert ece([0],[1.])==1.
    assert ece([1],[0.])==1.
    assert ece([0,1],[0.,1.])==0.

def test_tied_risk_is_permutation_invariant():
    p=np.array([.8,.8,.2,.2]);y=np.array([1,0,1,0])
    a=confidence_risk_coverage(p,y);b=confidence_risk_coverage(p[::-1],y[::-1])
    np.testing.assert_allclose(a[1],b[1]);np.testing.assert_allclose(a[1],.5)
    np.testing.assert_allclose(retention_weights(p,.5),.5)

def test_unit_weights_control_is_scale_invariant():
    p=[.1,.3,.6,.9];y=[0,0,1,1]
    a,_=weighted_mondrian(p,y,[1]*4,[.4,.6],.2)
    b,_=weighted_mondrian(p,y,[10]*4,[.4,.6],.2)
    assert a==b
