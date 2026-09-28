import numpy as np
from edu_shift.conformal import quantile,weighted_quantile,split_conformal,conformal_metrics
from edu_shift.selective import conformal_selective_metrics

def test_weighted_quantile_respects_high_weight_tail():
 assert weighted_quantile([.1,.2,.9],[1,1,20],alpha=.1)==.9

def test_prediction_sets_and_metrics():
 sets,q,_=split_conformal(np.array([.1,.2,.8,.9]),np.array([0,0,1,1]),np.array([.05,.95]),alpha=.1)
 m=conformal_metrics(sets,np.array([0,1]),.1); assert m['observed_coverage']==1
 sm=conformal_selective_metrics(sets,np.array([0,1])); assert 0<=sm['automatic_coverage']<=1

def test_finite_sample_quantile_handles_unattainable_rank():
 assert np.isinf(quantile([.1,.2,.3],.1))
 assert quantile([.1,.2,.3],.5)==.2
 assert np.isinf(quantile([],.1))
