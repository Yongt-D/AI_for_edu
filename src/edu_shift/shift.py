import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fit_domain_classifier(x_source,x_target,seed=42):
    x=np.vstack([x_source,x_target]); y=np.r_[np.zeros(len(x_source),int),np.ones(len(x_target),int)]
    it,iv=train_test_split(np.arange(len(y)),test_size=.25,random_state=seed,stratify=y)
    # Preserve empirical domain priors; density_ratio applies the prior correction explicitly.
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed))
    model.fit(x[it],y[it]); auc=roc_auc_score(y[iv],model.predict_proba(x[iv])[:,1]); model.fit(x,y)
    return model,float(auc),float((y==0).mean()),float((y==1).mean())

def density_ratio(model,x_source,source_prior,target_prior,max_weight=10.0,clip_percentile=None):
    d=np.clip(model.predict_proba(x_source)[:,1],1e-6,1-1e-6)
    w=(d/(1-d))*(source_prior/target_prior); raw=w.copy(); cap=float(np.quantile(raw,clip_percentile/100)) if clip_percentile is not None else max_weight; w=np.clip(w,0,cap)
    ess=float(w.sum()**2/np.square(w).sum()) if np.square(w).sum()>0 else 0.0
    return w,raw,ess

def weight_summary(w,raw,ess):
    w=np.asarray(w,float); raw=np.asarray(raw,float)
    return {'weight_min':float(w.min()),'weight_median':float(np.median(w)),'weight_mean':float(w.mean()),'weight_max':float(w.max()),'raw_weight_p99':float(np.quantile(raw,.99)),'effective_sample_size':float(ess)}
