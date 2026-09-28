import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        if method not in {'isotonic','platt','none'}: raise ValueError(method)
        self.method=method
    def fit(self, p, y):
        p=np.asarray(p,float); y=np.asarray(y,int); self._eps=1e-6
        if self.method=='isotonic': self.model=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(p,y)
        elif self.method=='platt': self.model=LogisticRegression(C=1e6).fit(np.log(np.clip(p,self._eps,1-self._eps)/(1-np.clip(p,self._eps,1-self._eps))).reshape(-1,1),y)
        return self
    def predict(self,p):
        p=np.asarray(p,float)
        if self.method=='none': return np.clip(p,0,1)
        if self.method=='isotonic': return np.asarray(self.model.predict(p),float)
        z=np.log(np.clip(p,self._eps,1-self._eps)/(1-np.clip(p,self._eps,1-self._eps))).reshape(-1,1)
        return self.model.predict_proba(z)[:,1]

def calibration_slope_intercept(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    if len(np.unique(y))<2: return np.nan,np.nan
    x=np.log(p/(1-p)).reshape(-1,1); m=LogisticRegression(C=1e6).fit(x,y)
    return float(m.coef_[0,0]),float(m.intercept_[0])
