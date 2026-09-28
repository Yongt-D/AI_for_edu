import numpy as np

def conformal_selective_metrics(sets,y):
    y=np.asarray(y,int); sizes=np.array([len(s) for s in sets]); auto=sizes==1
    pred=np.array([s[0] if len(s)==1 else -1 for s in sets]); err=(pred[auto]!=y[auto]).mean() if auto.any() else np.nan
    return {'automatic_coverage':float(auto.mean()),'automatic_error_rate':float(err),'deferral_rate':float((~auto).mean()),'n_automatic':int(auto.sum())}

def confidence_risk_coverage(p,y):
    p=np.asarray(p,float); y=np.asarray(y,int); conf=np.maximum(p,1-p); pred=(p>=.5).astype(int); order=np.argsort(-conf)
    errors=(pred[order]!=y[order]).astype(float)
    # Expected random tie breaking: all members of a tied confidence block have
    # the same expected error. No ordering by labels or arbitrary row order.
    _,starts,counts=np.unique(-conf[order],return_index=True,return_counts=True)
    for start,count in zip(starts,counts):errors[start:start+count]=errors[start:start+count].mean()
    risk=np.cumsum(errors)/np.arange(1,len(y)+1); coverage=np.arange(1,len(y)+1)/len(y)
    return coverage,risk,float(risk.mean())

def retention_weights(p,fraction):
    """Expected inclusion under uniform randomisation within the boundary tie."""
    p=np.asarray(p,float);conf=np.maximum(p,1-p);target=float(fraction)*len(p)
    if not 0<=fraction<=1:raise ValueError('fraction out of range')
    weights=np.zeros(len(p));remaining=target
    for value in np.unique(conf)[::-1]:
        mask=conf==value;n=mask.sum();weights[mask]=min(1,max(0,remaining/n));remaining-=min(remaining,n)
        if remaining<=0:break
    return weights
