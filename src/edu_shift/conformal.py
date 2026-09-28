import numpy as np

def quantile(scores, alpha=0.1):
    s=np.sort(np.asarray(scores,float)); n=len(s)
    if not 0<alpha<1: raise ValueError('alpha must be between zero and one')
    if n==0: return np.inf
    # finite-sample split-conformal quantile (ceil((n+1)(1-alpha))/n)
    k=int(np.ceil((n+1)*(1-alpha)))-1
    return np.inf if k>=n else float(s[max(k,0)])

def prediction_sets(p, q, q_by_class=None):
    p=np.asarray(p,float); out=[]
    for prob in p:
        thresholds=q_by_class if q_by_class is not None else {0:q,1:q}
        labels=[y for y in (0,1) if 1-(1-prob if y==0 else prob) <= thresholds[y]]
        out.append(labels)
    return out

def split_conformal(p_cal,y_cal,p_test,alpha=.1,mondrian=False):
    p_cal=np.asarray(p_cal,float); y_cal=np.asarray(y_cal,int)
    scores=1-np.where(y_cal==1,p_cal,1-p_cal); q=quantile(scores,alpha)
    qs=None
    if mondrian:
        qs={c:quantile(scores[y_cal==c],alpha) for c in (0,1)}
        # score threshold for candidate label is calibrated conditionally on that label
        sets=[]
        for prob in np.asarray(p_test,float):
            sets.append([c for c in (0,1) if (1-(prob if c else 1-prob)) <= qs.get(c,q)])
    else: sets=prediction_sets(p_test,q)
    return sets, q, qs

def conformal_metrics(sets,y,nominal):
    y=np.asarray(y,int); sizes=np.array([len(s) for s in sets]); covered=np.array([int(v in s) for v,s in zip(y,sets)])
    return {'nominal_coverage':1-nominal,'observed_coverage':float(covered.mean()),'coverage_gap':float(abs(covered.mean()-(1-nominal))),
      'average_set_size':float(sizes.mean()),'singleton_rate':float((sizes==1).mean()),'ambiguity_rate':float((sizes==2).mean()),'empty_set_rate':float((sizes==0).mean()),
      'pass_coverage':float(covered[y==0].mean()) if np.any(y==0) else np.nan,'fail_coverage':float(covered[y==1].mean()) if np.any(y==1) else np.nan}

def weighted_quantile(scores, weights, alpha=.1):
    scores=np.asarray(scores,float); weights=np.asarray(weights,float)
    ok=np.isfinite(scores)&np.isfinite(weights)&(weights>=0); scores,weights=scores[ok],weights[ok]
    if len(scores)==0 or weights.sum()<=0: return np.inf
    order=np.argsort(scores); scores,weights=scores[order],weights[order]
    target=(1-alpha)*weights.sum(); return float(scores[min(np.searchsorted(np.cumsum(weights),target,side='left'),len(scores)-1)])

def weighted_mondrian(p_cal,y_cal,w_cal,p_test,alpha=.1):
    p_cal=np.asarray(p_cal,float); y_cal=np.asarray(y_cal,int); w_cal=np.asarray(w_cal,float)
    scores=1-np.where(y_cal==1,p_cal,1-p_cal)
    qs={c:weighted_quantile(scores[y_cal==c],w_cal[y_cal==c],alpha) for c in (0,1)}
    sets=[]
    for prob in np.asarray(p_test,float): sets.append([c for c in (0,1) if (1-(prob if c else 1-prob))<=qs[c]])
    return sets,qs
