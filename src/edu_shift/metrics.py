import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,brier_score_loss,log_loss
def ece(y,p,bins=10):
 y=np.asarray(y);p=np.asarray(p,float)
 if not len(p):return np.nan
 indices=np.minimum((p*bins).astype(int),bins-1)
 return float(sum((indices==i).mean()*abs(y[indices==i].mean()-p[indices==i].mean()) for i in range(bins) if np.any(indices==i)))
def evaluate(y,p): return {'roc_auc':roc_auc_score(y,p) if len(set(y))>1 else np.nan,'pr_auc':average_precision_score(y,p),'macro_f1':f1_score(y,p>=.5,average='macro'),'balanced_accuracy':balanced_accuracy_score(y,p>=.5),'brier':brier_score_loss(y,p),'nll':log_loss(y,p,labels=[0,1]),'ece':ece(np.asarray(y),np.asarray(p))}
