"""Standard-library physical-unit metrics with explicit provisional edge policies.

Implementation is testable before numerical lock acceptance. No fitting, data
loading, encoder execution or outcome-based metric selection occurs here.
"""
import math
from collections import defaultdict

NUMERICAL_POLICY = {
    'status': 'implemented_proposal_requires_lock_review',
    'classification_ties': 'first_class_for_multiclass;malignant_for_binary_p>=0.5',
    'multiclass_Brier': 'sum_over_classes_mean_over_physical_units',
    'binary_Brier': 'mean_squared_malignant_probability_error',
    'AP': 'score_tie_groups_stepwise_precision_weighted_recall_increments',
    'AURC': 'whole_confidence_tie_blocks_right_endpoint_risk_integral',
    'undefined': 'null_with_reason;macro_requires_all_configured_class_metrics',
    'ECE': '15_equal_width_bins;1_in_last_bin;empty_bins_zero_weight',
}

def na(reason): return {'value': None, 'reason': reason}
def value(x):
    if not math.isfinite(x): raise ValueError('Nonfinite metric; no replicate rejection/retry')
    return {'value': float(x), 'reason': None}

def validated(logits, labels, temperature=1., weights=None):
    z=tuple(tuple(r) for r in logits);y=tuple(labels)
    if not z or len(y)!=len(z) or len(z[0])<2:raise ValueError('Aligned nonempty class logits required')
    k=len(z[0])
    if any(len(r)!=k or any(type(x) not in (int,float) or not math.isfinite(x) for x in r) for r in z):raise ValueError('Finite rectangular logits required')
    if any(type(t) is not int or not 0<=t<k for t in y):raise ValueError('Canonical integer labels required')
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required')
    w=tuple(weights) if weights is not None else (1,)*len(z)
    if len(w)!=len(z) or any(type(t) is not int or t<0 for t in w) or not sum(w):raise ValueError('Nonzero integer physical multiplicities required')
    return z,y,w,k

def probabilities(logits, temperature=1.):
    rows=[];log_norm=[]
    for r in logits:
        s=tuple(x/temperature for x in r)
        if not s or any(not math.isfinite(x) for x in s):raise ValueError('Temperature overflow')
        m=max(s);d=math.fsum(math.exp(x-m) for x in s)
        rows.append(tuple(math.exp(x-m)/d for x in s));log_norm.append((s,m,math.log(d)))
    return tuple(rows),tuple(log_norm)

def binary_ranking(labels, scores, weights):
    groups=defaultdict(lambda:[0,0]);positive=negative=0
    for y,s,w in zip(labels,scores,weights):
        if not w:continue
        groups[s][y]+=w;positive+=w*y;negative+=w*(1-y)
    if not positive:ap=na('no_positive_units')
    else:
        tp=fp=0;terms=[]
        for s in sorted(groups,reverse=True):
            neg,pos=groups[s];tp+=pos;fp+=neg;terms.append(pos/positive*tp/(tp+fp))
        ap=value(math.fsum(terms))
    if not positive or not negative:auc=na('AUROC_requires_both_classes')
    else:
        below=0;terms=[]
        for s in sorted(groups):
            neg,pos=groups[s];terms.append(pos*(below+.5*neg));below+=neg
        auc=value(math.fsum(terms)/(positive*negative))
    return auc,ap

def core(logits,labels,temperature=1.,weights=None):
    z,y,w,k=validated(logits,labels,temperature,weights);p,logs=probabilities(z,temperature);n=sum(w)
    pred=tuple((1 if r[1]>=.5 else 0) if k==2 else max(range(k),key=r.__getitem__) for r in p)
    cm=[[0]*k for _ in range(k)]
    for a,b,m in zip(y,pred,w):cm[a][b]+=m
    support=[sum(r) for r in cm];columns=[sum(cm[i][j] for i in range(k)) for j in range(k)]
    f=[]
    for c in range(k):
        den=support[c]+columns[c];f.append(2*cm[c][c]/den if den else None)
    mf=na('macro_F1_requires_every_true_class') if not all(support) else value(math.fsum(f)/k)
    brier=value(math.fsum(m*math.fsum((q-(j==a))**2 for j,q in enumerate(r)) for r,a,m in zip(p,y,w))/n)
    nll=value(math.fsum(m*(mx-s[a]+ln) for (s,mx,ln),a,m in zip(logs,y,w))/n)
    return {'macro_F1':mf,'multiclass_Brier':brier,'NLL':nll},(z,y,w,k,p,pred,cm,support,columns)

def mean_metric(items):
    return na('macro_requires_all_configured_classes') if any(t['value'] is None for t in items) else value(math.fsum(t['value'] for t in items)/len(items))

def scores(logits,labels,temperature=1.,weights=None,cutoffs=()):
    result,state=core(logits,labels,temperature,weights);z,y,w,k,p,pred,cm,support,columns=state;n=sum(w)
    correct=sum(cm[c][c] for c in range(k));result['accuracy']=value(correct/n)
    per=[]
    for c in range(k):
        tp=cm[c][c];den=support[c]+columns[c];auc,ap=binary_ranking(tuple(int(a==c) for a in y),tuple(r[c] for r in p),w)
        per.append({'class':c,'support':support[c],'precision':value(tp/columns[c]) if columns[c] else na('no_predicted_units'),'recall':value(tp/support[c]) if support[c] else na('no_true_units'),'F1':value(2*tp/den) if den else na('no_true_or_predicted_units'),'AUROC':auc,'AP':ap})
    result.update(per_class=per,confusion_matrix=cm,balanced_accuracy=mean_metric([r['recall'] for r in per]),macro_AUROC=mean_metric([r['AUROC'] for r in per]),macro_AP=mean_metric([r['AP'] for r in per]))
    den=(n*n-sum(x*x for x in support))*(n*n-sum(x*x for x in columns))
    result['MCC']=value((correct*n-sum(a*b for a,b in zip(support,columns)))/math.sqrt(den)) if den else na('MCC_zero_denominator')
    if k==2:
        result['binary_Brier']=value(math.fsum(m*(r[1]-a)**2 for r,a,m in zip(p,y,w))/n)
        result['AUROC']=per[1]['AUROC'];result['AP']=per[1]['AP'];result['sensitivity']=per[1]['recall'];result['specificity']=per[0]['recall']
    confidence=tuple(max(r) for r in p);bins=[{'count':0,'confidence_sum':0.,'correct':0} for _ in range(15)]
    for a,b,c,m in zip(y,pred,confidence,w):
        q=bins[min(14,int(c*15))];q['count']+=m;q['confidence_sum']+=m*c;q['correct']+=m*(a==b)
    ece=0.;out=[]
    for i,q in enumerate(bins):
        count=q['count'];mean=q['confidence_sum']/count if count else None;acc=q['correct']/count if count else None
        if count:ece+=count/n*abs(mean-acc)
        out.append({'bin':i,'count':count,'mean_confidence':mean,'accuracy':acc})
    result['ECE']=value(ece);result['ECE_bins']=out
    tied=defaultdict(lambda:[0,0])
    for a,b,c,m in zip(y,pred,confidence,w):tied[c][0]+=m;tied[c][1]+=m*(a!=b)
    accepted=errors=0;area=0.;curve=[]
    for c in sorted(tied,reverse=True):
        count,bad=tied[c];accepted+=count;errors+=bad
        if not count:continue
        risk=errors/accepted;area+=risk*count/n;curve.append({'confidence':c,'accepted':accepted,'coverage':accepted/n,'risk':risk})
    result['AURC']=value(area);result['risk_coverage']=curve
    selective=[]
    for cutoff in cutoffs:
        if type(cutoff) not in (int,float) or not math.isfinite(cutoff) or not 0<=cutoff<=1:raise ValueError('Frozen finite confidence cutoff required')
        counts=[0]*k;bad=[0]*k
        for a,b,c,m in zip(y,pred,confidence,w):
            if c>=cutoff:counts[a]+=m;bad[a]+=m*(a!=b)
        total=sum(counts)
        selective.append({'cutoff':cutoff,'coverage':total/n,'selective_error':value(sum(bad)/total) if total else na('no_accepted_units'),'class_balanced_selective_error':value(math.fsum(e/c for e,c in zip(bad,counts))/k) if all(counts) else na('accepted_subset_missing_class'),'accepted_class_counts':counts})
    result['selective']=selective;return result
