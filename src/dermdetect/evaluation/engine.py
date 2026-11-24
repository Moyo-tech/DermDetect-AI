"""Four-system evaluator over pinned logits, with no automatic unseal or locks."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from .metrics import core, scores, NUMERICAL_POLICY

CLASSES=('akiec','bcc','bkl','df','mel','nv','vasc')
BINARY={'akiec':None,'bcc':1,'bkl':0,'df':0,'mel':1,'nv':0,'vasc':0}
SYSTEMS=('dense7','med7','dense2','med2')
SEED=20260729
REPLICATES=1000

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def authority(partition,lock=None,unseal=None,grant=None):
    if partition in ('synthetic','validation'):return
    from .authorization import RunGrant
    if not isinstance(grant, RunGrant):
        raise PermissionError('Verified scoped grant required; dictionaries confer no authority')
    grant.check(partition)

def aggregate(rows, image_ids, logits, classes):
    if classes not in (2,7):raise ValueError('Canonical seven/binary systems only')
    rows=tuple(rows);image_ids=tuple(image_ids);logits=tuple(tuple(z) for z in logits)
    if not rows or image_ids!=tuple(r['image_id'] for r in rows) or len(rows)!=len(logits) or len(set(image_ids))!=len(rows):raise ValueError('Exact ordered unique image coverage required')
    owner={};members=defaultdict(list)
    for r,z in zip(rows,logits):
        if r['label'] not in CLASSES or not r.get('physical_id') or not r.get('cluster_id') or r.get('active',True) is not True:raise ValueError('Active canonical physical membership required')
        label=CLASSES.index(r['label']) if classes==7 else BINARY[r['label']]
        if label is None:raise ValueError('Binary rows must exclude akiec before aggregation')
        key=(r['cluster_id'],label)
        if r['physical_id'] in owner and owner[r['physical_id']]!=key:raise ValueError('Physical identity crosses cluster/label')
        if len(z)!=classes or any(type(x) not in (int,float) or not math.isfinite(x) for x in z):raise ValueError('Finite canonical class logits required')
        owner[r['physical_id']]=key;members[r['physical_id']].append(z)
    ids=tuple(sorted(members))
    x=tuple(tuple(math.fsum(z[c] for z in members[u])/len(members[u]) for c in range(classes)) for u in ids)
    return ids,x,tuple(owner[u][1] for u in ids),tuple(owner[u][0] for u in ids)

def paired_draws(ids,labels,clusters,classes,replicates=REPLICATES,seed=SEED):
    if replicates!=REPLICATES or seed!=SEED:raise ValueError('Exact1000paired draws/seed20260729 required')
    ids=tuple(ids);labels=tuple(labels);clusters=tuple(clusters)
    if not ids or len(ids)!=len(set(ids)) or len(ids)!=len(labels) or len(ids)!=len(clusters) or set(labels)!=set(range(classes)):raise ValueError('Exact complete paired physical-class population required')
    groups=defaultdict(list)
    for u,c,y in zip(ids,clusters,labels):
        if not isinstance(c,str) or not c:raise ValueError('Canonical conservative cluster required')
        groups[c].append((u,y))
    strata=defaultdict(list)
    for c,m in sorted(groups.items()):
        if len({y for _,y in m})!=1:raise ValueError('Mixed-label evaluation cluster requires review')
        strata[m[0][1]].append(c)
    rng=random.Random(seed)
    for _ in range(REPLICATES):
        count=dict.fromkeys(ids,0)
        for y in range(classes):
            bucket=strata[y]
            for _ in bucket:
                c=bucket[rng.randrange(len(bucket))]
                for u,_ in groups[c]:count[u]+=1
        yield tuple(count[u] for u in ids)

def interval(values):
    if len(values)!=REPLICATES or any(not math.isfinite(v) for v in values):raise ValueError('All1000finite replicates required; no rejection/retry')
    v=sorted(values)
    def q(p):
        x=(len(v)-1)*p;i=math.floor(x);j=math.ceil(x);return v[i]+(v[j]-v[i])*(x-i)
    return [q(.025),q(.975)]

def paired_intervals(a,b,*,partition,check=lambda:None):
    if partition not in ('synthetic','validation','internal_test'):raise ValueError('HAM paired conservative-group intervals only')
    if a['ids']!=b['ids'] or a['labels']!=b['labels'] or a['clusters']!=b['clusters']:raise ValueError('Identical paired membership/labels/clusters required')
    classes=len(a['logits'][0]);values={m:{'a':[],'b':[],'b_minus_a':[]} for m in ['macro_F1','multiclass_Brier']}
    for weights in paired_draws(a['ids'],a['labels'],a['clusters'],classes):
        check()
        sa,_=core(a['logits'],a['labels'],a['temperature'],weights);sb,_=core(b['logits'],b['labels'],b['temperature'],weights)
        for metric in values:
            x,y=sa[metric]['value'],sb[metric]['value']
            if x is None or y is None:raise ValueError('Undefined replicate; no retry')
            values[metric]['a'].append(x);values[metric]['b'].append(y);values[metric]['b_minus_a'].append(y-x)
    return {'replicates':REPLICATES,'seed':SEED,'resampling':'class_stratified_conservative_groups;physical_point_estimates','difference':'MedSigLIP_minus_DenseNet','intervals':{m:{name:interval(v) for name,v in group.items()} for m,group in values.items()}}

def evaluate(rows,image_logits,specs,*,partition,lock=None,unseal=None,bootstrap=False,grant=None,check=lambda:None):
    authority(partition,lock,unseal,grant)
    if set(image_logits)!=set(SYSTEMS) or set(specs)!=set(SYSTEMS):raise ValueError('Four-system exact coverage required')
    rows=tuple(rows)
    if grant is not None:
        grant.check(partition,specs);grant.check_rows(rows)
    if any(r.get('partition')!=partition for r in rows):raise ValueError('Cross-partition evaluation forbidden')
    result={'partition':partition,'systems':{},'numerical_policy':NUMERICAL_POLICY,'lock_SHA':grant.lock_SHA if grant else None,'held_out_access':partition in ('internal_test','DDI')};tasks={}
    for name in SYSTEMS:
        classes=7 if name.endswith('7') else 2
        keep=rows if classes==7 else tuple(r for r in rows if BINARY.get(r['label']) is not None)
        supplied=image_logits[name]
        if supplied.get('payload_SHA')!=digest({'image_ids':supplied['image_ids'],'logits':supplied['logits']}):raise ValueError('Exact ordered logits payload hash required')
        if tuple(supplied.get('image_SHAs',()))!=tuple(r.get('sha256') for r in keep) or any(not r.get('sha256') for r in keep):raise ValueError('Exact source-image hash membership required')
        ids,z,y,clusters=aggregate(keep,supplied['image_ids'],supplied['logits'],classes)
        spec=specs[name]
        if spec.get('classes')!=list(range(classes)) or spec.get('confidence_tie_policy')!='include_all_ties':raise ValueError('Exact frozen class/confidence policy required')
        if classes==2 and (spec.get('malignant_probability')!=.5 or spec.get('binary_equality')!='malignant'):raise ValueError('Exact frozen binary equality rule required')
        if not spec.get('artifact_SHA'):raise ValueError('Pinned selected artifact binding required')
        t=spec['temperature'];cutoffs=spec['cutoffs']
        if len(cutoffs)!=2:raise ValueError('Frozen90/80coverage cutoffs required')
        tasks[name]=dict(ids=ids,logits=z,labels=y,clusters=clusters,temperature=t)
        result['systems'][name]={'physical_units':len(ids),'membership_SHA':digest({'ids':ids,'labels':y,'clusters':clusters}),'artifact_SHA':spec['artifact_SHA'],'metrics':scores(z,y,t,cutoffs=cutoffs)}
    if bootstrap:
        if partition=='DDI':raise ValueError('DDI verified patient/image subgroup resampling requires separate adapter')
        result['paired_intervals']={kind:paired_intervals(tasks['dense'+kind],tasks['med'+kind],partition=partition,check=check) for kind in ('7','2')}
    return result

def record_once(path,result,*,input_SHA,source_SHA):
    if not input_SHA or not source_SHA:raise ValueError('Input/source provenance required before recording')
    payload={'schema':'evaluation-record-v1','result':result,'input_SHA':input_SHA,'source_SHA':source_SHA,'lock_written':False}
    body=json.dumps(payload,sort_keys=True,allow_nan=False,indent=2)+'\n'
    # Exclusive creation: repeated evaluation cannot overwrite recorded outcomes.
    with Path(path).open('x',encoding='utf8') as f:f.write(body)
    return hashlib.sha256(body.encode()).hexdigest()

def permit_error_review(path,expected_SHA):
    p=Path(path)
    if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected_SHA:raise PermissionError('Recorded immutable metrics must precede error review')
    record=json.loads(p.read_text())
    if record.get('schema')!='evaluation-record-v1' or not record.get('input_SHA') or not record.get('source_SHA') or set(record.get('result',{}).get('systems',{}))!=set(SYSTEMS):raise PermissionError('Complete metric record required')
    return True
