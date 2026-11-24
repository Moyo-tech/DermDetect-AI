"""Finite CPU-only M5 orchestration. Callers enforce artifact access/approval."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from scipy.special import softmax, logsumexp
from .adaptation import (Probe, aggregate, physical_sample_weights, selection_key,
                         fit_temperature, confidence_cutoff, save_probe, load_probe)

CLASSES = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')
MAPPING = {'akiec': None, 'bcc': 1, 'bkl': 0, 'df': 0, 'mel': 1, 'nv': 0, 'vasc': 0}
POLICY = dict(seed=20260729, C_grid=[.01, .1, 1., 10.], solver='lbfgs', max_iter=1000,
              tol=.0001, scaler='unweighted_train_active_views', binary_mapping=MAPPING,
              weights='mean_one_physical_class_equal_unit_per_view',
              selection=['physical_macro_f1_desc', 'physical_NLL_asc', 'C_asc'],
              temperature_log_bounds=[-5, 5], temperature_xatol=1e-8,
              binary_classification_probability=.5, binary_equality='malignant',
              coverages=[.9, .8], confidence_ties='include_all_ties',
              tiny_overfit='first_sorted_physical_unit_per_class_C10',
              dummy='stratified_train_physical_priors_seed20260729',
              probe_reload_atol=1e-12, probe_reload_rtol=1e-12)


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2,
                             default=lambda x: x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n')
    tmp.replace(path)


def membership(rows):
    seen = set(); owners = {}; groups = {}; units = {'train': set(), 'validation': set()}
    for row in rows:
        s = row['split']; i = row['image_id']; u = row['physical_id']; y = row['target']
        if s not in units or i in seen or not i or not u or type(y) is not int or not 0 <= y < 7:
            raise ValueError('Unique canonical TRAIN/VAL membership required')
        seen.add(i); units[s].add(u)
        if u in owners and owners[u] != y: raise ValueError('Mixed physical label')
        owners[u] = y
        g = row.get('split_group_id', row.get('cluster_id', u))
        if g in groups and groups[g] != s: raise ValueError('Conservative group crosses split')
        groups[g] = s
    if not all(units.values()) or units['train'] & units['validation']:
        raise ValueError('Nonempty disjoint development physical units required')


def physical(rows, logits):
    ids, z = aggregate([r['image_id'] for r in rows], [r['physical_id'] for r in rows], logits)
    labels = {r['physical_id']: r['target'] for r in rows}
    return ids, z, [labels[u] for u in ids]


def fitted(c, x, y, units, classes):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probe = Probe(c, range(classes))
        probe.model.set_params(tol=POLICY['tol'])
        probe.fit(x, y, units, partition='train')
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError('Solver convergence warning; no automatic iteration increase: '+repr([str(w.message) for w in caught]))
    return probe, [str(w.message) for w in caught]


def candidate_task(name, rows, x, classes, output, check):
    output = Path(output); output.mkdir()
    train = np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val = np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    tr = [rows[i] for i in train]; vr = [rows[i] for i in val]
    y = [r['target'] for r in tr]; units = [r['physical_id'] for r in tr]
    if set(y) != set(range(classes)) or set(r['target'] for r in vr) != set(range(classes)):
        raise ValueError('Every configured class required in TRAIN and VAL')
    trials = []; selected = None
    for c in POLICY['C_grid']:
        check(); start = time.monotonic()
        probe, notices = fitted(c, x[train], y, units, classes)
        ids, z, vy = physical(vr, probe.logits(x[val])); key = selection_key(z, vy, c=c)
        directory = output/('C_'+str(c)); conf = dict(task=name, policy=POLICY)
        meta = save_probe(directory, probe, conf)
        restored = load_probe(directory, conf, meta)
        difference = float(np.max(np.abs(restored.logits(x[val])-probe.logits(x[val]))))
        if not np.allclose(restored.logits(x[val]), probe.logits(x[val]), atol=1e-12, rtol=1e-12):
            raise ValueError('Portable probe reload changed predictions')
        row = dict(C=c, selection_key=list(key), iterations=probe.model.n_iter_.tolist(),
                   wall_seconds=time.monotonic()-start, warnings=notices, metadata_SHA=meta,
                   reload_max_abs_difference=difference)
        trials.append(row); write(output/'candidates.json', trials)
        if selected is None or key < selected[0]: selected = (key, probe, ids, z, vy, row)
        check()
    # Prespecified diagnostics are separate from selection; all results retained.
    _, probe, ids, z, vy, chosen = selected
    owner = {r['physical_id']: r['target'] for r in tr}; counts = Counter(owner.values())
    rng = np.random.default_rng(POLICY['seed'])
    dy = rng.choice(classes, len(vy), p=[counts[k]/len(owner) for k in range(classes)])
    dummy_f1 = float(f1_score(vy, dy, labels=range(classes), average='macro', zero_division=0))
    first = {k:min(u for u in owner if owner[u]==k) for k in range(classes)}
    small = [i for i,u in enumerate(units) if u in first.values()]
    check(); tiny, notices = fitted(10., x[train][small], [y[i] for i in small], [units[i] for i in small], classes)
    _, tz, ty = physical([tr[i] for i in small], tiny.logits(x[train][small]))
    overfit_f1 = -selection_key(tz, ty, c=10.)[0]
    sanity = dict(dummy_macro_f1=dummy_f1, selected_validation_macro_f1=-selected[0][0],
                  tiny_train_macro_f1=overfit_f1, tiny_physical_ids=sorted(first.values()),
                  review_required=(-selected[0][0] < dummy_f1 or overfit_f1 < 1.), warnings=notices)
    write(output/'sanity.json', sanity)
    return dict(physical_ids=list(ids), logits=z, labels=vy, selected=chosen, sanity=sanity)


def calibration_scores(logits, labels, temperature):
    p=softmax(logits/temperature,axis=1);y=np.asarray(labels);confidence=p.max(axis=1)
    correct=p.argmax(axis=1)==y;bins=[];ece=0.
    for b in range(15):
        keep=(confidence>=b/15)&((confidence<(b+1)/15) if b<14 else (confidence<=1))
        n=int(keep.sum());mean=float(confidence[keep].mean()) if n else None;accuracy=float(correct[keep].mean()) if n else None
        if n:ece+=n/len(y)*abs(mean-accuracy)
        bins.append(dict(bin=b,count=n,mean_confidence=mean,accuracy=accuracy))
    z=logits/temperature
    return dict(NLL=float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])),
                Brier=float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1))),ECE=ece,ECE_bins=bins)


def decision_artifact(name, ids, logits, labels, output, check):
    check(); result = fit_temperature(logits, labels, partition='validation')
    np.save(Path(output)/(name+'_physical_validation_logits.npy'), logits, allow_pickle=False)
    write(Path(output)/(name+'_physical_validation_labels.json'), dict(physical_ids=list(ids), labels=list(labels)))
    write(Path(output)/(name+'_temperature.json'), result)
    if result['at_bound']: raise ValueError('Temperature reached numerical bound; review required')
    cutoffs = [confidence_cutoff(logits, result['temperature'], q, partition='validation',
                               tie_policy='include_all_ties') for q in POLICY['coverages']]
    return dict(physical_ids=list(ids), temperature=result, abstention=cutoffs,
                validation_scores_before=calibration_scores(logits,labels,1.),
                validation_scores_after=calibration_scores(logits,labels,result['temperature']),
                binary_classification=dict(malignant_probability=.5, equality='malignant') if logits.shape[1]==2 else None)


def run(rows, med_ids, med_features, dense_ids, dense_features, dense_reference, output, *,
        policy, max_seconds, after_check=lambda: None):
    if policy != POLICY or max_seconds <= 0: raise ValueError('Exact reviewed M5 policy and budget required')
    membership(rows); ids = [r['image_id'] for r in rows]
    if list(med_ids) != ids or list(dense_ids) != ids: raise ValueError('Ordered cache membership differs')
    for x in (med_features, dense_features):
        if x.ndim != 2 or len(x)!=len(rows) or not np.isfinite(x).all() or not np.any(np.var(x,axis=0)>0):
            raise ValueError('Finite nondegenerate exact cache required')
    output = Path(output); output.mkdir(); start=time.monotonic()
    def check():
        if time.monotonic()-start >= max_seconds: raise TimeoutError('M5 CPU budget exhausted')
        after_check()
    try:
        vr=[r for r in rows if r['split']=='validation']; expected=sorted({r['physical_id'] for r in vr})
        labels={r['physical_id']:r['target'] for r in vr}; ref=np.asarray(dense_reference['logits'],dtype=np.float64)
        if dense_reference['physical_ids'] != expected or ref.shape!=(len(expected),7) or not np.isfinite(ref).all():
            raise ValueError('Selected DenseNet physical-validation reference mismatch')
        task=candidate_task('med7',rows,med_features,7,output/'med7',check)
        tasks={'med7':task, 'dense7':dict(physical_ids=expected,logits=ref,labels=[labels[u] for u in expected])}
        keep=[i for i,r in enumerate(rows) if MAPPING[CLASSES[r['target']]] is not None]
        binary=[dict(rows[i],target=MAPPING[CLASSES[rows[i]['target']]]) for i in keep]
        for name,x in [('med2',med_features),('dense2',dense_features)]:
            tasks[name]=candidate_task(name,binary,x[keep],2,output/name,check)
        decisions={name:decision_artifact(name,t['physical_ids'],t['logits'],t['labels'],output,check) for name,t in tasks.items()}
        check(); result=dict(status='passed',wall_seconds=time.monotonic()-start,policy=policy,
                             decisions=decisions,review_required=any(t.get('sanity',{}).get('review_required',False) for t in tasks.values()),
                             primary_candidate_fits=12,diagnostic_fits=3,held_out_access=False)
        write(output/'result.json',result); return result
    except BaseException as error:
        write(output/'failure.json',dict(error_type=type(error).__name__,message=str(error),wall_seconds=time.monotonic()-start))
        raise
