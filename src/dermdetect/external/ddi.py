"""Locked external DDI benign-vs-malignant evaluation of the frozen HAM binary systems.

Measures only: no fitting, calibration, threshold or model change. The HAM joint lock
and its system manifest (scientific digest 3398a83c...) are reused unchanged; this
adapter must itself be bound by a separately approved DDI add-on lock before any DDI
label is read. Evaluation unit is the image (the public DDI metadata has no patient
identifier); intervals therefore ignore within-patient dependence, a stated limitation.
"""
from dataclasses import dataclass
from pathlib import Path
import csv, hashlib, json, math, time

from dermdetect.evaluation import arrays, engine, metrics
from dermdetect.evaluation.authorization import checked_file, checked_json, runtime_specs
from dermdetect.evaluation.metrics import NUMERICAL_POLICY
from dermdetect.evaluation.pipeline import write_once

SCOPE = 'DDI_EXTERNAL_BINARY_V1'
SYSTEMS = ('dense2', 'med2')
SKIN_GROUPS = ('12', '34', '56')            # Fitzpatrick I-II, III-IV, V-VI
LABELS = {'1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
SCIENTIFIC_SYSTEMS_SHA = '3398a83c941cf2202674280649595c04aa4e69d6bdd1c19a72e9293a25cc88c2'
POLICY_SHA = '4cc2dcf40933490dd0721059eb5668152e0e7c55c6245030f5486a9d0ff39f0f'
INTERVAL_METRICS = ('AUROC', 'AP', 'binary_Brier')
_TOKEN = object()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True, init=False)
class DDIGrant:
    root: Path
    def __init__(self, token, *, root, approval, systems, evaluator, addon):
        if token is not _TOKEN: raise PermissionError('Verified DDI grant required')
        object.__setattr__(self, 'root', Path(root))
        for k, v in (('_approval', approval), ('_systems', systems), ('_evaluator', evaluator), ('_addon', addon)):
            object.__setattr__(self, k, json.dumps(v, sort_keys=True))
    @property
    def approval(self): return json.loads(self._approval)
    @property
    def systems(self): return json.loads(self._systems)
    @property
    def evaluator(self): return json.loads(self._evaluator)
    @property
    def addon(self): return json.loads(self._addon)
    def dataset_dir(self):
        d = Path(self.approval['dataset']['directory'])
        if not d.is_absolute() or '..' in d.parts or any(p.is_symlink() for p in (d, *d.parents)) or not d.is_dir():
            raise PermissionError('Approved absolute non-symlink DDI directory required')
        return d
    def reverify(self):
        for b in self.evaluator['files']: checked_file(self.root, b)
        checked_file(self.root, self.systems['M5_source_configuration'])
        for name in ('dense7', 'med7', 'dense2', 'med2'):
            s = self.systems['systems'][name]
            for k in ('probe', 'probe_metadata', 'temperature_record'):
                if k in s: checked_file(self.root, s[k])
            w = s['selected_backbone']
            for b in (w if isinstance(w, list) else [w]): checked_file(self.root, b)
        meta = self.dataset_dir()/'ddi_metadata.csv'
        if sha(meta) != self.approval['dataset']['metadata_SHA256']: raise PermissionError('DDI metadata changed')


def verify_ddi_grant(root, run_approval):
    """All authority checks happen before any DDI metadata row or image is read."""
    root = Path(root).resolve(strict=True)
    a = checked_json(root, run_approval)
    if (a.get('status') != 'user_approved' or not a.get('user_reference') or a.get('scope') != SCOPE
            or a.get('external_outcome_access') is not True or a.get('compute_authorized') is not True
            or a.get('systems') != list(SYSTEMS) or a.get('evaluation_unit') != 'image'
            or type(a.get('per_image_retention')) is not bool or a.get('DDI_tuning') is not False):
        raise PermissionError('Explicit scoped DDI outcome-unseal approval required')
    lock = checked_json(root, a['ham_lock']); la = checked_json(root, lock['lock_approval'])
    if (lock.get('status') != 'joint_locked' or la.get('status') != 'user_approved' or la.get('lock_authorized') is not True
            or lock.get('system_manifest') != la.get('system_manifest') or a.get('system_manifest') != lock['system_manifest']):
        raise PermissionError('Approved HAM joint lock with the same system manifest required')
    systems = checked_json(root, lock['system_manifest'])
    if engine.digest(systems['systems']) != SCIENTIFIC_SYSTEMS_SHA: raise PermissionError('Frozen scientific systems changed')
    runtime_specs(systems)
    addon = checked_json(root, a['ddi_addon_lock']); aa = checked_json(root, addon['addon_approval'])
    if (addon.get('status') != 'ddi_addon_locked' or aa.get('status') != 'user_approved' or aa.get('lock_authorized') is not True
            or not aa.get('user_reference') or addon.get('ham_lock') != a['ham_lock']
            or aa.get('ddi_evaluator_manifest') != addon.get('ddi_evaluator_manifest') or a.get('ddi_evaluator_manifest') != addon['ddi_evaluator_manifest']):
        raise PermissionError('Approved DDI add-on lock bound to the HAM lock required')
    ev = checked_json(root, addon['ddi_evaluator_manifest'])
    if ev.get('numerical_policy_SHA') != POLICY_SHA or engine.digest(NUMERICAL_POLICY) != POLICY_SHA:
        raise PermissionError('Accepted numerical policy required')
    integrity = checked_json(root, addon['integrity_record'])
    ds = a.get('dataset', {})
    if (integrity.get('status') != 'passed' or integrity.get('schema', {}).get('patient_identifier_present') is not False
            or ds.get('metadata_SHA256') != integrity.get('metadata_SHA256') or ds.get('metadata_SHA256') != ev.get('metadata_SHA256')
            or ds.get('image_inventory_SHA256') != integrity.get('image_inventory_SHA256') or ev.get('image_inventory_SHA256') != ds.get('image_inventory_SHA256')):
        raise PermissionError('Passed integrity record matching the bound DDI dataset required')
    inventory = checked_json(root, addon['image_inventory'])
    if hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest() != ds['image_inventory_SHA256']:
        raise PermissionError('Image inventory mismatch')
    g = DDIGrant(_TOKEN, root=root, approval=a, systems=systems, evaluator=ev, addon=addon)
    g.reverify()
    return g, {x['file']: x for x in inventory}


def load_rows(grant, inventory):
    """Read labels only through a verified grant; returns image-level rows."""
    if not isinstance(grant, DDIGrant): raise PermissionError('Verified DDI grant required')
    meta = grant.dataset_dir()/'ddi_metadata.csv'
    if sha(meta) != grant.approval['dataset']['metadata_SHA256']: raise PermissionError('DDI metadata changed')
    with meta.open(newline='') as f:
        raw = list(csv.DictReader(f))
    column = 'malignant' if raw and 'malignant' in raw[0] else 'malignancy(malig=1)'
    rows = []
    for r in raw:
        name = r['DDI_file']; lab = r[column].strip(); tone = r['skin_tone'].strip()
        if name not in inventory or lab not in LABELS or tone not in SKIN_GROUPS:
            raise ValueError('Row fails the validated DDI schema')
        rows.append({'image_id': 'DDI_'+r['DDI_ID'], 'file': name, 'sha256': inventory[name]['SHA256'],
                     'label': LABELS[lab], 'skin_group': tone})
    if len({r['image_id'] for r in rows}) != len(rows) or len(rows) != len(inventory):
        raise ValueError('DDI membership mismatch')
    return tuple(rows)


def read_image(grant, row):
    p = grant.dataset_dir()/row['file']
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 20*2**20: raise PermissionError('Bounded regular DDI image required')
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']: raise ValueError('DDI image checksum mismatch')
    return data


def image_draws(ids, labels, check=lambda: None):
    """1000 label-stratified image bootstrap weights, seed 20260729 (no patient IDs)."""
    return engine.paired_draws(ids, labels, tuple(ids), 2)


def interval_block(subset, logits, specs, check=lambda: None):
    """Point metrics per system plus paired 95% intervals (MedSigLIP - DenseNet)."""
    ids = tuple(r['image_id'] for r in subset); y = tuple(r['label'] for r in subset)
    out = {'units': len(ids)}
    for name in SYSTEMS:
        z = tuple(logits[name][i] for i in ids)
        out[name] = metrics.scores(z, y, specs[name]['temperature'], cutoffs=tuple(specs[name]['cutoffs']))
    if set(y) != {0, 1}:
        out['paired_intervals'] = metrics.na('intervals_require_both_classes'); return out
    values = {m: {'dense2': [], 'med2': [], 'med_minus_dense': []} for m in INTERVAL_METRICS}
    probs = {n: tuple(p[1] for p in metrics.probabilities(tuple(logits[n][i] for i in ids), specs[n]['temperature'])[0]) for n in SYSTEMS}
    for w in image_draws(ids, y):
        check(); res = {}
        for n in SYSTEMS:
            auc, ap = metrics.binary_ranking(y, probs[n], w)
            total = sum(w); brier = math.fsum(m*(q-t)**2 for q, t, m in zip(probs[n], y, w))/total
            res[n] = {'AUROC': auc['value'], 'AP': ap['value'], 'binary_Brier': brier}
        for m in INTERVAL_METRICS:
            a, b = res['dense2'][m], res['med2'][m]
            if a is None or b is None: raise ValueError('Undefined replicate; no retry')
            values[m]['dense2'].append(a); values[m]['med2'].append(b); values[m]['med_minus_dense'].append(b-a)
    out['paired_intervals'] = {'replicates': engine.REPLICATES, 'seed': engine.SEED, 'level': 0.95,
                               'resampling': 'label_stratified_image_bootstrap;no_patient_ids',
                               'intervals': {m: {k: engine.interval(v) for k, v in g.items()} for m, g in values.items()}}
    return out


def strip_per_image(block):
    """Aggregate-only mode: drop per-confidence curve points derived from single images."""
    for n in SYSTEMS:
        if isinstance(block.get(n), dict): block[n].pop('risk_coverage', None)
    return block


def run_ddi(grant, inventory, backend, output, *, check=lambda: None, read=read_image):
    if not isinstance(grant, DDIGrant): raise PermissionError('Verified DDI grant required')
    output = Path(output)
    if output.absolute() != Path(grant.approval['output']).absolute(): raise PermissionError('Output must equal approved destination')
    grant.reverify(); check()
    rows = load_rows(grant, inventory); specs = runtime_specs(grant.systems)
    probes = {n: arrays.portable_probe(checked_file(grant.root, grant.systems['systems'][n]['probe'])) for n in SYSTEMS}
    output.mkdir()
    began = time.monotonic(); stage = {'payload_seconds': 0., 'dense_encode_seconds': 0., 'med_encode_seconds': 0., 'probe_seconds': 0.}
    logits = {n: {} for n in SYSTEMS}
    for row in rows:
        check(); t = time.monotonic(); payload = read(grant, row); stage['payload_seconds'] += time.monotonic()-t
        t = time.monotonic(); _, dense_features = backend.dense(payload); stage['dense_encode_seconds'] += time.monotonic()-t
        check(); t = time.monotonic(); med_features = backend.med(payload); stage['med_encode_seconds'] += time.monotonic()-t
        t = time.monotonic()
        logits['dense2'][row['image_id']] = tuple(arrays.probe_logits(probes['dense2'], dense_features))
        logits['med2'][row['image_id']] = tuple(arrays.probe_logits(probes['med2'], med_features))
        stage['probe_seconds'] += time.monotonic()-t
    invariance = backend.finish(); check()
    if invariance.get('frozen_state_unchanged') is not True or invariance.get('RNG_unchanged') is not True:
        raise ValueError('Inference altered frozen model/RNG state')
    ordered = {n: {'image_ids': [r['image_id'] for r in rows], 'logits': [logits[n][r['image_id']] for r in rows]} for n in SYSTEMS}
    predictions_SHA = engine.digest(ordered)
    retain = grant.approval['per_image_retention']
    if retain:
        write_once(output/'predictions.json', {'partition': 'DDI', 'per_image_logits': ordered, 'labels_included': False})
    t = time.monotonic()
    result = {'partition': 'DDI', 'evaluation_unit': 'image', 'systems': list(SYSTEMS), 'predictions_SHA': predictions_SHA,
              'per_image_retained': retain, 'overall': interval_block(rows, logits, specs, check), 'skin_groups': {}}
    for g in SKIN_GROUPS:
        result['skin_groups'][g] = interval_block(tuple(r for r in rows if r['skin_group'] == g), logits, specs, check)
    if not retain:
        strip_per_image(result['overall'])
        for g in SKIN_GROUPS: strip_per_image(result['skin_groups'][g])
    stage['metrics_and_bootstrap_seconds'] = time.monotonic()-t
    metric_SHA = engine.record_once(output/'metrics.json', result, input_SHA=grant.approval['dataset']['metadata_SHA256'],
                                    source_SHA=grant.approval['ddi_evaluator_manifest']['SHA'])
    grant.reverify(); check()
    for row in rows: check(); read(grant, row)
    compute_SHA = write_once(output/'compute.json', {'stages': stage, 'backend': invariance, 'pipeline_wall_seconds': time.monotonic()-began,
                                                     'scope': 'locked DDI inference; development costs reported separately'})
    receipt = {'status': 'complete', 'schema': 'ddi-evaluation-receipt-v1', 'partition': 'DDI', 'metrics_SHA': metric_SHA, 'compute_SHA': compute_SHA,
               'predictions_SHA': predictions_SHA, 'per_image_retained': retain, 'dataset_metadata_SHA': grant.approval['dataset']['metadata_SHA256'],
               'ham_lock_SHA': grant.approval['ham_lock']['SHA'], 'ddi_addon_lock_SHA': grant.approval['ddi_addon_lock']['SHA'],
               'recorded_before_error_review': True, 'DDI_tuning': False}
    write_once(output/'receipt.json', receipt)
    return receipt
