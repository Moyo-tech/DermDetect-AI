"""Read-only verification of separately approved manifests and scoped run grants.

This verifies trusted operator-supplied approval hashes, not a signature system or
protection against the machine owner. No API in this module creates a lock.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from .engine import SYSTEMS, CLASSES, BINARY, digest
from .metrics import NUMERICAL_POLICY

_TOKEN = object()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def route(root, relative):
    root = Path(root).resolve(strict=True)
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise PermissionError('Canonical relative artifact route required')
    p = root / p
    if any(q.is_symlink() for q in (p, *p.parents)) or not p.resolve().is_relative_to(root):
        raise PermissionError('Artifact symlink/escape forbidden')
    return p

def checked_file(root, binding):
    if not isinstance(binding, dict) or not re.fullmatch('[0-9a-f]{64}', binding.get('SHA', '')):
        raise PermissionError('Exact artifact SHA required')
    p = route(root, binding['path'])
    if not p.is_file() or ('bytes' in binding and p.stat().st_size != binding['bytes']) or file_sha(p) != binding['SHA']:
        raise PermissionError('Artifact integrity mismatch')
    return p

def input_root(root, reference):
    """Only for explicitly approved dataset locations, not code/model artifacts."""
    p=Path(reference)
    if not p.is_absolute():return route(root,reference)
    if '..' in p.parts or any(q.is_symlink() for q in (p,*p.parents)):
        raise PermissionError('Dataset input symlink/escape forbidden')
    return p.resolve(strict=True)

def checked_input_file(root,binding):
    p=input_root(root,binding['path'])
    if not re.fullmatch('[0-9a-f]{64}',binding.get('SHA','')) or not p.is_file() or ('bytes' in binding and p.stat().st_size!=binding['bytes']) or file_sha(p)!=binding['SHA']:
        raise PermissionError('Exact approved dataset-source integrity required')
    return p


def checked_runtime_file(binding):
    """Exact native runtime dependency; never a broad Homebrew read grant."""
    p=Path(binding['path'])
    if not p.is_absolute() or not str(p).startswith(('/opt/homebrew/opt/','/opt/homebrew/Cellar/')) or p.suffix not in ('.dylib','.so'):
        raise PermissionError('Named native runtime library required')
    resolved=p.resolve(strict=True)
    if not str(resolved).startswith('/opt/homebrew/Cellar/') or not resolved.is_file() or file_sha(resolved)!=binding['SHA']:
        raise PermissionError('Native runtime dependency hash mismatch')
    return p,resolved


def checked_json(root, binding):
    p = checked_file(root, binding)
    if p.stat().st_size > 16 * 1024 ** 2:
        raise PermissionError('Bounded JSON contract required')
    return json.loads(p.read_text())

def runtime_specs(system_manifest):
    """Normalize the unchanged M6 review system representation for the evaluator."""
    if set(system_manifest.get('systems', {})) != set(SYSTEMS):
        raise PermissionError('Joint four-system manifest required')
    specs = {}
    for name, s in system_manifest['systems'].items():
        k = 7 if name.endswith('7') else 2
        if s['classes'] != (list(CLASSES) if k == 7 else ['benign', 'malignant']):
            raise PermissionError('Canonical class order required')
        rules = s['confidence_rules']
        if [r['target_coverage'] for r in rules] != [.9, .8] or any(r['tie_policy'] != 'include_all_ties' for r in rules):
            raise PermissionError('Original confidence rules required')
        if k == 2 and (s['source_binary_mapping'] != BINARY or s['binary_classification'] != {'equality':'malignant','malignant_probability':.5}):
            raise PermissionError('Original binary rule required')
        artifact = s['selected_backbone'] if name == 'dense7' else s['probe']
        specs[name] = {'classes': list(range(k)), 'temperature': s['temperature']['temperature'],
                       'cutoffs': [r['cutoff'] for r in rules], 'confidence_tie_policy': 'include_all_ties',
                       'artifact_SHA': artifact['SHA']}
        if k == 2:
            specs[name].update(malignant_probability=.5, binary_equality='malignant')
    return specs

@dataclass(frozen=True, init=False)
class RunGrant:
    """Opaque grant created only after exact approval/source/artifact verification."""
    def __init__(self, token, *, root, partition, dataset, systems, evaluator, approval, lock_SHA, authority_files):
        if token is not _TOKEN:
            raise PermissionError('Use exact-file approval verification')
        object.__setattr__(self,'root',root)
        object.__setattr__(self,'partition',partition)
        object.__setattr__(self,'lock_SHA',lock_SHA)
        for key,value in [('dataset',dataset),('systems',systems),('evaluator',evaluator),('approval',approval)]:
            object.__setattr__(self,'_'+key,json.dumps(value,sort_keys=True,allow_nan=False))
        object.__setattr__(self,'_authority_files',json.dumps(authority_files,sort_keys=True))
        object.__setattr__(self, 'specs_SHA', digest(runtime_specs(systems)))
    @property
    def authority_files(self):return json.loads(self._authority_files)
    @property
    def dataset(self):return json.loads(self._dataset)
    @property
    def systems(self):return json.loads(self._systems)
    @property
    def evaluator(self):return json.loads(self._evaluator)
    @property
    def approval(self):return json.loads(self._approval)
    def check(self, partition, specs=None):
        if partition != self.partition or partition == 'DDI':
            raise PermissionError('Grant does not authorize this partition/adapter')
        if specs is not None and digest(specs) != self.specs_SHA:
            raise PermissionError('System/rule substitution forbidden')
    def check_rows(self, rows):
        from .pipeline import load_rows
        if digest(rows) != digest(load_rows(self)):
            raise PermissionError('Dataset membership/labels/routes differ from approval')
    def reverify(self):
        for binding in self.evaluator.get('runtime_files',[]):checked_runtime_file(binding)
        for binding in self.evaluator['files']:
            checked_file(self.root, binding)
        checked_file(self.root,self.systems['M5_source_configuration'])
        checked_file(self.root,self.dataset['manifest'])
        if self.dataset.get('payload_mode')=='zip':
            for binding in self.dataset['sources'].values():checked_input_file(self.root,binding)
        for name, system in self.systems['systems'].items():
            for key in ('probe', 'probe_metadata', 'temperature_record'):
                if key in system:
                    checked_file(self.root, system[key])
            weights = system['selected_backbone']
            for binding in weights if isinstance(weights, list) else [weights]:
                checked_file(self.root, binding)


def verify_grant(root, *, run_approval, system_manifest, evaluator_manifest,
                 acceptance, lock=None, lock_approval=None, execution_context="local_cpu"):
    """Bindings must come from the separately reviewed operator approval, not data.

    validation requires an approved development run; internal_test additionally
    requires a joint lock plus its separately approved lock hash. DDI is refused
    until its independently approved data/unit/dependence adapter exists.
    """
    root = Path(root).resolve(strict=True)
    a = checked_json(root, run_approval)
    if a.get('status') != 'user_approved' or not a.get('user_reference'):
        raise PermissionError('Explicit run approval required')
    partition = a.get('partition')
    if partition not in ('validation', 'internal_test'):
        raise PermissionError('HAM adapter only; DDI gates remain separate')
    for key, supplied in [('system_manifest', system_manifest), ('evaluator_manifest', evaluator_manifest), ('metric_acceptance', acceptance)]:
        if a.get(key) != supplied:
            raise PermissionError('Approval manifest binding mismatch')
    s = checked_json(root, system_manifest); e = checked_json(root, evaluator_manifest)
    accepted = checked_json(root, acceptance)
    if (accepted.get('status') != 'user_accepted' or not accepted.get('user_reference')
            or accepted.get('accepted_policy_SHA') != digest(NUMERICAL_POLICY)
            or set(accepted.get('accepted_conventions', [])) != {'AP','AURC','binary_Brier','undefined'}
            or e.get('numerical_policy_SHA') != digest(NUMERICAL_POLICY)):
        raise PermissionError('Exact accepted numerical policy required')
    runtime_specs(s)
    if not isinstance(e.get('files'), list) or not e['files'] or not e.get('runtime'):
        raise PermissionError('Exact evaluator/runtime source inventory required')
    # No held-out manifest or payload file is opened above or before these gates.
    lock_SHA = None
    if partition == 'internal_test':
        if a.get('held_out_access') is not True or not lock or not lock_approval or a.get('lock') != lock:
            raise PermissionError('Separate scoped unseal and compute approval required')
        l = checked_json(root, lock); la = checked_json(root, lock_approval)
        if (l.get('status') != 'joint_locked' or l.get('systems') != list(SYSTEMS)
                or l.get('system_manifest') != system_manifest or l.get('evaluator_manifest') != evaluator_manifest
                or l.get('metric_acceptance') != acceptance or l.get('lock_approval') != lock_approval
                or la.get('status') != 'user_approved' or la.get('lock_authorized') is not True
                or not la.get('user_reference') or la.get('system_manifest') != system_manifest
                or la.get('evaluator_manifest') != evaluator_manifest or la.get('metric_acceptance') != acceptance):
            raise PermissionError('Approved joint lock mismatch')
        lock_SHA = lock['SHA']
    elif a.get('held_out_access') is not False:
        raise PermissionError('Development approval must explicitly exclude held-out access')
    if execution_context in ('cloud_cpu','cloud_gpu'):
        from .cloud_ham import validate_contract
        validate_contract(a,execution_context)
        if execution_context=='cloud_gpu':
            from .gpu_parity import verify_receipt_before_unseal
            verify_receipt_before_unseal(root,a)
    elif execution_context != 'local_cpu' or a.get('compute_authorized') is not True or a.get('cloud') is not False:
        raise PermissionError('This entry supports separately approved local CPU compute only')
    if not isinstance(a.get('dataset'), dict) or 'manifest' not in a['dataset'] or a['dataset'].get('format') not in ('m4_development_json','m2_analysis_csv'):
        raise PermissionError('Exact evaluation dataset binding required')
    grant = RunGrant(_TOKEN, root=root, partition=partition, dataset=a['dataset'], systems=s,
                     evaluator=e, approval=a, lock_SHA=lock_SHA, authority_files=[run_approval,system_manifest,evaluator_manifest,acceptance]+([lock,lock_approval] if lock else []))
    grant.reverify()
    return grant
