"""Fail-closed configuration and explicit execution gates for full M4 workers."""
import hashlib
import json
import math
from pathlib import Path

CLASS_ORDER = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate_execution_approval(config_path, approval_path, *, stage):
    """No credential setup. Caller must gate before decoding/model loading.

    Approval is an authority record, not a malicious-code security primitive.
    It binds one exact reviewed package; no guessed or inherited authorization.
    """
    if stage not in {'training', 'extraction'}:
        raise ValueError('Unknown full-run stage')
    config = json.loads(Path(config_path).read_text())
    approval = json.loads(Path(approval_path).read_text())
    if (approval.get('full_execution_approved') is not True or
        approval.get('config_sha256') != sha256_file(config_path) or
        not isinstance(approval.get('user_authority'), str) or not approval['user_authority'].strip() or
        stage not in approval.get('stages', [])):
        raise PermissionError('Exact full-run execution approval is absent or mismatched')
    if config.get('class_order') != list(CLASS_ORDER) or config.get('seed') != 20260729:
        raise ValueError('Frozen class order/seed mismatch')
    if config.get('configuration_decisions_accepted') is not True:
        raise PermissionError('Explicit numerical configuration decisions remain unaccepted')
    training=config.get('training', {})
    fixed={'batch_size':16,'head_epochs':2,'final_block_epochs':20,'dropout':.2,'head_lr':.001,'final_block_lr':.0001,'weight_decay':.0001,'patience':5,'precision':'float32','TF32':False}
    if any(training.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Carried-forward DenseNet numerical configuration changed')
    if training.get('phase2_head_lr') != .001 or training.get('patience_scope') != 'all_evaluated_epochs' or training.get('optimizer_transition') != 'retain_head_add_final_block':
        raise ValueError('Explicit reviewed optimizer/patience policy mismatch')
    extraction=config.get('extraction', {})
    if extraction.get('batch_size')!=1 or extraction.get('binding', {}).get('dimension')!=1152 or extraction.get('binding', {}).get('revision')!='9cea28a1a1195f665105faa6e8544c112fd960a4':
        raise ValueError('Pinned extraction contract changed')
    if config.get('runtime_environment')!={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}:
        raise ValueError('Reviewed deterministic CUDA environment changed')
    if config.get('resource_guards', {}).get('persistent_storage_max_bytes') != 42*10**9 or approval.get('persistent_storage_max_bytes') != 42*10**9:
        raise ValueError('Exact approval-bound42GB persistent storage ceiling required')
    seconds = approval.get('allocation_seconds', {}).get(stage)
    credits = approval.get('additional_existing_credit_cap')
    if (type(seconds) is not int or not 300 < seconds <= {'training': 10800, 'extraction': 3600}[stage] or
        type(credits) not in (int, float) or not math.isfinite(credits) or not 0 < credits <= 3.30):
        raise ValueError('Reviewed allocation/credit ceilings required')
    if (approval.get('same_private_Studio') is not True or
        approval.get('manual_Sleep_required') is not True or
        approval.get('local_IPC_residual_risk_accepted') is not True or
        approval.get('held_out_authorized') is not False):
        raise PermissionError('Destination/stop/data scope is not accepted')
    return config, approval

def rooted_file(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p in {'..', '.', ''} for p in rel.parts):
        raise ValueError('Canonical relative task file required')
    target = root / rel
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root):
        raise ValueError('Symlink task file forbidden')
    if root not in target.resolve().parents or not target.is_file():
        raise ValueError('File escapes approved task root or is absent')
    return target
