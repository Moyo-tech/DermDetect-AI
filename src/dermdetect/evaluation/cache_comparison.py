"""Post-run comparison of fresh two-TRAIN features with the stored development caches.

Runs on the controller after a passed compatibility benchmark has been retrieved.
Reads only the retrieved ``benchmark.json`` and the two hash-pinned TRAIN/VAL
feature caches. Opens no images, loads no model, fits nothing and touches no
held-out data.

Pass rule fixed by the user before any run (2026-11-22, human review log):
elementwise float32 atol=rtol=1e-4 for both systems, unchanged. DenseNet is
compared with the M5 CPU cache; MedSigLIP with the M4 GPU-extracted cache. If
MedSigLIP fails only this cross-device comparison, the recorded consequence is
that M7 MedSigLIP inference uses a GPU. The tolerance is never loosened here.
"""
from pathlib import Path
import hashlib, json, math

ATOL = 1e-4
RTOL = 1e-4
DIMENSIONS = {'dense': 1024, 'med': 1152}
FEATURE_KEYS = ('features_SHA', 'features_sha256')
GPU_FALLBACK = 'M7 MedSigLIP inference uses a GPU (separate GPU budget/approval); tolerance unchanged'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(directory, metadata_sha256, image_ids, dimension):
    """Return {image_id: list[float]} for the requested ids from one pinned cache."""
    import numpy as np
    directory = Path(directory)
    if _sha(directory/'cache.json') != metadata_sha256:
        raise ValueError('Cache metadata hash differs from pinned value')
    metadata = json.loads((directory/'cache.json').read_text())
    declared = [metadata[k] for k in FEATURE_KEYS if k in metadata]
    if len(declared) != 1 or _sha(directory/'features.npy') != declared[0]:
        raise ValueError('Cache features hash differs from its metadata')
    if metadata.get('binding', {}).get('dimension') != dimension or metadata['binding'].get('dtype') != 'float32':
        raise ValueError('Cache dimension/dtype binding differs')
    ids = list(metadata['ids'])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate cache ids')
    array = np.load(directory/'features.npy', allow_pickle=False, mmap_mode='r')
    if array.dtype != np.float32 or array.shape != (len(ids), dimension):
        raise ValueError('Cache array shape/dtype differs')
    index = {v: i for i, v in enumerate(ids)}
    missing = [i for i in image_ids if i not in index]
    if missing:
        raise ValueError('Benchmark image ids absent from cache: ' + ','.join(missing))
    return {i: [float(x) for x in array[index[i]]] for i in image_ids}


def compare_vectors(fresh, cached):
    if len(fresh) != len(cached) or any(not math.isfinite(v) for v in fresh):
        raise ValueError('Finite equal-length vectors required')
    differences = [abs(a - b) for a, b in zip(fresh, cached)]
    within = [math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL) for a, b in zip(fresh, cached)]
    return {'passed': all(within), 'elements': len(fresh), 'elements_outside_tolerance': within.count(False),
            'max_absolute_difference': max(differences),
            'mean_absolute_difference': math.fsum(differences)/len(differences)}


def compare_benchmark(benchmark, caches):
    """benchmark: parsed benchmark.json; caches: {'dense': {id: vec}, 'med': {id: vec}}."""
    if benchmark.get('status') != 'passed' or benchmark.get('held_out_access') is not False:
        raise ValueError('Only a passed, non-held-out benchmark can be compared')
    reference = benchmark['reference']
    ids = [r['image_id'] for r in reference]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError('Exactly two distinct TRAIN reference views required')
    systems = {}
    for name, field in (('dense', 'dense_features'), ('med', 'med_features')):
        per_image = {r['image_id']: compare_vectors(r[field], caches[name][r['image_id']]) for r in reference}
        systems[name] = {'passed': all(v['passed'] for v in per_image.values()), 'images': per_image,
                         'max_absolute_difference': max(v['max_absolute_difference'] for v in per_image.values())}
    dense_ok, med_ok = systems['dense']['passed'], systems['med']['passed']
    if dense_ok and med_ok:
        decision = 'passed: CPU features reproduce both stored caches; CPU route admissible for M7'
    elif dense_ok:
        decision = 'MedSigLIP cross-device cache comparison failed: ' + GPU_FALLBACK
    else:
        decision = 'DenseNet CPU cache comparison failed: stop and report; no fallback is predefined'
    return {'status': 'passed' if dense_ok and med_ok else 'failed', 'decision': decision,
            'atol': ATOL, 'rtol': RTOL, 'image_ids': ids, 'systems': systems,
            'cache_sources': {'dense': 'M5 CPU cache', 'med': 'M4 GPU-extracted cache'},
            'held_out_access': False, 'model_loaded': False, 'fitting_performed': False}
