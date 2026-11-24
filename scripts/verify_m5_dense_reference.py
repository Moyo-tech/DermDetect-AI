#!/usr/bin/env python3
"""Read-only pre-fitting check: CPU Dense features reproduce stored VAL logits.

Reads only the selected checkpoint and the Dense feature cache directory
(features.npy + cache.json). Opens no images, fits nothing, touches no
held-out data and makes no network calls. Writes one new JSON record.
Peak memory is roughly the checkpoint plus a ~35 MB feature matrix.

Example (paths relative to the private project root):
  python -B scripts/verify_m5_dense_reference.py \
    --checkpoint outputs/m4-trained-private-handoff-20261004/best_epoch_021.pt \
    --dense-cache outputs/m5-CPU-reactivation-execution-20261007/download_20261007T142013Z/features/artifacts \
    --output outputs/m5-dense-reference-check/result.json
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dense-cache', type=Path, required=True, help='directory holding features.npy and cache.json')
    p.add_argument('--output', type=Path, required=True, help='new JSON record path (never overwritten)')
    p.add_argument('--expected-checkpoint-sha256', help='optional pinned checkpoint hash')
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Refusing to overwrite an existing check record')
    if a.checkpoint.name != 'best_epoch_021.pt':
        raise ValueError('Use the selected best_epoch_021.pt, not recovery.pt')
    import numpy as np
    import torch
    from dermdetect.models.dense_reference_check import verify_dense_reference
    started = time.monotonic()
    checkpoint_sha = sha(a.checkpoint)
    if a.expected_checkpoint_sha256 and checkpoint_sha != a.expected_checkpoint_sha256:
        raise ValueError('Checkpoint hash differs from pinned value')
    metadata = json.loads((a.dense_cache/'cache.json').read_text())
    features_sha = sha(a.dense_cache/'features.npy')
    if metadata.get('features_SHA') != features_sha:
        raise ValueError('Dense features.npy hash differs from cache.json')
    features = np.load(a.dense_cache/'features.npy', allow_pickle=False, mmap_mode='r')
    if features.dtype != np.float32:
        raise ValueError('Dense cache must be float32')
    value = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if value.get('state', {}).get('best_key', [None]*3)[2] != 21:
        raise ValueError('Checkpoint is not the selected epoch-21 model')
    result = verify_dense_reference(value, metadata['ids'], features)
    result.update(checkpoint_sha256=checkpoint_sha, dense_features_sha256=features_sha,
                  dense_metadata_sha256=sha(a.dense_cache/'cache.json'),
                  wall_seconds=time.monotonic()-started)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
