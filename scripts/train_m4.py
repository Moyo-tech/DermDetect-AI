#!/usr/bin/env python3
"""Approved single-process M4 DenseNet launch; never downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import importlib.metadata
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dermdetect.models.densenet import DenseNetSystem, transform
from dermdetect.models.training_run import TrainingRun
from dermdetect.data.manifests import rooted_file, file_sha256


def main():
    parser = argparse.ArgumentParser()
    for key in ('configuration', 'approval', 'train', 'validation', 'data-root', 'checkpoint', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-worker-seconds', '--max-seconds', dest='max_worker_seconds', type=float, required=True)
    parser.add_argument('--stop-file')
    args = parser.parse_args()
    # Central approval contract is intentionally mandatory before payload/model access.
    from dermdetect.models.full_run_contracts import validate_execution_approval
    package, approval = validate_execution_approval(Path(args.configuration), Path(args.approval), stage='training')
    if args.max_worker_seconds > approval['allocation_seconds']['training'] - 300:
        raise ValueError('Worker allowance must retain five-minute allocation reserve')
    for name, version in package['software'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Pinned runtime software mismatch')
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    config = package['training']; binding = dict(package['binding'])
    binding['configuration_sha256'] = file_sha256(Path(args.configuration))
    rows = {}
    for split in ('train', 'validation'):
        path = Path(getattr(args, split))
        if file_sha256(path) != package['manifest_hashes'][split+'.json'] or binding['manifest_sha256'][split] != package['manifest_hashes'][split+'.json']:
            raise ValueError('Manifest binding mismatch')
        rows[split] = json.loads(path.read_text())
    checkpoint_hash = package['checkpoint_hashes']['weights/densenet/densenet121-a639ec97.pth']
    if file_sha256(Path(args.checkpoint)) != checkpoint_hash or binding['model_checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Pinned model checkpoint mismatch')
    root = Path(args.data_root)
    # Verify all bytes before starting optimizer work; route only explicit manifest payloads.
    for row in rows['train'] + rows['validation']:
        if file_sha256(rooted_file(root, row['path'])) != row['sha256']:
            raise ValueError('Image payload changed')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = DenseNetSystem(config['dropout'])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # Official torchvision DenseNet legacy keys must be canonicalized exactly.
    import re
    state = {re.sub(r'(denselayer\d+\.(?:norm|relu|conv))\.([12]\.)', r'\1\2', k): v
             for k, v in state.items() if not k.startswith('classifier.')}
    result = model.backbone.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {'classifier.1.weight', 'classifier.1.bias'} or result.unexpected_keys:
        raise ValueError('Pinned DenseNet state does not match the required encoder')
    def load(row, training, generator):
        path = rooted_file(root, row['path'])
        if file_sha256(path) != row['sha256']:
            raise ValueError('Image changed during run')
        with Image.open(path) as image:
            return transform(image, training=training, generator=generator)
    run = TrainingRun(model, rows['train'], rows['validation'], load, config, binding, args.output, device=args.device, allow_resume=bool(args.resume))
    if args.resume:
        run.resume(args.resume)
    carried = float(run.state['worker_seconds'])
    if carried + args.max_worker_seconds > approval['allocation_seconds']['training'] - 300:
        raise ValueError('Cumulative resumed worker allowance exceeds approved allocation reserve')
    result = run.run(max_worker_seconds=carried + args.max_worker_seconds,
                     stop_requested=lambda: stop[0] or (args.stop_file is not None and Path(args.stop_file).exists()))
    if result['complete']:
        started_reload = __import__('time').monotonic()
        result['selected_reload'] = run.verify_selected_reload(stop_requested=lambda: stop[0] or
            (args.stop_file is not None and Path(args.stop_file).exists()) or
            result['worker_seconds'] + __import__('time').monotonic() - started_reload >= carried + args.max_worker_seconds)
        result['worker_seconds'] += result['selected_reload']['worker_seconds']
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
