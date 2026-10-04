"""Explicitly approved offline M4 batch-one extraction; no automatic downloads."""
import argparse
import json
import importlib.metadata
import signal
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def validate_budget(seconds, config, approval):
    if (not math.isfinite(seconds) or not 0 < seconds <= approval['allocation_seconds']['extraction'] - 300
            or config.get('shard_size') != 64):
        raise ValueError('Finite approved execution budget and reviewed shard size required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    from dermdetect.models.full_run_contracts import validate_execution_approval
    full_config, approval = validate_execution_approval(args.config, args.approval, stage='extraction')
    config = full_config['extraction']
    validate_budget(args.max_seconds, config, approval)
    if args.max_seconds <= 0 or config['binding']['dimension'] != 1152 or config.get('batch_size') != 1:
        raise ValueError('Positive execution budget and fixed batch-one 1152 contract required')
    if config['binding'].get('checkpoint_files_sha256') != config['snapshot_files']:
        raise ValueError('Exact checkpoint file hashes must be retained in cache binding')
    for distribution, expected in config['binding']['software'].items():
        if importlib.metadata.version(distribution) != expected:
            raise ValueError('Installed extraction software version changed')
    stop_requested = [False]
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.__setitem__(0, True))
    import torch
    from PIL import Image
    from dermdetect.models.embeddings import load_local_encoder, freeze_image_features
    from dermdetect.models.tf_source_resize import evaluation_resize
    from dermdetect.models.extraction_run import run_extraction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    began = time.monotonic()
    model, processor = load_local_encoder(args.snapshot, expected_files=config['snapshot_files'], revision=config['binding']['revision'])
    model.to(args.device)
    raw_processor = json.loads((args.snapshot / 'preprocessor_config.json').read_text())
    if raw_processor != config['binding']['processor']:
        raise ValueError('Processor binding changed')
    if (processor.image_mean != [.5, .5, .5] or processor.image_std != [.5, .5, .5]
            or processor.size != {'height': 448, 'width': 448}):
        raise ValueError('Pinned MedSigLIP processor normalization/size required')
    items = []
    for row in config['items']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['sha256'] != config['binding']['image_sha256'][row['image_id']]:
            raise ValueError('Invalid bound payload route')
        path = args.payload_root / relative
        if not path.resolve().is_relative_to(args.payload_root.resolve()):
            raise ValueError('Payload route escapes approved root')
        items.append((row['image_id'], path))

    def encode(identifier, path):
        with Image.open(path) as image:
            pixels = processor(images=evaluation_resize(image), do_resize=False, return_tensors='pt')['pixel_values']
        if pixels.dtype != torch.float32:
            raise ValueError('Float32 processor output required')
        result = freeze_image_features(model, pixels.to(args.device))[0].numpy()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        return result

    result = run_extraction(items=items, key=config['binding'], work_directory=args.work,
                            cache_directory=args.cache, encode=encode, shard_size=64,
                            should_stop=lambda: (stop_requested[0] or (args.stop_file is not None and args.stop_file.exists())
                                                or time.monotonic() - began >= args.max_seconds))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
