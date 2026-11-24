#!/usr/bin/env python3
"""Compare a retrieved two-TRAIN benchmark.json with the pinned development caches.

Controller-side and read-only: no images, model, fitting, network or held-out data.
Writes one new JSON record and never overwrites. Exit 0 pass, 2 fail.

Example (paths relative to the private project root):
  python -B scripts/compare_benchmark_to_caches.py \
    --benchmark <retrieved>/benchmark.json \
    --dense-cache outputs/m5-CPU-reactivation-execution-20261007/download_20261007T142013Z/features/artifacts \
    --med-cache outputs/m4-Med-private-cache-20261004 \
    --output <new>/cache_comparison.json
"""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

# Pinned metadata hashes, as bound by the M6 system review manifest.
DENSE_METADATA_SHA256 = '0de96b0a98988f75020b4aa03801c651a506762e01f7be432515212f47e7ff01'
MED_METADATA_SHA256 = '8feb14d1f088a2fc2188705b0b17812ed1a4b605c793997e36e7d1000ecde6c8'


def main(argv=None):
    from dermdetect.evaluation.cache_comparison import DIMENSIONS, compare_benchmark, load_rows
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--dense-cache', type=Path, required=True)
    p.add_argument('--med-cache', type=Path, required=True)
    p.add_argument('--dense-metadata-sha256', default=DENSE_METADATA_SHA256)
    p.add_argument('--med-metadata-sha256', default=MED_METADATA_SHA256)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Refusing to overwrite an existing comparison record')
    benchmark = json.loads(a.benchmark.read_text())
    ids = [r['image_id'] for r in benchmark['reference']]
    caches = {'dense': load_rows(a.dense_cache, a.dense_metadata_sha256, ids, DIMENSIONS['dense']),
              'med': load_rows(a.med_cache, a.med_metadata_sha256, ids, DIMENSIONS['med'])}
    result = compare_benchmark(benchmark, caches)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({'status': result['status'], 'decision': result['decision']}))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    sys.exit(main())
