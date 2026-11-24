"""Invented-data tests for the post-run benchmark-to-cache comparison."""
from pathlib import Path
import hashlib, importlib.util, json, sys, tempfile, unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dermdetect.evaluation.cache_comparison import ATOL, RTOL, compare_benchmark, load_rows

IDS = ['ISIC_A', 'ISIC_B', 'ISIC_C']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_cache(root, dimension, key, seed):
    root.mkdir()
    array = np.random.default_rng(seed).standard_normal((len(IDS), dimension)).astype(np.float32)
    np.save(root/'features.npy', array, allow_pickle=False)
    meta = {'binding': {'dimension': dimension, 'dtype': 'float32'}, 'ids': IDS, key: sha(root/'features.npy')}
    (root/'cache.json').write_text(json.dumps(meta))
    return array, sha(root/'cache.json')


def benchmark(dense, med, ids=('ISIC_A', 'ISIC_B')):
    ref = [{'image_id': i, 'dense_features': [float(x) for x in dense[IDS.index(i)]],
            'med_features': [float(x) for x in med[IDS.index(i)]], 'logits': {}} for i in ids]
    return {'status': 'passed', 'held_out_access': False, 'reference': ref}


class CacheComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        self.dense, self.dense_sha = make_cache(d/'dense', 1024, 'features_SHA', 1)
        self.med, self.med_sha = make_cache(d/'med', 1152, 'features_sha256', 2)
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def caches(self, ids=('ISIC_A', 'ISIC_B')):
        return {'dense': load_rows(self.d/'dense', self.dense_sha, list(ids), 1024),
                'med': load_rows(self.d/'med', self.med_sha, list(ids), 1152)}

    def test_identical_features_pass(self):
        r = compare_benchmark(benchmark(self.dense, self.med), self.caches())
        self.assertEqual(r['status'], 'passed'); self.assertEqual((r['atol'], r['rtol']), (1e-4, 1e-4))
        self.assertEqual(r['systems']['med']['max_absolute_difference'], 0.0)
        self.assertFalse(r['held_out_access']); self.assertFalse(r['model_loaded'])

    def test_tolerance_is_fixed_at_1e4(self):
        self.assertEqual((ATOL, RTOL), (1e-4, 1e-4))

    def test_small_difference_within_tolerance_passes(self):
        med = self.med.copy(); med[0, 5] += 5e-5
        self.assertEqual(compare_benchmark(benchmark(self.dense, med), self.caches())['status'], 'passed')

    def test_med_only_failure_selects_gpu_fallback(self):
        med = self.med.copy(); med[1, 0] += 10.0
        r = compare_benchmark(benchmark(self.dense, med), self.caches())
        self.assertEqual(r['status'], 'failed'); self.assertTrue(r['systems']['dense']['passed'])
        self.assertIn('uses a GPU', r['decision']); self.assertIn('tolerance unchanged', r['decision'])
        self.assertEqual(r['systems']['med']['images']['ISIC_B']['elements_outside_tolerance'], 1)

    def test_dense_failure_stops_without_fallback(self):
        dense = self.dense.copy(); dense[0, 0] += 1.0
        r = compare_benchmark(benchmark(dense, self.med), self.caches())
        self.assertEqual(r['status'], 'failed'); self.assertIn('no fallback', r['decision'])

    def test_refuses_failed_or_heldout_benchmark(self):
        for change in ({'status': 'failed'}, {'held_out_access': True}):
            with self.assertRaises(ValueError):
                compare_benchmark(dict(benchmark(self.dense, self.med), **change), self.caches())

    def test_requires_exactly_two_distinct_views(self):
        with self.assertRaises(ValueError):
            compare_benchmark(benchmark(self.dense, self.med, ids=('ISIC_A',)), self.caches(('ISIC_A',)))

    def test_cache_hash_and_id_checks(self):
        with self.assertRaises(ValueError):
            load_rows(self.d/'med', '0'*64, ['ISIC_A'], 1152)
        with self.assertRaises(ValueError):
            load_rows(self.d/'med', self.med_sha, ['ISIC_Z'], 1152)
        with self.assertRaises(ValueError):
            load_rows(self.d/'med', self.med_sha, ['ISIC_A'], 1024)
        np.save(self.d/'med'/'features.npy', self.med*2, allow_pickle=False)
        with self.assertRaises(ValueError):
            load_rows(self.d/'med', self.med_sha, ['ISIC_A'], 1152)

    def test_script_writes_once_and_exit_codes(self):
        spec = importlib.util.spec_from_file_location('cmp', ROOT/'scripts/compare_benchmark_to_caches.py')
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        b = self.d/'benchmark.json'; b.write_text(json.dumps(benchmark(self.dense, self.med)))
        args = ['--benchmark', str(b), '--dense-cache', str(self.d/'dense'), '--med-cache', str(self.d/'med'),
                '--dense-metadata-sha256', self.dense_sha, '--med-metadata-sha256', self.med_sha,
                '--output', str(self.d/'out/cmp.json')]
        self.assertEqual(m.main(args), 0)
        with self.assertRaises(FileExistsError):
            m.main(args)


if __name__ == '__main__':
    unittest.main()
