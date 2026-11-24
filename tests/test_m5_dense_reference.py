"""Invented checkpoint/features only: CPU Dense features versus stored VAL logits."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dermdetect.models.adaptation import aggregate
from dermdetect.models.dense_reference_check import verify_dense_reference
from dermdetect.models.densenet import DenseNetSystem


def invented(seed=20260729):
    rng = np.random.default_rng(seed)
    rows = [dict(image_id=f'v{i:03d}', physical_id=f'u{i//2:03d}', split='validation', target=(i//2) % 7) for i in range(28)]
    train = [dict(image_id=f't{i:03d}', physical_id=f'w{i:03d}', split='train', target=i % 7) for i in range(14)]
    ids = sorted(r['image_id'] for r in rows+train)
    x = np.maximum(rng.normal(0, 1, (len(ids), 1024)), 0).astype(np.float32)
    model = DenseNetSystem(.2).state_dict()
    w = model['backbone.classifier.1.weight'].double().numpy(); b = model['backbone.classifier.1.bias'].double().numpy()
    pos = {i: k for k, i in enumerate(ids)}
    logits = x[[pos[r['image_id']] for r in rows]].astype(np.float64) @ w.T + b
    units, z = aggregate([r['image_id'] for r in rows], [r['physical_id'] for r in rows], logits)
    checkpoint = dict(model=model, membership=dict(train=train, validation=rows),
                      validation_reference=dict(physical_ids=list(units), logits=z.tolist()),
                      state=dict(best_key=[-.5, 1., 21]))
    return checkpoint, ids, x


class DenseReferenceTests(unittest.TestCase):
    def test_exact_cpu_head_reproduces_reference(self):
        c, ids, x = invented(); r = verify_dense_reference(c, ids, x)
        self.assertTrue(r['passed']); self.assertEqual(r['physical_units'], 14); self.assertEqual(r['validation_views'], 28)
        self.assertLess(r['max_absolute_difference'], 1e-5); self.assertEqual(r['argmax_agreement'], 1.)
        self.assertFalse(r['fitting_performed']); self.assertFalse(r['held_out_access'])

    def test_device_scale_noise_passes_but_drift_fails(self):
        c, ids, x = invented()
        small = x + np.float32(1e-7)
        self.assertTrue(verify_dense_reference(c, ids, small)['passed'])
        drift = x.copy(); drift[ids.index('v000')] += np.float32(.05)
        r = verify_dense_reference(c, ids, drift)
        self.assertFalse(r['passed']); self.assertGreater(r['max_absolute_difference'], 1e-4)

    def test_cache_order_is_resolved_by_id_not_position(self):
        c, ids, x = invented(); order = np.random.default_rng(1).permutation(len(ids))
        self.assertTrue(verify_dense_reference(c, [ids[i] for i in order], x[order])['passed'])

    def test_structural_mismatches_raise(self):
        c, ids, x = invented()
        k = ids.index('v000'); keep = [i for i in range(len(ids)) if i != k]
        with self.assertRaisesRegex(ValueError, 'lacks'): verify_dense_reference(c, [ids[i] for i in keep], x[keep])
        with self.assertRaisesRegex(ValueError, '1024'): verify_dense_reference(c, ids, x[:, :512])
        bad = dict(c, validation_reference=dict(c['validation_reference'], physical_ids=c['validation_reference']['physical_ids'][::-1]))
        with self.assertRaisesRegex(ValueError, 'units'): verify_dense_reference(bad, ids, x)
        with self.assertRaisesRegex(ValueError, 'lacks'): verify_dense_reference({k: v for k, v in c.items() if k != 'model'}, ids, x)
        nan = x.copy(); nan[ids.index('v003'), 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): verify_dense_reference(c, ids, nan)

    def test_script_records_once_and_rejects_recovery(self):
        import importlib.util, hashlib
        p = Path(__file__).resolve().parents[1]/'scripts/verify_m5_dense_reference.py'
        spec = importlib.util.spec_from_file_location('verify_dense', p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        c, ids, x = invented()
        with tempfile.TemporaryDirectory() as d:
            D = Path(d); cache = D/'artifacts'; cache.mkdir()
            np.save(cache/'features.npy', x, allow_pickle=False)
            (cache/'cache.json').write_text(json.dumps(dict(ids=ids, features_SHA=hashlib.sha256((cache/'features.npy').read_bytes()).hexdigest())))
            ck = D/'best_epoch_021.pt'; torch.save(c, ck); out = D/'check/result.json'
            self.assertEqual(m.main(['--checkpoint', str(ck), '--dense-cache', str(cache), '--output', str(out)]), 0)
            self.assertTrue(json.loads(out.read_text())['passed'])
            with self.assertRaises(FileExistsError): m.main(['--checkpoint', str(ck), '--dense-cache', str(cache), '--output', str(out)])
            rec = D/'recovery.pt'; torch.save(c, rec)
            with self.assertRaisesRegex(ValueError, 'recovery'): m.main(['--checkpoint', str(rec), '--dense-cache', str(cache), '--output', str(D/'x.json')])
            with self.assertRaisesRegex(ValueError, 'pinned'):
                m.main(['--checkpoint', str(ck), '--dense-cache', str(cache), '--output', str(D/'y.json'), '--expected-checkpoint-sha256', '0'*64])


if __name__ == '__main__':
    unittest.main()
