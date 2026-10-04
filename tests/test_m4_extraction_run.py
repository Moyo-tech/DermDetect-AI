"""Invented files/tiny CPU encoder only; interrupted extraction recovery contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from dermdetect.models.embeddings import binding, load_cache, freeze_image_features
from dermdetect.models.extraction_run import run_extraction, file_sha


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]]))
            self.linear.bias.fill_(.1)

    def get_image_features(self, pixel_values):
        return self.linear(pixel_values)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = []
        for index in range(5):
            path = self.root / f'image-{index}.txt'
            path.write_text(str(index + 1))
            self.items.append((f'invented-{index}', path))
        self.key = binding(manifest_sha256='a'*64, image_sha256={i:file_sha(p) for i,p in self.items},
                           revision='b'*40, processor={'invented':True}, software={'fixture':'1'}, dimension=4)
        self.encoder = TinyEncoder()
        self.calls = []

    def encode(self, identifier, path):
        self.calls.append(identifier)
        value = float(path.read_text())
        pixels = torch.tensor([[value, value*value, 1.]])
        return freeze_image_features(self.encoder, pixels)[0].numpy()

    def run_case(self, suffix='one', **kwargs):
        return run_extraction(items=self.items, key=self.key, work_directory=self.root / (suffix+'-work'),
                              cache_directory=self.root / (suffix+'-cache'), encode=self.encode, shard_size=2, **kwargs)

    def test_cpu_encoder_end_to_end_immutable_reload(self):
        state = self.run_case()
        self.assertTrue(state['complete'])
        self.assertFalse(any(p.requires_grad for p in self.encoder.parameters()))
        ids, matrix = load_cache(self.root/'one-cache', self.key, state['metadata_sha256'])
        self.assertEqual(ids, tuple(i for i,_ in self.items))
        self.assertEqual(matrix.shape, (5,4))
        count = len(self.calls)
        again = self.run_case()
        self.assertTrue(again['complete'])
        self.assertEqual(count, len(self.calls))

    def test_stop_resume_matches_uninterrupted_without_duplicate_encoding(self):
        full = self.run_case('full')
        self.calls.clear()
        stopped = self.run_case(should_stop=lambda: len(self.calls) == 3)
        self.assertFalse(stopped['complete'])
        resumed = self.run_case()
        self.assertEqual(self.calls, [i for i,_ in self.items])
        _, expected = load_cache(self.root/'full-cache', self.key, full['metadata_sha256'])
        _, actual = load_cache(self.root/'one-cache', self.key, resumed['metadata_sha256'])
        np.testing.assert_array_equal(actual, expected)

    def test_encoder_failure_redoes_only_uncommitted_rows(self):
        original = self.encode
        def fail(identifier, path):
            if identifier == 'invented-3':
                raise RuntimeError('invented interruption')
            return original(identifier,path)
        with self.assertRaises(RuntimeError):
            with patch.object(self, 'encode', fail):
                self.run_case()
        self.calls.clear()
        self.assertTrue(self.run_case()['complete'])
        self.assertEqual(self.calls, ['invented-2','invented-3','invented-4'])

    def test_altered_source_or_shard_rejected_before_new_encoder_calls(self):
        self.run_case(should_stop=lambda: len(self.calls) == 2)
        self.calls.clear()
        shard = self.root/'one-work'/'shard-000000.npy'
        shard.write_bytes(shard.read_bytes()+b'changed')
        with self.assertRaises(ValueError): self.run_case()
        self.assertEqual(self.calls, [])
        self.items[0][1].write_text('changed')
        with self.assertRaises(ValueError): self.run_case()
        self.assertEqual(self.calls, [])

    def test_changed_order_binding_and_duplicates_rejected(self):
        self.run_case(should_stop=lambda: True)
        self.items.reverse()
        with self.assertRaises(ValueError): self.run_case()
        self.items.reverse()
        self.key['processor'] = {'changed': True}
        with self.assertRaises(ValueError): self.run_case()
        self.items.append(self.items[0])
        with self.assertRaises(ValueError): self.run_case()

    def test_invalid_feature_and_zero_variance_never_finalize(self):
        with patch.object(self, 'encode', lambda i,p: np.ones(4,dtype=np.float32)):
            with self.assertRaises(ValueError): self.run_case()
        self.assertFalse((self.root/'one-cache').exists())
        with patch.object(self, 'encode', lambda i,p: np.array([1.,0.,0.,0.],dtype=np.float32)):
            with self.assertRaises(ValueError): self.run_case('constant')
        self.assertFalse((self.root/'constant-cache').exists())

    def test_nonfinite_or_mid_encode_image_change_rejected(self):
        with patch.object(self, 'encode', lambda i,p: np.full(4, np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError): self.run_case('nan')
        original = self.encode
        def change(identifier, path):
            row = original(identifier, path)
            path.write_text('mutated during callback')
            return row
        with patch.object(self, 'encode', change):
            with self.assertRaises(ValueError): self.run_case('mutated')
        self.assertFalse((self.root/'mutated-cache').exists())

    def test_final_metadata_write_interruption_recovers_without_encoder(self):
        from dermdetect.models import extraction_run
        def interrupted_save(directory, ids, matrix, key):
            directory.mkdir()
            np.save(directory/'features.npy', matrix, allow_pickle=False)
            raise RuntimeError('before metadata write')
        with patch.object(extraction_run, 'save_cache', interrupted_save):
            with self.assertRaises(RuntimeError): self.run_case()
        self.calls.clear()
        self.assertTrue(self.run_case()['complete'])
        self.assertEqual(self.calls, [])

    def test_crash_between_shard_rename_and_ledger_recovers(self):
        from dermdetect.models import extraction_run
        original = extraction_run._atomic
        def interrupt(path, data):
            if path.name == 'state.json' and json.loads(data).get('shards'):
                raise RuntimeError('power-loss fixture')
            return original(path,data)
        with patch.object(extraction_run, '_atomic', interrupt):
            with self.assertRaises(RuntimeError): self.run_case()
        self.calls.clear()
        self.assertTrue(self.run_case()['complete'])
        self.assertEqual(self.calls, [i for i,_ in self.items])

    def test_truncated_uncommitted_final_payload_rebuilt_from_shards(self):
        from dermdetect.models import extraction_run
        def partial_save(directory, ids, matrix, key):
            directory.mkdir()
            (directory/'features.npy').write_bytes(b'partial-npy-header')
            raise RuntimeError('mid payload write')
        with patch.object(extraction_run, 'save_cache', partial_save):
            with self.assertRaises(RuntimeError): self.run_case()
        self.calls.clear()
        self.assertTrue(self.run_case()['complete'])
        self.assertEqual(self.calls, [])
        self.assertEqual((self.root/'one-work'/'final-cache-interrupted-000000.tmp'/'features.npy').read_bytes(), b'partial-npy-header')

    def test_committed_final_payload_corruption_stays_failure(self):
        from dermdetect.models import extraction_run
        with patch.object(extraction_run.os, 'rename', side_effect=RuntimeError('publication interruption')):
            with self.assertRaises(RuntimeError): self.run_case()
        (self.root/'one-work'/'final-cache.tmp'/'features.npy').write_bytes(b'corrupted committed payload')
        self.calls.clear()
        with self.assertRaises(ValueError): self.run_case()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root/'one-cache').exists())

    def test_cli_budget_rejects_nonfinite_and_excess(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('extract_m4_cli', Path(__file__).resolve().parents[1]/'scripts'/'extract_m4.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        approval = {'allocation_seconds': {'extraction': 3600}}
        config = {'shard_size':64}
        module.validate_budget(3300, config, approval)
        for seconds in (float('inf'), float('nan'), -1, 0, 3301):
            with self.assertRaises(ValueError): module.validate_budget(seconds, config, approval)
        with self.assertRaises(ValueError): module.validate_budget(100, {'shard_size':1}, approval)

    def test_final_publication_crash_recovers_verified_cache(self):
        from dermdetect.models import extraction_run
        original = extraction_run.os.rename
        with patch.object(extraction_run.os, 'rename', side_effect=RuntimeError('publication interruption')):
            with self.assertRaises(RuntimeError): self.run_case()
        self.calls.clear()
        state = self.run_case()
        self.assertTrue(state['complete'])
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
