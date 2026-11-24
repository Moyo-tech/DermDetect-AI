"""Invented-data tests for the locked DDI external adapter (no real DDI data)."""
from pathlib import Path
import csv, hashlib, json, tempfile, unittest
from unittest.mock import patch

from dermdetect.evaluation import engine as e
from dermdetect.evaluation.metrics import NUMERICAL_POLICY
from dermdetect.external import ddi
from test_m6_integration import Fixture


class FakeBackend:
    def __init__(self): self.calls = 0
    def _f(self, payload):
        h = hashlib.sha256(payload).digest(); return (h[0]/64.-2., h[1]/64.-2.)
    def dense(self, payload): self.calls += 1; return (0.,)*7, self._f(payload)
    def med(self, payload): return self._f(payload[::-1])
    def finish(self): return {'frozen_state_unchanged': True, 'RNG_unchanged': True, 'device': 'invented'}


class DDIFixture:
    def __init__(self, root, n=36, single_class_group=None):
        self.f = Fixture(Path(root)/'ham', 'validation'); self.root = self.f.root
        self.dir = Path(root)/'ddi_invented'; self.dir.mkdir(); rows = []; inventory = []
        for i in range(n):
            name = f'{i+1:06d}.png'; data = ('INVENTED DDI IMAGE %d' % i).encode(); (self.dir/name).write_bytes(data)
            group = ddi.SKIN_GROUPS[i % 3]
            label = 0 if group == single_class_group else (i // 3) % 2
            rows.append({'': str(i), 'DDI_ID': str(i+1), 'DDI_file': name, 'skin_tone': group,
                         'malignant': 'True' if label else 'False', 'disease': 'invented'})
            inventory.append({'file': name, 'SHA256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        with (self.dir/'ddi_metadata.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        meta_sha = ddi.sha(self.dir/'ddi_metadata.csv')
        inv_sha = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
        put = self.f.put
        self.inventory = put('ddi/image_inventory.json', inventory)
        self.integrity = put('ddi/integrity_record.json', {'status': 'passed', 'metadata_SHA256': meta_sha, 'image_inventory_SHA256': inv_sha,
                                                          'schema': {'patient_identifier_present': False}})
        self.ev = put('ddi/ddi_evaluator.json', {'files': [self.f.code], 'runtime': {'invented': True}, 'numerical_policy_SHA': e.digest(NUMERICAL_POLICY),
                                                 'metadata_SHA256': meta_sha, 'image_inventory_SHA256': inv_sha})
        self.addon_approval = put('ddi/addon_approval.json', {'status': 'user_approved', 'lock_authorized': True, 'user_reference': 'INVENTED', 'ddi_evaluator_manifest': self.ev})
        self.addon = put('ddi/addon_lock.json', {'status': 'ddi_addon_locked', 'addon_approval': self.addon_approval, 'ham_lock': self.f.lock,
                                                 'ddi_evaluator_manifest': self.ev, 'integrity_record': self.integrity, 'image_inventory': self.inventory})
        self.out = Path(root)/'out'/'ddi'; self.out.parent.mkdir()
        self.approval = {'status': 'user_approved', 'user_reference': 'INVENTED', 'scope': ddi.SCOPE, 'external_outcome_access': True,
                         'compute_authorized': True, 'systems': list(ddi.SYSTEMS), 'evaluation_unit': 'image', 'per_image_retention': False,
                         'DDI_tuning': False, 'ham_lock': self.f.lock, 'system_manifest': self.f.system, 'ddi_addon_lock': self.addon,
                         'ddi_evaluator_manifest': self.ev, 'output': str(self.out),
                         'dataset': {'directory': str(self.dir.resolve()), 'metadata_SHA256': meta_sha, 'image_inventory_SHA256': inv_sha}}
        self.request = put('ddi/run_approval.json', self.approval)
        self.digest = e.digest(json.loads((self.root/self.f.system['path']).read_text())['systems'])
    def change(self, key, value):
        self.approval[key] = value; self.request = self.f.put('ddi/run_approval.json', self.approval)
    def grant(self):
        with patch.object(ddi, 'SCIENTIFIC_SYSTEMS_SHA', self.digest):
            return ddi.verify_ddi_grant(self.root, self.request)


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup); self.x = DDIFixture(self.t.name)
    def refused(self):
        with self.assertRaises(PermissionError): self.x.grant()
    def test_valid_grant(self): g, inv = self.x.grant(); self.assertIsInstance(g, ddi.DDIGrant); self.assertEqual(len(inv), 36)
    def test_requires_user_approval(self): self.x.change('status', 'pending'); self.refused()
    def test_requires_external_unseal(self): self.x.change('external_outcome_access', False); self.refused()
    def test_refuses_tuning(self): self.x.change('DDI_tuning', True); self.refused()
    def test_requires_image_unit_contract(self): self.x.change('evaluation_unit', 'patient'); self.refused()
    def test_requires_explicit_retention_decision(self): self.x.change('per_image_retention', None); self.refused()
    def test_requires_binary_systems_only(self): self.x.change('systems', ['dense2', 'med2', 'med7']); self.refused()
    def test_requires_unchanged_frozen_systems(self):
        with self.assertRaises(PermissionError): ddi.verify_ddi_grant(self.x.root, self.x.request)  # unpatched real digest differs
    def test_requires_addon_lock(self):
        lock = json.loads((self.x.root/self.x.addon['path']).read_text()); lock['status'] = 'proposal'
        self.x.addon = self.x.f.put('ddi/addon_lock.json', lock); self.x.change('ddi_addon_lock', self.x.addon); self.refused()
    def test_requires_passed_integrity(self):
        rec = json.loads((self.x.root/self.x.integrity['path']).read_text()); rec['status'] = 'failed'
        p = self.x.root/self.x.integrity['path']; p.write_text(json.dumps(rec)); self.refused()   # addon binding SHA now stale too
    def test_refuses_changed_metadata(self):
        with (self.x.dir/'ddi_metadata.csv').open('a') as fh: fh.write('\n')
        self.refused()


class RunTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup)
    def run_fixture(self, **kw):
        x = DDIFixture(self.t.name, **kw.pop('fixture', {}))
        for k, v in kw.items(): x.change(k, v)
        g, inv = x.grant(); return x, ddi.run_ddi(g, inv, FakeBackend(), x.out)
    def test_aggregate_only_run(self):
        x, r = self.run_fixture()
        self.assertEqual(r['status'], 'complete'); self.assertTrue(r['recorded_before_error_review']); self.assertFalse(r['per_image_retained'])
        self.assertFalse((x.out/'predictions.json').exists())
        res = json.loads((x.out/'metrics.json').read_text())['result']
        self.assertEqual(res['overall']['units'], 36); self.assertEqual(set(res['skin_groups']), set(ddi.SKIN_GROUPS))
        self.assertNotIn('risk_coverage', res['overall']['med2'])
        iv = res['overall']['paired_intervals']; self.assertEqual(iv['replicates'], 1000); self.assertEqual(set(iv['intervals']), set(ddi.INTERVAL_METRICS))
        lo, hi = iv['intervals']['AUROC']['med_minus_dense']; self.assertLessEqual(lo, hi)
    def test_retention_mode_writes_predictions_without_labels(self):
        x, r = self.run_fixture(per_image_retention=True)
        p = json.loads((x.out/'predictions.json').read_text()); self.assertFalse(p['labels_included'])
        self.assertNotIn('label', json.dumps(p['per_image_logits']))
    def test_single_class_group_reports_na(self):
        x, r = self.run_fixture(fixture={'single_class_group': '56'})
        res = json.loads((x.out/'metrics.json').read_text())['result']
        self.assertIsNone(res['skin_groups']['56']['paired_intervals']['value'])
        self.assertIsNotNone(res['skin_groups']['12']['paired_intervals'].get('intervals'))
    def test_predictions_hash_independent_of_retention(self):
        _, a = self.run_fixture()
        t2 = tempfile.TemporaryDirectory(); self.addCleanup(t2.cleanup)
        x = DDIFixture(t2.name); x.change('per_image_retention', True); g, inv = x.grant()
        b = ddi.run_ddi(g, inv, FakeBackend(), x.out); self.assertEqual(a['predictions_SHA'], b['predictions_SHA'])
    def test_tampered_image_refused(self):
        x = DDIFixture(self.t.name); g, inv = x.grant(); (x.dir/'000001.png').write_bytes(b'tampered')
        with self.assertRaises(ValueError): ddi.run_ddi(g, inv, FakeBackend(), x.out)
    def test_no_overwrite_and_exact_output(self):
        x = DDIFixture(self.t.name); g, inv = x.grant(); ddi.run_ddi(g, inv, FakeBackend(), x.out)
        with self.assertRaises(FileExistsError): ddi.run_ddi(g, inv, FakeBackend(), x.out)
        with self.assertRaises(PermissionError): ddi.run_ddi(g, inv, FakeBackend(), Path(self.t.name)/'elsewhere')
    def test_unverified_grant_refused(self):
        with self.assertRaises(PermissionError): ddi.load_rows(object(), {})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        S = importlib.util.spec_from_file_location('ddi_runner', Path(__file__).resolve().parents[1]/'scripts/run_ddi_evaluation.py')
        self.m = importlib.util.module_from_spec(S); S.loader.exec_module(self.m)
        from dermdetect.evaluation.supervision import LIMITS
        import time
        self.a = {'cloud': True, 'device': 'cpu', 'resource_limits': LIMITS, 'destination_id': 'pod', 'provider': 'runpod', 'transfer_approval_reference': 'INVENTED',
                  'max_wall_seconds': 7200, 'provider_work_deadline_unix': time.time()+7200, 'max_total_usd': 1., 'dedicated_memory_bytes': 32*10**9,
                  'currency': 'USD', 'memory_accounting': 'cgroup_v2', **{k: True for k in self.m.REQUIRED_TRUE}}
    def test_contract_accepts_complete_envelope(self): self.m.validate_cloud_contract(self.a)
    def test_contract_requires_no_sandbox_acknowledgment(self):
        self.a['trusted_process_no_OS_isolation_acknowledged'] = False
        with self.assertRaises(PermissionError): self.m.validate_cloud_contract(self.a)
    def test_contract_refuses_host_memory_accounting(self):
        self.a['memory_accounting'] = 'dedicated_host'
        with self.assertRaises(PermissionError): self.m.validate_cloud_contract(self.a)
    def test_contract_refuses_expired_deadline(self):
        self.a['provider_work_deadline_unix'] = 1.
        with self.assertRaises(PermissionError): self.m.validate_cloud_contract(self.a)


if __name__ == '__main__':
    unittest.main()
