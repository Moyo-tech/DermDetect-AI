"""Invented-data tests for the restricted DDI integrity validator."""
from pathlib import Path
import csv, importlib.util, io, json, tempfile, unittest
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
S = importlib.util.spec_from_file_location('ddi_integrity', ROOT/'scripts/validate_ddi_integrity.py')
m = importlib.util.module_from_spec(S); S.loader.exec_module(m)


def make(d, n=6, label='malignant', bad_label=False, extra=None, missing=False):
    from PIL import Image
    d = Path(d); rows = []
    for i in range(n):
        name = f'{i+1:06d}.png'
        Image.new('RGB', (8, 8), (i*20 % 255, 10, 10)).save(d/name)
        rows.append({'DDI_ID': str(i+1), 'DDI_file': name, 'skin_tone': ['12', '34', '56'][i % 3],
                     label: ('maybe' if bad_label and i == 0 else ['True', 'False'][i % 2]), 'disease': 'invented'})
    if missing:(d/rows[0]['DDI_file']).unlink()
    if extra:(d/extra).write_text('x')
    with (d/'ddi_metadata.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return d


class DDIIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup); self.old = m.EXPECTED_IMAGES; m.EXPECTED_IMAGES = 6
        self.addCleanup(setattr, m, 'EXPECTED_IMAGES', self.old)
    def test_valid_invented_folder_passes(self):
        r, inv = m.validate(make(self.t.name)); self.assertEqual(r['status'], 'passed'); self.assertEqual(len(inv), 6)
        self.assertFalse(r['schema']['patient_identifier_present'])
    def test_fallback_label_column(self):
        r, _ = m.validate(make(self.t.name, label='malignancy(malig=1)')); self.assertEqual(r['schema']['label_column'], 'malignancy(malig=1)')
    def test_invalid_label_fails_without_revealing_values(self):
        r, _ = m.validate(make(self.t.name, bad_label=True)); self.assertFalse(r['checks']['label_values_valid'])
        self.assertNotIn('maybe', json.dumps(r))
    def test_missing_and_unexpected_files_fail(self):
        r, _ = m.validate(make(self.t.name, missing=True)); self.assertFalse(r['checks']['every_listed_file_present'])
    def test_unexpected_entry_fails(self):
        r, _ = m.validate(make(self.t.name, extra='notes.txt')); self.assertFalse(r['checks']['no_unexpected_entries'])
    def test_record_and_stdout_expose_no_labels_or_group_sizes(self):
        ddi = Path(self.t.name)/'ddi'; ddi.mkdir(); d = make(ddi); out = Path(self.t.name)/'out_rec'
        buf = io.StringIO()
        with redirect_stdout(buf): code = m.main(['--ddi-dir', str(d), '--output', str(out)])
        self.assertEqual(code, 0)
        blobs = [(out/'integrity_record.json').read_text(), (out/'image_inventory.json').read_text(), buf.getvalue()]
        for blob in blobs:
            for leak in ('"True"', '"False"', 'invented', '"12"', '"34"', '"56"'):
                self.assertNotIn(leak, blob)
        self.assertNotIn('skin', blobs[1]); self.assertNotIn('malignant', blobs[1])
    def test_never_overwrites(self):
        ddi = Path(self.t.name)/'ddi'; ddi.mkdir(); d = make(ddi); out = Path(self.t.name)/'o'; out.mkdir()
        with self.assertRaises(FileExistsError): m.main(['--ddi-dir', str(d), '--output', str(out)])


if __name__ == '__main__':
    unittest.main()
