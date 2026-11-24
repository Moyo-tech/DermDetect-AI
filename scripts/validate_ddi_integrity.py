#!/usr/bin/env python3
"""Restricted pre-unseal DDI schema/integrity validation (protocol §5 item 5).

Reports schema compatibility and pass/fail status ONLY. Never prints or records
per-image labels, examples, diagnosis/prevalence/skin-tone distributions or
subgroup sizes. Label and skin-tone values are checked in memory against
prespecified allowed sets; only a boolean result leaves this process.

Writes (never overwrites):
  <output>/integrity_record.json   pass/fail per check + non-identifying hashes
  <output>/image_inventory.json    restricted local per-file SHA-256 list (no labels)
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys

ID_COLUMN = 'DDI_ID'
FILE_COLUMN = 'DDI_file'
LABEL_COLUMNS = ('malignant', 'malignancy(malig=1)')   # official field, documented fallback
SKIN_COLUMN = 'skin_tone'
LABEL_VALUES = {'1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}   # official loader semantics
SKIN_VALUES = {'12', '34', '56'}                         # Fitzpatrick I-II, III-IV, V-VI
PATIENT_COLUMN_CANDIDATES = ('patient_id', 'patient', 'PatientID', 'DDI_patient_id')
EXPECTED_IMAGES = 656                                    # official public count
MAX_IMAGE_BYTES = 20 * 2**20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate(directory):
    """Return (record, inventory). Record holds booleans/schema only."""
    d = Path(directory)
    if d.is_symlink() or not d.is_dir():
        raise PermissionError('Real DDI directory required')
    meta = d/'ddi_metadata.csv'
    checks = {}
    with meta.open(newline='') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    label_column = next((c for c in LABEL_COLUMNS if c in columns), None)
    patient_column = next((c for c in PATIENT_COLUMN_CANDIDATES if c in columns), None)
    checks['required_columns_present'] = all(c in columns for c in (ID_COLUMN, FILE_COLUMN, SKIN_COLUMN)) and label_column is not None
    checks['row_count_matches_official'] = len(rows) == EXPECTED_IMAGES
    ids = [r.get(ID_COLUMN, '') for r in rows]; files = [r.get(FILE_COLUMN, '') for r in rows]
    checks['ids_unique_nonempty'] = all(ids) and len(set(ids)) == len(ids)
    checks['files_unique_nonempty'] = all(files) and len(set(files)) == len(files)
    checks['label_values_valid'] = label_column is not None and all(r.get(label_column, '').strip() in LABEL_VALUES for r in rows)
    checks['skin_tone_values_valid'] = all(r.get(SKIN_COLUMN, '').strip() in SKIN_VALUES for r in rows)
    png = sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    others = sorted(p.name for p in d.iterdir() if p.name not in set(png) | {'ddi_metadata.csv'})
    checks['no_unexpected_entries'] = not others
    checks['every_listed_file_present'] = set(files) <= set(png)
    checks['no_unlisted_images'] = set(png) <= set(files)
    inventory = []; decodable = True; bounded = True; regular = True
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for name in sorted(set(files) & set(png)):
        p = d/name
        if p.is_symlink() or not p.is_file():
            regular = False; continue
        size = p.stat().st_size; bounded &= 0 < size <= MAX_IMAGE_BYTES
        if Image is not None:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.convert('RGB').load()
            except Exception:
                decodable = False
        inventory.append({'file': name, 'SHA256': sha(p), 'bytes': size})
    checks['all_regular_files'] = regular
    checks['all_sizes_bounded'] = bounded
    checks['decoder_available'] = Image is not None
    checks['all_images_decode'] = Image is not None and decodable
    checks['image_hashes_unique'] = len({x['SHA256'] for x in inventory}) == len(inventory)
    record = {
        'status': 'passed' if all(checks.values()) else 'failed',
        'checks': checks,
        'schema': {'columns': columns, 'label_column': label_column, 'skin_tone_column': SKIN_COLUMN,
                   'patient_identifier_column': patient_column, 'patient_identifier_present': patient_column is not None},
        'counts': {'metadata_rows': len(rows), 'png_files': len(png)},
        'metadata_SHA256': sha(meta), 'metadata_bytes': meta.stat().st_size,
        'image_inventory_SHA256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        'exposed': 'schema and pass/fail only; no labels, examples, prevalence, diagnosis or subgroup distributions',
        'outcomes_unsealed': False}
    return record, inventory


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--ddi-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    record, inventory = validate(a.ddi_dir)
    with (a.output/'image_inventory.json').open('x') as f:
        json.dump(inventory, f, indent=1, sort_keys=True); f.write('\n')
    record['image_inventory_file_SHA256'] = sha(a.output/'image_inventory.json')
    with (a.output/'integrity_record.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps({'status': record['status'], 'checks': record['checks'], 'schema': record['schema'], 'counts': record['counts']}, indent=1))
    return 0 if record['status'] == 'passed' else 2


if __name__ == '__main__':
    sys.exit(main())
