"""M9 reproduction check: recompute the recorded M7 HAM metrics and paired intervals
from the retrieved M7 physical predictions with the locked evaluator code.

Reads only already-unsealed M7 outputs and the joint lock; no model, no encoder,
no DDI. Run from the project root: python3 -I scripts/m9/replay_m7_metrics.py
"""
import hashlib, json, sys
from pathlib import Path
R = Path.cwd(); sys.path.insert(0, str(R/'src'))
from dermdetect.evaluation.metrics import scores
from dermdetect.evaluation.engine import paired_intervals

M7 = R/'outputs/m7-run-20261122/retrieved/m7_output/internal_test'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = json.loads((M7/'receipt.json').read_text())
assert sha(M7/'predictions.json') == receipt['predictions_SHA'] and sha(M7/'metrics.json') == receipt['metrics_SHA']
lock = json.loads((R/'outputs/m6-rebind-20261122/joint_lock.json').read_text())
assert sha(R/'outputs/m6-rebind-20261122/joint_lock.json') == '560952b7f5d3c88c5cd9ea77b7f04f250d3dc6c8cdac0cd359f44af95ff6d7e8'
specs = json.loads((R/lock['system_manifest']['path']).read_text())['systems']
pred = json.loads((M7/'predictions.json').read_text())['physical']
recorded = json.loads((M7/'metrics.json').read_text())['result']

def strip(x):  # compare the full metric tree exactly
    return json.dumps(x, sort_keys=True)

report = {'systems': {}, 'intervals': {}}; tasks = {}
for name, rows in pred.items():
    s = specs[name]; t = s['temperature']['temperature']; cut = [c['cutoff'] for c in s['confidence_rules']]
    z = tuple(tuple(r['mean_logits']) for r in rows); y = tuple(r['label'] for r in rows)
    ids = tuple(r['physical_id'] for r in rows); cl = tuple(r['cluster_id'] for r in rows)
    got = scores(z, y, t, cutoffs=cut); want = recorded['systems'][name]['metrics']
    report['systems'][name] = {'units': len(rows), 'exact_match': strip(got) == strip(want)}
    tasks[name] = dict(ids=ids, logits=z, labels=y, clusters=cl, temperature=t)
for kind in ('7', '2'):
    got = paired_intervals(tasks['dense'+kind], tasks['med'+kind], partition='internal_test')
    report['intervals'][kind] = strip(got) == strip(recorded['paired_intervals'][kind])
report['all_exact'] = all(v['exact_match'] for v in report['systems'].values()) and all(report['intervals'].values())
print(json.dumps(report, indent=1))
