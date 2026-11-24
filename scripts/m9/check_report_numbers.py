"""M9: check that key figures quoted in the report docs match the source records
(rounded as printed). Run from project root with python3 -I."""
import json, sys
from pathlib import Path
R = Path.cwd()
M7 = json.loads((R/'outputs/m7-run-20261122/retrieved/m7_output/internal_test/metrics.json').read_text())['result']
M8 = json.loads((R/'outputs/m8-ddi-run-20261123/retrieved/dermdetect_runpod/m8_output/ddi/metrics.json').read_text())['result']
C7 = json.loads((R/'outputs/m7-run-20261122/retrieved/m7_output/internal_test/compute.json').read_text())
C8 = json.loads((R/'outputs/m8-ddi-run-20261123/retrieved/dermdetect_runpod/m8_output/ddi/compute.json').read_text())
docs = {n: (R/'docs'/n).read_text() for n in ('final_experiment_report.md', 'model_card.md', 'm8_ddi_results_20261123.md')}
def v(x): return x['value'] if isinstance(x, dict) else x
s7 = {k: x['metrics'] for k, x in M7['systems'].items()}
iv7 = M7['paired_intervals']; iv8 = M8['overall']['paired_intervals']['intervals']
def f(x, d=3): return f'{x:.{d}f}'
def sg(x, d=3): return ('+' if x > 0 else '−' if x < 0 else '') + f'{abs(x):.{d}f}'
checks = []
def need(doc, text, why): checks.append((doc, text, why, text in docs[doc]))
R_ = 'final_experiment_report.md'
for k, d in (('dense7', 'macro_F1'), ('med7', 'macro_F1'), ('dense7', 'multiclass_Brier'), ('med7', 'multiclass_Brier'), ('dense7', 'ECE'), ('med7', 'ECE'),
             ('dense7', 'balanced_accuracy'), ('med7', 'balanced_accuracy'), ('dense7', 'macro_AUROC'), ('med7', 'macro_AUROC'), ('dense7', 'macro_AP'), ('med7', 'macro_AP'),
             ('dense7', 'NLL'), ('med7', 'NLL'), ('dense7', 'AURC'), ('med7', 'AURC'), ('dense2', 'AUROC'), ('med2', 'AUROC'), ('dense2', 'AP'), ('med2', 'AP'),
             ('dense2', 'macro_F1'), ('med2', 'macro_F1'), ('dense2', 'binary_Brier'), ('med2', 'binary_Brier'), ('dense2', 'sensitivity'), ('med2', 'sensitivity')):
    need(R_, f(v(s7[k][d])), f'HAM {k} {d}')
for kind, m in (('7', 'macro_F1'), ('7', 'multiclass_Brier'), ('2', 'macro_F1'), ('2', 'multiclass_Brier')):
    lo, hi = iv7[kind]['intervals'][m]['b_minus_a']; need(R_, f'[{sg(lo)}, {sg(hi)}]', f'HAM {kind} {m} diff CI')
for sysn in ('dense2', 'med2'):
    o = M8['overall'][sysn]
    for d in ('AUROC', 'AP', 'binary_Brier', 'NLL', 'ECE', 'AURC'): need(R_, f(v(o[d])), f'DDI {sysn} {d}')
    lo, hi = iv8['AUROC'][sysn]; need(R_, f'[{f(lo)}, {f(hi)}]', f'DDI {sysn} AUROC CI')
for m in ('AUROC', 'AP', 'binary_Brier'):
    lo, hi = iv8[m]['med_minus_dense']; need(R_, f'[{sg(lo)}, {sg(hi)}]', f'DDI diff {m} CI')
for g in ('12', '34', '56'):
    b = M8['skin_groups'][g]
    for sysn in ('dense2', 'med2'):
        lo, hi = b['paired_intervals']['intervals']['AUROC'][sysn]
        need(R_, f'{f(v(b[sysn]["AUROC"]))} [{f(lo)}, {f(hi)}]', f'DDI group {g} {sysn} AUROC+CI')
    lo, hi = b['paired_intervals']['intervals']['AUROC']['med_minus_dense']; need(R_, f'[{sg(lo)}, {sg(hi)}]', f'DDI group {g} diff CI')
need(R_, f'{C7["stages"]["med_encode_seconds"]:,.0f} s', 'M7 Med encode s'); need(R_, f'{C8["stages"]["med_encode_seconds"]:,.0f} s', 'M8 Med encode s')
for k in ('dense2', 'med2'): need('model_card.md', f(v(M8['overall'][k]['AUROC'])), f'model card DDI {k}')
bad = [c for c in checks if not c[3]]
print(json.dumps({'checks': len(checks), 'failed': [(c[0], c[2], c[1]) for c in bad]}, indent=1, ensure_ascii=False))
sys.exit(1 if bad else 0)
