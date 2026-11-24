"""M9: build aggregate result tables and SVG figures from the recorded M7/M8 runs.

Inputs are only the hash-checked recorded metrics/compute/receipt files (no model,
no encoder, no raw images, no per-image DDI data). Standard library only.
Run from project root: python3 -I scripts/m9/build_m9_tables.py
"""
import csv, hashlib, json
from pathlib import Path
R = Path.cwd()
M7 = R/'outputs/m7-run-20261122/retrieved/m7_output/internal_test'
M8 = R/'outputs/m8-ddi-run-20261123/retrieved/dermdetect_runpod/m8_output/ddi'
RES = R/'results'; FIG = R/'figures'; RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())

r7, r8 = load(M7/'receipt.json'), load(M8/'receipt.json')
assert sha(M7/'metrics.json') == r7['metrics_SHA'] and sha(M7/'compute.json') == r7['compute_SHA']
assert sha(M8/'metrics.json') == r8['metrics_SHA'] and sha(M8/'compute.json') == r8['compute_SHA']
assert r8['per_image_retained'] is False and not (M8/'predictions.json').exists()
ham = load(M7/'metrics.json')['result']; ddi = load(M8/'metrics.json')['result']
NAME = {'dense7': 'DenseNet121', 'med7': 'MedSigLIP probe', 'dense2': 'DenseNet121', 'med2': 'MedSigLIP probe'}
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
def v(x): return x['value'] if isinstance(x, dict) and 'value' in x else x
SCAL = ['macro_F1', 'balanced_accuracy', 'accuracy', 'MCC', 'macro_AUROC', 'macro_AP', 'AUROC', 'AP', 'sensitivity', 'specificity',
        'NLL', 'multiclass_Brier', 'binary_Brier', 'ECE', 'AURC']
written = {}
def out_csv(name, header, rows):
    p = RES/name
    with p.open('w', newline='') as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)
    written[str(p.relative_to(R))] = sha(p)

def metric_rows(m, prefix):
    rows = [prefix + [k, v(m[k])] for k in SCAL if k in m and v(m[k]) is not None]
    if 'per_class' in m and len(m['per_class']) == 2: rows.append(prefix + ['F1_malignant', v(m['per_class'][1]['F1'])])
    for s, t in zip(m['selective'], ('90', '80')):
        rows += [prefix + [f'coverage_at_{t}pct_cutoff', s['coverage']], prefix + [f'selective_error_at_{t}pct_cutoff', v(s['selective_error'])],
                 prefix + [f'class_balanced_selective_error_at_{t}pct_cutoff', v(s['class_balanced_selective_error'])]]
    return rows

rows = []
for sysname, x in ham['systems'].items():
    task = 'seven_class' if sysname.endswith('7') else 'binary_bcc_mel_vs_benign'
    rows += metric_rows(x['metrics'], [task, sysname, NAME[sysname], x['physical_units']])
out_csv('ham_internal_test_metrics.csv', ['task', 'system_id', 'system', 'physical_units', 'metric', 'value'], rows)
rows = []
for sysname in ('dense7', 'med7'):
    for c in ham['systems'][sysname]['metrics']['per_class']:
        rows.append([sysname, NAME[sysname], CLASSES[c['class']], c['support'], v(c['precision']), v(c['recall']), v(c['F1']), v(c['AUROC']), v(c['AP'])])
out_csv('ham_seven_class_per_class.csv', ['system_id', 'system', 'class', 'support_units', 'precision', 'recall', 'F1', 'AUROC', 'AP'], rows)
rows = []
for kind, x in ham['paired_intervals'].items():
    for metric, g in x['intervals'].items():
        rows.append(['seven_class' if kind == '7' else 'binary', metric + ('' if kind == '7' else '_two_class_sum' if metric == 'multiclass_Brier' else ''),
                     *g['a'], *g['b'], *g['b_minus_a'], x['replicates'], x['seed'], x['resampling']])
out_csv('ham_paired_intervals.csv', ['task', 'metric', 'dense_lo', 'dense_hi', 'med_lo', 'med_hi', 'med_minus_dense_lo', 'med_minus_dense_hi', 'replicates', 'seed', 'resampling'], rows)
rows = []
for sysname in ('dense7', 'med7', 'dense2', 'med2'):
    for i, row in enumerate(ham['systems'][sysname]['metrics']['confusion_matrix']):
        rows.append([sysname, (CLASSES if sysname.endswith('7') else ['benign', 'malignant'])[i], *row])
out_csv('ham_confusion_matrices.csv', ['system_id', 'true_class', 'predicted_0..k'], rows)

GROUPS = {'overall': 'All DDI', '12': 'Fitzpatrick I-II', '34': 'Fitzpatrick III-IV', '56': 'Fitzpatrick V-VI'}
def ddi_block(g): return ddi['overall'] if g == 'overall' else ddi['skin_groups'][g]
rows, irows = [], []
for g in GROUPS:
    b = ddi_block(g)
    for sysname in ('dense2', 'med2'):
        m = b[sysname]; cm = m['confusion_matrix']
        pre = [g, GROUPS[g], sysname, NAME[sysname], b['units'], sum(cm[0]), sum(cm[1]), sum(cm[1])/b['units']]
        rows += [pre + r[4:] for r in metric_rows(m, [None]*4)]
        rows += [pre + ['confusion_TN_FP_FN_TP', ' '.join(map(str, cm[0] + cm[1]))]]
    pi = b['paired_intervals']
    for metric, gi in (pi.get('intervals') or {}).items():
        irows.append([g, GROUPS[g], metric, *gi['dense2'], *gi['med2'], *gi['med_minus_dense'], pi['replicates']])
out_csv('ddi_metrics.csv', ['group', 'group_label', 'system_id', 'system', 'images', 'benign', 'malignant', 'prevalence', 'metric', 'value'], rows)
out_csv('ddi_paired_intervals.csv', ['group', 'group_label', 'metric', 'dense_lo', 'dense_hi', 'med_lo', 'med_hi', 'med_minus_dense_lo', 'med_minus_dense_hi', 'replicates'], irows)

# ---------------- SVG figures ----------------
COL = {'dense': '#1f5fa8', 'med': '#c8611a'}; INK = '#1d1d1f'; GRID = '#d9d9de'
def svg(name, w, h, body, title):
    p = FIG/name
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Helvetica,Arial,sans-serif">'
                 f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>\n')
    written[str(p.relative_to(R))] = sha(p)
def panel(x0, y0, size, heading, xlabel, ylabel, ticks=(0, .25, .5, .75, 1)):
    s = [f'<text x="{x0}" y="{y0-12}" font-size="14" font-weight="600" fill="{INK}">{heading}</text>',
         f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="none" stroke="{INK}" stroke-width="1"/>']
    for t in ticks:
        X = x0 + t*size; Y = y0 + size - t*size
        s += [f'<line x1="{X}" y1="{y0}" x2="{X}" y2="{y0+size}" stroke="{GRID}"/>', f'<line x1="{x0}" y1="{Y}" x2="{x0+size}" y2="{Y}" stroke="{GRID}"/>',
              f'<text x="{X}" y="{y0+size+16}" font-size="11" text-anchor="middle" fill="{INK}">{t:g}</text>',
              f'<text x="{x0-6}" y="{Y+4}" font-size="11" text-anchor="end" fill="{INK}">{t:g}</text>']
    s += [f'<text x="{x0+size/2}" y="{y0+size+34}" font-size="12" text-anchor="middle" fill="{INK}">{xlabel}</text>',
          f'<text transform="translate({x0-38},{y0+size/2}) rotate(-90)" font-size="12" text-anchor="middle" fill="{INK}">{ylabel}</text>']
    return s
def legend(x, y):
    return [f'<rect x="{x}" y="{y-9}" width="12" height="12" fill="{COL["dense"]}"/><text x="{x+18}" y="{y+1}" font-size="12" fill="{INK}">DenseNet121</text>',
            f'<rect x="{x+120}" y="{y-9}" width="12" height="12" fill="{COL["med"]}"/><text x="{x+138}" y="{y+1}" font-size="12" fill="{INK}">MedSigLIP probe</text>']

def reliability(metrics_by_sys, heading, x0, y0, size=240):
    s = panel(x0, y0, size, heading, 'Mean confidence', 'Observed accuracy')
    s.append(f'<line x1="{x0}" y1="{y0+size}" x2="{x0+size}" y2="{y0}" stroke="#8e8e93" stroke-dasharray="4 4"/>')
    for key, m in metrics_by_sys:
        pts = [(b['mean_confidence'], b['accuracy'], b['count']) for b in m['ECE_bins'] if b['count']]
        total = sum(c for *_, c in pts); colour = COL[key]
        s.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>' % (colour, ' '.join(f'{x0+c*size:.1f},{y0+size-a*size:.1f}' for c, a, _ in pts)))
        for c, a, n in pts:
            s.append(f'<circle cx="{x0+c*size:.1f}" cy="{y0+size-a*size:.1f}" r="{2.5+9*(n/total)**.5:.1f}" fill="{colour}" fill-opacity="0.55" stroke="{colour}"/>')
    return s
body = []
body += reliability([('dense', ham['systems']['dense7']['metrics']), ('med', ham['systems']['med7']['metrics'])], 'HAM test, seven-class', 70, 50)
body += reliability([('dense', ham['systems']['dense2']['metrics']), ('med', ham['systems']['med2']['metrics'])], 'HAM test, binary', 390, 50)
body += reliability([('dense', ddi['overall']['dense2']), ('med', ddi['overall']['med2'])], 'DDI, binary (all images)', 710, 50)
body += legend(70, 360) + [f'<text x="330" y="361" font-size="11" fill="#55555a">15 equal-width bins; marker area ∝ bin count; dashed line = perfect calibration. Confidence = max calibrated probability.</text>']
svg('reliability_diagrams.svg', 990, 380, ''.join(body), 'Reliability diagrams: HAM internal test and DDI')

def risk_cov(key_metrics, heading, x0, y0, size=260, ymax=.5):
    s = panel(x0, y0, size, heading, 'Coverage', 'Selective risk (error rate)', ticks=(0, .25, .5, .75, 1))
    for i, t in enumerate((0, .25, .5, .75, 1)):  # relabel y ticks to ymax scale
        s = [q.replace(f'text-anchor="end" fill="{INK}">{t:g}<', f'text-anchor="end" fill="{INK}">{t*ymax:g}<') for q in s]
    for key, m in key_metrics:
        pts = [(p['coverage'], p['risk']) for p in m['risk_coverage']]
        s.append('<polyline fill="none" stroke="%s" stroke-width="1.6" points="%s"/>' % (COL[key], ' '.join(f'{x0+c*size:.1f},{y0+size-min(r,ymax)/ymax*size:.1f}' for c, r in pts)))
        for q in m['selective']:
            s.append(f'<circle cx="{x0+q["coverage"]*size:.1f}" cy="{y0+size-v(q["selective_error"])/ymax*size:.1f}" r="4" fill="#ffffff" stroke="{COL[key]}" stroke-width="2"/>')
    return s
body = risk_cov([('dense', ham['systems']['dense7']['metrics']), ('med', ham['systems']['med7']['metrics'])], 'HAM test, seven-class', 80, 50)
body += risk_cov([('dense', ham['systems']['dense2']['metrics']), ('med', ham['systems']['med2']['metrics'])], 'HAM test, binary', 440, 50)
body += legend(80, 380) + ['<text x="340" y="381" font-size="11" fill="#55555a">Open circles: realized coverage/error at the frozen 90% and 80% validation cutoffs. DDI curves were not retained (aggregate-only).</text>']
svg('risk_coverage_ham.svg', 900, 400, ''.join(body), 'Risk-coverage curves on the HAM internal test')

# AUROC forest: DDI overall + groups with 95% intervals; HAM binary point estimates for context
rows = [('HAM test binary (context)', v(ham['systems']['dense2']['metrics']['AUROC']), None, v(ham['systems']['med2']['metrics']['AUROC']), None)]
for g in GROUPS:
    b = ddi_block(g); iv = b['paired_intervals']['intervals']['AUROC']
    rows.append((('DDI ' if g == 'overall' else '') + GROUPS[g] + ('' if g == 'overall' else f' (n={b["units"]})'), v(b['dense2']['AUROC']), iv['dense2'], v(b['med2']['AUROC']), iv['med2']))
x0, w, y0, step = 260, 460, 50, 46; X = lambda a: x0 + (a - .2)/.8*w
body = [f'<text x="20" y="28" font-size="14" font-weight="600" fill="{INK}">AUROC with 95% bootstrap intervals (DDI) and HAM binary point estimates</text>']
for t in (.2, .3, .4, .5, .6, .7, .8, .9, 1.):
    body += [f'<line x1="{X(t)}" y1="{y0-6}" x2="{X(t)}" y2="{y0+step*len(rows)}" stroke="{"#8e8e93" if t == .5 else GRID}" {"stroke-dasharray=\"4 4\"" if t == .5 else ""}/>',
             f'<text x="{X(t)}" y="{y0+step*len(rows)+16}" font-size="11" text-anchor="middle" fill="{INK}">{t:.1f}</text>']
for i, (label, d, di, m, mi) in enumerate(rows):
    y = y0 + i*step + 14
    body.append(f'<text x="{x0-12}" y="{y+8}" font-size="12" text-anchor="end" fill="{INK}">{label}</text>')
    for off, val, iv, key in ((0, d, di, 'dense'), (14, m, mi, 'med')):
        if iv: body.append(f'<line x1="{X(iv[0]):.1f}" y1="{y+off}" x2="{X(iv[1]):.1f}" y2="{y+off}" stroke="{COL[key]}" stroke-width="2"/>')
        body.append(f'<circle cx="{X(val):.1f}" cy="{y+off}" r="4.5" fill="{COL[key]}"/><text x="{X(val)+8:.1f}" y="{y+off-5}" font-size="10" fill="{COL[key]}">{val:.3f}</text>')
body += legend(x0, y0 + step*len(rows) + 40) + [f'<text x="{x0}" y="{y0+step*len(rows)+60}" font-size="11" fill="#55555a">Dashed line: chance (0.5). DDI unit is the image; groups are descriptive, not fairness claims.</text>']
svg('auroc_ham_vs_ddi_groups.svg', 760, y0 + step*len(rows) + 75, ''.join(body), 'AUROC: HAM binary and DDI by Fitzpatrick group')

# seven-class confusion matrices (row-normalised recall heat map with counts)
def confusion(cm, heading, x0, y0, cell=38):
    s = [f'<text x="{x0}" y="{y0-30}" font-size="14" font-weight="600" fill="{INK}">{heading}</text>',
         f'<text x="{x0+cell*3.5}" y="{y0-10}" font-size="11" text-anchor="middle" fill="{INK}">Predicted</text>']
    for j, c in enumerate(CLASSES): s.append(f'<text x="{x0+j*cell+cell/2}" y="{y0+7*cell+14}" font-size="10" text-anchor="middle" fill="{INK}">{c}</text>')
    for i, row in enumerate(cm):
        n = sum(row) or 1
        s.append(f'<text x="{x0-6}" y="{y0+i*cell+cell/2+4}" font-size="10" text-anchor="end" fill="{INK}">{CLASSES[i]}</text>')
        for j, c in enumerate(row):
            f = c/n; s.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell-1}" height="{cell-1}" fill="#1f5fa8" fill-opacity="{0.06+0.9*f:.3f}"/>'
                              f'<text x="{x0+j*cell+cell/2}" y="{y0+i*cell+cell/2+4}" font-size="10" text-anchor="middle" fill="{"#ffffff" if f > .5 else INK}">{c}</text>')
    return s
body = confusion(ham['systems']['dense7']['metrics']['confusion_matrix'], 'DenseNet121 (HAM test, physical units)', 70, 60)
body += confusion(ham['systems']['med7']['metrics']['confusion_matrix'], 'MedSigLIP probe (HAM test, physical units)', 440, 60)
body += ['<text x="20" y="372" font-size="11" fill="#55555a">Rows = true class; shading = row share (recall). Counts are physical lesion units (n=1,117).</text>']
svg('confusion_ham_seven_class.svg', 760, 385, ''.join(body), 'Seven-class confusion matrices on the HAM internal test')

manifest = {'inputs': {'M7_receipt': sha(M7/'receipt.json'), 'M7_metrics': r7['metrics_SHA'], 'M7_compute': r7['compute_SHA'],
                       'M8_receipt': sha(M8/'receipt.json'), 'M8_metrics': r8['metrics_SHA'], 'M8_compute': r8['compute_SHA']},
            'outputs': written, 'per_image_DDI_data_used': False, 'model_or_encoder_run': False}
(RES/'m9_build_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
print(json.dumps(manifest, indent=1))
