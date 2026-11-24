"""Authorized HAM inference -> physical predictions -> recorded paired metrics.

No fitting or automatic model acquisition. DDI is deliberately a separate gate.
"""
import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from . import arrays, engine, metrics
from .authorization import RunGrant, checked_file, checked_json, runtime_specs, file_sha, route, input_root, checked_input_file


def normalize_rows(raw, partition, *, csv_format=False):
    out = []
    for r in raw:
        active = r.get('active', 'true' if csv_format else True)
        if type(active) is not bool and (type(active) is not str or active not in ('true','false')):
            raise ValueError('Canonical active flag required')
        if active in (False, 'false'):
            continue
        split = r.get('partition', r.get('split', partition if csv_format else None))
        label = r.get('label', r.get('dx'))
        if split != partition or label not in engine.CLASSES:
            raise ValueError('Cross-partition or noncanonical HAM label')
        row = {'image_id':r['image_id'], 'physical_id':r.get('physical_id', r.get('analysis_lesion_id')),
               'cluster_id':r.get('cluster_id', r.get('split_group_id')), 'label':label,
               'sha256':r['sha256'], 'path':r['path'], 'partition':partition, 'active':True}
        if (not re.fullmatch('[A-Za-z0-9_-]+', row['image_id']) or not row['physical_id'] or not row['cluster_id']
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Canonical membership fields required')
        p = Path(row['path'])
        if p.is_absolute() or '..' in p.parts or len(p.parts) < 1:
            raise ValueError('Relative bound payload route required')
        if 'target' in r and r['target'] != engine.CLASSES.index(label):
            raise ValueError('Target/class order mismatch')
        out.append(row)
    if not out or len({r['image_id'] for r in out}) != len(out):
        raise ValueError('Unique nonempty active image population required')
    return tuple(out)


def load_rows(grant):
    if not isinstance(grant, RunGrant):
        raise PermissionError('Verified grant required before data open')
    grant.check(grant.partition)
    p = checked_file(grant.root, grant.dataset['manifest'])
    if p.stat().st_size > 16 * 1024 ** 2:
        raise ValueError('Bounded evaluation manifest required')
    fmt = grant.dataset['format']
    if fmt == 'm2_analysis_csv':
        with p.open(newline='') as f:
            raw = tuple(csv.DictReader(f))
    elif fmt == 'm4_development_json':
        raw = json.loads(p.read_text())
    else:
        raise ValueError('Explicit HAM manifest format required')
    rows = normalize_rows(raw, grant.partition, csv_format=fmt == 'm2_analysis_csv')
    expected = grant.dataset.get('canonical_rows_SHA')
    if expected is not None and engine.digest(rows) != expected:
        raise PermissionError('Canonical evaluation membership mismatch')
    return rows


class PayloadReader:
    """Read only approved row routes; verify exact bytes before decoding."""
    def __init__(self, grant):
        if not isinstance(grant, RunGrant):
            raise PermissionError('Verified grant required')
        self.grant = grant; self.members = {r['image_id']:r for r in load_rows(grant)}
        self.archives = {}
        mode = grant.dataset['payload_mode']
        if mode == 'zip':
            for alias, binding in grant.dataset['sources'].items():
                p = checked_input_file(grant.root, binding)
                self.archives[alias] = zipfile.ZipFile(p)
        elif mode != 'files':
            raise ValueError('Explicit files/zip payload mode required')
    def read(self, row):
        if self.members.get(row['image_id']) != row:
            raise PermissionError('Unapproved payload row')
        if self.grant.dataset['payload_mode'] == 'zip':
            parts = Path(row['path']).parts
            if len(parts) != 2 or parts[0] not in self.archives or parts[1] != row['image_id'] + '.jpg':
                raise PermissionError('Only inventoried HAM image route permitted')
            archive = self.archives[parts[0]]
            matches = [i for i in archive.infolist() if i.filename == parts[1]]
            if len(matches) != 1 or matches[0].file_size > 10 * 1024 ** 2:
                raise ValueError('Unique bounded image archive member required')
            payload = archive.read(matches[0])
        else:
            base = input_root(self.grant.root, self.grant.dataset['payload_root'])
            p = route(base, row['path'])
            if p.stat().st_size > 10 * 1024 ** 2:
                raise ValueError('Bounded image payload required')
            payload = p.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('Image checksum mismatch before decoder')
        return payload
    def close(self):
        for archive in self.archives.values():
            archive.close()


def infer(rows, backend, probes, read_payload, check=lambda:None):
    """One encoder pass per model/view; shared features feed both system branches."""
    rows = tuple(rows); supplied = {n:{'image_ids':[], 'image_SHAs':[], 'logits':[]} for n in engine.SYSTEMS}
    stage = {'dense_encode_seconds':0., 'med_encode_seconds':0., 'probe_seconds':0., 'payload_seconds':0.}
    for row in rows:
        check(); started=time.monotonic(); payload=read_payload(row); stage['payload_seconds']+=time.monotonic()-started
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError('Unverified bytes cannot enter encoder')
        started=time.monotonic(); dense7, dense_features=backend.dense(payload); stage['dense_encode_seconds']+=time.monotonic()-started
        check(); started=time.monotonic(); med_features=backend.med(payload); stage['med_encode_seconds']+=time.monotonic()-started
        check(); started=time.monotonic()
        values={'dense7':tuple(dense7), 'med7':arrays.probe_logits(probes['med7'],med_features)}
        if engine.BINARY[row['label']] is not None:
            values['dense2']=arrays.probe_logits(probes['dense2'],dense_features)
            values['med2']=arrays.probe_logits(probes['med2'],med_features)
        for name, z in values.items():
            item=supplied[name];item['image_ids'].append(row['image_id']);item['image_SHAs'].append(row['sha256']);item['logits'].append(z)
        stage['probe_seconds']+=time.monotonic()-started
    for item in supplied.values():
        item['payload_SHA']=engine.digest({'image_ids':item['image_ids'],'logits':item['logits']})
    return supplied,stage


def physical_predictions(rows, supplied, specs):
    out={}
    for name,item in supplied.items():
        k=7 if name.endswith('7') else 2;keep=rows if k==7 else tuple(r for r in rows if engine.BINARY[r['label']] is not None)
        ids,z,y,clusters=engine.aggregate(keep,item['image_ids'],item['logits'],k)
        probs,_=metrics.probabilities(z,specs[name]['temperature']);records=[]
        for u,group,logits,label,p in zip(ids,clusters,z,y,probs):
            prediction=(1 if p[1]>=.5 else 0) if k==2 else max(range(k),key=p.__getitem__)
            records.append({'physical_id':u,'cluster_id':group,'label':label,'mean_logits':logits,'probabilities':p,
                            'prediction':prediction,'confidence':max(p),'accepted':[max(p)>=c for c in specs[name]['cutoffs']]})
        out[name]=records
    return out


def write_once(path, data):
    body=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x') as f:
        f.write(body);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(body.encode()).hexdigest()


def run_ham(grant, backend, output, *, check=lambda:None, reader_factory=PayloadReader):
    """Run only under verified authority; reserve new output, retain failed runs."""
    if not isinstance(grant,RunGrant):
        raise PermissionError('Verified grant required before opening any input')
    grant.check(grant.partition);grant.reverify();check()
    rows=load_rows(grant);specs=runtime_specs(grant.systems);grant.check_rows(rows)
    output=Path(output)
    if output.absolute()!=Path(grant.approval['output']).absolute():
        raise PermissionError('Output must equal approved destination')
    output.mkdir()  # Existing/partial runs cannot be overwritten or silently retried.
    began=time.monotonic();reader=None
    try:
        probes={n:arrays.portable_probe(checked_file(grant.root,s['probe'])) for n,s in grant.systems['systems'].items() if n!='dense7'}
        reader=reader_factory(grant);check()
        supplied,stage=infer(rows,backend,probes,reader.read,check)
        invariance=backend.finish();check()
        if invariance.get('frozen_state_unchanged') is not True or invariance.get('RNG_unchanged') is not True:
            raise ValueError('Inference altered frozen model/RNG state')
        predictions={'partition':grant.partition,'image_logits':supplied,'physical':physical_predictions(rows,supplied,specs)}
        pred_SHA=write_once(output/'predictions.json',predictions)
        started=time.monotonic()
        result=engine.evaluate(rows,supplied,specs,partition=grant.partition,grant=grant,bootstrap=True,check=check)
        stage['metrics_and_bootstrap_seconds']=time.monotonic()-started
        result['predictions_SHA']=pred_SHA
        metric_SHA=engine.record_once(output/'metrics.json',result,input_SHA=grant.dataset['manifest']['SHA'],source_SHA=grant.approval['evaluator_manifest']['SHA'])
        grant.reverify();check()
        # Repeat exact active file hashes; archives are reverified by the grant.
        for row in rows:
            check();reader.read(row)
        accounting={'pipeline_wall_seconds':time.monotonic()-began,'supervisor_end_to_end_record_required':True,'stages':stage,'backend':invariance,'metrics_GPU_seconds':0.,
                    'scope':'locked inference; excludes historical development costs, which must be reported separately',
                    'development_costs_excluded_from_efficiency_claim':False,'first_run_and_cached_adaptation_must_be_reported_separately':True}
        cost_SHA=write_once(output/'compute.json',accounting)
        receipt={'schema':'ham-evaluation-receipt-v1','status':'complete','partition':grant.partition,'lock_SHA':grant.lock_SHA,
                 'dataset_SHA':grant.dataset['manifest']['SHA'],'predictions_SHA':pred_SHA,'metrics_SHA':metric_SHA,'compute_SHA':cost_SHA,
                 'lock_written':False,'held_out_access':grant.partition=='internal_test','recorded_before_error_review':True}
        receipt_SHA=write_once(output/'receipt.json',receipt)
        return dict(receipt,receipt_SHA=receipt_SHA)
    except BaseException as exc:
        write_once(output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc),'partial_outputs_not_final':True})
        raise
    finally:
        if reader is not None:reader.close()
        if hasattr(backend,'close'):backend.close()


def permit_recorded_review(output, expected_receipt_SHA, *, supervision_record=None, expected_supervision_SHA=None):
    if supervision_record is None or expected_supervision_SHA is None:
        raise PermissionError('Reaped, source-verified supervision record required')
    supervision_record=Path(supervision_record)
    if supervision_record.is_symlink() or file_sha(supervision_record)!=expected_supervision_SHA:
        raise PermissionError('Supervision record hash mismatch')
    supervision=json.loads(supervision_record.read_text())
    if supervision.get('status')!='passed' or supervision.get('worker_reaped') is not True or supervision.get('source_integrity_after') is not True or supervision.get('receipt_SHA')!=expected_receipt_SHA:
        raise PermissionError('Unfinished or stopped supervised run cannot enter review')
    output=Path(output);p=output/'receipt.json'
    if p.is_symlink() or file_sha(p)!=expected_receipt_SHA:
        raise PermissionError('Complete hash-bound receipt required before error review')
    r=json.loads(p.read_text())
    if r.get('status')!='complete' or r.get('recorded_before_error_review') is not True:
        raise PermissionError('Failed/partial evaluation cannot enter error review')
    for name,key in [('predictions.json','predictions_SHA'),('metrics.json','metrics_SHA'),('compute.json','compute_SHA')]:
        q=output/name
        if q.is_symlink() or file_sha(q)!=r[key]:raise PermissionError('Recorded output changed')
    return engine.permit_error_review(output/'metrics.json',r['metrics_SHA'])
