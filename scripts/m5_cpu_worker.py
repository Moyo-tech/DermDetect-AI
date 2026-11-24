#!/usr/bin/env python3
"""Offline M5 stage entry; real inputs require exact external approval first."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, sys, time
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dermdetect.models.m5_run import POLICY, membership, run, write
from dermdetect.models.densenet import DenseNetSystem, transform
from dermdetect.models.embeddings import load_cache
from dermdetect.data.manifests import file_sha256, rooted_file


def authorized(configuration, approval, stage):
    c=json.loads(Path(configuration).read_text()); a=json.loads(Path(approval).read_text())
    if (a.get('status')!='user_approved' or not a.get('user_reference') or
        a.get('configuration_SHA')!=file_sha256(Path(configuration)) or
        stage not in a.get('stages',[]) or a.get('held_out_access') is not False or
        a.get('cloud_or_GPU') is not False or c['policy']!=POLICY):
        raise ValueError('Exact M5 stage/configuration/CPU approval missing')
    for name,version in c['software'].items():
        if importlib.metadata.version(name)!=version: raise ValueError('Pinned local runtime changed')
    return c


def selected(c, rows):
    p=Path(c['inputs']['checkpoint']); assert file_sha256(p)==c['hashes']['checkpoint']
    value=torch.load(p,map_location='cpu',weights_only=True)
    if (value['configuration']!=c['Dense_training_configuration'] or value['binding']!=c['Dense_binding'] or
        value['membership']!={'train':[r for r in rows if r['split']=='train'],
                              'validation':[r for r in rows if r['split']=='validation']} or
        value['state']['best_checkpoint']!='best_epoch_021.pt' or value['state']['best_key'][2]!=21):
        raise ValueError('Selected DenseNet configuration/membership mismatch')
    if not value.get('validation_reference'): raise ValueError('Selected validation reference missing')
    return value


def extract(model, rows, root, check):
    before={k:v.detach().clone() for k,v in model.state_dict().items()}; rng=torch.get_rng_state().clone()
    model.eval().requires_grad_(False); values=[]
    with torch.inference_mode():
        for row in rows:
            check(); p=rooted_file(root,row['path'])
            if file_sha256(p)!=row['sha256']: raise ValueError('Development image hash differs')
            with Image.open(p) as im: x=transform(im,training=False).unsqueeze(0)
            z=model.features(x).cpu().numpy()
            if z.shape!=(1,1024) or not np.isfinite(z).all(): raise ValueError('DenseNet feature contract')
            values.append(z[0].copy())
            if file_sha256(p)!=row['sha256']: raise ValueError('Image changed during inference')
    if not torch.equal(rng,torch.get_rng_state()) or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):
        raise ValueError('Frozen DenseNet parameters/buffers/RNG changed')
    check(); return np.asarray(values,dtype=np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--approval',required=True)
    p.add_argument('--stage',choices=['benchmark','features','fitting'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); c=authorized(a.configuration,a.approval,a.stage) # before any scientific input open
    started=time.monotonic(); limit=c['budgets_seconds'][a.stage]
    def check():
        if time.monotonic()-started>=limit: raise TimeoutError('Stage CPU wall cap')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for s in ['train','validation']:
        path=Path(c['inputs'][s]);assert file_sha256(path)==c['hashes'][s];rows+=json.loads(path.read_text())
    membership(rows); value=selected(c,rows); rows.sort(key=lambda r:r['image_id']); output=Path(a.output)
    if a.stage=='fitting':
        ids,med=load_cache(c['inputs']['med_cache'],c['Med_binding'],c['hashes']['med_metadata'])
        dense=Path(c['inputs']['dense_cache']);metadata=json.loads((dense/'cache.json').read_text())
        if metadata['binding']!=c['Dense_feature_binding'] or file_sha256(dense/'features.npy')!=metadata['features_SHA']:
            raise ValueError('Dense feature cache binding/hash mismatch')
        receipt=json.loads((dense/'cache_receipt.json').read_text())
        if (receipt['source_configuration_SHA']!=file_sha256(Path(a.configuration)) or
            file_sha256(dense/'cache.json')!=receipt['metadata_SHA']):raise ValueError('Dense metadata receipt')
        x=np.load(dense/'features.npy',allow_pickle=False)
        if x.dtype!=np.float32 or x.shape!=(len(rows),1024):raise ValueError('Dense cache shape/dtype')
        result=run(rows,ids,med,metadata['ids'],x,value['validation_reference'],output,
                   policy=c['policy'],max_seconds=limit-(time.monotonic()-started),after_check=check)
        print(json.dumps({'status':result['status'],'review_required':result['review_required']}));return
    output.mkdir(); model=DenseNetSystem(value['configuration']['dropout']);model.load_state_dict(value['model'],strict=True)
    model.set_phase(value['state']['phase']); check()
    if a.stage=='features':
        b=json.loads(Path(c['inputs']['benchmark_result']).read_text())
        if b['configuration_SHA']!=file_sha256(Path(a.configuration)) or b['projected_full_feature_seconds']>c['budgets_seconds']['features']:
            raise ValueError('Benchmark projection exceeds fixed feature budget')
    chosen=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['image_id'])[:16] if a.stage=='benchmark' else rows
    encoded_start=time.monotonic(); x=extract(model,chosen,Path(c['inputs']['images']),check);seconds=time.monotonic()-encoded_start
    if a.stage=='benchmark':
        result=dict(configuration_SHA=file_sha256(Path(a.configuration)),rows=len(chosen),ids=[r['image_id'] for r in chosen],
                    encode_seconds=seconds,projected_full_feature_seconds=1.5*seconds/len(chosen)*len(rows)+(encoded_start-started),
                    projection_is_estimate=True,model_state_and_RNG_unchanged=True,wall_seconds=time.monotonic()-started)
        write(output/'benchmark.json',result)
    else:
        if not np.any(np.var(x,axis=0)>0): raise ValueError('Degenerate Dense features')
        np.save(output/'features.npy',x,allow_pickle=False)
        result=dict(binding=c['Dense_feature_binding'],ids=[r['image_id'] for r in rows],features_SHA=file_sha256(output/'features.npy'),
                    encode_seconds=seconds,wall_seconds=time.monotonic()-started,model_state_and_RNG_unchanged=True)
        write(output/'cache.json',result)
        write(output/'cache_receipt.json',dict(source_configuration_SHA=file_sha256(Path(a.configuration)),
                                             metadata_SHA=file_sha256(output/'cache.json')))
    print(json.dumps({'stage':a.stage,'status':'passed','rows':len(chosen),'wall_seconds':result['wall_seconds']}))

if __name__=='__main__': main()
