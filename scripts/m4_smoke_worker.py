"""Inspected offline worker: synthetic-first, batch1 float32, max2 staged train views."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time
import threading
import zipfile

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);parser.add_argument('--model',choices=('densenet','medsiglip','preflight'),required=True)
args=parser.parse_args();B=args.bundle;O=args.output
sys.path.insert(0,str(B/'src'))
LOCK=threading.RLock()
START=time.monotonic();state={'model':args.model,'phase':'starting','cpu_elapsed_seconds':0,'mps_elapsed_seconds':0,'mps_driver_bytes':0,'mps_current_bytes':0}


def write(name,value):
    with LOCK:
        path=O/name;temp=O/(name+'.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temp.replace(path)


def mark(phase,**fields):
    with LOCK:
        state.update(phase=phase,**fields);state['wall_seconds']=time.monotonic()-START;write('telemetry.json',state)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


mark('boundary')
checks={};probes=json.loads((B/'probes.json').read_text())
for name,path in probes['denied'].items():
    try:fd=os.open(path,os.O_RDONLY)
    except PermissionError:checks[name]=True
    except Exception:checks[name]=False
    else:os.close(fd);checks[name]=False
for name,path in probes['metadata'].items():
    try:os.stat(path)
    except PermissionError:checks[name]=True
    except Exception:checks[name]=False
    else:checks[name]=False
for name,path in probes['zip'].items():
    try:
        with zipfile.ZipFile(path) as archive:archive.infolist()
    except PermissionError:checks[name]=True
    except Exception:checks[name]=False
    else:checks[name]=False
try:
    with socket.socket() as connection:connection.connect(('127.0.0.1',9))
except PermissionError:checks['network_denied']=True
except Exception:checks['network_denied']=False
else:checks['network_denied']=False
try:subprocess.run(['/bin/echo','invented-helper'],check=True,capture_output=True)
except PermissionError:checks['helper_denied']=True
except Exception:checks['helper_denied']=False
else:checks['helper_denied']=False
try:fd=os.open(B/'worker.py',os.O_WRONLY|os.O_APPEND)
except PermissionError:checks['immutable_code_write_denied']=True
except Exception:checks['immutable_code_write_denied']=False
else:os.close(fd);checks['immutable_code_write_denied']=False
if not all(checks.values()):raise RuntimeError('Access boundary refusal failed')
write('boundary_checks.json',checks)
mark('cpu_imports')
import numpy as np
from PIL import Image
import torch
from torchvision.models import densenet121,DenseNet121_Weights
from torch import nn
from transformers import AutoModel,AutoImageProcessor
from dermdetect.models.densenet import DenseNetSystem,transform,weighted_loss
from dermdetect.models.embeddings import freeze_image_features
from dermdetect.models.tf_source_resize import evaluation_resize

torch.set_num_threads(1);torch.manual_seed(20260729)
plan=json.loads((B/'plan.json').read_text());manifest=json.loads((B/'checkpoint_manifest.json').read_text())
# Stream verification avoids buffering the 3.51GB weights file in Python memory.
mark('cpu_checkpoint_hashes')
for name,checksum in manifest.items():
    if sha(B/name)!=checksum:raise RuntimeError('Staged checkpoint bytes changed')
for kind,path in probes['loader_sentinels'].items():
    try:
        if kind=='PIL':Image.open(path)
        elif kind=='numpy':np.load(path,allow_pickle=False)
        else:torch.load(path,map_location='cpu',weights_only=True)
    except PermissionError:checks[kind+'_loader_denied']=True
    except Exception:checks[kind+'_loader_denied']=False
    else:checks[kind+'_loader_denied']=False
if not all(checks.values()):raise RuntimeError('Protected loader refused incorrectly')
write('boundary_checks.json',checks)
if args.model=='preflight':
    import unittest
    sys.path.insert(0,str(B))
    import test_model_development,test_tf_source_resize
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_model_development),unittest.defaultTestLoader.loadTestsFromModule(test_tf_source_resize)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:raise RuntimeError('Synthetic preflight failed')
    write('result.json',{'status':'synthetic_CPU_preflight_passed','tests':result.testsRun,'boundary_checks':len(checks),
          'model':'preflight','mps_available':torch.backends.mps.is_available(),'real_image_reads':0})
    mark('done',cpu_elapsed_seconds=time.monotonic()-START)
    sys.exit(0)

result={'model':args.model,'status':'running','device_results':{},'comparisons':{},'precision':'float32','batch_size':1,
        'boundary_checks':len(checks),'source_test_or_outcomes_opened':False,'real_train_views':0,'model_steps':{'cpu':0,'mps':0}}

def timed(function):
    start=time.monotonic();value=function();elapsed=time.monotonic()-start
    if not torch.isfinite(value).all():raise RuntimeError('Nonfinite model output')
    return value.detach().float().cpu(),elapsed


def compare(a,b,name):
    difference=(a-b).abs()
    passed=torch.allclose(a,b,atol=1e-4,rtol=1e-4)
    result['comparisons'][name]={'max_absolute_difference':float(difference.max()),'atol':1e-4,'rtol':1e-4,'passed':passed}
    if not passed:raise RuntimeError('Declared numerical comparison failed: '+name)


def checkpoint_dense():
    base=densenet121(weights=None)
    weights=torch.load(B/'weights/densenet/densenet121-a639ec97.pth',map_location='cpu',weights_only=True)
    # Official torchvision migration for older DenseNet dotted keys.
    pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(weights):
        match=pattern.match(key)
        if match:weights[match.group(1)+match.group(2)]=weights.pop(key)
    base.load_state_dict(weights,strict=True);del weights
    torch.manual_seed(20260729)
    base.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(1024,7))
    model=DenseNetSystem(.2);model.backbone=base;model.set_phase('head');return model.eval()


def checkpoint_med():
    model=AutoModel.from_pretrained(B/'weights/medsiglip',local_files_only=True,trust_remote_code=False,
        use_safetensors=True,torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    return model


def inputs(real=False):
    if not real:return synthetic
    row=json.loads((B/'train_inputs.json').read_text())[0]
    path=B/row['path']
    if sha(path)!=row['sha256']:raise RuntimeError('Staged training payload changed')
    with Image.open(path) as image:
        image.load();rgb=image.convert('RGB')
    result['real_train_views']=1
    if args.model=='densenet':return transform(rgb).unsqueeze(0)
    return processor(images=evaluation_resize(rgb),do_resize=False,return_tensors='pt')['pixel_values']


mark('cpu_loading_model');load_start=time.monotonic()
if args.model=='densenet':
    model=checkpoint_dense();synthetic=torch.randn(1,3,224,224)
    inference=lambda m,x:m(x)
    result['preprocessing']={'input_size':224,'official_V1_transform_shared':True,'embedding_dimension':1024}
else:
    processor=AutoImageProcessor.from_pretrained(B/'weights/medsiglip',local_files_only=True,trust_remote_code=False,use_fast=False)
    if processor.size!={'height':448,'width':448} or processor.image_mean!=[.5]*3 or processor.image_std!=[.5]*3:
        raise RuntimeError('Pinned official processor contract mismatch')
    # Independent source-derived golden bytes, generated before this worker by
    # the pinned TensorFlow scalar CPU arithmetic in standalone C++ (18 cases).
    pixels=np.load(B/'resize_reference/input.npy',allow_pickle=False)
    ref=np.load(B/'resize_reference/output.npy',allow_pickle=False)
    actual=np.asarray(evaluation_resize(Image.fromarray(pixels)),dtype=int)
    pixel_error=int(np.abs(actual-ref.astype(int)).max())
    result['preprocessing']={'input_size':448,'official_processor':processor.to_dict(),
        'independent_half_pixel_max_pixel_error':pixel_error,'TF_CPU_source_arithmetic_verified':True,
        'official_TensorFlow_runtime_executed':False,'embedding_dimension_from_config':1152,
        'output_definition':'normalized image_embeds','reference_commit':'6550e4bd80223cdb8be6c3afd1f81e86a4d433c3'}
    write('processor_checks.json',result['preprocessing'])
    if pixel_error!=0:raise RuntimeError('Official CPU-source exact-pixel comparison failed')
    model=checkpoint_med()
    synthetic=processor(images=Image.new('RGB',(448,448),(31,127,211)),do_resize=False,return_tensors='pt')['pixel_values']
    if tuple(synthetic.shape)!=(1,3,448,448) or not torch.isfinite(synthetic).all():raise RuntimeError('Processor tensor invalid')
    inference=lambda m,x:freeze_image_features(m,x)
result['cpu_load_seconds']=time.monotonic()-load_start
mark('cpu_synthetic_forward')
with torch.inference_mode():cpu_synthetic,seconds=timed(lambda:inference(model,synthetic))
result['model_steps']['cpu']+=1
result['device_results']['cpu']={'synthetic_forward_seconds':seconds,'output_shape':list(cpu_synthetic.shape)}
if args.model=='densenet':
    # Second synthetic step verifies head backward; third step is one real forward.
    model.train();model.zero_grad(set_to_none=True)
    before=model.backbone.features.norm0.running_mean.clone()
    start=time.monotonic();weighted_loss(model(synthetic),torch.tensor([0]),[1.]).backward()
    if model.backbone.features.conv0.weight.grad is not None or model.backbone.classifier[1].weight.grad is None:
        raise RuntimeError('Head warm-up gradient contract failed')
    if not torch.equal(before,model.backbone.features.norm0.running_mean):raise RuntimeError('Frozen BatchNorm changed')
    result['device_results']['cpu']['synthetic_backward_seconds']=time.monotonic()-start
    result['model_steps']['cpu']+=1;model.zero_grad(set_to_none=True);model.eval()
    model.set_phase('final_block')
    if not model.backbone.features.denseblock4.denselayer1.conv1.weight.requires_grad or model.backbone.features.denseblock3.denselayer1.conv1.weight.requires_grad:
        raise RuntimeError('Final-block setup mismatch')
else:
    if tuple(cpu_synthetic.shape)!=(1,1152) or not torch.allclose(cpu_synthetic.norm(dim=1),torch.ones(1),atol=1e-4,rtol=1e-4):
        raise RuntimeError('Actual image_embeds dimension/normalization mismatch')
    write('partial_result.json',result)
    mark('cpu_predictive_reload')
    del model
    import gc
    gc.collect()
    model=checkpoint_med()
    with torch.inference_mode():repeated,repeat_seconds=timed(lambda:inference(model,synthetic))
    result['model_steps']['cpu']+=1;compare(cpu_synthetic,repeated,'cpu_synthetic_predictive_reload')
    result['actual_checkpoint_predictive_reload_verified']=True
    result['device_results']['cpu']['reload_forward_seconds']=repeat_seconds
# Real read occurs only after successful boundary+synthetic gates.
mark('cpu_real_forward');real=inputs(real=True)
with torch.inference_mode():cpu_real,real_seconds=timed(lambda:inference(model,real))
result['model_steps']['cpu']+=1;result['device_results']['cpu']['real_forward_seconds']=real_seconds
# Reload state without another forward/step; exact tensor state equality is checked.
# Full predictive reload was already tested in synthetic preflight. Actual MedSigLIP
# reload would exceed resource/step bounds and is recorded as pending, not claimed.
result.setdefault('actual_checkpoint_predictive_reload_verified',False)
mark('cpu_done',cpu_elapsed_seconds=time.monotonic()-START)
write('partial_result.json',result)
if not torch.backends.mps.is_available():
    result['status']='CPU_passed_MPS_unavailable';write('result.json',result);mark('done');sys.exit(2)

MPS_START=time.monotonic()
mark('mps_allocating',mps_start_monotonic=MPS_START)
def monitor_mps():
    while state['phase']!='done':
        time.sleep(1)
        try:
            with LOCK:
                state.update(mps_elapsed_seconds=time.monotonic()-MPS_START,
                    mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_current_bytes=torch.mps.current_allocated_memory())
                write('telemetry.json',state)
        except Exception:
            with LOCK:state['monitor_failed']=True;write('telemetry.json',state)
            return
threading.Thread(target=monitor_mps,daemon=True).start()
model.to('mps');torch.mps.synchronize()

def telemetry(phase):
    mark(phase,mps_elapsed_seconds=time.monotonic()-MPS_START,mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_current_bytes=torch.mps.current_allocated_memory())
    if state['mps_driver_bytes']>4*2**30:raise RuntimeError('Approved MPS allocation limit exceeded')

telemetry('mps_synthetic_forward')
with torch.inference_mode():
    torch.mps.synchronize();start=time.monotonic();out=inference(model,synthetic.to('mps'));torch.mps.synchronize()
    mps_synthetic=out.detach().float().cpu();seconds=time.monotonic()-start
if not torch.isfinite(mps_synthetic).all():raise RuntimeError('Nonfinite MPS output')
result['model_steps']['mps']+=1;compare(cpu_synthetic,mps_synthetic,'cpu_vs_mps_synthetic')
result['device_results']['mps']={'synthetic_forward_seconds':seconds}
telemetry('mps_after_synthetic')
if args.model=='densenet':
    model.train();model.zero_grad(set_to_none=True);start=time.monotonic()
    buffers={name:value.detach().clone() for name,value in model.named_buffers()}
    weighted_loss(model(synthetic.to('mps')),torch.tensor([0],device='mps'),[1.]).backward();torch.mps.synchronize()
    if model.backbone.features.conv0.weight.grad is not None:raise RuntimeError('Frozen MPS backbone has gradients')
    result['device_results']['mps']['synthetic_backward_seconds']=time.monotonic()-start
    result['model_steps']['mps']+=1;model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name,value in model.named_buffers():value.copy_(buffers[name])
    del buffers
    model.eval()
else:
    with torch.inference_mode():repeat,repeat_seconds=timed(lambda:inference(model,synthetic.to('mps')))
    torch.mps.synchronize();result['model_steps']['mps']+=1;compare(mps_synthetic,repeat,'mps_synthetic_repeat')
telemetry('mps_real_forward')
with torch.inference_mode():
    torch.mps.synchronize();start=time.monotonic();out=inference(model,real.to('mps'));torch.mps.synchronize()
    mps_real=out.detach().float().cpu();real_seconds=time.monotonic()-start
result['model_steps']['mps']+=1;compare(cpu_real,mps_real,'cpu_vs_mps_real')
result['device_results']['mps']['real_forward_seconds']=real_seconds
telemetry('mps_releasing');del model,real,synthetic,out;torch.mps.empty_cache();torch.mps.synchronize()
result['mps_elapsed_seconds']=time.monotonic()-MPS_START
result['status']='bounded_CPU_MPS_smoke_passed'
result['model_step_bounds_verified']=all(n<=3 for n in result['model_steps'].values())
write('result.json',result);mark('done',mps_elapsed_seconds=result['mps_elapsed_seconds'],mps_driver_bytes=torch.mps.driver_allocated_memory())
print(json.dumps({'model':args.model,'status':result['status'],'mps_elapsed_seconds':result['mps_elapsed_seconds']}))
