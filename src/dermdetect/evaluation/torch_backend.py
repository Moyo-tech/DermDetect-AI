"""Offline CPU inference of the unchanged selected DenseNet and pinned MedSigLIP.

Imported lazily after grant, runtime, boundary and resource admission. No fitting,
checkpoint download, tokenizer/text inference or alternate preprocessing.
"""
import hashlib
import importlib.metadata
import io
import json
import resource
import sys
import time
from pathlib import Path
from .authorization import checked_file, checked_json, file_sha


def runtime_versions():
    return {'python':'.'.join(map(str,sys.version_info[:3])),
            'distributions':{n:importlib.metadata.version(n) for n in
                             ('torch','torchvision','numpy','scipy','scikit-learn','transformers','Pillow','safetensors')}}


def native_runtime_files():
    # Named dependencies of this installed CPython's _lzma/_ssl extensions.
    # _ssl is imported by the offline HF/requests stack; network stays denied.
    # Pin individual existing dylibs, never grant the Homebrew tree or install.
    paths=['/opt/homebrew/opt/xz/lib/liblzma.5.dylib',
           '/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib',
           '/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib']
    return [{'path':p,'SHA':file_sha(p)} for p in paths]


def peak_rss_bytes(usage=None,platform_name=None):
    value=(usage if usage is not None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return int(value)*(1024 if (platform_name or sys.platform).startswith('linux') else 1)


def state_sha(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str((a.shape,a.dtype)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


class TorchCPUBackend:
    def __init__(self,grant,check):
        if runtime_versions()!=grant.evaluator['runtime']:
            raise PermissionError('Actual CPU runtime differs from reviewed evaluator')
        check();self.started=time.monotonic();self.check=check
        import torch
        from dermdetect.models.densenet import DenseNetSystem
        from transformers import AutoModel, AutoImageProcessor
        self.torch=torch;torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        c=checked_json(grant.root,grant.systems['M5_source_configuration'])
        s=grant.systems['systems']['dense7'];p=checked_file(grant.root,s['selected_backbone'])
        t=time.monotonic();saved=torch.load(p,map_location='cpu',weights_only=True)
        if (saved['configuration']!=c['Dense_training_configuration'] or saved['binding']!=c['Dense_binding']
                or saved['state']['best_checkpoint']!='best_epoch_021.pt' or saved['state']['best_key'][2]!=21):
            raise ValueError('Selected checkpoint scientific contract mismatch')
        self.dense_model=DenseNetSystem(saved['configuration']['dropout']);self.dense_model.load_state_dict(saved['model'],strict=True)
        self.dense_model.set_phase(saved['state']['phase']);self.dense_model.eval().requires_grad_(False)
        del saved
        self.dense_load_seconds=time.monotonic()-t;check()
        bindings=grant.systems['systems']['med7']['selected_backbone'];paths=[checked_file(grant.root,b) for b in bindings]
        snapshot=paths[0].parent
        if any(p.parent!=snapshot for p in paths) or {p.name for p in snapshot.iterdir()}!={p.name for p in paths}:
            raise ValueError('Exact native snapshot composition required')
        binding=grant.systems['systems']['med7']['feature_binding']
        if binding['revision']!=grant.systems['Med_revision'] or binding['normalization']!='l2' or binding['dimension']!=1152:
            raise ValueError('Original MedSigLIP feature binding required')
        t=time.monotonic()
        self.med_model=AutoModel.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,use_safetensors=True).cpu().eval().requires_grad_(False)
        self.processor=AutoImageProcessor.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,use_fast=False)
        if json.loads((snapshot/'preprocessor_config.json').read_text())!=binding['processor']:
            raise ValueError('Pinned MedSigLIP processor changed')
        self.med_load_seconds=time.monotonic()-t;check()
        self.initial=(state_sha(self.dense_model),state_sha(self.med_model));self.rng=torch.get_rng_state().clone()
        self.setup_seconds=time.monotonic()-self.started;self.check()
    def dense(self,payload):
        from PIL import Image
        from dermdetect.models.densenet import transform
        self.check()
        with Image.open(io.BytesIO(payload)) as image:
            x=transform(image,training=False).unsqueeze(0)
        with self.torch.inference_mode():
            features=self.dense_model.features(x)
            # Exactly one DenseNet backbone pass; its original selected head uses
            # the same pooled features supplied to the binary linear probe.
            logits=self.dense_model.backbone.classifier(features)
        if features.shape!=(1,1024) or logits.shape!=(1,7) or not self.torch.isfinite(features).all() or not self.torch.isfinite(logits).all():
            raise ValueError('DenseNet inference shape/finite contract')
        return tuple(logits[0].tolist()),tuple(features[0].tolist())
    def med(self,payload):
        from PIL import Image
        from dermdetect.models.tf_source_resize import evaluation_resize
        from dermdetect.models.embeddings import freeze_image_features
        self.check()
        with Image.open(io.BytesIO(payload)) as image:
            pixels=self.processor(images=evaluation_resize(image),do_resize=False,return_tensors='pt')['pixel_values']
        if pixels.dtype!=self.torch.float32:raise ValueError('Pinned float32 processor required')
        z=freeze_image_features(self.med_model,pixels)
        if z.shape!=(1,1152):raise ValueError('MedSigLIP feature shape mismatch')
        return tuple(z[0].tolist())
    def finish(self):
        self.check();final=(state_sha(self.dense_model),state_sha(self.med_model))
        return {'frozen_state_unchanged':final==self.initial,'RNG_unchanged':bool(self.torch.equal(self.rng,self.torch.get_rng_state())),
                'dense_load_seconds':self.dense_load_seconds,'med_load_seconds':self.med_load_seconds,
                'setup_including_hashes_seconds':self.setup_seconds,'device':'cpu','inference_GPU_seconds':0.,
                'worker_peak_RSS_bytes':peak_rss_bytes(),
                'no_fit_or_calibration':True,'one_backbone_pass_per_model_per_view':True}
    def close(self):
        self.dense_model=None;self.med_model=None
