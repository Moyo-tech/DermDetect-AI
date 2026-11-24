"""Opt-in CUDA inference adapter; scientific CPU implementation is unchanged.

Construction is not authorization. The separate runner must approve the CUDA
runtime, destination and numerical parity before any scientific execution.
"""
import io
import os
import time
from .torch_backend import TorchCPUBackend, state_sha


MIN_DEVICE_BYTES = 20 * 2**30
RESERVE_BYTES = 2 * 2**30
MAX_ALLOCATED_BYTES = 20 * 2**30


def configure_cuda(torch):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise PermissionError('Explicit deterministic cuBLAS configuration required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PermissionError('Exactly one visible approved CUDA device required')
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')


def check_device_memory(torch, extra_bytes=0):
    free, total = torch.cuda.mem_get_info(0)
    if (torch.cuda.memory_allocated(0) > MAX_ALLOCATED_BYTES
            or torch.cuda.max_memory_allocated(0) > MAX_ALLOCATED_BYTES):
        raise PermissionError('CUDA current or peak allocation exceeds reviewed cap')
    if total < MIN_DEVICE_BYTES or free < RESERVE_BYTES + extra_bytes:
        raise PermissionError('CUDA allocation or free VRAM reserve insufficient')
    return {'free_bytes': free, 'total_bytes': total}


class TorchGPUBackend(TorchCPUBackend):
    def __init__(self, grant, check):
        # Reject the environment before the inherited loader imports torch.
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise PermissionError('Explicit deterministic cuBLAS configuration required')
        super().__init__(grant, check)
        t = self.torch
        configure_cuda(t)
        self.device = t.device('cuda:0')
        self.inference_gpu_seconds = 0.
        self.gpu_transfer_seconds = 0.
        self.device_memory = check_device_memory(t)
        required = sum(x.numel() * x.element_size() for model in (self.dense_model, self.med_model)
                       for x in list(model.parameters()) + list(model.buffers()))
        check_device_memory(t, required)
        t.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        self.dense_model.to(device=self.device, dtype=t.float32)
        self.med_model.to(device=self.device, dtype=t.float32)
        t.cuda.synchronize(0)
        self.gpu_transfer_seconds = time.monotonic() - started
        check_device_memory(t)
        if (state_sha(self.dense_model), state_sha(self.med_model)) != self.initial:
            raise PermissionError('CUDA transfer changed selected model state')
        self.rng = t.get_rng_state().clone()
        self.cuda_rng = t.cuda.get_rng_state(0).clone()
        self.setup_seconds = time.monotonic() - self.started
        self.check()

    def _start(self):
        self.check()
        check_device_memory(self.torch)
        self.torch.cuda.synchronize(0)
        return time.monotonic()

    def _end(self, started):
        self.torch.cuda.synchronize(0)
        self.inference_gpu_seconds += time.monotonic() - started
        check_device_memory(self.torch)
        self.check()

    def dense(self, payload):
        from PIL import Image
        from dermdetect.models.densenet import transform
        self.check()
        with Image.open(io.BytesIO(payload)) as image:
            x = transform(image, training=False).unsqueeze(0)
        if x.dtype != self.torch.float32:
            raise ValueError('Pinned float32 transform required')
        started = self._start()
        x = x.to(device=self.device, dtype=self.torch.float32)
        with self.torch.inference_mode():
            features = self.dense_model.features(x)
            logits = self.dense_model.backbone.classifier(features)
        if features.shape != (1,1024) or logits.shape != (1,7) or not self.torch.isfinite(features).all() or not self.torch.isfinite(logits).all():
            raise ValueError('DenseNet inference shape/finite contract')
        result = tuple(logits[0].cpu().tolist()), tuple(features[0].cpu().tolist())
        self._end(started)
        return result

    def med(self, payload):
        from PIL import Image
        from dermdetect.models.tf_source_resize import evaluation_resize
        from dermdetect.models.embeddings import freeze_image_features
        self.check()
        with Image.open(io.BytesIO(payload)) as image:
            pixels = self.processor(images=evaluation_resize(image), do_resize=False, return_tensors='pt')['pixel_values']
        if pixels.dtype != self.torch.float32:
            raise ValueError('Pinned float32 processor required')
        started = self._start()
        z = freeze_image_features(self.med_model, pixels.to(device=self.device, dtype=self.torch.float32))
        if z.shape != (1,1152):
            raise ValueError('MedSigLIP feature shape mismatch')
        result = tuple(z[0].tolist())
        self._end(started)
        return result

    def finish(self):
        self.torch.cuda.synchronize(0)
        result = super().finish()
        cuda_unchanged = bool(self.torch.equal(self.cuda_rng, self.torch.cuda.get_rng_state(0)))
        result.update(device='cuda:0', inference_GPU_seconds=self.inference_gpu_seconds,
                      inference_GPU_timing='synchronized wall seconds including transfers; excludes preprocessing',
                      CPU_RNG_unchanged=result['RNG_unchanged'], CUDA_RNG_unchanged=cuda_unchanged,
                      RNG_unchanged=result['RNG_unchanged'] and cuda_unchanged,
                      gpu_transfer_seconds=self.gpu_transfer_seconds,
                      GPU_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(0),
                      GPU_peak_reserved_bytes=self.torch.cuda.max_memory_reserved(0),
                      GPU_name=self.torch.cuda.get_device_name(0),
                      GPU_memory=check_device_memory(self.torch))
        return result

    def close(self):
        super().close()
        self.torch.cuda.synchronize(0)
        self.torch.cuda.empty_cache()
