"""Invented CUDA interfaces only: never imports torch or loads a model."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from dermdetect.evaluation.torch_gpu_backend import (
    TorchGPUBackend, configure_cuda, check_device_memory, RESERVE_BYTES, MAX_ALLOCATED_BYTES)


def fake_torch():
    t = Mock()
    t.cuda.is_available.return_value = True
    t.cuda.device_count.return_value = 1
    t.cuda.mem_get_info.return_value = (12*2**30, 24*2**30)
    t.cuda.memory_allocated.return_value = 0
    t.cuda.max_memory_allocated.return_value = 0
    t.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False))
    return t


class GPUAdapterTests(unittest.TestCase):
    def test_missing_cublas_rejected_before_cpu_loader(self):
        with patch.dict(os.environ, {}, clear=True), patch('dermdetect.evaluation.torch_gpu_backend.TorchCPUBackend.__init__') as loader:
            with self.assertRaises(PermissionError): TorchGPUBackend(None, lambda: None)
            loader.assert_not_called()

    def test_deterministic_configuration(self):
        t = fake_torch()
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}): configure_cuda(t)
        self.assertFalse(t.backends.cuda.matmul.allow_tf32)
        self.assertFalse(t.backends.cudnn.allow_tf32)
        self.assertFalse(t.backends.cudnn.benchmark)
        self.assertTrue(t.backends.cudnn.deterministic)
        t.use_deterministic_algorithms.assert_called_once_with(True)
        t.cuda.set_device.assert_called_once_with(0)
        t.set_float32_matmul_precision.assert_called_once_with('highest')

    def test_device_visibility(self):
        for available, count in ((False,1),(True,0),(True,2)):
            t = fake_torch(); t.cuda.is_available.return_value = available; t.cuda.device_count.return_value = count
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), self.assertRaises(PermissionError): configure_cuda(t)

    def test_memory_reserve_and_model_preallocation(self):
        t = fake_torch()
        self.assertEqual(check_device_memory(t)['total_bytes'], 24*2**30)
        for free,total,extra in ((RESERVE_BYTES-1,24*2**30,0),(12*2**30,19*2**30,0),(3*2**30,24*2**30,2*2**30)):
            t.cuda.mem_get_info.return_value=(free,total)
            with self.assertRaises(PermissionError): check_device_memory(t,extra)

    def test_synchronized_timing_and_checks(self):
        b=object.__new__(TorchGPUBackend);b.torch=fake_torch();b.check=Mock();b.inference_gpu_seconds=0.
        with patch('dermdetect.evaluation.torch_gpu_backend.time.monotonic',side_effect=[10.,12.5]):
            started=b._start();b._end(started)
        self.assertEqual(b.inference_gpu_seconds,2.5)
        self.assertEqual(b.torch.cuda.synchronize.call_count,2)
        self.assertEqual(b.check.call_count,2)

    def test_current_and_peak_allocation_cap(self):
        for current, peak in ((MAX_ALLOCATED_BYTES+1, 0),(0, MAX_ALLOCATED_BYTES+1)):
            t=fake_torch();t.cuda.memory_allocated.return_value=current;t.cuda.max_memory_allocated.return_value=peak
            with self.assertRaises(PermissionError):check_device_memory(t)
        t=fake_torch();t.cuda.memory_allocated.return_value=MAX_ALLOCATED_BYTES;t.cuda.max_memory_allocated.return_value=MAX_ALLOCATED_BYTES
        check_device_memory(t)

    def test_close_releases_references_then_cached_allocator(self):
        b=object.__new__(TorchGPUBackend);b.torch=fake_torch();b.dense_model=object();b.med_model=object()
        def synchronized(device):
            self.assertIsNone(b.dense_model);self.assertIsNone(b.med_model)
            b.torch.cuda.empty_cache.assert_not_called()
        b.torch.cuda.synchronize.side_effect=synchronized
        b.close()
        b.torch.cuda.synchronize.assert_called_once_with(0)
        b.torch.cuda.empty_cache.assert_called_once_with()

    def test_finish_reports_gpu_and_both_rngs(self):
        for same in (True,False):
            b=object.__new__(TorchGPUBackend);b.torch=fake_torch();b.cuda_rng=object()
            b.torch.equal.return_value=same;b.inference_gpu_seconds=3.;b.gpu_transfer_seconds=1.
            with patch('dermdetect.evaluation.torch_gpu_backend.TorchCPUBackend.finish',return_value={'RNG_unchanged':True,'device':'cpu','inference_GPU_seconds':0.}):
                result=b.finish()
            self.assertEqual(result['device'],'cuda:0')
            self.assertEqual(result['inference_GPU_seconds'],3.)
            self.assertEqual(result['RNG_unchanged'],same)
            self.assertTrue(result['CPU_RNG_unchanged'])
            self.assertEqual(result['CUDA_RNG_unchanged'],same)


if __name__ == '__main__': unittest.main()
