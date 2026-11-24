"""CPU port of pinned TensorFlow ResizeBilinear arithmetic; no TF dependency.

Sources: TensorFlow v2.18.0 core/kernels/image/resize_bilinear_op.cc
(compute_interpolation_weights, compute_lerp), core/util/image_resizer_state.h
(CalculateResizeScale, HalfPixelScaler); immutable source record accompanies M4.
Float32 half-pixel coordinates, nested difference lerps, no antialias/crop.
The former Torch interpolate proposal remains preserved as M3 evidence.
"""
import numpy as np
from PIL import Image


def bilinear_uint8(pixels, size=(448, 448)):
    pixels = np.asarray(pixels)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8 or min(pixels.shape[:2]) < 1:
        raise ValueError("Nonempty RGB uint8 pixels required")
    if len(size) != 2 or any(type(s) is not int or s < 1 for s in size):
        raise ValueError("Positive integer target dimensions required")
    if pixels.shape[:2] == tuple(size):
        return pixels.copy()
    def axis(source, target):
        scale = np.float32(source) / np.float32(target)
        coordinate = (np.arange(target, dtype=np.float32) + np.float32(.5)) * scale - np.float32(.5)
        floored = np.floor(coordinate)
        lower = np.maximum(floored.astype(np.int64), 0)
        upper = np.minimum(np.ceil(coordinate).astype(np.int64), source - 1)
        return lower, upper, coordinate - floored
    y0, y1, wy = axis(pixels.shape[0], size[0])
    x0, x1, wx = axis(pixels.shape[1], size[1])
    data = pixels.astype(np.float32)
    tl, tr = data[y0[:, None], x0[None, :]], data[y0[:, None], x1[None, :]]
    bl, br = data[y1[:, None], x0[None, :]], data[y1[:, None], x1[None, :]]
    top = tl + (tr - tl) * wx[None, :, None]
    bottom = bl + (br - bl) * wx[None, :, None]
    out = top + (bottom - top) * wy[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def evaluation_resize(image):
    return Image.fromarray(bilinear_uint8(np.asarray(image.convert("RGB"))))
