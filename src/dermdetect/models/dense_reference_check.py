"""Read-only DenseNet CPU-feature versus stored physical-VAL reference check.

The M4 selected checkpoint stores physical-validation logits computed on the
training device. M5 re-extracts pooled 1024-d DenseNet features on CPU. Applying
the checkpoint's own classifier head (dropout is inactive in eval) to those CPU
features must reproduce the stored logits before any binary DenseNet probe,
temperature or confidence rule is fitted on them. No fitting, no image reads,
no held-out data and no writes happen here.
"""
import numpy as np
from .adaptation import aggregate

WEIGHT = 'backbone.classifier.1.weight'
BIAS = 'backbone.classifier.1.bias'
ATOL = 1e-4
RTOL = 1e-4  # Same tolerance as the M4 selected predictive-reload check.


def _array(value):
    detach = getattr(value, 'detach', None)
    if detach is not None:
        value = detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def verify_dense_reference(checkpoint, dense_ids, dense_features, *, atol=ATOL, rtol=RTOL):
    """Return a JSON-safe record; ``passed`` is False on a numerical mismatch.

    Structural problems (missing keys, misaligned membership/ids, nonfinite
    values) raise ValueError, because they mean the inputs are not the
    approved artifacts rather than a measurable device difference.
    """
    try:
        validation = checkpoint['membership']['validation']
        reference = checkpoint['validation_reference']
        state = checkpoint['model']
        weight, bias = _array(state[WEIGHT]), _array(state[BIAS])
    except (KeyError, TypeError) as error:
        raise ValueError('Selected checkpoint lacks membership/reference/classifier head') from error
    if not isinstance(reference, dict) or weight.shape != (7, 1024) or bias.shape != (7,):
        raise ValueError('Seven-class 1024-d classifier head and reference required')
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Nonfinite classifier head')
    expected = np.asarray(reference.get('logits'), dtype=np.float64)
    expected_ids = list(reference.get('physical_ids', []))
    if expected.shape != (len(expected_ids), 7) or not np.isfinite(expected).all():
        raise ValueError('Finite aligned stored validation reference required')

    ids = list(dense_ids)
    x = np.asarray(dense_features)
    if x.ndim != 2 or x.shape != (len(ids), 1024) or len(set(ids)) != len(ids):
        raise ValueError('Unique ids aligned with 1024-d Dense features required')
    position = {image_id: i for i, image_id in enumerate(ids)}
    missing = [r['image_id'] for r in validation if r['image_id'] not in position]
    if not validation or missing:
        raise ValueError('Dense cache lacks %d selected validation views' % len(missing))
    rows = np.array([position[r['image_id']] for r in validation])
    features = x[rows].astype(np.float64)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite Dense validation features')

    logits = features @ weight.T + bias
    units, actual = aggregate([r['image_id'] for r in validation],
                              [r['physical_id'] for r in validation], logits)
    if list(units) != expected_ids:
        raise ValueError('Physical validation units differ from stored reference')
    difference = np.abs(actual - expected)
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    agree = float(np.mean(actual.argmax(axis=1) == expected.argmax(axis=1)))
    return dict(passed=passed, physical_units=len(units), validation_views=len(validation),
                max_absolute_difference=float(difference.max()),
                mean_absolute_difference=float(difference.mean()),
                argmax_agreement=agree, atol=atol, rtol=rtol,
                comparison='checkpoint classifier head on CPU features vs stored training-device logits',
                fitting_performed=False, held_out_access=False)
