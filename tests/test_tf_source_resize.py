"""Golden source-derived fixture checks: no model/data/GPU operation."""
import hashlib
import json
from pathlib import Path
import unittest
try:
    import numpy as np
    from PIL import Image
except ImportError as error:
    raise unittest.SkipTest('M3 installed stack required: '+str(error))
from dermdetect.models.tf_source_resize import bilinear_uint8,evaluation_resize


class TensorFlowSourceResizeTests(unittest.TestCase):
    def test_official_scalar_reference_golden_bytes(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/m4_tf_bilinear_source.json').read_text())
        for row in fixture['cases']:
            h,w=row['source']
            pixels=(np.arange(h*w*3,dtype=np.uint32).reshape(h,w,3)%256).astype(np.uint8) if row['kind']=='gradient' else np.full((h,w,3),255,np.uint8)
            out=bilinear_uint8(pixels,tuple(row['target']))
            self.assertEqual(hashlib.sha256(out.tobytes()).hexdigest(),row['reference_uint8_sha256'],row)

    def test_rgb_pil_wrapper_and_repeated_bytes(self):
        image=Image.new('L',(7,5),255)
        a=evaluation_resize(image);b=evaluation_resize(image)
        self.assertEqual((a.mode,a.size),('RGB',(448,448)))
        self.assertEqual(a.tobytes(),b.tobytes())
        self.assertEqual(set(a.tobytes()),{255})

    def test_bad_inputs_rejected(self):
        for pixels in (np.zeros((2,3),np.uint8),np.zeros((2,3,3),np.float32),np.zeros((0,3,3),np.uint8)):
            with self.assertRaises(ValueError):bilinear_uint8(pixels)
        with self.assertRaises(ValueError):bilinear_uint8(np.zeros((2,3,3),np.uint8),(0,448))
