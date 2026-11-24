"""Invented CPU tensors/PIL images only. No downloads, real input or accelerator."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import SiglipConfig, SiglipModel, SiglipImageProcessor
except ImportError as error:
    raise unittest.SkipTest("Separate M3 environment required: " + str(error))
from dermdetect.models.densenet import DenseNetSystem, transform, weighted_loss, save_checkpoint, load_checkpoint
from dermdetect.models.embeddings import binding, save_cache, load_cache, freeze_image_features, evaluation_resize, load_local_encoder
from dermdetect.models.adaptation import (Probe, aggregate, physical_sample_weights, fit_temperature,
    confidence_cutoff, selection_key, binary_labels, save_probe, load_probe)
from dermdetect.data.numerical_policy import View, CLASS_ORDER, training_weights

torch.set_num_threads(1)


class DenseNetContracts(unittest.TestCase):
    def test_224_shared_transform_and_seeded_augmentation(self):
        image = Image.fromarray(np.arange(301*257*3, dtype=np.uint32).reshape(301,257,3).astype(np.uint8))
        x = transform(image)
        self.assertEqual(tuple(x.shape), (3,224,224))
        from torchvision.models import DenseNet121_Weights
        torch.testing.assert_close(x, DenseNet121_Weights.IMAGENET1K_V1.transforms()(image))
        a = transform(image, training=True, generator=torch.Generator().manual_seed(12))
        b = transform(image, training=True, generator=torch.Generator().manual_seed(12))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError): transform(image, training=True)

    def test_forward_backward_frozen_state_and_reload(self):
        torch.manual_seed(20260729)
        model = DenseNetSystem(.2).cpu().train()
        images = torch.randn(2,3,224,224)
        frozen = model.backbone.features.norm0.running_mean.clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss = weighted_loss(model(images), torch.tensor([1,6]), [1.,1.])
        self.assertTrue(torch.isfinite(loss)); loss.backward(); optimizer.step()
        self.assertIsNone(model.backbone.features.conv0.weight.grad)
        self.assertIsNotNone(model.backbone.classifier[1].weight.grad)
        torch.testing.assert_close(frozen, model.backbone.features.norm0.running_mean, rtol=0, atol=0)
        model.set_phase("final_block"); model.zero_grad(set_to_none=True)
        weighted_loss(model(images), torch.tensor([1,6]), [1.,1.]).backward()
        self.assertIsNotNone(model.backbone.features.denseblock4.denselayer1.conv1.weight.grad)
        self.assertIsNone(model.backbone.features.denseblock3.denselayer1.conv1.weight.grad)
        self.assertFalse(model.backbone.features.denseblock3.training)
        model.eval(); expected = model(images).detach()
        self.assertEqual(tuple(model.features(images).shape), (2,1024))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic.pt"
            save_checkpoint(path, model, {"dropout":.2,"data_kind":"synthetic"})
            restored = load_checkpoint(path, {"dropout":.2,"data_kind":"synthetic"})
            torch.testing.assert_close(expected, restored(images), rtol=0, atol=0)
            with self.assertRaises(ValueError): load_checkpoint(path, {"dropout":.3})
        with self.assertRaises(ValueError): model.set_phase("full_backbone")

    def test_approved_loss_objective_matches_physical_weights(self):
        views = [View(f"image{i}", f"unit{i}", f"group{i}", c, f"hash{i}") for i,c in enumerate(CLASS_ORDER)]
        views.append(View("second", "unit0", "group0", CLASS_ORDER[0], "second_hash"))
        official = training_weights(views)["image_weights"]
        custom = physical_sample_weights([v.label for v in views], [v.physical_id for v in views])
        np.testing.assert_allclose(custom, [float(official[v.image_id]) for v in views])
        logits = torch.randn(8,7, requires_grad=True)
        target = torch.tensor([CLASS_ORDER.index(v.label) for v in views])
        expected = (torch.nn.functional.cross_entropy(logits,target,reduction="none") * torch.tensor(custom)).mean()
        torch.testing.assert_close(weighted_loss(logits,target,custom).double(),expected,rtol=1e-6,atol=1e-6)
        with self.assertRaises(ValueError): weighted_loss(logits,target,[-1]*8)


class EmbeddingContracts(unittest.TestCase):
    def fixture(self):
        return binding(manifest_sha256="a"*64, image_sha256={"synthetic_a":"b"*64,"synthetic_b":"c"*64},
                       revision="d"*40, processor={"size":448,"synthetic":True},
                       software={"torch":torch.__version__},dimension=3)

    def test_cache_reload_prediction_equivalence_and_invalidation(self):
        key = self.fixture(); x = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"cache"
            checksum = save_cache(directory, ["synthetic_a","synthetic_b"],x,key)
            ids,z = load_cache(directory,key,checksum)
            np.testing.assert_array_equal(z @ np.arange(3),x @ np.arange(3))
            changed = dict(key,revision="e"*40)
            with self.assertRaises(ValueError): load_cache(directory,changed,checksum)
            with self.assertRaises(FileExistsError): save_cache(directory,ids,x,key)
            payload = directory/"features.npy"
            payload.write_bytes(payload.read_bytes()+b"tamper")
            with self.assertRaises(ValueError): load_cache(directory,key,checksum)

    def test_cache_rejects_missing_duplicate_nonfinite_or_zero_variance(self):
        key = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for ids,x in [(["synthetic_a"] ,np.array([[1,0,0]])),
                          (["synthetic_a"]*2,np.array([[1,0,0],[0,1,0]])),
                          (["synthetic_a","synthetic_b"],np.full((2,3),np.nan)),
                          (["synthetic_a","synthetic_b"],np.array([[1,0,0],[1,0,0]]))]:
                with self.assertRaises(ValueError): save_cache(Path(tmp)/"bad",ids,x,key)
        with self.assertRaises(ValueError): binding(manifest_sha256="a"*64,image_sha256={"x":"b"*64},
            revision="main",processor={"x":1},software={"x":1},dimension=3)

    def test_real_siglip_api_with_tiny_random_configuration(self):
        torch.manual_seed(7)
        config = SiglipConfig(text_config={"vocab_size":16,"hidden_size":16,"intermediate_size":32,
            "num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":4},
            vision_config={"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                "num_attention_heads":2,"image_size":32,"patch_size":16})
        model = SiglipModel(config).cpu().eval()
        pixels = torch.randn(2,3,32,32)
        a = freeze_image_features(model,pixels); b = freeze_image_features(model,pixels)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        with torch.inference_mode():
            expected = model(pixel_values=pixels,input_ids=torch.tensor([[1,2,3,4],[4,3,2,1]])).image_embeds
        torch.testing.assert_close(a,expected)
        self.assertFalse(a.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        torch.testing.assert_close(a.norm(dim=1),torch.ones(2))
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)/"random-siglip"
            model.save_pretrained(snapshot,safe_serialization=True)
            SiglipImageProcessor(size={"height":32,"width":32}).save_pretrained(snapshot)
            hashes = {p.relative_to(snapshot).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in snapshot.rglob("*") if p.is_file()}
            restored, processor = load_local_encoder(snapshot,expected_files=hashes,revision="a"*40)
            torch.testing.assert_close(a,freeze_image_features(restored,pixels),rtol=0,atol=0)
            self.assertEqual(processor.size,{"height":32,"width":32})
            with self.assertRaises(ValueError): load_local_encoder(snapshot,expected_files={},revision="main")

    def test_448_resize_and_processor_contract(self):
        image = Image.new("RGB",(7,13),(17,93,201))
        resized = evaluation_resize(image)
        self.assertEqual(resized.size,(448,448))
        # Float arithmetic plus uint8 truncation can differ by one intensity.
        self.assertLessEqual(np.abs(np.asarray(resized,dtype=int)-[17,93,201]).max(),1)
        processor = SiglipImageProcessor(size={"height":448,"width":448},do_resize=False)
        output = processor(images=resized,return_tensors="pt")["pixel_values"]
        self.assertEqual(tuple(output.shape),(1,3,448,448))
        self.assertTrue(torch.isfinite(output).all())


class AdaptationContracts(unittest.TestCase):
    def test_train_only_probe_reload_arrays_and_binary(self):
        rng = np.random.default_rng(1)
        y = list(range(7))*4; units = [f"unit{i}" for i in range(28)]
        x = np.eye(7)[y] + rng.normal(0,.01,(28,7))
        probe = Probe(1,range(7)).fit(x,y,units,partition="train")
        before = probe.scaler.mean_.copy()
        logits = probe.logits(x)
        self.assertEqual(logits.shape,(28,7)); self.assertEqual(np.argmax(logits,axis=1).tolist(),y)
        probe.logits(x+100)
        np.testing.assert_array_equal(before,probe.scaler.mean_)
        with self.assertRaises(ValueError): Probe(1,range(7)).fit(x,y,units,partition="validation")
        binary = Probe(.1,range(2)).fit(x,[i%2 for i in y],units,partition="train")
        self.assertEqual(binary.logits(x).shape,(28,2))
        candidates = [Probe(c,range(7)).fit(x,y,units,partition="train") for c in (.01,.1,1,10)]
        self.assertEqual(len(candidates),4)
        with tempfile.TemporaryDirectory() as tmp:
            config = {"data_kind":"synthetic", "train_manifest":"invented"}
            checksum = save_probe(Path(tmp)/"probe",probe,config)
            restored = load_probe(Path(tmp)/"probe",config,checksum)
            np.testing.assert_array_equal(restored.logits(x),logits)
            with self.assertRaises(ValueError): load_probe(Path(tmp)/"probe",{},checksum)

    def test_aggregation_calibration_and_ties(self):
        ids,z = aggregate(["a","b","c"],["u","u","v"],[[5.,0.],[1.,0.],[0.,3.]])
        self.assertEqual(ids,("u","v")); np.testing.assert_array_equal(z,[[3,0],[0,3]])
        result = fit_temperature([[5,0],[5,0],[0,5],[0,5]],[0,1,1,1],partition="validation")
        self.assertLessEqual(result["nll_after"],result["nll_before"])
        threshold = confidence_cutoff(z,1,.5,partition="validation",tie_policy="include_all_ties")
        self.assertEqual(threshold["realized_validation_coverage"],1.)
        with self.assertRaises(ValueError): fit_temperature(z,[0,1],partition="internal_test")
        with self.assertRaises(ValueError): confidence_cutoff(z,1,.9,partition="internal_test",tie_policy="include_all_ties")
        with self.assertRaises(ValueError): confidence_cutoff(z,0,.9,partition="validation",tie_policy="include_all_ties")
        with self.assertRaises(ValueError): aggregate(["a","a"],["u","v"],z)

    def test_selection_and_explicit_binary_policy(self):
        logits = [[2.,0.],[0.,2.]]
        self.assertLess(selection_key(logits,[0,1],epoch=1),selection_key(logits,[0,1],epoch=2))
        self.assertLess(selection_key(logits,[0,1],c=.01),selection_key(logits,[0,1],c=.1))
        self.assertEqual(binary_labels(["nv","akiec","mel"],{"nv":0,"mel":1,"akiec":None}),([0,2],[0,1]))
        with self.assertRaises(ValueError): binary_labels(["unknown"],{"nv":0,"mel":1,"akiec":None})
        with self.assertRaises(ValueError): physical_sample_weights([0,1],["same","same"])
        with self.assertRaises(ValueError): fit_temperature([[float("nan"),1]],[0],partition="validation")
