"""Invented CPU tensors only: real AdamW/dropout/resume trajectory tests."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from dermdetect.models.training_run import TrainingRun, validate_configuration

torch.set_num_threads(1)


class TinySystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.block = nn.Linear(4, 4)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(4, 7))
        self.set_phase('head')

    def set_phase(self, phase):
        self.phase = phase
        for p in self.frozen.parameters(): p.requires_grad_(False)
        for p in self.block.parameters(): p.requires_grad_(phase == 'final_block')
        for p in self.head.parameters(): p.requires_grad_(True)

    def forward(self, x):
        return self.head(torch.relu(self.block(self.frozen(x))))


def seed():
    random.seed(20260729); np.random.seed(20260729); torch.manual_seed(20260729)


def config():
    return dict(seed=20260729,batch_size=3,head_epochs=1,final_block_epochs=2,
                head_lr=.001,final_block_lr=.0001,phase2_head_lr=.001,weight_decay=.0001,
                patience=5,patience_scope='all_evaluated_epochs',
                optimizer_transition='retain_head_add_final_block',recovery_every_steps=1)


def rows(split):
    return [dict(image_id=f'{split}{i}',physical_id=f'{split}u{i//2}',target=(i//2)%7,
                 weight=.5+(i%3),split=split) for i in range(10)]


def load(row, training, generator):
    i = int(row['image_id'].removeprefix(row['split']))
    x = torch.tensor([i/10.,1.,-.5,i%2],dtype=torch.float32)
    if training:
        x = x + torch.rand(4, generator=generator)*.01
    return x


class TrainingTests(unittest.TestCase):
    def run_object(self, directory, **kwargs):
        return TrainingRun(TinySystem(), rows('train'), rows('validation'), load,
                           config(), dict(synthetic=True,software='test'), directory, **kwargs)

    def test_mid_epoch_resume_exact_with_optimizer_dropout_sampler_augmentation(self):
        with tempfile.TemporaryDirectory() as d:
            seed(); full = self.run_object(Path(d)/'full'); frozen = copy.deepcopy(full.model.frozen.state_dict())
            expected = full.run(max_worker_seconds=60)
            seed(); part = self.run_object(Path(d)/'part')
            stop = [False]
            part.run(max_worker_seconds=60,stop_requested=lambda: stop[0],
                     after_step=lambda r: stop.__setitem__(0,r.state['steps']==6))
            self.assertEqual(part.state['phase'],'final_block'); self.assertEqual(part.state['cursor'],6)
            # Construction consumes global RNG, resume must restore it afterwards.
            resumed = self.run_object(Path(d)/'resumed'); resumed.resume(Path(d)/'part'/'recovery.pt')
            actual = resumed.run(max_worker_seconds=60)
            for key in ('steps','best_key','epoch','complete','bad_epochs'):
                self.assertEqual(expected[key],actual[key],key)
            without_clock=lambda history:[{k:v for k,v in item.items() if k!='epoch_worker_seconds'} for item in history]
            self.assertEqual(without_clock(expected['history']),without_clock(actual['history']))
            for key,value in full.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
            for key,value in frozen.items():
                torch.testing.assert_close(value,full.model.frozen.state_dict()[key],rtol=0,atol=0)
            a,b = full.optimizer.state_dict(),resumed.optimizer.state_dict()
            self.assertEqual(a['param_groups'],b['param_groups'])
            for k in a['state']:
                for field in a['state'][k]:
                    torch.testing.assert_close(a['state'][k][field],b['state'][k][field],rtol=0,atol=0)
            best = torch.load(Path(d)/'full'/full.state['best_checkpoint'],weights_only=True)
            model = TinySystem(); model.load_state_dict(best['model']); model.eval()
            predictions = model(torch.stack([load(r,False,None) for r in rows('validation')]))
            reloaded = TinySystem(); reloaded.load_state_dict(best['model']); reloaded.eval()
            torch.testing.assert_close(predictions,reloaded(torch.stack([load(r,False,None) for r in rows('validation')])),rtol=0,atol=0)

    def test_uncommitted_batch_failure_replays_from_last_durable_rng(self):
        with tempfile.TemporaryDirectory() as d:
            seed(); full=self.run_object(Path(d)/'full'); full.c['recovery_every_steps']=25
            full.run(max_worker_seconds=60)
            seed(); failed=self.run_object(Path(d)/'failed'); failed.c['recovery_every_steps']=25
            def interrupt(run):
                if run.state['steps']==2: raise RuntimeError('invented interruption')
            with self.assertRaises(RuntimeError): failed.run(max_worker_seconds=60,after_step=interrupt)
            saved=torch.load(Path(d)/'failed'/'recovery.pt',weights_only=True)
            self.assertEqual(saved['state']['steps'],0)
            resumed=self.run_object(Path(d)/'resumed'); resumed.c['recovery_every_steps']=25
            resumed.resume(Path(d)/'failed'/'recovery.pt'); resumed.run(max_worker_seconds=60)
            for key,value in full.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)

    def test_crash_before_new_best_pointer_commit_preserves_old_selected_file(self):
        with tempfile.TemporaryDirectory() as d:
            seed(); run=self.run_object(d)
            run.validate=lambda: (-.1*(run.state['epoch']+1),2.-run.state['epoch']*.1,run.state['epoch'])
            stopped=[False]
            run.run(max_worker_seconds=60,stop_requested=lambda: stopped[0],
                    after_step=lambda r: stopped.__setitem__(0,r.state['steps']==5))
            old=run.state['best_checkpoint']; old_hash=run.state['best_checkpoint_sha256']
            save=run.save
            def fail_commit(*,path=None):
                if path is None and run.state['epoch']==2: raise RuntimeError('invented commit interruption')
                return save(path=path)
            run.save=fail_commit
            with self.assertRaises(RuntimeError): run.run(max_worker_seconds=60)
            durable=torch.load(Path(d)/'recovery.pt',weights_only=True)
            self.assertEqual(durable['state']['best_checkpoint'],old)
            self.assertEqual(durable['state']['best_checkpoint_sha256'],old_hash)
            self.assertTrue((Path(d)/old).is_file())
            resumed=self.run_object(d,allow_resume=True); resumed.resume(Path(d)/'recovery.pt')
            self.assertEqual(resumed.state['best_checkpoint'],old)
            resumed.validate=lambda: (-.1*(resumed.state['epoch']+1),2.-resumed.state['epoch']*.1,resumed.state['epoch'])
            resumed.run(max_worker_seconds=60)
            self.assertFalse((Path(d)/old).exists())
            self.assertEqual(len(list(Path(d).glob('best_epoch_*.pt'))),1)

    def test_existing_output_needs_explicit_validated_resume(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.run_object(d); run.save()
            original=(Path(d)/'recovery.pt').read_bytes()
            with self.assertRaises(FileExistsError): self.run_object(d)
            candidate=self.run_object(d,allow_resume=True)
            with self.assertRaises(PermissionError): candidate.run(max_worker_seconds=60)
            self.assertEqual((Path(d)/'recovery.pt').read_bytes(),original)

    def test_selected_predictive_reload_and_corruption_preserve_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            seed(); run=self.run_object(d); run.run(max_worker_seconds=60)
            before=copy.deepcopy(run.state); rng=run._rng()
            original={k:v.clone() for k,v in run.model.state_dict().items()}
            proof=run.verify_selected_reload()
            self.assertTrue(proof['passed']); self.assertEqual(proof['max_absolute_difference'],0.)
            self.assertEqual(run.state,before)
            torch.testing.assert_close(run._rng()['torch'],rng['torch'],rtol=0,atol=0)
            torch.testing.assert_close(run._rng()['augmentation'],rng['augmentation'],rtol=0,atol=0)
            for k,v in original.items(): torch.testing.assert_close(run.model.state_dict()[k],v,rtol=0,atol=0)
            path=Path(d)/run.state['best_checkpoint']; saved=torch.load(path,weights_only=True)
            saved['validation_reference']['logits'][0][0]+=1.
            torch.save(saved,path)
            # Check byte tampering first; independently test predictive discrepancy after a rebound hash.
            with self.assertRaisesRegex(ValueError,'payload changed'): run.verify_selected_reload()
            import hashlib
            run.state['best_checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'predictive reload mismatch'): run.verify_selected_reload()

    def test_binding_change_rejected_and_missing_explicit_settings(self):
        with tempfile.TemporaryDirectory() as d:
            seed(); run = self.run_object(Path(d)/'one'); run.save()
            changed = self.run_object(Path(d)/'two'); changed.binding['software']='changed'
            with self.assertRaisesRegex(ValueError,'mismatch'): changed.resume(Path(d)/'one'/'recovery.pt')
        for key in ('phase2_head_lr','optimizer_transition','patience_scope','recovery_every_steps'):
            c=config(); del c[key]
            with self.assertRaises(ValueError): validate_configuration(c)

    def test_pre_work_stop_and_nonfinite_input_leave_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.run_object(Path(d)/'stop')
            result=run.run(max_worker_seconds=60,stop_requested=lambda: True)
            self.assertEqual(result['steps'],0); self.assertFalse(result['complete'])
            self.assertTrue((Path(d)/'stop'/'recovery.pt').is_file())
            run.load_view=lambda *args: torch.full((4,),float('nan'))
            with self.assertRaises(ValueError): run.run(max_worker_seconds=60)
            self.assertEqual(run.state['steps'],0)

    def test_physical_aggregation_selection_is_equal_unit(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.run_object(d)
            key=run.validate()
            from dermdetect.models.adaptation import aggregate,selection_key
            run.model.eval()
            with torch.no_grad(): z=run.model(torch.stack([load(r,False,None) for r in rows('validation')])).numpy()
            units,physical=aggregate([r['image_id'] for r in rows('validation')],[r['physical_id'] for r in rows('validation')],z)
            owners={r['physical_id']:r['target'] for r in rows('validation')}
            expected=selection_key(physical,[owners[u] for u in units],epoch=0)
            self.assertEqual(key[0],expected[0]); self.assertAlmostEqual(key[1],expected[1],places=7); self.assertEqual(key[2],0)


if __name__ == '__main__': unittest.main()
