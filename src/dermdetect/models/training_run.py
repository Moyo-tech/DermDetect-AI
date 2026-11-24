"""Single-process M4 epoch driver with atomic, trajectory-preserving recovery.

Callers supply verified development rows, explicit numerical policy and software/data
binding. This module grants no data access or execution authorization.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .densenet import weighted_loss
from .adaptation import aggregate, selection_key


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_configuration(c):
    required = ('seed', 'batch_size', 'head_epochs', 'final_block_epochs', 'head_lr',
                'final_block_lr', 'phase2_head_lr', 'weight_decay', 'patience',
                'patience_scope', 'optimizer_transition', 'recovery_every_steps')
    if any(k not in c for k in required):
        raise ValueError('Explicit complete numerical configuration required')
    if c['seed'] != 20260729 or c['optimizer_transition'] != 'retain_head_add_final_block':
        raise ValueError('Unsupported seed or optimizer transition')
    if c['patience_scope'] not in ('all_evaluated_epochs', 'final_block_only'):
        raise ValueError('Explicit supported patience scope required')
    for key in ('batch_size', 'head_epochs', 'final_block_epochs', 'patience', 'recovery_every_steps'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Positive integer schedule required')
    for key in ('head_lr', 'phase2_head_lr', 'final_block_lr', 'weight_decay'):
        if not isinstance(c[key], (int, float)) or not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Finite positive optimizer settings required')


def validate_rows(train, validation):
    ids = set(); owners = {}; units_by_split = {}
    for split, rows in [('train', train), ('validation', validation)]:
        if not rows:
            raise ValueError('Nonempty train and validation required')
        units_by_split[split] = set()
        for r in rows:
            if r['split'] != split or r['image_id'] in ids:
                raise ValueError('Development-only unique image membership required')
            ids.add(r['image_id']); units_by_split[split].add(r['physical_id'])
            if type(r['target']) is not int or not 0 <= r['target'] < 7:
                raise ValueError('Canonical seven-class targets required')
            if r['physical_id'] in owners and owners[r['physical_id']] != r['target']:
                raise ValueError('Mixed-label physical identity')
            owners[r['physical_id']] = r['target']
            if split == 'train' and (not np.isfinite(r['weight']) or r['weight'] <= 0):
                raise ValueError('Explicit finite positive population weights required')
    if units_by_split['train'] & units_by_split['validation']:
        raise ValueError('Physical identity crosses development splits')


class TrainingRun:
    def __init__(self, model, train_rows, validation_rows, load_view, configuration,
                 binding, output_directory, *, device='cpu', allow_resume=False):
        validate_configuration(configuration)
        validate_rows(train_rows, validation_rows)
        if not binding or any(not isinstance(k, str) for k in binding):
            raise ValueError('Explicit configuration/data/software binding required')
        self.c = copy.deepcopy(configuration); self.binding = copy.deepcopy(binding)
        self.train_rows = tuple(copy.deepcopy(train_rows)); self.validation_rows = tuple(copy.deepcopy(validation_rows))
        self.load_view = load_view; self.device = torch.device(device)
        self.output = Path(output_directory); self.output.mkdir(parents=True, exist_ok=True)
        self._resume_required = any(self.output.iterdir())
        self._resumed = False
        self._validation_reference = None
        if self._resume_required and not allow_resume:
            raise FileExistsError('Existing training output requires explicit validated resume')
        self.model = model.to(self.device); self.model.set_phase('head')
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=self.c['head_lr'], weight_decay=self.c['weight_decay'])
        self.sampler = torch.Generator().manual_seed(self.c['seed'])
        self.augmentation = torch.Generator().manual_seed(self.c['seed'])
        self.state = dict(epoch=0, cursor=0, permutation=None, phase='head', best_key=None,
                          bad_epochs=0, best_checkpoint=None, best_checkpoint_sha256=None, history=[], steps=0, worker_seconds=0., complete=False,
                          epoch_loss_sum=0., epoch_views=0, epoch_started_seconds=None)

    def _rng(self):
        n = np.random.get_state()
        return dict(python=random.getstate(), numpy=(n[0], n[1].tolist(), n[2], n[3], n[4]),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
                    sampler=self.sampler.get_state(), augmentation=self.augmentation.get_state())

    def _restore_rng(self, r):
        random.setstate(r['python']); n = r['numpy']
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(r['torch']); self.sampler.set_state(r['sampler']); self.augmentation.set_state(r['augmentation'])
        if self.device.type == 'cuda':
            if len(r['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA RNG device count changed')
            torch.cuda.set_rng_state_all(r['cuda'])

    def _transition(self):
        if self.state['phase'] == 'head' and self.state['epoch'] >= self.c['head_epochs']:
            existing = {id(p) for group in self.optimizer.param_groups for p in group['params']}
            self.model.set_phase('final_block')
            for group in self.optimizer.param_groups:
                group['lr'] = self.c['phase2_head_lr']
            new = [p for p in self.model.parameters() if p.requires_grad and id(p) not in existing]
            if not new:
                raise ValueError('Final-block transition adds no parameters')
            self.optimizer.add_param_group(dict(params=new, lr=self.c['final_block_lr']))
            self.state['phase'] = 'final_block'

    def save(self, *, path=None):
        atomic_save(dict(version=1, configuration=self.c, binding=self.binding,
                         membership=dict(train=list(self.train_rows), validation=list(self.validation_rows)),
                         model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                         state=self.state, rng=self._rng(), validation_reference=self._validation_reference), path or self.output / 'recovery.pt')
        if path is None:
            progress = {k:self.state[k] for k in ('epoch','cursor','phase','steps','worker_seconds','complete','best_checkpoint','best_key','history')}
            temporary = self.output/'progress.json.partial'
            with temporary.open('w') as handle:
                json.dump(progress,handle,sort_keys=True,allow_nan=False); handle.write('\n')
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary,self.output/'progress.json')

    def resume(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['version'] != 1 or saved['configuration'] != self.c or saved['binding'] != self.binding or
            saved['membership'] != dict(train=list(self.train_rows), validation=list(self.validation_rows))):
            raise ValueError('Recovery binding/configuration/membership mismatch')
        self._validation_reference = saved.get('validation_reference')
        self.state = saved['state']
        if self.state['phase'] == 'final_block':
            # Reconstruct identical group order before restoring optimizer moments.
            self.state['phase'] = 'head'; self._transition()
        self.model.load_state_dict(saved['model'], strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        # AdamW.load_state_dict handles device casting, preserving CPU step scalars.
        if self.state['best_key'] is not None:
            best_path = Path(path).parent / self.state['best_checkpoint']
            if hashlib.sha256(best_path.read_bytes()).hexdigest() != self.state['best_checkpoint_sha256']:
                raise ValueError('Selected best checkpoint payload changed')
            best = torch.load(best_path, map_location='cpu', weights_only=True)
            if (best['configuration'] != self.c or best['binding'] != self.binding or
                    best['membership'] != saved['membership'] or
                    best['state']['best_key'] != self.state['best_key']):
                raise ValueError('Best checkpoint does not match recovery selection')
            target_best = self.output / self.state['best_checkpoint']
            if best_path.resolve() != target_best.resolve():
                temporary = target_best.with_name(target_best.name+'.partial')
                with temporary.open('wb') as handle:
                    handle.write(best_path.read_bytes()); handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, target_best)
        self._restore_rng(saved['rng'])
        self._resumed = True

    def _physical_predictions(self, model, *, stop_requested=lambda: False):
        model.eval(); values = []
        with torch.no_grad():
            for start in range(0, len(self.validation_rows), self.c['batch_size']):
                if stop_requested():
                    raise RuntimeError('Selected reload verification stopped before completion')
                rows = self.validation_rows[start:start+self.c['batch_size']]
                x = torch.stack([self.load_view(r, False, None) for r in rows]).to(self.device)
                logits = model(x)
                if logits.shape != (len(rows), 7) or not torch.isfinite(logits).all():
                    raise ValueError('Finite seven-logit validation required')
                values.extend(logits.cpu().double().tolist())
        return aggregate([r['image_id'] for r in self.validation_rows],
                         [r['physical_id'] for r in self.validation_rows], values)

    def validate(self):
        units, logits = self._physical_predictions(self.model)
        self._validation_reference = dict(physical_ids=list(units), logits=logits.tolist())
        labels = {r['physical_id']: r['target'] for r in self.validation_rows}
        return selection_key(logits, [labels[u] for u in units], epoch=self.state['epoch'])

    def verify_selected_reload(self, *, stop_requested=lambda: False):
        """Read-only trained-selection check; preserve final model, optimizer and all RNG.

        Full launch must treat completion as provisional until this succeeds. Runtime is
        charged by the caller/supervisor; recovery trajectory is deliberately unchanged.
        """
        if not self.state['complete'] or not self.state['best_checkpoint']:
            raise ValueError('Completed schedule and selected checkpoint required')
        path = self.output/self.state['best_checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.state['best_checkpoint_sha256']:
            raise ValueError('Selected best checkpoint payload changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['configuration'] != self.c or saved['binding'] != self.binding or
                saved['membership'] != dict(train=list(self.train_rows),validation=list(self.validation_rows)) or
                saved['state']['best_key'] != self.state['best_key']):
            raise ValueError('Selected predictive checkpoint binding mismatch')
        reference = saved.get('validation_reference')
        if not isinstance(reference, dict):
            raise ValueError('Selected validation prediction reference missing')
        expected = np.asarray(reference['logits'],dtype=np.float64)
        if expected.shape != (len(reference['physical_ids']),7) or not np.isfinite(expected).all():
            raise ValueError('Finite aligned selected validation reference required')
        rng = self._rng(); started = time.monotonic()
        try:
            selected = copy.deepcopy(self.model)
            selected.load_state_dict(saved['model'],strict=True)
            selected.set_phase(saved['state']['phase'])
            units, actual = self._physical_predictions(selected,stop_requested=stop_requested)
            if list(units) != reference['physical_ids'] or not np.allclose(actual,expected,atol=1e-4,rtol=1e-4):
                raise ValueError('Selected trained checkpoint predictive reload mismatch')
            return dict(passed=True,physical_units=len(units),max_absolute_difference=float(np.abs(actual-expected).max()),
                        atol=1e-4,rtol=1e-4,worker_seconds=time.monotonic()-started,
                        checkpoint_sha256=self.state['best_checkpoint_sha256'])
        finally:
            self._restore_rng(rng)

    def run(self, *, max_worker_seconds, stop_requested=lambda: False, after_step=None):
        """Stop at safe batch/validation boundaries; all invocations charge cumulative time.

        Outer supervisor must enforce hard limits: a running batch, validation or fsync
        cannot be interrupted safely by this cooperative clock alone.
        """
        if not np.isfinite(max_worker_seconds) or max_worker_seconds <= self.state['worker_seconds']:
            raise ValueError('Positive remaining cumulative execution allowance required')
        if self._resume_required and not self._resumed:
            raise PermissionError('Existing output cannot run before recovery validation')
        self.save()  # Durable pre-work trajectory; failures never replace it with partial batch state.
        started = time.monotonic(); carried = self.state['worker_seconds']
        def charge():
            self.state['worker_seconds'] = carried + time.monotonic() - started
        try:
            while not self.state['complete']:
                charge()
                if stop_requested() or self.state['worker_seconds'] >= max_worker_seconds:
                    break
                self._transition()
                if self.state['permutation'] is None:
                    self.state['epoch_started_seconds'] = self.state['worker_seconds']
                    self.state['permutation'] = torch.randperm(len(self.train_rows), generator=self.sampler).tolist()
                self.model.train()
                p = self.state['permutation']; cursor = self.state['cursor']
                if cursor < len(p):
                    rows = [self.train_rows[i] for i in p[cursor:cursor+self.c['batch_size']]]
                    x = torch.stack([self.load_view(r, True, self.augmentation) for r in rows]).to(self.device)
                    y = torch.tensor([r['target'] for r in rows], device=self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(x)
                    if logits.shape != (len(rows), 7) or not torch.isfinite(logits).all():
                        raise ValueError('Finite seven-logit training required')
                    loss = weighted_loss(logits, y, [r['weight'] for r in rows])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training objective')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                        raise ValueError('Nonfinite gradients')
                    self.optimizer.step()
                    if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                        raise ValueError('Nonfinite parameter update')
                    self.state['cursor'] += len(rows); self.state['steps'] += 1
                    self.state['epoch_loss_sum'] += float(loss.detach()) * len(rows)
                    self.state['epoch_views'] += len(rows)
                    charge()
                    if self.state['steps'] % self.c['recovery_every_steps'] == 0:
                        self.save()
                    if after_step is not None:
                        after_step(self)
                    continue
                key = self.validate(); previous = self.state['best_key']
                improved = previous is None or tuple(key) < tuple(previous)
                if improved:
                    self.state['best_key'] = list(key); self.state['bad_epochs'] = 0
                elif self.c['patience_scope'] == 'all_evaluated_epochs' or self.state['phase'] == 'final_block':
                    self.state['bad_epochs'] += 1
                charge()
                self.state['history'].append(dict(epoch=self.state['epoch'], phase=self.state['phase'],
                    selection_key=list(key), epoch_worker_seconds=self.state['worker_seconds']-self.state['epoch_started_seconds'], training_objective=self.state['epoch_loss_sum']/self.state['epoch_views']))
                self.state['epoch'] += 1; self.state['cursor'] = 0; self.state['permutation'] = None
                self.state['epoch_loss_sum'] = 0.; self.state['epoch_views'] = 0; self.state['epoch_started_seconds'] = None
                self.state['complete'] = (self.state['epoch'] >= self.c['head_epochs'] + self.c['final_block_epochs'] or
                                         self.state['bad_epochs'] >= self.c['patience'])
                charge()
                previous_best = self.state['best_checkpoint']
                if improved:
                    self.state['best_checkpoint'] = f"best_epoch_{self.state['epoch']-1:03d}.pt"
                    self.save(path=self.output/self.state['best_checkpoint'])
                    self.state['best_checkpoint_sha256'] = hashlib.sha256((self.output/self.state['best_checkpoint']).read_bytes()).hexdigest()
                self.save()
                # Prune only after the new pointer and hash are durably committed.
                if improved and previous_best and previous_best != self.state['best_checkpoint']:
                    (self.output/previous_best).unlink(missing_ok=True)
        except BaseException:
            charge()
            # Supervisor separately charges failed/uncommitted compute. Preserve last valid recovery.
            raise
        else:
            charge(); self.save()
        return copy.deepcopy(self.state)
