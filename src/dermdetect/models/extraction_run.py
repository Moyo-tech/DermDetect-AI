"""Sequential offline extraction with verified atomic shards and immutable final cache.

The caller owns authorization, resource supervision and wall/compute accounting.
Incomplete rows are recomputed after interruption; committed rows are never encoded again.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import time
import numpy as np
from .embeddings import save_cache, load_cache, validate


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode() + b'\n'


def _fsync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic(path, data):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _rows(rows, dimension):
    array = np.asarray(rows)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError('Exact float32 output dimension required')
    if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError('Finite normalized features required')
    return array


def run_extraction(*, items, key, work_directory, cache_directory, encode,
                   shard_size=64, should_stop=lambda: False):
    """Return complete/incomplete status; errors retain committed recovery evidence.

    items is an exact ordered sequence of (id, local_path). encode is batch-one.
    should_stop is checked before every callback; it must include external budget gates.
    A shard rename commits rows; ledger checksums reject any subsequent modification.
    """
    if type(shard_size) is not int or shard_size < 1:
        raise ValueError('Positive shard size required')
    items = tuple((identifier, Path(path)) for identifier, path in items)
    ids = tuple(identifier for identifier, _ in items)
    if len(ids) < 2 or len(ids) != len(set(ids)) or set(ids) != set(key['image_sha256']):
        raise ValueError('Exact unique bound image coverage required')
    for identifier, path in items:
        if path.is_symlink() or file_sha(path) != key['image_sha256'][identifier]:
            raise ValueError('Image bytes changed')
    contract = {'binding': key, 'ids': ids, 'shard_size': shard_size}
    contract = json.loads(_json(contract))
    work, cache = Path(work_directory), Path(cache_directory)
    if work.is_symlink() or cache.is_symlink():
        raise ValueError('Symlink recovery/cache directories forbidden')
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / 'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Extraction binding/order changed')
    else:
        if any(work.iterdir()):
            raise ValueError('Unbound recovery contents')
        _atomic(contract_path, _json(contract))
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'complete': False, 'shards': [], 'encode_seconds': 0.0, 'encoded_attempts': 0}
    allowed = {'contract.json', 'state.json', 'state.json.tmp', 'contract.json.tmp', 'final-cache.tmp'}
    allowed.update(s['file'] for s in state['shards'])
    # An interrupted shard written before its ledger commit is explicitly uncommitted.
    for path in work.iterdir():
        if re.fullmatch(r'final-cache-interrupted-[0-9]{6}\.tmp', path.name) and path.is_dir() and not path.is_symlink():
            continue
        if path.name not in allowed and not (path.name.startswith('shard-') and path.name.endswith('.tmp')):
            if path.name == f'shard-{len(state["shards"]):06d}.npy' and not path.is_symlink():
                os.replace(path, work / (path.name + '.uncommitted.tmp'))
            else:
                raise ValueError('Uncommitted or unexpected recovery file')
    arrays, covered = [], 0
    for index, shard in enumerate(state['shards']):
        path = work / shard['file']
        if shard['file'] != f'shard-{index:06d}.npy' or path.is_symlink() or file_sha(path) != shard['sha256']:
            raise ValueError('Recovery shard changed')
        array = _rows(np.load(path, allow_pickle=False), key['dimension'])
        if shard['ids'] != list(ids[covered:covered + len(array)]) or len(array) > shard_size or len(array) == 0:
            raise ValueError('Recovery shard membership changed')
        arrays.append(array)
        covered += len(array)
    if covered > len(ids):
        raise ValueError('Duplicate recovery coverage')
    if cache.exists():
        if not state.get('metadata_sha256'):
            raise ValueError('Existing cache lacks committed recovery binding')
        cached_ids, _ = load_cache(cache, key, state['metadata_sha256'])
        if cached_ids != ids:
            raise ValueError('Cache order changed')
        state.update(complete=True, rows=len(ids))
        _atomic(state_path, _json(state))
        return state
    state['complete'] = False
    _atomic(state_path, _json(state))
    pending = []

    def commit():
        nonlocal covered, pending
        if not pending:
            return
        array = _rows(np.stack(pending), key['dimension'])
        name = f'shard-{len(arrays):06d}.npy'
        path = work / name
        temporary = work / (name + '.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        # Only the ledger commits a shard; an orphan is preserved and recomputed on resume.
        os.replace(temporary, path)
        state['shards'].append({'file': name, 'ids': list(ids[covered:covered + len(array)]), 'sha256': file_sha(path)})
        arrays.append(array)
        covered += len(array)
        pending = []
        _atomic(state_path, _json(state))

    try:
        for identifier, path in items[covered:]:
            if should_stop():
                commit()
                return state
            if file_sha(path) != key['image_sha256'][identifier]:
                raise ValueError('Image changed before encoding')
            began = time.monotonic()
            try:
                row = encode(identifier, path)
            finally:
                state['encode_seconds'] += time.monotonic() - began
                state['encoded_attempts'] += 1
            if file_sha(path) != key['image_sha256'][identifier]:
                raise ValueError('Image changed during encoding')
            row = _rows(np.asarray(row)[None, :], key['dimension'])[0]
            pending.append(row)
            if len(pending) == shard_size:
                commit()
        commit()
        matrix = np.concatenate(arrays)
        validate(ids, matrix, key)
        temporary_cache = work / 'final-cache.tmp'
        if temporary_cache.exists():
            if temporary_cache.is_symlink() or set(p.name for p in temporary_cache.iterdir()) - {'features.npy', 'cache.json', 'cache.json.tmp'}:
                raise ValueError('Unexpected temporary cache contents')
            if state.get('metadata_sha256'):
                metadata_sha = state['metadata_sha256']
                # Once ledger-bound, changed/truncated bytes are corruption, never repaired.
                load_cache(temporary_cache, key, metadata_sha)
            else:
                # Final publication not committed: preserve partial bytes and rebuild solely
                # from verified committed shards, without further encoder calls.
                index = 0
                while (work / f'final-cache-interrupted-{index:06d}.tmp').exists():
                    index += 1
                os.rename(temporary_cache, work / f'final-cache-interrupted-{index:06d}.tmp')
                _fsync_directory(work)
                metadata_sha = save_cache(temporary_cache, ids, matrix, key)
        else:
            metadata_sha = save_cache(temporary_cache, ids, matrix, key)
        for name in ('features.npy', 'cache.json'):
            with (temporary_cache / name).open('rb') as handle:
                os.fsync(handle.fileno())
        _fsync_directory(temporary_cache)
        actual_ids, actual_matrix = load_cache(temporary_cache, key, metadata_sha)
        if actual_ids != ids or not np.array_equal(actual_matrix, matrix):
            raise ValueError('Final cache reload mismatch')
        # Persist expected digest before rename, enabling recovery across final publication.
        state['metadata_sha256'] = metadata_sha
        _atomic(state_path, _json(state))
        if cache.exists():
            raise ValueError('Cache already exists')
        os.rename(temporary_cache, cache)
        _fsync_directory(work)
        _fsync_directory(cache.parent)
        state['complete'] = True
        state['rows'] = len(ids)
        _atomic(state_path, _json(state))
        return state
    finally:
        _atomic(state_path, _json(state))
