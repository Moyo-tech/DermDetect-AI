"""Approved exact-file acquisition only; no imports of torch or model execution."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/'configs/model-development/m4_acquisition_smoke_proposal.json').read_text())
EVIDENCE = ROOT/'outputs/m4-acquisition'
DESTINATION = ROOT/PLAN['acquisition']['local_destination']


def write(name, value):
    (EVIDENCE/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def hashes(path):
    h=hashlib.sha256();git=hashlib.sha1();git.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block);git.update(block)
    return h.hexdigest(),git.hexdigest()


def disk():
    size=sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file()) if DESTINATION.exists() else 0
    free=shutil.disk_usage(ROOT).free
    if size>8*2**30 or free<8*2**30: raise RuntimeError('Approved disk limit reached')
    return {'destination_bytes':size,'free_bytes':free}


def main():
    EVIDENCE.mkdir();DESTINATION.mkdir(parents=True)
    write('approval.json',{'recorded_utc':now(),'source_thread':'01a0fd4f-bf8a-7310-bc3d-2d76d82610df',
        'user_reply':'Yes','request':'3.30 GiB exact pinned acquisitions and local smoke capped at10minutes GPU time',
        'package':'docs/m4_acquisition_smoke_plan.md','package_sha256':hashes(ROOT/'docs/m4_acquisition_smoke_plan.md')[0],
        'proposal_sha256':hashes(ROOT/'configs/model-development/m4_acquisition_smoke_proposal.json')[0],
        'full_training_or_extraction_approved':False,'held_out_approved':False})
    revision=PLAN['medsiglip']['revision']
    head=get_hf_file_metadata(hf_hub_url(PLAN['medsiglip']['id'],'config.json',revision=revision))
    if head.commit_hash!=revision: raise RuntimeError('Pinned client access revision mismatch')
    write('access.json',{'status':'authenticated_normal_client_HEAD_passed','revision':head.commit_hash,'recorded_utc':now()})
    start=time.monotonic();started=now();records=[]
    model_dir=DESTINATION/'medsiglip';model_dir.mkdir()
    # One sequential download at a time. Disable Xet on invocation so plain HTTP
    # transfer avoids additional parallel connections and duplicate chunk caches.
    files=sorted(PLAN['medsiglip']['required_image_only_files'],key=lambda row:row['bytes'])
    for row in files:
        disk();begin=time.monotonic()
        path=Path(hf_hub_download(PLAN['medsiglip']['id'],row['path'],revision=revision,local_dir=model_dir))
        checksum,git=hashes(path)
        if path.stat().st_size!=row['bytes']:raise RuntimeError('Pinned file size mismatch')
        if row['lfs']:
            if checksum!=row['lfs']['sha256']: raise RuntimeError('Published weight SHA256 mismatch')
        elif git!=row['blob_id']: raise RuntimeError('Pinned Git blob mismatch')
        records.append({'model':'medsiglip','file':row['path'],'bytes':path.stat().st_size,'sha256':checksum,
                        'git_blob':git,'elapsed_seconds':time.monotonic()-begin,'integrity_verified':True})
        write('progress.json',{'recorded_utc':now(),'files':records,'disk':disk()})
        print(json.dumps(records[-1]),flush=True)
    dense=PLAN['densenet'];dense_dir=DESTINATION/'densenet';dense_dir.mkdir()
    path=dense_dir/'densenet121-a639ec97.pth';begin=time.monotonic();count=0;last_check=0
    with requests.get(dense['url'],stream=True,timeout=(20,60)) as response:
        response.raise_for_status()
        with path.open('xb') as handle:
            for block in response.iter_content(chunk_size=1024*1024):
                count+=len(block)
                if count>dense['bytes']:raise RuntimeError('DenseNet transfer exceeds pinned size')
                handle.write(block)
                if time.monotonic()-last_check>=1:disk();last_check=time.monotonic()
    checksum,_=hashes(path)
    if count!=dense['bytes'] or not checksum.startswith(dense['sha256_prefix']):raise RuntimeError('DenseNet size/hash prefix mismatch')
    records.append({'model':'densenet','file':path.name,'bytes':count,'sha256':checksum,
                    'elapsed_seconds':time.monotonic()-begin,'integrity_verified':True})
    result={'status':'approved_exact_acquisitions_hash_verified','start_utc':started,'end_utc':now(),
        'elapsed_seconds':time.monotonic()-start,'files':records,'disk':disk(),
        'payload_bytes':sum(row['bytes'] for row in records),'model_execution':False,'credentials_modified':False}
    write('acquisition.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if EVIDENCE.exists():write('failure.json',{'recorded_utc':now(),'error_type':type(error).__name__,
            'message':'Acquisition failed; retain partial artifacts. No model run begun.'})
        print(json.dumps({'status':'failed','error_type':type(error).__name__}),flush=True)
        raise SystemExit(2)
