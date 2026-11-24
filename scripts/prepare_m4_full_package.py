"""Prepare exact development metadata/transfer manifest locally. No network/decode.
Original seal/manifests/configuration remain unchanged; no image bytes copied.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dermdetect.data.numerical_policy import CLASS_ORDER,View,training_weights
from dermdetect.models.full_run_contracts import sha256_file,rooted_file

def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def prepare(bundle, output, checkpoint_payload):
    output.mkdir(exist_ok=False,parents=True)
    seal=ROOT/'outputs/m2-production-closeout/production_seal.json'
    original=json.loads((bundle/'bundle_manifest.json').read_text())
    files=original['files'];rows={}
    for split in ['train','validation']:
        source=rooted_file(bundle,'manifests/'+split+'.json')
        if sha256_file(source)!=files['manifests/'+split+'.json']:raise ValueError('Sealed development manifest changed')
        raw=json.loads(source.read_text());normalized=[]
        for r in raw:
            if r['dx'] not in CLASS_ORDER or r['path'] not in files or r['sha256']!=files[r['path']]:raise ValueError('Membership/payload binding changed')
            p=rooted_file(bundle,r['path'])
            normalized.append({'image_id':r['image_id'],'physical_id':r['analysis_lesion_id'],'cluster_id':r['split_group_id'],'target':CLASS_ORDER.index(r['dx']),'label':r['dx'],'path':r['path'],'sha256':r['sha256'],'bytes':p.stat().st_size,'split':split})
        rows[split]=normalized
    train,validation=rows['train'],rows['validation']
    if len(train)!=7000 or len(validation)!=1505 or len({r['image_id'] for r in train+validation})!=8505:raise ValueError('Exact accepted composition required')
    if len({r['physical_id'] for r in train})!=5214 or len({r['physical_id'] for r in validation})!=1118:raise ValueError('Physical-unit composition changed')
    if {r['physical_id'] for r in train}&{r['physical_id'] for r in validation} or {r['cluster_id'] for r in train}&{r['cluster_id'] for r in validation}:raise ValueError('Development partition overlap')
    weights=training_weights([View(r['image_id'],r['physical_id'],r['cluster_id'],r['label'],r['sha256']) for r in train])
    for r in train:r['weight']=float(weights['image_weights'][r['image_id']])
    for split in rows:write(output/(split+'.json'),rows[split])
    old=json.loads((checkpoint_payload/'payload_manifest.json').read_text())
    reuse={n:row for n,row in old['files'].items() if n.startswith('weights/')}
    entries=[]
    for r in train+validation:
        existing_path='data/'+r['image_id']+'.payload';oldrow=old['files'].get(existing_path);existing=bool(oldrow and oldrow['sha256']==r['sha256'])
        entries.append({'destination':r['path'],'source':str((bundle/r['path']).resolve()),'sha256':r['sha256'],'bytes':r['bytes'],'kind':'development_image','existing_remote_source':existing_path if existing else None,'action':'verify_existing_and_link' if existing else 'upload_after_approval'})
    for n,row in reuse.items():entries.append({'destination':n,'source':str((checkpoint_payload/n).resolve()),'sha256':row['sha256'],'bytes':row['bytes'],'kind':'pinned_checkpoint','action':'verify_existing'})
    write(output/'transfer_manifest.json',{'status':'proposal_not_upload_authorization','destination':'same_private_Lightning_Studio','development_images':8505,'existing_images_verified_by_saved_manifest':2,'entries':entries,'source_seal_sha256':sha256_file(seal),'source_bundle_manifest_sha256':sha256_file(bundle/'bundle_manifest.json'),'train_manifest_sha256':sha256_file(output/'train.json'),'validation_manifest_sha256':sha256_file(output/'validation.json'),'payload_hashes_from_seal_not_reread':True})
    return {'train':len(train),'validation':len(validation),'weights_sum':sum(r['weight'] for r in train),'payload_bytes':sum(r['bytes'] for r in train+validation),'new_images':sum(e['kind']=='development_image' and e['action']=='upload_after_approval' for e in entries),'checkpoints_reused':len(reuse)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'env/m2-production/inputs');p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-payload',type=Path,default=ROOT/'outputs/m4-lightning-preflight/payload');a=p.parse_args();print(json.dumps(prepare(a.bundle,a.output,a.checkpoint_payload)))
