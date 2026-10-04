#!/usr/bin/env python3
"""Controller stages only an explicitly approved exact development payload."""
from pathlib import Path
import argparse,hashlib,json,shutil,time,os
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--approval',required=True);a=p.parse_args()
    cp=Path(a.configuration);c=json.loads(cp.read_text());v=json.loads(Path(a.approval).read_text())
    if v.get('status')!='user_approved' or not v.get('user_reference') or v.get('configuration_SHA')!=sha(cp) or v.get('development_staging') is not True:
        raise ValueError('Exact development staging approval required before reading source payloads')
    dest=Path(c['staged_input_root']);dest.mkdir(exist_ok=False);records=[];started=time.monotonic()
    st=os.statvfs(dest)
    if st.f_bavail*st.f_frsize<11*1024**3:raise ValueError('Staging needs11GiB free to retain8GiB reserve')
    def copy(source,target,expected):
        if time.monotonic()-started>=180:raise TimeoutError('Staging180second cap')
        source=Path(source);target=Path(target)
        if not source.is_file() or source.is_symlink() or not target.is_relative_to(dest) or sha(source)!=expected:
            raise ValueError('Pinned staged input mismatch')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if sha(target)!=expected:raise ValueError('Staged copy differs')
        records.append({'path':str(target),'SHA':expected})
    try:
        rows=[]
        for name in ['train','validation']:
            copy(c['source_paths'][name],c['inputs'][name],c['hashes'][name]);rows+=json.loads(Path(c['inputs'][name]).read_text())
        seen=set();root=Path(c['source_paths']['images']).resolve(strict=True)
        for r in rows:
            if r['split'] not in ('train','validation') or r['image_id'] in seen:raise ValueError('Development-only unique copy scope')
            seen.add(r['image_id']);q=Path(r['path'])
            if q.is_absolute() or '..' in q.parts or (root/q).is_symlink() or not (root/q).resolve().is_relative_to(root):raise ValueError('Unsafe image reference')
            copy(root/q,Path(c['inputs']['images'])/q,r['sha256'])
        if len(seen)!=8505:raise ValueError('Exact8505 development views required')
        copy(c['source_paths']['checkpoint'],c['inputs']['checkpoint'],c['hashes']['checkpoint'])
        for name,key in [('cache.json','med_metadata'),('features.npy','med_features')]:
            copy(Path(c['source_paths']['med_cache'])/name,Path(c['inputs']['med_cache'])/name,c['hashes'][key])
        (dest/'staging_record.json').write_text(json.dumps({'configuration_SHA':sha(cp),'files':records,'verified':True,'wall_seconds':time.monotonic()-started},indent=2)+'\n')
    except BaseException as e:
        (dest/'staging_failure.json').write_text(json.dumps({'files':records,'error':str(e)},indent=2)+'\n');raise
    print(json.dumps({'status':'passed','development_images':8505,'files':len(records)}))
if __name__=='__main__':main()
