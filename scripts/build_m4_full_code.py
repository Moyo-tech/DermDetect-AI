"""Curated local source package builder; no image copy, download or cloud action."""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dermdetect.models.full_run_contracts import sha256_file
SOURCES=['src/dermdetect/__init__.py','src/dermdetect/data/__init__.py','src/dermdetect/data/manifests.py','src/dermdetect/data/schemas.py','src/dermdetect/models/__init__.py','src/dermdetect/models/densenet.py','src/dermdetect/models/embeddings.py','src/dermdetect/models/tf_source_resize.py','src/dermdetect/models/adaptation.py','src/dermdetect/models/training_run.py','src/dermdetect/models/extraction_run.py','src/dermdetect/models/full_run_contracts.py','scripts/train_m4.py','scripts/extract_m4.py','scripts/supervise_m4.py']

def build(config_path,output):
    output.mkdir(exist_ok=False,parents=True)
    for n in SOURCES:
        q=output/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,q)
    runtime=ROOT/'outputs/m4-full-execution-proposal/runtime'
    for q in runtime.iterdir():
        if q.is_file() and q.suffix in {'.py','.json'}:shutil.copyfile(q,output/q.name)
    files={p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob('*') if p.is_file()}
    manifest=output/'full_code_manifest.json';manifest.write_text(json.dumps({'policy':'curated-full-M4-code-v1','files':files},sort_keys=True,indent=2)+'\n')
    config=json.loads(config_path.read_text());config['full_code_manifest_sha256']=sha256_file(manifest);config_path.write_text(json.dumps(config,sort_keys=True,indent=2)+'\n')
    return {'source_files':len(files),'code_manifest_sha256':sha256_file(manifest),'config_sha256':sha256_file(config_path)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.config,a.output)))
