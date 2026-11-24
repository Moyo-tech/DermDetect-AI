"""Read-only isolated local audit of the authorized downloaded dev-feature cache."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cache=a.cache.resolve();assert cache.is_relative_to(ROOT/'outputs') and cache.is_dir() and not a.cache.is_symlink();a.output.mkdir(exist_ok=False);B=a.output.resolve()/'bundle';W=a.output.resolve()/'worker-output';B.mkdir();W.mkdir()
 # Stage only read-only numerical cache checks and exact approved metadata; no model weights.
 C=ROOT/'outputs/m4-full-execution-proposal/timer_repair_candidate/full_code';(B/'src/dermdetect/models').mkdir(parents=True)
 for n in ['src/dermdetect/__init__.py','src/dermdetect/models/__init__.py','src/dermdetect/models/embeddings.py']:shutil.copyfile(C/n,B/n)
 cfg=json.loads((ROOT/'outputs/m4-full-execution-proposal/timer_repair_candidate/full_config.json').read_text());download=json.loads((cache/'download_record.json').read_text());assert download['all_downloaded_verified'];control={'cache':str(cache),'expected_key':cfg['extraction']['binding'],'metadata_SHA':download['metadata_sha256'],'ordered_ids':[r['image_id'] for r in cfg['extraction']['items']]};(B/'control.json').write_text(json.dumps(control)+'\n')
 worker='''from pathlib import Path
import sys,json,numpy as np
B=Path(__file__).parent;sys.path.insert(0,str(B/'src'))
from dermdetect.models.embeddings import load_cache
c=json.loads((B/'control.json').read_text());ids,x=load_cache(c['cache'],c['expected_key'],c['metadata_SHA']);assert list(ids)==c['ordered_ids'] and x.shape==(8505,1152) and x.dtype==np.float32 and np.isfinite(x).all();norms=np.linalg.norm(x,axis=1);assert np.allclose(norms,1,atol=1e-5,rtol=1e-5);assert np.any(np.var(x,axis=0)>0)
print(json.dumps({'status':'passed','rows':len(ids),'dimension':x.shape[1],'dtype':str(x.dtype),'ordered_ids_binding_hashes_exact':True,'finite':True,'norm_min':float(norms.min()),'norm_max':float(norms.max()),'nonzero_variance':True,'model_loaded_or_fitted':False,'GPU_used':False}))
''';(B/'worker.py').write_text(worker);old=ROOT/'outputs/m3-runtime-verification-final';sb=(old/'boundary.sb').read_text().replace(str(old/'bundle'),str(B)).replace(str(old/'worker-output'),str(W));needle='(subpath "'+str(B)+'")';assert needle in sb;sb=sb.replace(needle,needle+' (subpath "'+str(cache)+'")',1);(a.output/'boundary.sb').write_text(sb)
 env={'PATH':'/usr/bin:/bin','LC_ALL':'C','TMPDIR':str(W),'HOME':str(W),'HF_HOME':str(W/'hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':''};start=time.monotonic();r=subprocess.run(['/usr/bin/sandbox-exec','-f',str(a.output.resolve()/'boundary.sb'),str(ROOT/'env/m3-model-dev/bin/python'),'-B','-I',str(B/'worker.py')],cwd=B,env=env,stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=30);(W/'stdout.txt').write_text(r.stdout);(W/'stderr.txt').write_text(r.stderr);value={'recorded_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'seconds':time.monotonic()-start,'worker':json.loads(r.stdout) if not r.returncode else None,'error_tail':r.stderr[-1800:] if r.returncode else None,'read_only_cache_input':True,'original_data_network_helper_denials_retained':True};(a.output/'verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
