"""Bounded-memory staging with the existing approved bundle input contract."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil

from .development_bundle import (POLICY, APPROVAL_SCOPES, BundleApproval, _inputs,
                                 _safe_location)
from .manifests import canonical_json_bytes, sha256_bytes
from .schemas import ContractError


def stage_streamed_bundle(*, destination, provenance_path, records, partitions,
                          allocation_report, code_files, open_payload, approvals,
                          protected_roots, protected_content_sha256, evidence_context,
                          grouping):
    rows,assigned,development,verification,code,protected,hashes=_inputs(
        records,partitions,code_files,protected_content_sha256,grouping)
    if canonical_json_bytes(dict(allocation_report))!=canonical_json_bytes(verification):
        raise ContractError('unvalidated_allocation','Exact allocation verification required')
    approvals=tuple(approvals)
    if (len(approvals)!=len(APPROVAL_SCOPES) or any(not isinstance(a,BundleApproval) for a in approvals)
            or {a.scope for a in approvals}!=set(APPROVAL_SCOPES)):
        raise ContractError('missing_bundle_approval','Each exact input scope needs approval')
    if any(a.approved is not True or a.input_sha256!=hashes[a.scope] or not a.reviewer.strip() or not a.evidence.strip() for a in approvals):
        raise ContractError('unapproved_bundle_inputs','Approval must bind current inputs')
    target,provenance=_safe_location(destination),_safe_location(provenance_path)
    roots=tuple(_safe_location(Path(p),existing=True) for p in protected_roots)
    if (not roots or any(target.is_relative_to(p) or p.is_relative_to(target) for p in roots)
            or provenance.is_relative_to(target) or target.is_relative_to(provenance)
            or not any(provenance.is_relative_to(p) for p in roots)):
        raise ContractError('unsafe_bundle_location','Separate evaluator and development locations required')
    if not callable(open_payload) or not isinstance(evidence_context,str) or not evidence_context.strip():
        raise ContractError('missing_bundle_context','Explicit stream opener and evaluator context required')
    files={};target.mkdir();provenance_created=False
    def write(reference,payload):
        path=target/reference;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(payload)
        files[reference]=sha256_bytes(payload)
    try:
        for name,payload in sorted(code_files.items()):write('code/'+name,payload)
        for partition in ('train','validation'):
            safe=[]
            for row in assigned[partition]:
                if not row.active:continue
                image=row.original.image_id;reference=f'data/{partition}/{image}.payload'
                path=target/reference;path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
                with open_payload(row) as source,path.open('xb') as output:
                    while True:
                        block=source.read(1024*1024)
                        if not block:break
                        if type(block) is not bytes:raise ContractError('invalid_bundle_payload','Byte stream required')
                        h.update(block);output.write(block)
                checksum=h.hexdigest()
                if checksum!=row.original.sha256 or checksum in protected:
                    raise ContractError('bundle_payload_mismatch','Stream must match approved active payload')
                files[reference]=checksum
                safe.append({'image_id':image,'dx':row.original.dx,'analysis_lesion_id':row.analysis_lesion_id,
                    'split_group_id':row.split_group_id,'path':reference,'sha256':checksum})
            write(f'manifests/{partition}.json',canonical_json_bytes(safe))
        manifest={'policy':POLICY,'files':dict(sorted(files.items())),
                  'production_seal_validated':False,'status':'staged_only'}
        manifest_bytes=canonical_json_bytes(manifest)
        with (target/'bundle_manifest.json').open('xb') as handle:handle.write(manifest_bytes)
        evaluator={'policy':POLICY,'status':'staged_only','production_seal_validated':False,
            'bundle_manifest_sha256':sha256_bytes(manifest_bytes),'input_hashes':hashes,
            'source_review_context':evidence_context,'approvals':[asdict(a) for a in approvals],
            'analysis':[r.to_dict() for r in rows],
            'partitions':{p:[r.original.image_id for r in members] for p,members in assigned.items()},
            'allocation_report':verification,'protected_roots':[str(p) for p in roots],'code_manifest':code,
            'physical_analysis_anchor':[r.to_dict() for r in grouping.physical_records],
            'reviewed_uncertainties':[asdict(link) for link in grouping.uncertainties],
            'stream_chunk_bytes':1024*1024,'limitations':['Staging only; final runtime and boundary verification required.']}
        provenance_bytes=canonical_json_bytes(evaluator)
        with provenance.open('xb') as handle:
            provenance_created=True;handle.write(provenance_bytes)
    except BaseException:
        shutil.rmtree(target)
        if provenance_created:provenance.unlink()
        raise
    return {'status':'staged_only','production_seal_validated':False,
        'bundle_manifest_sha256':sha256_bytes(manifest_bytes),'evaluator_provenance_sha256':sha256_bytes(provenance_bytes),
        'exported_active_images':len(development),'code_files':len(code),'stream_chunk_bytes':1024*1024}
