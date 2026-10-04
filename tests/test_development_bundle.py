"""Invented byte-only composition and refusal tests; no production data access."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.analysis import resolve_analysis
from dermdetect.data.conservative_groups import ReviewedUncertainty, build_conservative_groups
from dermdetect.data.constrained_allocation import allocate_constrained_groups, verify_constrained_allocation
from dermdetect.data.development_bundle import (
    BundleApproval, bundle_input_hashes, stage_development_bundle, verify_staged_bundle)
from dermdetect.data.group_allocation import allocate_groups, verify_allocation
from dermdetect.data.manifests import sha256_bytes
from dermdetect.data.schemas import CLASS_ORDER, ContractError, ManifestRecord


class DevelopmentBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "evaluator").mkdir()
        source = []
        self.all_payloads = {}
        for dx in CLASS_ORDER:
            for i in range(10):
                image = f"invented_{dx}_{i}"
                payload = f"INVENTED {dx} image {i}".encode()
                self.all_payloads[image] = payload
                source.append(ManifestRecord(image, image, dx, "invented", "v1",
                    f"evaluator/original/{image}.txt", sha256_bytes(payload)))
        original = source[0]
        source.append(replace(original, image_id=original.image_id + "_copy"))
        self.all_payloads[source[-1].image_id] = self.all_payloads[original.image_id]
        self.rows = resolve_analysis(source, [])
        self.assigned = allocate_groups(self.rows)
        self.protected_hashes = [sha256_bytes(b"INVENTED FULL SOURCE ARCHIVE"),
                                 sha256_bytes(b"INVENTED CANDIDATE REVIEW")]
        self.arguments = dict(destination=self.root / "development",
            provenance_path=self.root / "evaluator/staging_provenance.json", records=self.rows,
            partitions=self.assigned, allocation_report=verify_allocation(self.rows, self.assigned),
            code_files={"worker.py": b"print('INVENTED worker')\n"},
            payloads={row.original.image_id: self.all_payloads[row.original.image_id]
                      for name in ("train", "validation") for row in self.assigned[name] if row.active},
            protected_roots=[self.root / "evaluator"], protected_content_sha256=self.protected_hashes,
            evidence_context="INVENTED source and completed human-review evidence, never actual approval")
        self.approve()

    def approve(self):
        inputs={name:self.arguments[name] for name in (
            "records", "partitions", "code_files", "protected_content_sha256")}
        if "grouping" in self.arguments:inputs["grouping"]=self.arguments["grouping"]
        hashes = bundle_input_hashes(**inputs)
        self.arguments["approvals"] = [BundleApproval(scope, digest, "invented reviewer",
            "invented explicit approval record", True) for scope, digest in hashes.items()]

    def reject(self, code, **changes):
        with self.assertRaises(ContractError) as caught:
            stage_development_bundle(**{**self.arguments, **changes})
        self.assertEqual(caught.exception.code, code)
        self.assertFalse(self.arguments["destination"].exists())

    def test_evaluator_provenance_cannot_be_written_outside_protected_roots(self):
        unprotected = self.root / "unprotected_review.json"
        self.reject("unsafe_bundle_location", provenance_path=unprotected)
        self.assertFalse(unprotected.exists())

    def test_constrained_mixed_group_bundle_preserves_separate_units_and_export_gate(self):
        a=next(r for r in self.assigned["train"] if r.original.dx=="nv")
        b=next(r for r in self.assigned["internal_test"] if r.original.dx=="bcc")
        uncertainty=ReviewedUncertainty(a.original.image_id,b.original.image_id,"INVENTED view evidence",
            "invented reviewer","INVENTED unresolved physical identity","2026-10-02",full_view_inspected=True)
        grouping=build_conservative_groups(self.rows,[uncertainty])
        assigned=allocate_constrained_groups(grouping)
        self.arguments.update(records=grouping.records,partitions=assigned,grouping=grouping,
            allocation_report=verify_constrained_allocation(grouping,assigned),
            payloads={r.original.image_id:self.all_payloads[r.original.image_id]
                      for name in ("train","validation") for r in assigned[name] if r.active})
        # Old approvals do not authorize the new grouping/assignment.
        self.reject("unapproved_bundle_inputs")
        self.approve()
        report=stage_development_bundle(**self.arguments)
        self.assertFalse(report["production_seal_validated"])
        evaluator=json.loads(self.arguments["provenance_path"].read_text())
        self.assertEqual(evaluator["reviewed_uncertainties"][0]["rationale"],uncertainty.rationale)
        self.assertEqual(len(evaluator["physical_analysis_anchor"]),len(self.rows))
        self.assertEqual(evaluator["allocation_report"]["physical_units"],70)
        by_image={r['image_id']:r for r in evaluator['analysis']}
        self.assertEqual(by_image[a.original.image_id]['split_group_id'],by_image[b.original.image_id]['split_group_id'])
        self.assertNotEqual(by_image[a.original.image_id]['analysis_lesion_id'],by_image[b.original.image_id]['analysis_lesion_id'])
        dev_files=[p for p in self.arguments['destination'].rglob('*') if p.is_file()]
        self.assertFalse(any(uncertainty.rationale.encode() in p.read_bytes() for p in dev_files))
        internal={r.original.image_id for r in assigned['internal_test']}
        observed={p.stem for p in self.arguments['destination'].rglob('*.payload')}
        self.assertFalse(internal & observed)
        self.assertFalse(verify_staged_bundle(destination=self.arguments['destination'],
            expected_manifest_sha256=report['bundle_manifest_sha256'])['production_seal_validated'])

    def test_uncertainty_provenance_changes_bundle_input_hash_even_with_identical_assignment(self):
        a,b=next(r for r in self.rows if r.original.dx=='nv'),next(r for r in self.rows if r.original.dx=='bcc')
        uncertainty=ReviewedUncertainty(a.original.image_id,b.original.image_id,'INVENTED evidence',
            'invented reviewer','INVENTED original rationale','2026-10-02',full_view_inspected=True)
        first=build_conservative_groups(self.rows,[uncertainty])
        second=build_conservative_groups(self.rows,[replace(uncertainty,rationale='INVENTED revised rationale')])
        assigned=allocate_constrained_groups(first)
        self.assertEqual(assigned,allocate_constrained_groups(second))
        kwargs=dict(records=first.records,partitions=assigned,code_files=self.arguments['code_files'],
                    protected_content_sha256=self.protected_hashes)
        first_hashes=bundle_input_hashes(**kwargs,grouping=first)
        second_hashes=bundle_input_hashes(**kwargs,grouping=second)
        self.assertNotEqual(first_hashes['resolved_analysis'],second_hashes['resolved_analysis'])
        self.assertNotEqual(first_hashes['development_export'],second_hashes['development_export'])
        self.assertEqual(first_hashes['fixed_assignment'],second_hashes['fixed_assignment'])

    def test_constrained_adapter_refuses_other_quota_valid_membership(self):
        grouping=build_conservative_groups(self.rows,[])
        assigned=allocate_constrained_groups(grouping)
        a=next(r for r in assigned['train'] if r.original.dx=='nv')
        b=next(r for r in assigned['validation'] if r.original.dx=='nv')
        changed={s:list(rows) for s,rows in assigned.items()}
        changed['train'].remove(a);changed['validation'].remove(b)
        changed['train'].append(b);changed['validation'].append(a)
        self.assertTrue(verify_constrained_allocation(grouping,changed)['zero_overlap'])
        with self.assertRaises(ContractError) as caught:
            bundle_input_hashes(records=grouping.records,partitions=changed,grouping=grouping,
                code_files=self.arguments['code_files'],protected_content_sha256=self.protected_hashes)
        self.assertEqual(caught.exception.code,'nonprescribed_assignment')

    def test_only_active_development_data_and_safe_context_are_exported(self):
        report = stage_development_bundle(**self.arguments)
        target = self.arguments["destination"]
        manifest = json.loads((target / "bundle_manifest.json").read_text())
        self.assertFalse(report["production_seal_validated"])
        self.assertEqual(report["exported_active_images"], 63)
        for relative, digest in manifest["files"].items():
            self.assertEqual(sha256_bytes((target / relative).read_bytes()), digest)
        development_bytes = b"".join(path.read_bytes() for path in target.rglob("*") if path.is_file())
        for row in self.assigned["internal_test"]:
            self.assertNotIn(row.original.image_id.encode(), development_bytes)
            self.assertNotIn(self.all_payloads[row.original.image_id], development_bytes)
        self.assertNotIn(b"evaluator/original", development_bytes)
        self.assertNotIn(b"source_review_context", development_bytes)
        for members in self.assigned.values():
            for row in members:
                if not row.active:
                    self.assertFalse(any(row.original.image_id == path.stem for path in target.rglob("*.payload")))
        provenance = json.loads(self.arguments["provenance_path"].read_text())
        self.assertEqual(len(provenance["analysis"]), 71)
        self.assertIn("internal_test", provenance["partitions"])
        self.assertEqual(provenance["bundle_manifest_sha256"], report["bundle_manifest_sha256"])
        self.assertEqual(sha256_bytes(self.arguments["provenance_path"].read_bytes()),
                         report["evaluator_provenance_sha256"])
        checked = verify_staged_bundle(destination=target,
            expected_manifest_sha256=report["bundle_manifest_sha256"])
        self.assertFalse(checked["production_seal_validated"])

    def test_staged_verification_rejects_added_source_tampering_and_symlink(self):
        report = stage_development_bundle(**self.arguments)
        arguments = dict(destination=self.arguments["destination"],
                         expected_manifest_sha256=report["bundle_manifest_sha256"])
        extra = arguments["destination"] / "full_source.zip"
        extra.write_bytes(b"INVENTED SOURCE COPY")
        with self.assertRaises(ContractError) as caught:
            verify_staged_bundle(**arguments)
        self.assertEqual(caught.exception.code, "unexpected_bundle_files")
        extra.unlink()
        extra.symlink_to(self.arguments["provenance_path"])
        with self.assertRaises(ContractError) as caught:
            verify_staged_bundle(**arguments)
        self.assertEqual(caught.exception.code, "unsafe_bundle_path")
        extra.unlink()
        (arguments["destination"] / "code/worker.py").write_bytes(b"CHANGED CODE")
        with self.assertRaises(ContractError) as caught:
            verify_staged_bundle(**arguments)
        self.assertEqual(caught.exception.code, "bundle_payload_mismatch")

    def test_missing_denied_and_stale_approvals_leave_no_output(self):
        self.reject("missing_bundle_approval", approvals=[])
        approvals = self.arguments["approvals"]
        self.reject("unapproved_bundle_inputs", approvals=[replace(approvals[0], approved=False), *approvals[1:]])
        self.reject("unapproved_bundle_inputs", code_files={"worker.py": b"changed reviewed code"})

    def test_sealed_inactive_or_missing_payloads_and_tampering_stop(self):
        extra = dict(self.arguments["payloads"])
        row = self.assigned["internal_test"][0]
        extra[row.original.image_id] = self.all_payloads[row.original.image_id]
        self.reject("unexpected_bundle_membership", payloads=extra)
        self.reject("unexpected_bundle_membership", payloads=self.all_payloads)
        missing = dict(self.arguments["payloads"])
        missing.pop(next(iter(missing)))
        self.reject("unexpected_bundle_membership", payloads=missing)
        changed = dict(self.arguments["payloads"])
        changed[next(iter(changed))] = b"WRONG INVENTED PAYLOAD"
        self.reject("bundle_payload_mismatch", payloads=changed)

    def test_protected_copies_refused_even_if_relabelled_as_code(self):
        self.reject("protected_content_copy", code_files={"worker.py": b"INVENTED FULL SOURCE ARCHIVE"})
        row = self.assigned["internal_test"][0]
        self.reject("protected_content_copy", code_files={"worker.py": self.all_payloads[row.original.image_id]})

    def test_protected_overlaps_symlinks_traversal_and_context_in_bundle_refused(self):
        self.reject("unsafe_bundle_location", destination=self.root / "evaluator/development")
        self.reject("unsafe_bundle_location", protected_roots=[self.root])
        self.reject("unsafe_bundle_location", provenance_path=self.root / "development")
        alias = self.root / "alias"
        alias.symlink_to(self.root / "evaluator", target_is_directory=True)
        self.reject("unsafe_bundle_path", destination=alias / "development")
        self.reject("unsafe_bundle_path", destination=self.root / "evaluator/../development")
        for name in ("../worker.py", "/worker.py", "a/./worker.py", "a\\worker.py"):
            self.reject("unsafe_bundle_path", code_files={name: b"INVENTED"})
        self.reject("unsafe_bundle_path", code_files={"worker": b"INVENTED", "worker/child": b"INVENTED"})

    def test_prevalidation_required_and_no_overwrite(self):
        self.reject("unvalidated_allocation", allocation_report={"zero_overlap": True})
        changed = dict(self.assigned)
        changed["train"] = changed["train"][1:]
        self.reject("membership_mismatch", partitions=changed)
        stage_development_bundle(**self.arguments)
        before = self.arguments["provenance_path"].read_bytes()
        with self.assertRaises(ContractError) as caught:
            stage_development_bundle(**self.arguments)
        self.assertEqual(caught.exception.code, "bundle_overwrite")
        self.assertEqual(before, self.arguments["provenance_path"].read_bytes())

    def test_quotas_alone_cannot_authorize_a_different_seed_assignment(self):
        changed = {name: list(members) for name, members in self.assigned.items()}
        # Select singleton groups from the same class, preserving all quotas.
        a = next(row for row in changed["train"] if row.original.dx == "nv")
        b = next(row for row in changed["validation"] if row.original.dx == "nv")
        changed["train"].remove(a)
        changed["validation"].remove(b)
        changed["train"].append(b)
        changed["validation"].append(a)
        self.assertTrue(verify_allocation(self.rows, changed)["zero_overlap"])
        self.reject("nonprescribed_assignment", partitions=changed)

    def test_fingerprints_and_evaluator_context_cannot_be_omitted(self):
        self.reject("missing_protected_fingerprints", protected_content_sha256=[])
        self.reject("missing_bundle_context", evidence_context="")


if __name__ == "__main__":
    unittest.main()
