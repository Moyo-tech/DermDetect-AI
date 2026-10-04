"""Complete human-review drafts never inherit agent or hash identity decisions."""
import json
import re
import shutil
import subprocess
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.manifests import canonical_json_bytes, file_sha256
from dermdetect.data.review_policy import RANKED_PROCEDURE, RANKED_STATUS, APPROVAL_REFERENCE
from dermdetect.data.schemas import ContractError
from prepare_ham_review_package import prepare, REVIEW_JS
from resolve_ham_analysis import resolve
import test_streamed_candidate_review as fixtures


class ReviewPackageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StreamedReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args, self.old_review, self.old_value = self.fixture.reviewed_fixture()
        self.root = self.fixture.root
        self.index = self.root / "visual_index.json"

    def test_complete_draft_preserves_agent_recommendation_separately_and_cannot_resolve(self):
        index = json.loads(self.index.read_text())
        observation = {**index["pairs"][0], "agent_recommendation": "same_lesion", "rationale": "invented observation"}
        agent = self.root / "agent.json"
        agent.write_bytes(canonical_json_bytes({"manifest_sha256": self.old_value["manifest_sha256"], "pairs": [observation]}))
        summary = prepare(self.args[0], self.args[-1], self.index, self.root / "review", agent)
        draft_path = self.root / "review.draft.json"
        draft = json.loads(draft_path.read_text())
        self.assertEqual(summary["review_candidates"], 2)
        self.assertEqual(summary["human_decisions_recorded"], 0)
        self.assertTrue(all(row["decision"] == "pending" and row["reviewer"] == "" for row in draft["decisions"]))
        self.assertTrue(draft["draft"])
        self.assertFalse(draft["human_acceptance"]["accept_identity_and_participation"])
        self.assertEqual(draft["decisions"][0]["agent_recommendation"], "same_lesion")
        self.assertFalse(draft["decisions"][0]["agent_recommendation_is_human_decision"])
        with self.assertRaises(ContractError):
            resolve(self.args[0], self.args[-1], draft_path, self.root / "forbidden_analysis")
        self.assertFalse((self.root / "forbidden_analysis").exists())

    def test_queue_or_full_resolution_evidence_tampering_has_no_package_outputs(self):
        queue = self.args[-1] / "candidates.csv"
        original = queue.read_bytes()
        queue.write_bytes(original + b"\n")
        with self.assertRaises(ContractError):
            prepare(self.args[0], self.args[-1], self.index, self.root / "blocked")
        self.assertFalse((self.root / "blocked.summary.json").exists())
        queue.write_bytes(original)
        view = self.root / self.old_value["decisions"][0]["evidence"]
        view.write_bytes(b"CHANGED INVENTED VIEW")
        with self.assertRaises(ContractError):
            prepare(self.args[0], self.args[-1], self.index, self.root / "blocked")
        self.assertFalse((self.root / "blocked.draft.json").exists())

    def test_missing_views_remain_pending_and_reported_not_excluded(self):
        value = json.loads(self.index.read_text())
        value["pairs"].pop()
        self.index.write_bytes(canonical_json_bytes(value))
        summary = prepare(self.args[0], self.args[-1], self.index, self.root / "review")
        self.assertEqual(summary["candidates_needing_visual_evidence"], 1)
        draft = json.loads((self.root / "review.draft.json").read_text())
        self.assertEqual(len(draft["decisions"]), 2)
        self.assertEqual(sum(not item["evidence"] for item in draft["decisions"]), 1)
        self.assertFalse(summary["production_freeze"])
        with self.assertRaises(ContractError):
            prepare(self.args[0], self.args[-1], self.index, self.root / "review")

    def ranked_fixture(self):
        report_path = self.args[-1] / "screen_report.json"
        report = json.loads(report_path.read_text())
        outside = self.args[-1] / "broad_outside.csv"
        outside.write_text("left,right,dhash_distance,reasons,decision,reviewer,rationale,evidence\n")
        view_queue = self.args[-1] / "view_queue.json"
        view_queue.write_bytes(canonical_json_bytes({"manifest_sha256": report["manifest_sha256"], "candidates": []}))
        report.update({"procedure": RANKED_PROCEDURE, "status": RANKED_STATUS,
                       "procedure_approval": "approved_candidate_selection_only", "approval_reference": APPROVAL_REFERENCE,
                       "human_decisions_recorded": 0, "broad_outside_status": "unadjudicated",
                       "broad_outside_sha256": file_sha256(outside), "view_queue_sha256": file_sha256(view_queue),
                       "broad_pairs_outside_approved_review": 30072,
                       "completeness_limit": "Sensitivity-limited selection; unknown duplicates may remain.",
                       "source_applicability_limit": "Original ranking source image hashes unavailable."})
        report_path.write_bytes(canonical_json_bytes(report))
        index = json.loads(self.index.read_text());index["queue_sha256"] = file_sha256(view_queue)
        self.index.write_bytes(canonical_json_bytes(index))

    def test_approved_ranked_package_uses_actual_procedure_and_limits(self):
        self.ranked_fixture()
        summary = prepare(self.args[0], self.args[-1], self.index, self.root / "ranked")
        self.assertEqual(summary["procedure"], RANKED_PROCEDURE)
        draft = json.loads((self.root / "ranked.draft.json").read_text())
        self.assertEqual(draft["procedure"], RANKED_PROCEDURE)
        self.assertTrue(draft["draft"])
        self.assertTrue(all(not d["full_view_inspected"] for d in draft["decisions"]))
        page = (self.root / "ranked.html").read_text()
        self.assertIn("30,072 broad-screen pairs remain unadjudicated", page)
        self.assertIn("Sensitivity-limited", page)
        self.assertIn("Original ranking source image hashes unavailable", page)
        self.assertIn("max-width:none;max-height:none", page)
        self.assertNotIn("fetch(", page)

    def test_ranked_approval_and_evidence_binding_fail_before_output(self):
        self.ranked_fixture()
        report_path = self.args[-1] / "screen_report.json"
        original = report_path.read_bytes()
        value = json.loads(original);value["approval_reference"] = "wrong approval"
        report_path.write_bytes(canonical_json_bytes(value))
        with self.assertRaises(ContractError):
            prepare(self.args[0], self.args[-1], self.index, self.root / "blocked")
        report_path.write_bytes(original)
        value = json.loads(self.index.read_text());value["queue_sha256"] = "0" * 64
        self.index.write_bytes(canonical_json_bytes(value))
        with self.assertRaises(ContractError):
            prepare(self.args[0], self.args[-1], self.index, self.root / "blocked")
        self.assertFalse((self.root / "blocked.html").exists())

    def test_critical_observations_order_and_embedded_json_escape(self):
        index = json.loads(self.index.read_text())
        pairs = [{**index["pairs"][0], "agent_recommendation": "same_lesion", "rationale": "same structures"},
                 {**index["pairs"][1], "agent_recommendation": "unresolved", "rationale": "</script><script>unsafe()</script>"}]
        agent = self.root / "agent.json"
        agent.write_bytes(canonical_json_bytes({"manifest_sha256": self.old_value["manifest_sha256"], "pairs": pairs}))
        prepare(self.args[0], self.args[-1], self.index, self.root / "review", agent)
        draft = json.loads((self.root / "review.draft.json").read_text())
        self.assertEqual(draft["decisions"][0]["agent_recommendation"], "unresolved")
        self.assertEqual(draft["decisions"][1]["agent_recommendation"], "same_lesion")
        self.assertTrue(all(d["decision"] == "pending" for d in draft["decisions"]))
        page = (self.root / "review.html").read_text()
        self.assertNotIn("</script><script>unsafe()", page)
        embedded = re.search(r'<script id="review-data" type="application/json">(.*?)</script>', page, re.S).group(1)
        self.assertIn("\\u003c/script\\u003e", embedded)
        self.assertEqual(json.loads(embedded)["draft"]["decisions"], draft["decisions"])

    @unittest.skipUnless(shutil.which("node"), "Existing Node runtime required for JavaScript check")
    def test_javascript_syntax_and_annotation_import_export_contracts(self):
        syntax = subprocess.run([shutil.which("node"), "--check"], input=REVIEW_JS, text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        prepare(self.args[0], self.args[-1], self.index, self.root / "review")
        page = (self.root / "review.html").read_text()
        data = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', page, re.S).group(1))
        harness = r'''
const assert = require("node:assert/strict"), vm = require("node:vm"), fs = require("node:fs");
const {script, data} = JSON.parse(fs.readFileSync(0, "utf8"));
const elements = {}, storage = new Map();
const element = id => elements[id] ||= {textContent:"",value:"",checked:false,hidden:false,disabled:false,
 events:{},addEventListener(name,handler){this.events[name]=handler},removeAttribute(name){delete this[name]}};
element("review-data").textContent = JSON.stringify(data);
const document = {events:{}, getElementById:element, addEventListener(name,handler){this.events[name]=handler},
 createElement(){return {click(){}}}};
const context = vm.createContext({document, window:{}, localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
 Blob, URL:{createObjectURL:()=>"blob:local",revokeObjectURL:()=>{}},setTimeout:fn=>fn(), console});
vm.runInContext(script,context);
const api = context.window.ReviewDraft, copy = x => JSON.parse(JSON.stringify(x));
const original = data.draft, baseline = copy(original);
assert.equal(api.exportDraft(original.decisions).draft,true);
assert.equal(element("decision").value,"pending");
for (const mutate of [x=>x.decisions.pop(),x=>x.decisions[1]=copy(x.decisions[0]),
 x=>x.manifest_sha256="wrong",x=>x.draft=false,x=>x.human_acceptance.accept_procedure=true,
 x=>x.decisions[0].view_sha256="wrong",x=>x.decisions[0].source_payload_sha256.invented_a="wrong",
 x=>x.decisions[0].decision="same_lesion",x=>x.decisions[0].rationale="review without inspection"]) {
 const bad=copy(original);mutate(bad);assert.throws(()=>api.validateImported(bad));
}
const reviewed=copy(original);
Object.assign(reviewed.decisions[0],{decision:"different_lesions",reviewer:"Human",review_date:"2026-10-02",
 rationale:"invented independently inspected structures",full_view_inspected:true});
assert.equal(api.validateImported(reviewed)[0].decision,"different_lesions");
reviewed.decisions.reverse();assert.equal(api.validateImported(reviewed)[0].left,original.decisions[0].left);
assert.equal(api.validDate("2026-02-30"),false);assert.equal(api.validDate("2026-10-02"),true);
// Keyboard choice changes only the form; no source/agent recommendation enters human state.
document.events.keydown({target:{tagName:"BODY"},key:"1",preventDefault(){}});
assert.equal(element("decision").value,"same_lesion");assert.equal(storage.size,0);
element("save").events.click();assert.equal(storage.size,0); // no inspection/reviewer/rationale
element("reviewer").value="Human";element("review-date").value="2026-10-02";
element("rationale").value="invented visual observations";element("inspected").checked=true;
element("save").events.click();assert.equal(storage.size,1);
const saved=JSON.parse([...storage.values()][0]);
assert.equal(saved.decisions[0].decision,"same_lesion");assert.equal(saved.draft,true);
assert.equal(saved.human_acceptance.accept_identity_and_participation,false);
assert.deepEqual(data.draft,baseline);assert.match([...storage.keys()][0],/DermDetect-M2:/);
element("next").events.click();assert.equal(element("decision").value,"pending");
// Browser-storage failures never claim persistence; annotations remain exportable.
context.localStorage.setItem=()=>{throw new Error("disabled storage")};
element("reviewer").value="Human";element("review-date").value="2026-10-02";
element("rationale").value="unclear after inspection";element("inspected").checked=true;
element("save").events.click();assert.match(element("message").textContent,/storage unavailable/);
assert.match(element("message").textContent,/export JSON now/);
assert.equal(vm.runInContext("exportDraft(rows).decisions[1].decision",context),"pending");
(async()=>{
 const imported=vm.runInContext("exportDraft(rows)",context);
 await element("import").events.change({target:{files:[{text:async()=>JSON.stringify(imported)}],value:"chosen"}});
 assert.match(element("message").textContent,/Browser storage unavailable/);
 assert.match(element("message").textContent,/export JSON now/);
 const altered=copy(imported);altered.decisions[0].view_sha256="wrong";
 await element("import").events.change({target:{files:[{text:async()=>JSON.stringify(altered)}],value:"chosen"}});
 assert.match(element("message").textContent,/Import rejected/);
 assert.equal(vm.runInContext("exportDraft(rows).decisions[0].view_sha256",context),original.decisions[0].view_sha256);
 console.log("offline annotations and immutable import/export verified");
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        result = subprocess.run([shutil.which("node"), "-e", harness],
                                input=json.dumps({"script": REVIEW_JS, "data": data}), text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("immutable import/export verified", result.stdout)


if __name__ == "__main__":
    unittest.main()
