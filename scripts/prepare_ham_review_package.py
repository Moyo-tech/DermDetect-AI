"""Prepare pending human review from complete saved evidence; no image reads.

Copies no raw payloads, changes no identity decisions and assigns no partitions.
The HTML references existing full-resolution local integrity views only.
"""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.review_policy import BROAD_PROCEDURE, validate_review_screen
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, rooted_file
from dermdetect.data.schemas import ContractError
from resolve_ham_analysis import VIEW_TRANSFORM


REVIEW_JS = r'''
"use strict";
const packageData = JSON.parse(document.getElementById("review-data").textContent);
const BASE = packageData.draft;
const EDITABLE = new Set(["decision", "reviewer", "review_date", "rationale", "full_view_inspected"]);
const clone = value => JSON.parse(JSON.stringify(value));
const stable = value => JSON.stringify(value && typeof value === "object" && !Array.isArray(value)
  ? Object.fromEntries(Object.keys(value).sort().map(k => [k, JSON.parse(stable(value[k]))]))
  : Array.isArray(value) ? value.map(v => JSON.parse(stable(v))) : value);
const key = row => JSON.stringify([row.left, row.right]);
const immutable = row => Object.fromEntries(Object.entries(row).filter(([k]) => !EDITABLE.has(k)));
const validDate = text => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) return false;
  const date = new Date(text + "T00:00:00Z");
  return !Number.isNaN(date.getTime()) && date.toISOString().slice(0, 10) === text;
};
function validateRow(row, baseline) {
  if (!row || typeof row !== "object" || stable(immutable(row)) !== stable(immutable(baseline)))
    throw new Error("Immutable image/source/view evidence changed.");
  if (!["pending", "same_lesion", "different_lesions"].includes(row.decision))
    throw new Error("Decision must be same lesion, different lesions, or pending/unclear.");
  for (const field of ["reviewer", "review_date", "rationale"])
    if (typeof row[field] !== "string") throw new Error("Review fields must be text.");
  if (typeof row.full_view_inspected !== "boolean") throw new Error("Full-view inspection attestation is required.");
  const untouched = row.decision === "pending" && !row.full_view_inspected
    && [row.reviewer, row.review_date, row.rationale].every(v => v === "");
  if (!untouched && (!row.reviewer.trim() || !validDate(row.review_date)
      || !row.rationale.trim() || !row.full_view_inspected || !baseline.evidence || !baseline.view_sha256))
    throw new Error("Inspect the full-resolution view and supply reviewer, ISO date and rationale.");
  return clone(row);
}
function validateImported(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Import a complete review JSON object.");
  const top = value => Object.fromEntries(Object.entries(value).filter(([k]) => k !== "decisions"));
  if (stable(top(input)) !== stable(top(BASE))) throw new Error("Queue bindings, draft status or acceptance fields changed.");
  if (!Array.isArray(input.decisions) || input.decisions.length !== BASE.decisions.length)
    throw new Error("Import must cover every candidate exactly once.");
  const byKey = new Map();
  for (const row of input.decisions) {
    if (!row || typeof row !== "object" || byKey.has(key(row))) throw new Error("Repeated or invalid decision row.");
    byKey.set(key(row), row);
  }
  const checked = BASE.decisions.map(baseline => {
    const row = byKey.get(key(baseline));
    if (!row) throw new Error("Missing or substituted candidate.");
    return validateRow(row, baseline);
  });
  return checked;
}
function exportDraft(rows) {
  const output = clone(BASE);
  output.decisions = clone(rows);
  // The viewer records annotations only. Global approval stays outside it.
  output.draft = true;
  output.human_acceptance = clone(BASE.human_acceptance);
  validateImported(output);
  return output;
}
window.ReviewDraft = {validateImported, exportDraft, validateRow, validDate};
const storageKey = "DermDetect-M2:" + [BASE.manifest_sha256, BASE.candidate_queue_sha256,
  BASE.screen_report_sha256, BASE.visual_index_sha256].join(":");
let rows = clone(BASE.decisions), position = 0;
const el = id => document.getElementById(id);
const message = text => { el("message").textContent = text; };
function persist() {
  try { localStorage.setItem(storageKey, JSON.stringify(exportDraft(rows))); return true; }
  catch (error) { return false; }
}
try {
  const saved = localStorage.getItem(storageKey);
  if (saved) rows = validateImported(JSON.parse(saved));
} catch (error) { message("Saved annotations were not loaded: " + error.message); }
function render() {
  const row = rows[position], original = BASE.decisions[position];
  el("pair").textContent = row.left + " / " + row.right;
  el("counter").textContent = (position + 1) + " / " + rows.length;
  const finalCount = rows.filter(r => r.decision !== "pending").length;
  const unclear = rows.filter(r => r.decision === "pending" && r.full_view_inspected).length;
  el("progress").textContent = finalCount + " identity decisions; " + unclear + " inspected but unclear; "
    + (rows.length - finalCount - unclear) + " untouched. Final acceptance remains pending.";
  el("reviewer").value = row.reviewer || el("default-reviewer").value;
  el("review-date").value = row.review_date || el("default-date").value;
  el("rationale").value = row.rationale;
  el("inspected").checked = row.full_view_inspected;
  el("decision").value = row.decision;
  el("recommendation").textContent = original.agent_recommendation
    ? "Existing agent recommendation: " + original.agent_recommendation + ". " + (original.agent_rationale || "")
    : "No agent recommendation. Enter your own decision after inspection.";
  el("evidence").textContent = "View SHA-256: " + (row.view_sha256 || "not prepared")
    + " | dHash distance: " + row.dhash_distance + " | " + row.screen_reasons;
  const url = packageData.view_urls[position];
  el("image").hidden = !url;
  el("missing-view").hidden = !!url;
  if (url) { el("image").src = url; } else { el("image").removeAttribute("src"); }
  el("view").scrollTop = 0; el("view").scrollLeft = 0;
  el("save").disabled = !url;
  el("jump").value = position + 1;
}
function saveCurrent() {
  const value = {...rows[position], decision: el("decision").value,
    reviewer: el("reviewer").value.trim(), review_date: el("review-date").value,
    rationale: el("rationale").value.trim(), full_view_inspected: el("inspected").checked};
  // Explicit saves, including pending/unclear, require actual inspection.
  if (!value.full_view_inspected) { message("Confirm that you inspected both images at full resolution."); return false; }
  try { rows[position] = validateRow(value, BASE.decisions[position]); }
  catch (error) { message(error.message); return false; }
  const stored = persist();
  message(stored ? "Annotation saved locally. Export JSON regularly."
    : "Browser storage unavailable. Annotation is held in this tab only; export JSON now to preserve it.");
  render(); return true;
}
function move(delta) { position = Math.max(0, Math.min(rows.length - 1, position + delta)); render(); }
const today = new Date();
el("default-date").value = today.getFullYear() + "-" + String(today.getMonth()+1).padStart(2,"0")
  + "-" + String(today.getDate()).padStart(2,"0");
el("save").addEventListener("click", saveCurrent);
el("save-next").addEventListener("click", () => { if (saveCurrent()) move(1); });
el("previous").addEventListener("click", () => move(-1));
el("next").addEventListener("click", () => move(1));
el("jump-button").addEventListener("click", () => { const value = Number(el("jump").value);
  if (Number.isInteger(value) && value >= 1 && value <= rows.length) { position = value - 1; render(); } });
el("next-pending").addEventListener("click", () => {
  for (let delta = 1; delta <= rows.length; delta++) {
    const next = (position + delta) % rows.length;
    if (rows[next].decision === "pending") { position = next; render(); return; }
  }
  message("Every candidate has an identity decision; global acceptance is still pending.");
});
el("export").addEventListener("click", () => {
  try {
    const content = JSON.stringify(exportDraft(rows), null, 2) + "\n";
    const url = URL.createObjectURL(new Blob([content], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url;
    link.download = "m2-human-annotations.draft.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message("Exported draft annotations. No final acceptance or freeze was granted.");
  } catch (error) { message(error.message); }
});
el("import").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  try { const incoming = validateImported(JSON.parse(await file.text()));
    rows = incoming; const stored = persist(); render();
    message(stored ? "Imported complete, matching draft annotations."
      : "Imported into this tab only. Browser storage unavailable; export JSON now to preserve annotations.");
  } catch (error) { message("Import rejected: " + error.message); }
  event.target.value = "";
});
document.addEventListener("keydown", event => {
  const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault(); if (saveCurrent()) move(1); return;
  }
  if (editing || event.ctrlKey || event.metaKey || event.altKey) return;
  if (["1", "2", "3"].includes(event.key)) {
    event.preventDefault(); el("decision").value = {"1":"same_lesion","2":"different_lesions","3":"pending"}[event.key];
  } else if (event.key === "ArrowRight") { event.preventDefault(); move(1);
  } else if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); }
});
render();
'''


def render_review_page(draft, view_urls, summary, links):
    # Escaping '<' prevents even a source rationale containing '</script>'
    # from closing this inert JSON block. Dynamic text uses textContent only.
    data = json.dumps({"draft": draft, "view_urls": view_urls}, ensure_ascii=True)
    for raw, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        data = data.replace(raw, escaped)
    limitation = summary.get("completeness_limit", "Candidate screening does not certify unknown-duplicate completeness.")
    applicability = summary.get("source_applicability_limit", "Identity requires local visual adjudication.")
    scope = (f'Approved candidate selection only; {summary["broad_pairs_outside_approved_review"]:,} broad-screen pairs remain unadjudicated. '
             if "broad_pairs_outside_approved_review" in summary else "")
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>M2 human review</title>
<style>body{font:16px system-ui;margin:20px;line-height:1.45;background:#fafafa;color:#202020}
button,input,select,textarea{font:inherit;margin:4px;padding:7px}button{cursor:pointer}label{display:inline-block}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px}.view{overflow:auto;height:62vh;background:#eee;border:1px solid #aaa}
img{width:auto;height:auto;max-width:none;max-height:none;display:block}.rationale-field{display:block;width:100%}textarea{display:block;box-sizing:border-box;width:min(100%,900px);min-height:72px}
#message{min-height:1.5em;color:#683600}#evidence{font-size:12px;overflow-wrap:anywhere}.note{max-width:1000px}
</style></head><body><h1>M2 human annotations — draft</h1>
<p class="note">Inspect both full-resolution images using horizontal and vertical scrolling. Record physical-lesion identity only.
Ambiguity remains pending. Agent recommendations require your independent judgment. This viewer cannot grant final acceptance or split approval.</p>
<p>''' + html.escape(f'{summary["review_candidates"]:,} candidates; {summary["candidates_with_existing_full_resolution_views"]} full-resolution views. Procedure: {summary["procedure"]}.') + '</p><p class="note">' + html.escape(scope + limitation + " " + applicability) + '''</p>
<p>''' + ' · '.join('<a href="' + html.escape(url, quote=True) + '">' + html.escape(label) + '</a>' for label, url in links) + '''</p>
<div class="toolbar"><label>Default reviewer <input id="default-reviewer" autocomplete="name"></label>
<label>Default date <input id="default-date" type="date"></label><button id="export">Export draft JSON</button>
<label>Import matching draft <input id="import" type="file" accept=".json,application/json"></label></div>
<p id="progress"></p><p id="message" role="status" aria-live="polite"></p>
<div class="toolbar"><button id="previous">Previous ←</button><strong id="counter"></strong><button id="next">Next →</button>
<button id="next-pending">Next pending</button><label>Go to <input id="jump" type="number" min="1"></label><button id="jump-button">Go</button></div>
<h2 id="pair"></h2><p id="recommendation"></p><p id="evidence"></p>
<div id="view" class="view"><img id="image" alt="Full-resolution side-by-side integrity comparison">
<p id="missing-view" hidden>Full-resolution view is not prepared. This pair remains pending.</p></div>
<div class="toolbar"><label>Decision <select id="decision"><option value="pending">3 — Pending / unclear</option>
<option value="same_lesion">1 — Same physical lesion</option><option value="different_lesions">2 — Different lesions</option></select></label>
<label>Reviewer <input id="reviewer" required></label><label>Review date <input id="review-date" type="date" required></label></div>
<label class="rationale-field">Rationale <textarea id="rationale" required placeholder="Record visual observations and uncertainty."></textarea></label>
<label><input id="inspected" type="checkbox">I inspected both images at full resolution, including areas reached by scrolling.</label>
<div class="toolbar"><button id="save">Save annotation</button><button id="save-next">Save and next (Ctrl/⌘ Enter)</button></div>
<p class="note">Shortcuts when outside text fields: 1 same lesion, 2 different lesions, 3 pending/unclear; ←/→ navigate.
Navigation leaves unsaved edits behind. Save each annotation and export JSON regularly. Browser storage is local to this exact queue and evidence.</p>
<script id="review-data" type="application/json">''' + data + '</script><script>' + REVIEW_JS + '</script></body></html>'


def prepare(manifest: Path, screen_dir: Path, visual_index: Path,
            output_prefix: Path, agent_review: Path | None = None) -> dict:
    outputs = {name: output_prefix.with_suffix(suffix) for name, suffix in
               (("summary", ".summary.json"), ("draft", ".draft.json"), ("page", ".html"))}
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise ContractError("output_exists", "Refusing to overwrite human-review evidence.")
    root = output_prefix.parent.resolve(strict=True)
    index_path = visual_index.resolve(strict=True)
    if not index_path.is_relative_to(root):
        raise ContractError("unsafe_path", "Visual index must be beneath the review package's explicit root.")
    report_path, queue_path = screen_dir / "screen_report.json", screen_dir / "candidates.csv"
    report = json.loads(report_path.read_text())
    procedure = validate_review_screen(report, file_sha256(manifest), file_sha256(queue_path))
    if procedure == BROAD_PROCEDURE:
        if report.get("hash_matches_sha256") != file_sha256(screen_dir / "hash_matches.csv"):
            raise ContractError("review_binding_mismatch", "Review requires unchanged complete enumeration evidence.")
    elif (report.get("view_queue_sha256") != file_sha256(screen_dir / "view_queue.json")
            or report.get("broad_outside_sha256") != file_sha256(screen_dir / "broad_outside.csv")):
        raise ContractError("review_binding_mismatch", "Approved selection and preserved unadjudicated audit must remain unchanged.")
    by_id = {row.image_id: row for row in read_manifest(manifest)}
    with queue_path.open(newline="") as handle:
        queue = list(csv.DictReader(handle))
    keys = [tuple(sorted((row["left"], row["right"]))) for row in queue]
    key_set = set(keys)
    if (len(keys) != report.get("review_candidates") or len(keys) != len(key_set)
            or any(a == b or a not in by_id or b not in by_id for a, b in keys)):
        raise ContractError("candidate_membership_mismatch", "Every complete candidate must occur exactly once.")
    index = json.loads(index_path.read_text())
    if (index.get("manifest_sha256") != report["manifest_sha256"]
            or index.get("status") != "visual_evidence_prepared_not_adjudicated"
            or (procedure != BROAD_PROCEDURE and index.get("queue_sha256") != report.get("view_queue_sha256"))):
        raise ContractError("visual_evidence_mismatch", "Visual evidence must match the inventory.")
    views = {}
    for view in index["pairs"]:
        ids = view.get("image_ids")
        if (not isinstance(ids, list) or len(ids) != 2 or any(i not in by_id for i in ids)):
            raise ContractError("visual_evidence_mismatch", "View must name two inventory images.")
        key = tuple(sorted(ids))
        if (key in views or key not in key_set
                or view.get("source_payload_sha256") != [by_id[i].sha256 for i in ids]
                or view.get("view_transform") != VIEW_TRANSFORM
                or file_sha256(rooted_file(index_path.parent, view["view_file"])) != view.get("view_sha256")):
            raise ContractError("visual_evidence_mismatch", "Preserved full-resolution view evidence is inconsistent.")
        views[key] = view
    observations = {}
    if agent_review is not None:
        agent = json.loads(agent_review.read_text())
        if agent.get("manifest_sha256") != report["manifest_sha256"]:
            raise ContractError("review_binding_mismatch", "Agent observations refer to another inventory.")
        if len(agent["pairs"]) > 20:
            raise ContractError("invalid_agent_observations", "Only the existing 20 source-known observations may be reused.")
        for item in agent["pairs"]:
            key = tuple(sorted(item["image_ids"]))
            view = views.get(key)
            if (key in observations or view is None or item.get("view_sha256") != view["view_sha256"]
                    or item.get("source_payload_sha256") != view["source_payload_sha256"]):
                raise ContractError("visual_evidence_mismatch", "Agent observation must match the preserved view.")
            observations[key] = item
    # Candidate graph is workload evidence only. It does not resolve physical
    # identity or propose merging graph components into analysis/split groups.
    parent = {by_id[i].lesion_id: by_id[i].lesion_id for key in keys for i in key}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    label_pairs, degrees = Counter(), Counter()
    decisions, view_urls = [], {}
    for entry, key in zip(queue, keys):
        left, right = (by_id[i] for i in key)
        a, b = find(left.lesion_id), find(right.lesion_id)
        parent[max(a, b)] = min(a, b)
        label_pairs["/".join(sorted((left.dx, right.dx)))] += 1
        degrees.update(key)
        view, observation = views.get(key), observations.get(key)
        item = {"left": key[0], "right": key[1], "decision": "pending",
                "reviewer": "", "review_date": "", "rationale": "", "evidence": "",
                "source_payload_sha256": {i: by_id[i].sha256 for i in key},
                "source_applicability": {i: {"source": by_id[i].source, "source_version": by_id[i].source_version} for i in key},
                "view_repetition": "exact_content_repeat" if left.sha256 == right.sha256 else "byte_distinct_views",
                "view_sha256": "", "screen_reasons": entry["reasons"],
                "dhash_distance": int(entry["dhash_distance"]), "full_view_inspected": False}
        if view is not None:
            item.update({"evidence": view["view_file"], "view_sha256": view["view_sha256"]})
            view_urls[key] = (index_path.parent / view["view_file"]).relative_to(root).as_posix()
        if observation:
            item["agent_recommendation"] = observation.get("agent_recommendation", "unresolved")
            item["agent_rationale"] = observation.get("rationale", "")
            item["agent_recommendation_is_human_decision"] = False
        decisions.append(item)
    # Put the existing source-known ambiguities first, followed by existing
    # recommendations. All remaining candidates retain their queue order.
    decisions.sort(key=lambda item: (0 if "agent_recommendation" in item and item["agent_recommendation"] != "same_lesion"
                                    else 1 if "agent_recommendation" in item else 2))
    components = defaultdict(set)
    lesion_labels = {row.lesion_id: row.dx for row in by_id.values()}
    for lesion in parent:
        components[find(lesion)].add(lesion)
    summary = {"status": "complete_queue_pending_human_review", "production_freeze": False,
               "manifest_sha256": report["manifest_sha256"],
               "screen_report_sha256": file_sha256(report_path), "candidate_queue_sha256": file_sha256(queue_path),
               "visual_index_sha256": file_sha256(index_path), "script_sha256": file_sha256(Path(__file__)),
               "procedure": procedure, "review_candidates": len(decisions), "human_decisions_recorded": 0,
               "candidates_with_existing_full_resolution_views": len(views),
               "candidates_needing_visual_evidence": len(decisions) - len(views),
               "agent_recommendations": dict(Counter(o.get("agent_recommendation", "unresolved") for o in observations.values())),
               "candidate_label_pairs": dict(sorted(label_pairs.items())),
               "candidate_images": len(degrees), "largest_image_candidate_degree": max(degrees.values(), default=0),
               "candidate_graph_components": len(components),
               "candidate_graph_original_lesions": len(parent),
               "largest_candidate_graph_component_original_lesions": max(map(len, components.values()), default=0),
               "mixed_label_candidate_graph_components": sum(len({lesion_labels[i] for i in members}) > 1 for members in components.values()),
               "candidate_graph_limit": "Screen workload only; no physical identities, split groups, exclusions or assignments inferred.",
               "blocking_decisions": ["Human adjudication of every candidate with full-resolution evidence.",
                                      "Acceptance of procedure and actual resolved participation.",
                                      "Authorization of one fixed real assignment, then validation/review of production seal."]}
    for field in ("source_applicability_limit", "completeness_limit", "broad_pairs_outside_approved_review", "broad_outside_status", "approval_reference"):
        if field in report:
            summary[field] = report[field]
    if agent_review is not None:
        summary["agent_observations_sha256"] = file_sha256(agent_review)
    draft = {"manifest_sha256": report["manifest_sha256"], "procedure": procedure,
             "candidate_queue_sha256": file_sha256(queue_path), "screen_report_sha256": file_sha256(report_path),
             "visual_index": index_path.relative_to(root).as_posix(), "visual_index_sha256": file_sha256(index_path),
             "draft": True, "decisions": decisions,
             "human_acceptance": {"reviewer": "", "date": "", "approval_reference": "",
                                  "accept_procedure": False, "accept_identity_and_participation": False, "decision_sha256": ""}}
    queue_link = queue_path.resolve().relative_to(root).as_posix() if queue_path.resolve().is_relative_to(root) else ""
    page = render_review_page(draft, [view_urls.get((d["left"], d["right"]), "") for d in decisions], summary,
                             [("Complete pending draft", outputs["draft"].name),
                              ("Workload and gate summary", outputs["summary"].name), ("Candidate CSV", queue_link)])
    payloads = {outputs["summary"]: canonical_json_bytes(summary), outputs["draft"]: canonical_json_bytes(draft),
                outputs["page"]: page.encode()}
    created = []
    try:
        for path, content in payloads.items():
            with path.open("xb") as handle:
                created.append(path)
                handle.write(content)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "screen-dir", "visual-index", "output-prefix"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--agent-review", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.manifest, args.screen_dir, args.visual_index, args.output_prefix, args.agent_review)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({k: result[k] for k in ("status", "review_candidates", "candidates_needing_visual_evidence", "production_freeze")}))
