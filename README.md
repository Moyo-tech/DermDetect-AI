# DermDetect-R

**A research rebuild of DermDetect-AI: leakage-aware development, matched uncertainty evaluation, and measured compute.**

DermDetect-R is an ongoing comparison of an ImageNet-pretrained DenseNet121 system and frozen MedSigLIP embeddings. The aim is a reproducible systems comparison under a limited compute budget—not a new diagnostic model or evidence of clinical readiness.

[Research overview](docs/RESEARCH_OVERVIEW.md) · [Methods and status](docs/METHODS_AND_STATUS.md) · [Provenance and limits](docs/PROVENANCE.md)

## Research question

> Under limited compute, how do frozen MedSigLIP embeddings and an ImageNet-pretrained DenseNet121 baseline compare in class-balanced performance, calibration, selective-prediction behaviour and end-to-end computational cost when developed and evaluated on the same leakage-safe HAM10000 partitions?

The secondary question concerns external benign-versus-malignant transfer and descriptive subgroup performance on DDI, after both systems and their decision rules are frozen. No external outcomes have been accessed for development.

## Why this rebuild exists

The historical educational prototype included model-training notebooks and a Streamlit demo, but its saved outputs did not establish an independently tested, reproducible baseline. Selection-validation reuse, preprocessing differences, uncertain checkpoint lineage and uncalibrated displayed scores motivate a controlled rebuild. Historical outputs remain provenance; they are not current research results.

The practical question is whether a reusable frozen representation can offer a useful reliability–compute tradeoff against conventional task adaptation. Extraction has a first-run cost, even when later probes need no GPU. Architecture, input resolution, pretraining and adaptation all differ, so any observed difference cannot be attributed to medical pretraining alone. Negative or inconclusive comparisons are valid outcomes.

## What the project contributes

- Data-integrity review, physical-unit aggregation, conservative split grouping and explicit uncertainty.
- Matched development data, calibration/confidence arithmetic and planned paired evaluation.
- Tested offline contracts, immutable artifact bindings, access boundaries and bounded resource supervision.
- A record of scientific decisions, human gates, failed runs and actual compute—not a retrospective selection of successful outputs.

These are research-engineering aims and completed components, not claims of architectural novelty, clinical utility or a finished empirical contribution.

## Current progress — October 2026

| Milestone | Status |
|---|---|
| M0–M1: scope and contracts | Research design and invented-data contracts prepared |
| M2: data integrity and split | Approved physical split of 5,214 / 1,118 / 1,117 units; 8,505 active TRAIN/VAL views; uncertainties retained |
| M3: model-development runtime | Pinned CPU environment and synthetic model/data contracts verified |
| M4: bounded development compute | DenseNet completed 22 epochs, selected epoch 21; predictive CUDA reload matched exactly. MedSigLIP cache has 8,505 × 1,152 float32 features; private bindings/hashes/reload checked |
| M5: probes and calibration | CPU orchestration prepared; real fitting remains unapproved. Actual local memory readiness is a gate |
| M6: immutable systems/evaluator lock | Pending |
| M7: untouched internal evaluation | Pending separate unseal |
| M8: external DDI evaluation | Pending permissions, frozen binary systems and separate unseal |
| M9: reporting and interpretation | Pending completed evaluations |

**No rebuilt held-out or final comparison results are available.** Development validation is used for model selection and is not independent-test evidence. M4 completion does not imply M5 fitting, system freeze or clinical quality.

## Human and agent contributions

The user established the goals, approved scientific and compute boundaries, supplied/confirmed permitted source access, performed required visual audit decisions and controlled provider activation/shutdown. Agents implemented and tested contracts/drivers, inspected authorized evidence, prepared review proposals and documented failures/limits. Source-backed, agent-generated and human-adjudicated evidence remain distinct; an agent recommendation is not a human inspection attestation. Human authorization and review do not replace independent scientific evaluation.

## Data and artifact boundaries

This update adds reusable code, invented fixtures, dependency specifications and sanitized research documentation. It adds no research images, patient metadata, reviewer records, credentials, trained research checkpoints, feature caches or per-image outputs. Dataset/model access, source terms and real execution require separate permissions. Internal-test and DDI outcomes remain sealed during development.

The existing public history retains the original app, notebooks, sample images and legacy checkpoint. Their prior presence is not a new privacy/license clearance, a rebuilt baseline or endorsement for clinical use. This update does not delete or rewrite that history.

## Synthetic verification

The curated snapshot passed 281 CPU tests inside the OS allowlist. Its one JavaScript annotation-contract check passed separately with the existing Node runtime; no restricted scientific-worker permission was relaxed.

With a separately prepared environment matching the pinned model-development specifications:

```bash
python -B scripts/run_public_synthetic_tests.py
```

The JavaScript check needs an existing Node runtime. The core standard-library fixture demonstration is also available:

```bash
python3 -B scripts/run_synthetic_contract.py
```

Fixtures are invented software inputs, not research training data. Platform-specific isolation profiles must be regenerated and checked; synthetic verification does not authorize dataset access, real fitting or accelerator work.

## Limitations and remaining gates

Unknown duplicates may remain; recorded physical identities do not prove independent patients. The same validation units support selection, calibration and confidence cutoffs, creating optimistic development estimates. HAM dermoscopy and DDI clinical photographs differ in modality and population. Calibration cannot fix discrimination or establish safety; descriptive subgroup findings cannot prove fairness. There is no deployment threshold or clinical-use claim.

Earlier crashes, resource stops, diagnostic uncertainty and one missing historical recovery binary remain documented in the provenance summary. Successful continuation does not prove a crash's cause or future reliability. The next scientific step is the explicitly reviewed CPU M5 package, followed by system freeze and separate held-out evaluation. Publication of this code does not approve those experiments.
