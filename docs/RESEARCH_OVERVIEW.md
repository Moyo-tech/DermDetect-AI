# Research overview

## Motivation and background

DermDetect-R rebuilds a historical educational skin-lesion prototype into a controlled research workflow. A demo can display predictions without establishing independent generalization, reliable probabilities, clinical value or reproducible checkpoint provenance. The rebuild separates engineering verification from scientific evaluation and keeps original outputs as historical evidence.

Conventional transfer learning adapts a smaller ImageNet backbone to the task. A frozen medically oriented representation can reuse an expensive image-encoding pass and fit lightweight classifiers on CPU. Neither approach is automatically cheaper or more reliable: resolution, architecture, encoder size, extraction, backbone adaptation, probe selection, calibration and inference all contribute to first-run cost. Cached downstream cost is reported separately from that first-run total.

## Questions and interpretation

The primary question is the class-balanced performance, calibration, selective-prediction and compute comparison under the same leakage-aware HAM development/evaluation partitions. The secondary question is external benign-versus-malignant transfer and descriptive DDI subgroup heterogeneity after both systems are locked.

This is a comparison of complete systems. It has no matched non-medical foundation-model control, so it cannot identify medical pretraining as the sole mechanism. Macro-F1, proper scoring rules, class-specific behaviour, selective prediction and measured compute are interpreted together; disagreements are reported as tradeoffs. No model is called a clinical winner. Negative/inconclusive findings are retained after engineering/data integrity checks.

## Design in brief

The approved seven-class order is akiec, bcc, bkl, df, mel, nv, vasc. Recorded physical units guide membership and mean-image-logit aggregation. Exact-content duplicates have active representatives; conservative grouping for uncertainty does not merge physical identities. TRAIN class/physical/view weights distribute loss across classes and units while calibration and scoring remain equal physical-unit averages.

DenseNet121 uses one fixed task-adaptation schedule. MedSigLIP uses one pinned frozen image representation and a finite logistic-probe grid. Both selected systems receive the same physical-validation temperature/confidence procedures. The binary transfer task uses separately developed frozen-representation probes, not a sum of ambiguous seven-class probabilities. No internal-test/DDI data influences scaling, weights, model/C selection, calibration or thresholds.

Read [the detailed methods](METHODS_AND_STATUS.md) for approved versus proposed numerical choices, aggregation formulas, pinned model sources and current execution status. Methods for M5 and later stages remain proposals where indicated; their description is not an execution attestation.

## What is verified and what remains

M2 integrity/split contracts and M3 runtime preparation are completed within their inspected scopes. M4 completed bounded DenseNet development training and the immutable MedSigLIP development cache, with artifact integrity and reload checks. M5 orchestration has synthetic tests but has not performed real probe/calibration fits. M6 must lock all classifiers, temperatures, rules and evaluator before M7 internal or M8 external unseal. M9 integrates results, compute and limitations after evaluation.

The public code's software tests use invented metadata/text/pixels/features. They demonstrate implementation contracts, not diagnostic performance or complete duplicate discovery. Source permissions, exact data/artifact hashes, actual runtime/device isolation and approved resource budgets remain prerequisites for real reproduction.

## Agentic engineering and human scientific control

Agents support implementation, synthetic testing, authorized evidence inspection, failure diagnosis and structured decision proposals. The user controls research direction, permissions, bounded execution, policy acceptance and required human image-review decisions. Provider shutdown is user-attested where no independent control is available; it is not called invoice verification.

Source findings, agent recommendations and human decisions are recorded separately. Unclear findings are preserved rather than forced into certainty. Neither extensive automation nor human approval makes an untested scientific result valid. Independent held-out evaluation and transparent reporting are still required.

## Failure and provenance policy

Keep failed attempts and uncertain explanations. Two native training crashes preceded a timer-only wrapper change and successful continuation; causality was not demonstrated. Local diagnostic failures and CPU waiting overruns were retained. One earlier crash-recovery binary was overwritten before archival; its hash/inspection evidence remained, while later generations were preserved. There is no claim of perfect historical preservation.

The original public educational assets are inherited unchanged, with unverified privacy/license/performance lineage. New private data/reviews/artifacts and operational identifiers are excluded from this update. Read [provenance and reproduction limits](PROVENANCE.md) for the publication boundary.
