# Model card — DermDetect-R locked systems (research only)

**Status:** research artifacts from a controlled comparison, locked 22 November 2026 under joint lock `560952b7…`. **Not for clinical use.** They are not medical devices and have not been validated for diagnosis, triage or screening.

## Systems

| ID | Task | Backbone | Head | Calibration | Frozen rules |
|---|---|---|---|---|---|
| dense7 | HAM seven-class | DenseNet121, ImageNet init, final block + head fine-tuned (epoch 21), 224 px | linear (in network) | T = 1.729 | 90%/80% cutoffs 0.522 / 0.645 |
| med7 | HAM seven-class | MedSigLIP rev `9cea28a1…`, frozen, 448 px, 1,152-d | logistic regression, C = 0.01, train-only scaler | T = 0.965 | 0.572 / 0.701 |
| dense2 | bcc/mel vs benign | dense7 checkpoint, frozen 1,024-d features | logistic regression, C = 0.01 | T = 1.552 | malignant if p ≥ 0.5; 0.646 / 0.773 |
| med2 | bcc/mel vs benign | MedSigLIP, frozen | logistic regression, C = 1 | T = 1.775 | malignant if p ≥ 0.5; 0.665 / 0.806 |

**Common to all four systems:**
- Image logits are averaged within each lesion before temperature scaling.
- Confidence is the maximum calibrated probability.
- A prediction is accepted if confidence ≥ cutoff (ties included).

**Parameters:**
- DenseNet121: 6.96 M total. Trainable: 7.2 k in the head phase, 2.17 M in the final-block phase.
- MedSigLIP vision tower: 428.6 M, frozen.
- Probes: about 1–8 k parameters each.

## Intended use

Reproducible research on transfer learning, calibration and selective prediction for dermoscopic lesion classification under limited compute.

**Out of scope:**
- any clinical decision;
- patient-facing use;
- photographs or other modalities without new validation;
- akiec in the binary systems;
- populations or devices beyond the evaluated data.

## Training data

HAM10000 train partition only (5,214 lesion units). Selection and calibration used the validation partition (1,118 units). See the [data statement](data_statement.md).

## Evaluation (single locked runs)

| | dense7 | med7 | dense2 | med2 |
|---|---:|---:|---:|---:|
| HAM test macro-F1 | 0.644 | 0.710 | 0.736 | 0.793 |
| HAM test Brier (multiclass / binary) | 0.274 | 0.227 | 0.091 | 0.081 |
| HAM test ECE | 0.054 | 0.026 | 0.027 | 0.024 |
| HAM test AUROC (macro / binary) | 0.952 | 0.971 | 0.897 | 0.924 |
| DDI AUROC (656 clinical photos) | — | — | **0.583** | **0.447** |
| DDI binary Brier | — | — | 0.249 | 0.339 |

Intervals, per-group results and caveats are in the [final report](final_experiment_report.md).

## Known failure modes and caveats

**Domain shift:**
- Both binary systems perform poorly on clinical photographs.
- med2 ranks DDI images below chance.
- Both are over-confident on DDI (ECE 0.20 / 0.31).

**Class-specific weaknesses:**
- Melanoma recall on the HAM test is 0.64 (dense7) and 0.78 (med7).
- Many nv lesions are predicted as mel: 65 (dense7) and 62 (med7).
- df and vasc results rest on 11 and 15 test lesions.

**Skin tone:** DDI estimates differ by Fitzpatrick group, with both systems lowest in V–VI. These figures are descriptive, have wide intervals, and are not a fairness assessment.

**Evidence base:** one training seed and one split, with validation reused for selection and calibration.

**System differences:** the two systems differ in resolution, architecture, adaptation method and pretraining data. Differences are system-level.

## Compute

- **Training (L4 accelerator):** DenseNet 6,442 s; MedSigLIP extraction 1,959 s.
- **CPU inference per image (2 threads):** MedSigLIP about 6.8–7.9 s; DenseNet about 0.1 s.
- **Weights:** MedSigLIP 3.51 GB; DenseNet checkpoint 49 MB.

## Licences and terms

- **MedSigLIP:** Health AI Developer Foundations terms, accepted by the user.
- **DenseNet121:** torchvision ImageNet weights `IMAGENET1K_V1` (`densenet121-a639ec97.pth`, pinned).
- **HAM10000:** CC BY-NC 4.0, via Harvard Dataverse.
- **DDI:** used under its research terms (personal, non-commercial, non-clinical; no redistribution). No DDI images or per-image outputs are stored with these artifacts.
- **Redistribution:** weights, probes and caches stay private. Redistributing them needs a separate licence and privacy review.
