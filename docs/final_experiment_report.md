# DermDetect-R final experiment report

**23 November 2026. Results from single locked runs; not peer-reviewed.**

This is a matched comparison of two transfer-learning systems for skin-lesion classification under limited compute:
- an ImageNet-pretrained DenseNet121 adapted on HAM10000;
- a logistic-regression probe on frozen MedSigLIP image embeddings.

Both were developed on one leakage-safe HAM10000 split, locked, evaluated once on an untouched HAM internal test, and then evaluated once on the external Diverse Dermatology Images (DDI) set.

> **No clinical, diagnostic, safety or fairness claim follows from this work.** Dataset labels are research labels, not diagnoses. The DDI skin-tone groups are descriptive.

Companion documents:
- [model card](model_card.md)
- [data statement](data_statement.md)
- [reproducibility and evidence map](reproducibility.md)
- result tables in `docs/results/` and figures in `docs/figures/`, all built by `scripts/m9/build_m9_tables.py`

## 1. Summary

| Question (prespecified) | Result |
|---|---|
| H1 Internal performance vs task-adaptation compute | MedSigLIP probe is **better**, not just comparable, on seven-class macro-F1 (0.710 vs 0.644; paired 95% CI of the difference **+0.018 to +0.111**). It used about **3.3× less accelerator allocation** for adaptation (1,959 s vs 6,442 s on an NVIDIA L4). On CPU, however, MedSigLIP **inference is about 70–80× slower** (7.9 s vs 0.095 s per image) and its weights are **about 70× larger**. |
| H2 Probabilistic quality after calibration (seven-class Brier) | Supported: Brier 0.227 vs 0.274; difference CI **−0.074 to −0.019**. ECE (0.026 vs 0.054) and NLL are also lower. In the binary task the Brier difference is inconclusive (CI −0.043 to +0.005). |
| H3 Selective prediction | For both systems, error falls as coverage falls. MedSigLIP has the lower AURC (0.039 vs 0.058 seven-class; 0.027 vs 0.030 binary). This is descriptive; no AURC interval was prespecified as key. |
| H4 External transfer (DDI) | Strongly supported, and stronger than expected. Binary AUROC drops from 0.897 / 0.924 (HAM) to **0.583** (DenseNet) and **0.447** (MedSigLIP) on DDI. MedSigLIP is **below chance** (CI 0.400–0.494) and worse than DenseNet (difference CI **−0.201 to −0.077**). |
| H5 Skin-tone heterogeneity (exploratory) | Point estimates differ across Fitzpatrick groups: both systems are lowest in V–VI and DenseNet is highest in III–IV. Intervals are wide and overlap. **No fairness conclusion.** The prespecified I–II vs V–VI contrast interval was not computed (§7, deviation D1). |

In one sentence: frozen MedSigLIP embeddings gave a more accurate and better-calibrated HAM dermoscopy classifier for less training compute, but cost far more at inference on CPU and transferred worse than DenseNet121 to clinical photographs in DDI.

## 2. Background and research questions

The historical DermDetect prototype reported 91.68% "accuracy". The [claim ledger](PROVENANCE.md) shows this was validation-set arithmetic from a saved matrix, with:
- validation data reused as the test set;
- unverified lesion and patient separation;
- non-reproducible optimisation;
- training and app preprocessing that don't match.

DermDetect-R does not repair that number. It replaces it with a protocol-governed rebuild ([experiment protocol](METHODS_AND_STATUS.md); approvals in the human review log (private record)).

- **Primary question:** under limited compute, how do frozen MedSigLIP embeddings and an ImageNet-pretrained DenseNet121 compare in class-balanced performance, calibration, selective prediction and end-to-end compute, on the same leakage-safe HAM10000 partitions?
- **Secondary question:** how robust are the locked systems on an external clinical-image dataset (DDI), and how does benign-vs-malignant performance vary across DDI's Fitzpatrick groups?

This compares **complete systems**. Architecture, input resolution (448 vs 224 px), adaptation method (frozen probe vs final-block fine-tuning) and pretraining data all differ, so no difference can be attributed to medical pretraining alone.

## 3. Data

Full details are in the [data statement](data_statement.md).

**HAM10000** (Harvard Dataverse DOI 10.7910/DVN/DBW86T):
- 10,015 dermoscopic images, seven labels.
- Integrity review covered exact and near duplicates, 665 candidate image pairs reviewed with source, agent and human attribution, and 15 identity links kept as "unclear".
- Result: 7,449 physical lesion units.
- Group-aware split with approved targets 70/15/15 and seed 20260729:
  - train: 5,214 units
  - validation: 1,118 units
  - internal test: 1,117 units (1,508 images)
- The split was sealed before any modelling (M2). The test set stayed sealed until the joint lock (M6) was written.

**DDI** (official release; terms accepted by the user):
- 656 clinical photographs: 485 benign, 171 malignant (official `malignant` field).
- Fitzpatrick groups: I–II 208, III–IV 241, V–VI 207.
- No patient identifier is available, so the evaluation unit is the image.
- Used only once, for one locked evaluation. Nothing was trained, tuned or calibrated on DDI.

ISIC 2018 was excluded entirely.

## 4. Methods

**DenseNet121** (M4):
- ImageNet weights, 224 px.
- Training: 2 head epochs, then up to 20 final-block epochs. Final block is `denseblock4` + `norm5` + head: 2.17 M trainable of 6.96 M total parameters.
- AdamW learning rates 1e-3 / 1e-4 (weight decay 1e-4), batch 16, float32, deterministic algorithms, seed 20260729, early-stopping patience 5.
- Selection: validation physical-unit macro-F1, then NLL, then earlier epoch. Epoch 21 was selected.
- Trained on an NVIDIA L4. Two native crashes were recovered from checkpoints under approved stopping rules.

**MedSigLIP** (M4/M5):
- Pinned revision `9cea28a1…`, official processor, 448 px.
- 1,152-d frozen image embeddings for all 8,505 train and validation images, extracted once on an L4 into an immutable cache.
- Multinomial or binary logistic regression with a train-only scaler.
- C selected from a prespecified grid on validation: seven-class C = 0.01, binary C = 1.

**Binary systems** (for DDI):
- Malignant = bcc or mel; benign = nv, bkl, df or vasc; akiec excluded.
- DenseNet binary is a logistic probe (C = 0.01) on the selected checkpoint's 1,024-d CPU features.
- Rule: malignant if calibrated malignant probability ≥ 0.5.

**Shared rules** for all four systems:
- Image logits are averaged within each physical lesion before any calibration.
- One temperature per system, fitted by NLL on validation lesions: dense7 1.729, med7 0.965, dense2 1.552, med2 1.775.
- Confidence = maximum calibrated probability.
- Abstention cutoffs target 90% and 80% validation coverage, then are frozen.

**Locking and evaluation:**
- **Joint lock** `560952b7…` (M6) froze all four systems and the evaluator before any held-out outcome was seen.
- **DDI add-on lock** `126ff64f…` (M8) froze the DDI adapter and rules before any DDI label was read.
- **Held-out runs:** each was one supervised run with no retry. Metrics were recorded before any error review.
- **HAM intervals:** 1,000 paired, class-stratified conservative-group draws.
- **DDI intervals:** 1,000 paired, label-stratified image draws (overall and within each group).
- All intervals are 95% percentile intervals, seed 20260729.

The full numerical policy is `4cc2dcf4…`.

## 5. Results

### 5.1 HAM internal test, seven-class (1,117 lesions)

| Metric | DenseNet121 | MedSigLIP | Med − Dense 95% CI |
|---|---:|---:|---|
| **Macro-F1 (primary)** | 0.644 [0.598, 0.692] | 0.710 [0.664, 0.756] | **[+0.018, +0.111]** |
| **Multiclass Brier (primary, lower better)** | 0.274 [0.249, 0.300] | 0.227 [0.201, 0.253] | **[−0.074, −0.019]** |
| Balanced accuracy | 0.698 | 0.779 | |
| Accuracy | 0.811 | 0.835 | |
| MCC | 0.626 | 0.690 | |
| Macro AUROC | 0.952 | 0.971 | |
| Macro AP | 0.667 | 0.799 | |
| NLL | 0.530 | 0.434 | |
| ECE (15 bins) | 0.054 | 0.026 | |
| AURC | 0.058 | 0.039 | |
| Coverage / error at 90% cutoff | 90.2% / 0.160 | 87.0% / 0.104 | |
| Coverage / error at 80% cutoff | 80.9% / 0.124 | 77.7% / 0.071 | |

**Per-class F1**, DenseNet → MedSigLIP (test support in brackets):

| akiec (34) | bcc (49) | bkl (108) | df (11) | mel (92) | nv (808) | vasc (15) |
|---|---|---|---|---|---|---|
| 0.61 → 0.73 | 0.65 → 0.72 | 0.66 → 0.72 | 0.38 → 0.41 | 0.49 → 0.60 | 0.910 → 0.911 | 0.81 → 0.88 |

- Melanoma recall was 0.64 (DenseNet) and 0.78 (MedSigLIP).
- df and vasc have very small support, so their per-class values are unstable.
- See `results/ham_seven_class_per_class.csv` and `figures/confusion_ham_seven_class.svg`.

### 5.2 HAM internal test, binary (1,083 lesions; akiec excluded)

| Metric | DenseNet121 | MedSigLIP |
|---|---:|---:|
| AUROC | 0.897 | 0.924 |
| AP | 0.585 | 0.674 |
| Macro-F1 | 0.736 | 0.793 (diff CI **[+0.020, +0.093]**) |
| Sensitivity / specificity | 0.631 / 0.901 | 0.759 / 0.913 |
| Binary Brier | 0.091 | 0.081 (two-class-sum diff CI [−0.043, +0.005], inconclusive) |
| ECE / AURC | 0.027 / 0.030 | 0.024 / 0.027 |

### 5.3 DDI external evaluation (656 images)

| Metric | DenseNet121 | MedSigLIP | Med − Dense 95% CI |
|---|---:|---:|---|
| **AUROC (primary)** | **0.583** [0.530, 0.634] | **0.447** [0.400, 0.494] | **[−0.201, −0.077]** |
| AP (prevalence 0.261) | 0.358 [0.306, 0.420] | 0.235 [0.216, 0.268] | [−0.181, −0.066] |
| Binary Brier | 0.249 [0.226, 0.270] | 0.339 [0.314, 0.366] | [+0.061, +0.123] |
| Sensitivity / specificity | 0.292 / 0.821 | 0.222 / 0.707 | |
| Balanced accuracy / F1 | 0.557 / 0.325 | 0.465 / 0.217 | |
| NLL / ECE / AURC | 1.145 / 0.202 / 0.241 | 1.557 / 0.306 / 0.363 | |
| Coverage / error at 90%, 80% cutoffs | 89.6% / 0.294; 77.3% / 0.266 | 87.7% / 0.400; 74.8% / 0.385 | |

**By Fitzpatrick group (AUROC; descriptive only):**

| Group (images; malignant) | DenseNet | MedSigLIP | Med − Dense |
|---|---:|---:|---|
| I–II (208; 49) | 0.561 [0.461, 0.646] | 0.498 [0.407, 0.591] | [−0.197, +0.081] |
| III–IV (241; 74) | 0.647 [0.570, 0.725] | 0.433 [0.354, 0.515] | [−0.318, −0.110] |
| V–VI (207; 48) | 0.511 [0.420, 0.603] | 0.354 [0.268, 0.448] | [−0.259, −0.057] |

- Descriptive I–II minus V–VI AUROC differences are +0.050 (DenseNet) and +0.144 (MedSigLIP). No interval was computed (D1).
- Every group contained both classes, so no metric was undefined.
- See `results/ddi_metrics.csv` and `figures/auroc_ham_vs_ddi_groups.svg`.

### 5.4 Calibration and selective behaviour

Figures: `figures/reliability_diagrams.svg` and `figures/risk_coverage_ham.svg`.

- **On HAM:** both systems are close to the diagonal at high confidence. MedSigLIP seven-class tracks it more closely. Its realized test coverage (87.0%, 77.7%) fell slightly below the validation targets, so the frozen cutoffs proved mildly conservative. This is reported as is.
- **On DDI:** both systems are over-confident (ECE 0.20 and 0.31), as expected when a HAM-calibrated temperature meets a different domain. DDI risk-coverage curves were not retained, because per-image curve points were stripped under aggregate-only retention.

## 6. Compute and cost

**Locked inference** (CPU, Runpod cpu3m, 4 vCPU/32 GB, 2 threads, zero GPU-seconds):

| | HAM test (1,508 images) | DDI (656 images) |
|---|---:|---:|
| MedSigLIP encoding | 11,854 s (7.86 s/image) | 4,491 s (6.85 s/image) |
| DenseNet encoding | 143 s (0.095 s/image) | 65 s (0.099 s/image) |
| Metrics + bootstrap | 24.8 s | 2.7 s |
| Peak worker RSS | 4.41 GB | 4.39 GB |

**Development:**

| Stage | Resource | Measured |
|---|---|---|
| DenseNet training (22 epochs, incl. 2 crash recoveries) | Lightning AI Studio, NVIDIA L4 (23 GB) | 6,442 s paid allocation (1.79 h) |
| MedSigLIP extraction (8,505 images) | same L4 | 1,959 s paid allocation (0.54 h); encoding 1,413 s |
| Earlier smoke tests, failed or stopped accelerator attempts, local MPS | Lightning cloud and Mac MPS | 2,511 s |
| **Total accelerator allocation (M4)** | | **10,912 s = 3.03 h** of the 5.0 GPU-h protocol ceiling |
| DenseNet CPU features for the binary probe | Lightning free 4-CPU | 2,126 s |
| All 12 probe candidates, 4 temperatures, cutoffs (both systems) | same | 99.7 s |

**Cached downstream adaptation:** refitting every MedSigLIP and DenseNet-feature probe from existing embedding caches took under 100 s of CPU in total. Retraining DenseNet121 end to end would mean repeating the GPU training.

**Storage:**

| Artifact | Size |
|---|---|
| MedSigLIP weights (vision tower 428.6 M params; checkpoint also holds the 449.7 M-param text tower) | 3.51 GB |
| DenseNet checkpoint (6.96 M params, incl. optimizer state) | 49.4 MB |
| MedSigLIP embedding cache | 39.2 MB |
| DenseNet feature cache (8,505 × 1,024 float32) | about 34.8 MB |

**Money**, by balance difference (not finalized invoices; Lightning figures are user-attested):

| Provider and period | Spend |
|---|---|
| Lightning AI, M4 GPU (balance 21.61 → 19.61) | 2.00 credits |
| Lightning AI, before the M5 window (19.61 → 19.34; storage/billing lag, unreconciled) | 0.27 credits |
| Lightning AI, M5 CPU (rated 0 credits/h; final balance not independently verified) | n/a |
| Runpod, 17–18 Nov engineering and benchmark attempts with no scientific output ($10.00 → $8.0677, incl. any storage until 22 Nov) | $1.93 |
| Runpod, M6 actual-model compatibility check (3 attempts) | $0.09 |
| Runpod, M7 HAM internal test | $0.81 |
| Runpod, M8 DDI | $0.36 |
| **Total** | **about 2.27 Lightning credits + $3.19 Runpod** |

**Efficiency, read per the protocol's definition:**
- **First-run adaptation:** MedSigLIP needed about a third of DenseNet's accelerator time. DenseNet's figure is inflated by crash recovery; clean epoch cost was not separately measured.
- **Cached re-adaptation:** MedSigLIP probes refit in seconds on CPU.
- **Inference:** MedSigLIP is about 70–80× more expensive per image on CPU.
- **Conclusion:** neither system is cheaper end to end in general. Which one wins depends on how many images will be scored.

## 7. Deviations from the protocol (disclosed)

**D1 — I–II vs V–VI contrast interval not computed.**
- What was missed: protocol §14–15 prioritised a 95% interval for the descriptive Fitzpatrick I–II vs V–VI contrast. The locked DDI evaluator computed per-group and paired system intervals but not this contrast. The omission was made when the adapter was built (agent error) and was found while drafting this report.
- Why it can't be fixed: per-image predictions were deliberately not retained, so the interval can't be computed after the fact without a new DDI run, which the protocol does not permit.
- What is reported instead: point differences only. No conclusion depends on the contrast.

**D2 — Patient-count instability rule not applicable.** DDI has no patient IDs. The rule "fewer than 10 independent patients per class" could not be checked. Every subgroup AUROC and AP is treated as descriptive, with about 48–74 malignant images per group.

**D3 — DDI image-level intervals.** These ignore within-patient dependence, as the protocol foresaw, and may be too narrow.

**D4 — Compute accounting is approximate.**
- DenseNet's L4 time is paid allocation, not GPU-busy time, and includes crash recovery.
- Peak GPU memory, energy and carbon are not reported.
- Money figures are balance differences. The Lightning figures are user-attested.

**D5 — Cloud runs without OS sandboxing.** The M7 and M8 cloud runs used a trusted-process runner without OS-level sandboxing. This was explicitly acknowledged and approved; it was not a silent weakening.

## 8. Limitations

- **Same validation reuse:** the same HAM validation units chose checkpoints, probes, temperatures and cutoffs, so development estimates are optimistic. The test results are the unbiased ones.
- **Single seed:** one DenseNet training seed and one split, so no estimate of training-seed variance.
- **Rare classes:** df (11 test units) and vasc (15) have very wide implicit uncertainty.
- **Residual leakage risk:** 15 identity links were kept as unclear (co-located in one split, not merged). 30,072 broad candidate pairs outside the reviewed set remain unadjudicated. The authors' 960-d duplicate ranking could not be reproduced.
- **Pretraining overlap:** MedSigLIP's pretraining corpus may include public dermatology images related to HAM10000 or DDI. Overlap was not and cannot be audited here. ImageNet overlap is negligible but unverified.
- **System-level confounds:** see §2.
- **Domain shift:** DDI is clinical photography with a different disease mix. The DDI result measures transfer under modality and task shift, not performance on the task the systems were built for.
- **Unexplained below-chance result:** MedSigLIP's below-chance DDI AUROC is not explained.
  - A label-orientation error is unlikely, because DenseNet uses the same label pipeline and ranks normally. Without per-image predictions it can't be fully ruled out.
  - Plausible causes, all untested: HAM-specific probe directions, the benign DDI disease mix, and 448 px processing of phone and clinical photos.
  - Under the lock, no refit, recalibration or rerun is allowed.
- **Not clinical:** research labels, a retrospective design and no prospective or reader study. Confidence scores are operational, not clinical.

## 9. Negative and inconclusive findings kept on record

- Binary HAM Brier difference: inconclusive.
- MedSigLIP seven-class temperature scaling barely improved validation NLL, and slightly worsened validation Brier and ECE (M5). The test-set deficit did not appear.
- MedSigLIP realized test coverage fell below the validation targets.
- Both systems fail to transfer to DDI, and MedSigLIP fails worse.
- Earlier engineering failures are preserved in the dated updates (private record): memory-guard stops, crashes, descriptor and host defects, and stopped cloud attempts.

## 10. Conclusion

On leakage-safe HAM10000 dermoscopy, frozen MedSigLIP embeddings with a linear probe outperformed a fine-tuned DenseNet121:
- higher macro-F1;
- lower Brier;
- better selective behaviour;
- about a third of the accelerator training time.

The costs show up elsewhere: CPU inference about 70–80× slower and weights about 70× larger.

On DDI clinical photographs, both locked systems lost most of their discrimination. DenseNet121 kept weak above-chance ranking; MedSigLIP fell below chance.

The internal advantage of medical foundation-model embeddings therefore did not carry over to this external clinical-image set. Any practical use would need domain-matched validation, which this study does not provide.
