# Study methods and development status

This is an ongoing, resource-constrained research comparison of two complete systems: an ImageNet-pretrained DenseNet121 with task adaptation, and frozen MedSigLIP image embeddings with a CPU logistic probe. Architecture, input resolution, pretraining data and adaptation differ, so the comparison cannot establish medical pretraining as the causal mechanism. No clinical, fairness or diagnostic-readiness claim is made.

## Data units and leakage controls

HAM10000 is the development source, identified by Harvard Dataverse DOI[10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T). Original records are preserved. Active representatives retain one exact-content representative per recorded physical-analysis unit; byte-distinct views remain separate inputs. Original lesion identifiers, adjudicated physical-analysis identities and conservative split-only groups are distinct concepts. Unclear identity can impose split co-location without merging physical identities or changing labels.

The accepted physical-unit split is5,214TRAIN /1,118VAL /1,117internal-test units, seed20260729. Active development inputs total7,000TRAIN and1,505VAL views. No internal-test or DDI outcomes enter development. Stratification, fixed quotas, content/group overlap checks, source isolation, membership/hash verification and human audit are implemented. Historical audits, uncertain decisions and failures remain private evidence; this repository does not redistribute their per-image records or reviewed images.

The duplicate process is sensitivity-limited. Published cross-ID candidate evidence, a selected similarity queue and required human uncertainty/conflict/audit checks do not establish complete duplicate discovery or patient independence. Outside screened pairs remain unadjudicated. Recorded physical units are the aggregation/scoring units, not authenticated biological identity or independent patients.

Within a physical unit, average active-image logits before model selection, calibration and scoring. Do not average calibrated probabilities or merge uncertain physical identities. Calibration and point estimates give physical units equal weight; dependence-aware intervals are planned using paired conservative groups while preserving their separate physical identities.

For seven-class training, let M be active TRAIN views, n_c TRAIN physical units in class c, and m_l active views in unit l. Apply per-view weight w_li=M/(7*n_c*m_l) once. Their mean is1: each class contributes equally, and each unit within class has equal total contribution spread across its views. No additional class_weight, weighted sampling, inactive-copy contribution or validation-derived class weights. Labels are ordered[akiec,bcc,bkl,df,mel,nv,vasc].

## Models and completed M4 development work

DenseNet121 uses the torchvision IMAGENET1K_V1 preprocessing: RGB, shorter-edge resize256, center crop224, bilinear resize and its fixed ImageNet normalization. TRAIN-only seeded horizontal/vertical flips and right-angle rotations precede that transform. A dropout.2 and seven-logit head are trained for2head epochs, then at most20epochs adapting denseblock4 and norm5. AdamW has weight decay.0001; head learning rate.001 and final-block rate.0001. The phase transition retains head optimizer state while adding final-block parameters. Frozen BatchNorm statistics remain frozen. Batch16, float32, deterministic algorithms and TF32off are fixed.

Select a checkpoint by physical-VAL macroF1 descending, NLL ascending, then earlier epoch. Patience5counts every evaluated epoch. The completed development run executed22epochs/9,636updates and selected epoch21. Predictive reload on1,118physical VAL units matched exactly. These are engineering/development facts, not independent-test performance. The model's stored physical-validation logits can support later seven-class calibration after exact binding verification.

MedSigLIP is pinned to revision9cea28a1a1195f665105faa6e8544c112fd960a4 of[google/medsiglip-448](https://huggingface.co/google/medsiglip-448/tree/9cea28a1a1195f665105faa6e8544c112fd960a4). Acquisition and model access were separately authorized. Frozen batch1 extraction uses the pinned official image-feature path,448RGB processor contract, evaluated CPU-source resize arithmetic and L2-normalized1152-dimensional float32 image embeddings. The accepted cache has exactly8,505ordered development IDs; hashes/binding, dtype/shape, finite values, norms, nonzero variance and reload were checked. No MedSigLIP fine-tuning or alternate preprocessing search is in scope.

Both checkpoints and caches remain private. No images, real metadata, weights, features, predictions or evaluator outcomes are shipped here. Historical public demo files are separately identified and are not these rebuilt artifacts.

## M5 settings awaiting real-execution approval

MedSigLIP's seven-class probe uses train-only StandardScaler and multinomial L2 logistic regression with C={.01,.1,1,10}, seed20260729, lbfgs, proposed max_iter1000/tol1e-4. The proposed scaler is unweighted over active TRAIN views; the classifier receives the physical sample weights once. Selection is physical-VAL macroF1 descending, NLL ascending, then smaller C. Pinned sklearn1.7.2 lbfgs normalizes its L2 strength by C×sum(sample_weight); the mean-one convention preserves the usual sample-count C interpretation. Nonconvergence stops for review rather than expanding iterations/grid.

Separate binary probes are proposed on cached MedSigLIP1152features and global-pooled1024features from the selected seven-class DenseNet. Proposed benign labels are nv/bkl/df/vasc; malignant labels mel/bcc; akiec is excluded only from binary work because its combined source category is ambiguous. Binary mapping, recomputed K=2 physical weights and identical scaling/grid/selection need approval. Summing seven-class probabilities is not a substitute.

After selection, fit one positive scalar temperature for each of Dense7,Med7,Dense2,Med2 on equal-unit physical-VAL NLL. Proposed log-temperature bounds[-5,5], bounded scalar tolerance1e-8; failed/nonfinite optimization or a boundary solution stops for review. The proposed fixed binary classification rule is calibrated malignant probability≥.5, equality malignant. It is a research rule, not a clinical or prevalence-optimal threshold.

Confidence is maximum calibrated probability. Proposed90%/80%headline coverage cutoffs use the ceil(q*n)-th largest VAL confidence and retain all ties; realized coverage may exceed target.100%is the no-abstention reference;70%is supplementary. Test/DDI report realized coverage without adjusting thresholds. ECE uses15equal-width confidence bins, with empty bins contributing zero; NLL and multiclass-sum Brier remain proper-scoring anchors.

The same VAL units support Dense epoch selection, probe C selection, calibration and confidence rules. Development estimates are optimistic; no outcome-driven split, larger search or threshold change is allowed. Negative/inconclusive findings remain valid after engineering verification. Both seven-class and binary systems must be frozen before coordinated internal-test or external DDI unseal.

## Compute, isolation and remaining work

M4 used bounded approved local/cloud development work with explicit manual shutdown. Two native training crashes, local diagnostic failures, a timer-only wrapper change and CPU waiting overruns were retained rather than concealed. The timer change's successful continuation does not prove the crashes' cause or future crash freedom. A missing historical crash-recovery binary is documented; its recorded hash/inspection evidence remains, and later generations were preserved. No elapsed-time or credit number is treated as an independently reconciled invoice.

M5 has synthetic CPU orchestration/portable reload tests and OS allowlist checks. Proposed execution is localCPU only: a16TRAIN-image benchmark, conditional8505×1024DenseNet extraction,12fixed-C candidate fits,3TRAIN-only diagnostics, four temperatures and rules under an explicit resource/time cap. Real fitting is unapproved. Actual local memory readiness remains a gate; a synthetic launch test with simulated resource samples is not actual-host resource clearance.

The worker boundary restricts file reads to staged approved inputs/runtime roots, writes to the current stage output and denies network/helpers/protected sources. It is tested trusted-worker isolation, not owner-proof or hostile-code containment. Runtime/device changes require renewed verification. M6freeze, held-out/internal/DDI evaluation, reporting and publication claims retain separate gates. No M5fit or held-out outcome is implied by this code release.
