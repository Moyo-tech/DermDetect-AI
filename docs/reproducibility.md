# Reproducibility and evidence map — DermDetect-R

This document shows how each reported number traces to a saved, hash-bound record, and what can be re-checked without new experiments. Held-out HAM and DDI evaluations were single locked runs. **Do not re-run them to "reproduce" results**: the protocol forbids outcome-driven reruns.

## 1. Environment

| Use | Python | Pins |
|---|---|---|
| Local development, tests, report build | 3.11.13 (`env/m3-model-dev`) | `requirements-m3-model-dev.txt`, hash lock `requirements-m3-model-dev.lock.txt` (macOS arm64) |
| Cloud CPU inference (M6–M8) | 3.11.13 | 47 Linux distributions: `outputs/m8-ddi-run-20261123/setup_input/requirements-linux.lock.txt` (`uv`-installed, version-checked at runtime) |
| HAM integrity (M2) | 3.11 (`env/m2-integrity`) | `requirements-m2-integrity.txt` (Pillow 12.3.0) |

**Model artifacts** (private, not in Git):
- MedSigLIP revision `9cea28a1a1195f665105faa6e8544c112fd960a4`, `model.safetensors` SHA `3e1722b8…`.
- DenseNet selected checkpoint `outputs/m4-trained-private-handoff-20261004/best_epoch_021.pt`, SHA `5cfda594…`.

## 2. Checks anyone with the private outputs can run (no model, no new data)

Run from the project root.

1. **Regression suite** (647 tests; invented data only). Last run 23 Nov 2026: `Ran 647 tests … OK` in about 20 s.

```bash
PYTHONPATH=src:tests env/m3-model-dev/bin/python -m unittest discover -s tests
```

2. **M7 metric replay.** This recomputes every recorded HAM test metric and both paired-interval sets from the retrieved physical predictions, using the locked evaluator and frozen temperatures and cutoffs. Last run: `all_exact: true` for all four systems and both interval sets (about 18 s).

```bash
env/m3-model-dev/bin/python -I scripts/m9/replay_m7_metrics.py
```

3. **Report tables and figures.** This rebuilds `results/*.csv` and `figures/*.svg` from the hash-checked M7 and M8 records. It writes `results/m9_build_manifest.json` with the input and output hashes.

```bash
python3 -I scripts/m9/build_m9_tables.py
```

**DDI cannot be replayed at the prediction level.** Under the user-chosen aggregate-only retention, no per-image predictions exist. The DDI tables are re-derived from the hash-checked `metrics.json`, which is bound by receipt `74805d76…`.

## 3. Evidence map

| Claim area | Primary evidence | Binding |
|---|---|---|
| Split and lesion identities | `outputs/m2-production-closeout/production_seal.json`; M2 closeout (private record) | 5,214 / 1,118 / 1,117 units; seal recorded in M2 |
| DenseNet training and selection | `outputs/m4-full-execution-proposal/MedSigLIP_next_preparation/completed_private_handoff_closeout_pending.json`; checkpoint above | epoch 21; CUDA reload difference 0.0 |
| MedSigLIP embedding cache | `outputs/m4-Med-private-cache-20261004/`; verification `outputs/m4-Med-private-cache-verification-20261004/verification.json` | features `bde1f577…`, metadata `8feb14d1…` |
| Probes, temperatures, cutoffs | `outputs/m5-CPU-reactivation-execution-20261007/M5_summary_for_M6_proposal.json`; M5 closeout (private record) | 71 files hash-verified |
| CPU route equivalence | dated update, 22 Nov (private record) (M6 compatibility check) | DenseNet exact; MedSigLIP ≤ 7.4e-7 vs GPU cache |
| Joint lock (all four systems + evaluator) | `outputs/m6-rebind-20261122/joint_lock.json` | `560952b7…`; scientific digest `3398a83c…`; numerical policy `4cc2dcf4…` |
| HAM internal-test results | `outputs/m7-run-20261122/retrieved/m7_output/internal_test/` | receipt `f704be72…`; metrics `05d3cd36…`; predictions `4471d112…`; compute `be423658…` |
| DDI add-on lock | `outputs/m8-ddi-lock-20261123/ddi_addon_lock.json` | `126ff64f…`; evaluator `af158446…`; integrity `3b15b22a…` |
| DDI results | `outputs/m8-ddi-run-20261123/retrieved/dermdetect_runpod/m8_output/ddi/` | receipt `74805d76…`; metrics `eeb35776…`; compute `1241f943…`; run approval `68bee0a2…` |
| M4 GPU time and credits | `outputs/m4-full-execution-proposal/MedSigLIP_next_preparation/M4_user_attested_resource_cost_closeout.json` | 10,911.68 s; 21.61 → 19.61 credits (user-attested) |
| M5 CPU time | `outputs/m5-CPU-reactivation-execution-20261007/completed_private_handoff_Sleep_pending.json` | features 2,125.7 s; fits 99.7 s |
| Runpod spend | `benchmark_provider.json`/receipts in `outputs/runpod-*`, `outputs/m7-run-20261122/`, `outputs/m8-ddi-run-20261123/`; final balance read 23 Nov ($6.8151) | first recorded balance $10.00 (17 Nov) |
| Approvals and authority | human review log (private record), decision log (private record) | user's words recorded per gate |
| Historical-claim corrections | [claim ledger](PROVENANCE.md) | |
| Report tables and figures | `results/`, `figures/`, `results/m9_build_manifest.json` | input and output SHA-256 |

## 4. Rebuilding the systems from scratch

Rebuilding is possible in principle but needs:
- the HAM files;
- the pinned weights, with acceptance of MedSigLIP's terms;
- the M2 sealed manifests;
- an accelerator for DenseNet training and MedSigLIP extraction (about 0.5–1.8 L4-hours each in this study).

Follow the milestone guides in order: M2 (private record) → M3 (private record) → M4/M5 records → M6 lock package (private record).

Bitwise identity of a retrained DenseNet is **not** expected across hardware or driver versions. Deterministic algorithms were enabled, but this was a single seed. The probe and calibration stages are deterministic given the caches (reload differences 0.0).

## 5. What is excluded from any release

- Raw HAM or DDI images.
- DDI per-image data, which was never retained.
- Model weights and caches, which are private by licence and size.
- Credentials and tokens. Hugging Face access stayed with the user's local client; signed links were never logged.
- Private provider identifiers beyond what these records already contain.

Before public release:
- review privacy and provenance for every figure;
- confirm the terms allow publishing DDI aggregates;
- get explicit user approval for any commit, push or publication.
