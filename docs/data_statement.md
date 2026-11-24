# Data statement — DermDetect-R

## HAM10000 (development and internal test)

- **Source:** Harvard Dataverse, DOI 10.7910/DVN/DBW86T. Files used: `HAM10000_metadata`, `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`.
- **Untouched:** all `ISIC2018_Task3_Test_*` files and segmentation masks.
- **Licence:** CC BY-NC 4.0 (Dataverse V4.0 `termsOfUse`; non-commercial, attribution required). Images are not redistributed by this project.
- **Content:** 10,015 dermoscopic images, all of which decode. Seven diagnostic labels (akiec, bcc, bkl, df, mel, nv, vasc) are research labels from the source; a share were confirmed by histopathology.

**Unit of analysis:** physical lesion units. These are built from source lesion IDs, then corrected for exact and near duplicates:
- 665 selected candidate pairs were reviewed, attributed to their source, an agent or a human;
- the final human audit covered 366 entries: 22 same, 329 different, 15 unclear;
- the 15 unclear links were kept as split-only co-location, not identity merges.

Result: 7,449 units. Source labels were never changed.

**Split** (sealed in M2; `outputs/m2-production-closeout/production_seal.json`):
- train: 5,214 units
- validation: 1,118 units
- internal test: 1,117 units (1,508 images)

Targets were 70/15/15 with seed 20260729 and fixed class-wise quotas.

**Known limits:**
- Lesion IDs are not patient IDs, so patient-level separation is not guaranteed.
- 30,072 broad candidate pairs outside the reviewed set remain unadjudicated.
- The authors' duplicate ranking could not be reproduced.
- Class imbalance is severe: in the test set, nv has 808 units, df 11 and vasc 15.

## DDI — Diverse Dermatology Images (external evaluation only)

- **Source:** official DDI release, downloaded by the user, who accepted its terms. Local copy: 656 PNG images plus metadata (SHA `534bdb3d…`). Image inventory `756a7e07…`. Integrity record passed 14/14 checks.
- **Terms:** personal, non-commercial, non-clinical research. No redistribution.
- **Labels:** official `malignant` field, read the same way the official loader reads it: 485 benign, 171 malignant (prevalence 0.261). The official `skin_tone` encoding gives the groups:

| Group | Images | Malignant |
|---|---:|---:|
| Fitzpatrick I–II (`12`) | 208 | 49 |
| Fitzpatrick III–IV (`34`) | 241 | 74 |
| Fitzpatrick V–VI (`56`) | 207 | 48 |

- **No patient identifier:** the analysis unit is the image, and within-patient dependence cannot be handled.

**Handling:**
- DDI was sealed until the DDI add-on lock (`126ff64f…`) and an explicit run approval.
- The images were uploaded once to one private, temporary cloud CPU machine, which was stopped afterwards; stopping clears its disk.
- Only aggregate metrics were kept. No per-image predictions, labels or images are in the repository or the outputs.
- No training, tuning, calibration, threshold or exclusion decision used DDI.

## Not used

- ISIC 2018, in any role.
- Any historical augmented or processed images from the original prototype.

## Privacy and release

- Both datasets are public research releases of de-identified images.
- This project redistributes no images or derived per-image data.
- The aggregate tables in `results/` contain only counts and metrics.
- Before any public release, check provenance and privacy for every figure, and confirm the HAM and DDI terms allow publishing derived aggregates.
