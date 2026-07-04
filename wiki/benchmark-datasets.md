---
type: concept
timestamp: 2026-06-12T16:08:30-07:00
title: fMRI-FM Benchmark Datasets (Brainmarks)
tags: [benchmarks]
description: "introduced with the CortexMAE paper [@lane2025scaling]."
related: [foundation-models.md, fmri.md, connectomics.md, coordinate-systems.md]
---

# fMRI-FM Benchmark Datasets (Brainmarks)

**Brainmarks** (MedARC) is the open evaluation suite for fMRI foundation models
introduced with the CortexMAE paper [@lane2025scaling]. It frames FM evaluation as two
probe families over frozen representations:

- **State** — decode the cognitive condition from short trial- or movie-locked clips
  (e.g. which visual category is on screen).
- **Trait** — decode a stable subject attribute (age, sex, clinical dimension) from
  run-level resting-state patterns.

Probes run in either **volume space** (`mni`, `mni_cortex`,
`schaefer400_tians3_buckner7`) or **surface space** (`flat`, `fslr91k`); see
[coordinate-systems.md](coordinate-systems.md). The project family maintains four
local dataset plugins that extend Brainmarks beyond its shipped cohorts
(ABIDE / ADHD-200 / ADNI / PPMI / HCP-A), chosen so each adds a *generalization* test
the original benchmark lacks.

## The four local cohorts

| Plugin | Dataset | n | Ages | Adds | Key probes |
|---|---|---:|---|---|---|
| `brainmarks-hbn` | Healthy Brain Network (CMI) | ~1,556 | 5–21 (μ 10.4) | **Pediatric** OOD vs adult-pretrained CortexMAE; continuous CBCL traits | `hbn_p_factor`, `hbn_attention`, `hbn_internalizing`, `hbn_externalizing`, `hbn_age_bin`, `hbn_sex` |
| `brainmarks-wand` | Welsh Advanced Neuroimaging Database | ~165–170 | adult | **Both** state + trait in one cohort | `wand_catloc` (6-way), `wand_age_bin`, `wand_smoking`, `wand_alcohol` |
| `brainmarks-dlbs` | Dallas Lifespan Brain Study (ds004856) | 464 | 21–89 (μ 58.4) | **Widest adult lifespan**; MMSE; 3 longitudinal waves | `dlbs_ventralvisual` (7-way), `dlbs_taskid` (3-way), `dlbs_age_bin`, `dlbs_mmse_imp` |
| `brainmarks-syn` | Synaesthesia (ds004466) | 127 | adult | **Novel phenotype** unseen in pretraining | `syn_dx`, `syn_sex`, `syn_age_bin` |

All four use the paper's 4-way age-quartile framing (`*_age_bin`) for comparability
with the HCP-A age head.

## HBN — pediatric out-of-distribution

Healthy Brain Network (Child Mind Institute). The strongest paper-comparable cohort
locally accessible and **out-of-distribution** for CortexMAE (pretrained on adult
HCP-YA). Uses publicly available **C-PAC preprocessed** outputs
(`s3://fcp-indi/.../HBN/CPAC_preprocessed/`) — volumetric BOLD in MNI152 standard
space, so v0 drives Brainmarks via its volume-space readers and skips fMRIPrep
entirely (surface probes deferred to v1). Trait heads derive *continuous* CBCL
dimensions (general psychopathology `p_factor`, `attention`, `internalizing`,
`externalizing`), median-split to binary (quartile variants available). The same
subjects also have **HD-EEG** locally (`/data/datasets/hbn-eeg`), so a multimodal v1
extension is trivial.

## WAND — state + trait in one cohort

Welsh Advanced Neuroimaging Database. The only locally-available openly-shared cohort
giving **both halves** of the paper's evaluation:

- **State** — `task-categorylocaliser`, a 6-condition ventral-stream block-design
  localiser (`adult / body / car / corridor / word / baseline`), TR = 2 s, ~165
  subjects × 2 runs. A cleaner test than NSD's 24-way COCO probe (fewer classes,
  strong known signal in FFA/EBA/PPA/VWFA/LOC).
- **Trait** — `task-rest` from ~170 subjects with rich phenotype (age, sex, weight,
  blood pressure, smoking, alcohol).

## DLBS — adult lifespan, MMSE, longitudinal

Dallas Lifespan Brain Study (OpenNeuro ds004856). 464 subjects spanning the **full
adult lifespan** (21–89), wider than HCP-A, making it a strong generalization test of
the age head. Three task paradigms — Scenes (4 conditions), VentralVisual (7), Words
(3), all TR = 2 s — plus rest. Two features absent from the paper's benchmarks:
**MMSE** per wave (a clinical cognitive-impairment label, `dlbs_mmse_imp`: MMSE<26 vs
≥26) and **3 longitudinal waves** (n = 464 / 309 / 196, ~5 yr gaps) enabling
cognitive-decline probes (deferred for v0).

> **Cross-modal sibling.** The `smri-fm` *DLBS morphometry benchmark* uses the same
> ds004856 cohort on the **structural** side (FreeSurfer / T1Prep morphometry vs FM
> features) — the structural counterpart to this fMRI plugin. See
> [structural-mri.md](structural-mri.md) and [foundation-models.md](foundation-models.md).

## SYN — a phenotype never seen in pretraining

Synaesthesia (OpenNeuro ds004466): 102 synaesthetes + 25 controls (n = 127), HCP-D
protocol. A clean test of whether a CortexMAE backbone pretrained on HCP-YA
generalizes to a phenotype it has never seen — and whether FM features beat the
functional-connectivity baseline that is hard to dethrone on trait tasks. Caveats:
n = 127 is small for a 70/15/15 split (prefer 5-fold CV for headline numbers); HCP-D
TR = 0.8 s, and the 150-TR truncation (for ABIDE comparability) discards most of each
run, so a `_full` variant may be needed for power.

## Relevant Projects

- **brainmarks-hbn / -wand / -dlbs / -syn**: the four Brainmarks dataset plugins above.
- **smri-fm**: the structural-MRI FM that benchmarks against morphometry on the same
  cohorts (notably DLBS / ds004856) — see [foundation-models.md](foundation-models.md).

## Citations
- **lane2025scaling**: CortexMAE — cortical masked-autoencoder fMRI foundation model
  and the Brainmarks evaluation suite (arXiv 2510.13768).

## See Also

- [foundation-models.md](foundation-models.md) — the FMs these cohorts evaluate
- [fmri.md](fmri.md) — the modality
- [connectomics.md](connectomics.md) — the functional-connectivity baseline FMs must beat
- [coordinate-systems.md](coordinate-systems.md) — volume vs surface probe spaces
