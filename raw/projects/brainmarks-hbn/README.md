# brainmarks-hbn

Healthy Brain Network (HBN, Child Mind Institute) curation + loader plugin
for [Brainmarks](https://github.com/MedARC-AI/Brainmarks), the open eval
suite for fMRI foundation models from the CortexMAE paper
([arXiv 2510.13768](https://arxiv.org/abs/2510.13768)).

## Why HBN

HBN is the **strongest paper-comparable cohort** locally accessible:

- **Pediatric / adolescent** (ages 5–21, mean 10.4) — out-of-distribution
  for CortexMAE, which was pretrained on adult HCP-YA. Genuinely
  paper-worthy generalisation test.
- **Direct phenotype overlap** with the paper's trait benchmarks. HBN gives
  *continuous* CBCL-derived dimensions (`p_factor`, `attention`,
  `internalizing`, `externalizing`) that arguably out-shine the binary
  diagnostic labels in ABIDE/ADHD-200.
- **n ≈ 1,556** subjects with C-PAC preprocessed fMRI publicly available.
- **Multimodal** — same subjects have HD-EEG (locally at
  `/data/datasets/hbn-eeg`). v0 is fMRI-only on purpose; subjects are
  picked so EEG extension in v1 is trivial.

## v0 scope: rest-based trait probes, volumetric

To skip fMRIPrep entirely we use **C-PAC preprocessed outputs** from
`s3://fcp-indi/data/Projects/HBN/CPAC_preprocessed/`. C-PAC writes
volumetric BOLD in MNI152 standard space, so we drive Brainmarks via its
**volume-space readers** (`mni`, `mni_cortex`,
`schaefer400_tians3_buckner7`). Surface-space probes (`flat`, `fslr91k`)
are explicitly v1 — they need either fMRIPrep on Legion or a custom
`mri_vol2surf` shim.

## Plugin heads

### Trait (run-level rest, v0)
- `hbn_age_bin` — age quartile (matches paper's HCP-A 4-way framing)
- `hbn_sex` — M / F
- `hbn_p_factor` — general psychopathology, **median split** (binary).
  Quartile variant available as `hbn_p_factor_q`.
- `hbn_attention` — CBCL-derived attention dimension, median split
- `hbn_internalizing` — anxiety/mood, median split
- `hbn_externalizing` — conduct/aggression, median split

### State (movie-locked clips, v1 — plumbed but deferred)
- `hbn_movie` — 2-way (movieDM vs movieTP) which-movie state classification
- `hbn_movie_clip_aot` — arrow-of-time within a movie

## Layout

```
datasets/HBN/
  scripts/
    hbn_select_subjects.py   # intersect CPAC ∩ EEG-tsv ∩ full_pheno
    pull_cpac.sh             # aws s3 sync for selected subjects
    hbn_curation_trait.py    # walk C-PAC, build metadata + targets
    hbn_curation_state.py    # movie-task curation (v1, optional)
    make_hbn_rest_arrow.py   # ABIDE-style run-level shards
    make_hbn_movie_arrow.py  # NSD-style trial-clip shards (v1)
  metadata/                  # generated; gitignored

src/brainmarks/datasets/hbn.py
```

## Quick start (n≈20 demo, no fMRIPrep)

```bash
# 0. one-time env setup (only needed if XDG_CACHE_HOME points somewhere read-only)
export TEMPLATEFLOW_HOME=$HOME/.cache/templateflow
export MPLCONFIGDIR=$HOME/.cache/matplotlib
mkdir -p "$TEMPLATEFLOW_HOME" "$MPLCONFIGDIR"

# 0b. dependencies
pip install --user boto3 cloudpathlib
pip install --user "brainmarks @ git+https://github.com/MedARC-AI/brainmarks"

# 1. select multimodal-complete demo subjects
python datasets/HBN/scripts/hbn_select_subjects.py \
  --eeg-tsv /data/datasets/hbn-eeg/participants.tsv \
  --n 20 --out datasets/HBN/metadata/HBN_demo_subjects.txt

# 2. pull only those subjects' C-PAC outputs (~hundreds of MB each, rest only)
./datasets/HBN/scripts/pull_cpac.sh \
  datasets/HBN/metadata/HBN_demo_subjects.txt \
  /data/raw/hbn-cpac

# 3. curate
python datasets/HBN/scripts/hbn_curation_trait.py \
  --eeg-tsv /data/datasets/hbn-eeg/participants.tsv \
  --cpac-root /data/raw/hbn-cpac

# 4. generate arrow (volume-space; pick mni_cortex for paper-comparable framing)
python datasets/HBN/scripts/make_hbn_rest_arrow.py \
  --root /data/raw/hbn-cpac \
  --out-root /data/datasets/brainmarks \
  --space mni_cortex

# 5. drop hbn.py into a Brainmarks fork; run probes
brainmarks eval --dataset hbn_age_bin    --space mni_cortex --probe logistic
brainmarks eval --dataset hbn_p_factor   --space mni_cortex --probe attention
brainmarks eval --dataset hbn_attention  --space mni_cortex --probe attention
```

## Caveats / honest framing

- **C-PAC is not fMRIPrep.** Different preprocessing pipeline, different
  defaults (band-pass filtering, nuisance regression strategy). Numbers
  here are not strictly comparable to the paper's ABIDE/ADHD-200 numbers
  even though the spaces overlap. Worth flagging in any pitch deck.
- **Volume-only in v0.** The paper's headline state-decoding wins are on
  flat-map and surface representations. Volume-space reproduces only the
  HCP-A age and sparse-cortical-volume results. Surface support comes
  with v1.
- **Subject overlap with EEG, not equality.** Of ~1,556 C-PAC subjects
  and ~2,640 in the EEG release, the intersection is smaller. Demo
  subject-picker enforces the intersection so multimodal extension stays
  free.
- **Movie tasks not always present in C-PAC outputs.** Many subjects only
  have `_scan_rest` preprocessed; movie preprocessing is variable across
  C-PAC release. State head curation (`hbn_curation_state.py`) handles
  this gracefully but real n for movie probes will be smaller.
- **Pediatric motion is real.** Default mean-FD threshold is **0.3** (loose),
  not 0.2 like ABIDE; tighten before reporting.
- **Site is encoded in HBN BIDS_curated as session label** (e.g.
  `ses-HBNsiteSI`) but C-PAC subject dirs use `_ses-1` regardless of site.
  Site recovery for stratified splits requires cross-referencing with
  BIDS_curated metadata; v0 ignores site (single global split).
