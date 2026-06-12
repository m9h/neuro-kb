# brainmarks-dlbs

Dallas Lifespan Brain Study (OpenNeuro
[ds004856](https://openneuro.org/datasets/ds004856)) curation + loader plugin
for [Brainmarks](https://github.com/MedARC-AI/Brainmarks), the open evaluation
suite for fMRI foundation models from the CortexMAE paper
([arXiv 2510.13768](https://arxiv.org/abs/2510.13768)).

## Why DLBS

DLBS is the **broadest-coverage age dataset we have** locally:

- **464 subjects, ages 21–89** — full adult lifespan, mean 58.4 y. Wider than
  HCP-A (which is ~36–100) and so a stronger generalisation test of
  CortexMAE's age-prediction head.
- **3 longitudinal waves** (n=464 / 309 / 196 with ~5 yr gaps) — opens the
  door to longitudinal probes (cognitive decline prediction) that aren't in
  the paper. Deferred for v0; flagged for follow-up.
- **MMSE per wave** — clinical cognitive impairment label, a genuine biomarker
  use case missing from the paper's benchmarks.
- **Three task fMRI paradigms** — Scenes (4 conditions), VentralVisual
  (7 conditions), Words (3 conditions). All TR = 2 s.

## Plugin heads

### State (trial-locked clips, wave1)
- `dlbs_ventralvisual` — 7-way within-task decoding
- `dlbs_taskid` — 3-way: which of {Scenes, VentralVisual, Words} is running

### Trait (run-level rest, wave1)
- `dlbs_age_bin` — 4-way age quartile (matches paper's HCP-A framing)
- `dlbs_sex` — M / F
- `dlbs_mmse_imp` — MMSE < 26 vs ≥ 26 (standard cognitive-impairment cut)
- `dlbs_education` — high vs low (median split on `EduYrsEstCap`)

### Deferred (worth follow-up)
- Continuous age regression (paper uses classification; bonus head)
- Longitudinal cognitive decline (`CogW1toW3`) — predict 5+ yr trajectory
  from wave1 fMRI alone
- Wave2/3 separately or as test-retest pairs

## Layout

```
datasets/DLBS/
  scripts/
    dlbs_curation_state.py       # parse events.tsv → per-trial metadata
    dlbs_curation_trait.py       # rest QC + stratified split (wave1 only)
    make_dlbs_state_arrow.py     # NSD-style trial-clip arrow shards
    make_dlbs_rest_arrow.py      # ABIDE-style run-level arrow shards
  metadata/                      # generated; gitignored

src/brainmarks/datasets/dlbs.py  # plugin: dlbs_*
```

## Quick start

```bash
# 1. preprocess on Legion (x86) — wave1 only for v0
parallel -j 8 ./preprocess.sh {} ::: $(cat metadata/DLBS_subjects_wave1.txt)

# 2. trait curation
python datasets/DLBS/scripts/dlbs_curation_trait.py \
  --bids-root /data/raw/openneuro/ds004856 \
  --fmriprep-root /data/datasets/fmriprep/ds004856

# 3. state curation (all 3 tasks bundled)
python datasets/DLBS/scripts/dlbs_curation_state.py \
  --bids-root /data/raw/openneuro/ds004856 \
  --fmriprep-root /data/datasets/fmriprep/ds004856

# 4. arrow generation
python datasets/DLBS/scripts/make_dlbs_rest_arrow.py  --root <deriv> --out-root <arrow> --space schaefer400
python datasets/DLBS/scripts/make_dlbs_state_arrow.py --root <deriv> --out-root <arrow> --space schaefer400

# 5. drop dlbs.py into a Brainmarks fork; run probes
brainmarks eval --dataset dlbs_age_bin       --space schaefer400 --probe logistic
brainmarks eval --dataset dlbs_ventralvisual --space schaefer400 --probe attention
brainmarks eval --dataset dlbs_taskid        --space schaefer400 --probe attention
```

## Caveats

- **Wave1 only in v0.** Adding wave2/3 risks subject leakage if not handled
  carefully — split must be by *subject*, with all waves following the
  subject's split. Trivial extension once wave1 numbers are known.
- **VentralVisual trial types are integers 1–7** in events.tsv with no
  bundled labels; the canonical mapping (faces/places/objects/scrambled/etc.)
  needs to be looked up from the DLBS protocol docs. The plugin treats them
  as opaque class IDs; relabel via target map JSON if you want semantic
  category names.
- **MMSE binarisation at 26** is conventional but blunt. n_impaired in this
  cohort is small (most are healthy aging) — expect class imbalance and
  consider stratified loss / sklearn class_weight.
- **Age range 21–89 is wider than HCP-A.** Quartile boundaries will be
  ~21–46 / 46–60 / 60–73 / 73–89 (depends on exact distribution). Comparable
  to HCP-A's 4-way age head but not identical.
