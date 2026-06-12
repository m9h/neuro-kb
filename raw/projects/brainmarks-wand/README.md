# brainmarks-wand

WAND (Welsh Advanced Neuroimaging Database) curation + loader plugin for
[Brainmarks](https://github.com/MedARC-AI/Brainmarks), the open evaluation
suite for fMRI foundation models from the CortexMAE paper
([arXiv 2510.13768](https://arxiv.org/abs/2510.13768)).

## Why WAND

WAND is the only locally-available openly-shared dataset that gives us
**both halves of the paper's evaluation** in one cohort:

- **State decoding** — `task-categorylocaliser`, a 6-condition ventral-stream
  block-design localiser (`adult / body / car / corridor / word / baseline`),
  TR = 2 s, ~165 subjects × 2 runs. Direct analogue of HCP-YA Task21 and a
  cleaner test of CortexMAE's headline state-decoding result than NSD's
  24-way COCO probe (fewer classes, stronger known signal in
  FFA/EBA/PPA/VWFA/LOC).
- **Trait probes** — `task-rest` from ~170 subjects with rich phenotype
  (age, sex, weight, BP, smoking, alcohol). Run-level pattern mirroring
  `hcpya_rest1lr_*` and ABIDE.

## Plugin heads

### State (trial-locked clips, framing A)
- `wand_catloc` — 6-way visual category decoding (default)
- `wand_catloc5` — 5-way (drops `baseline`)

### Trait (run-level, framing "B")
- `wand_age_bin` — age quartile (matches paper's HCP-A 4-way framing)
- `wand_sex` — sex
- `wand_smoking` — smoker / non-smoker
- `wand_alcohol` — drinker / non-drinker

## Layout

```
datasets/WAND/
  README.md
  scripts/
    wand_curation_state.py       # parse events.tsv → per-trial metadata
    wand_curation_trait.py       # rest QC + stratified split
    make_wand_catloc_arrow.py    # NSD-style trial-clip arrow shards
    make_wand_rest_arrow.py      # ABIDE-style run-level arrow shards
  metadata/                      # generated; gitignored

src/brainmarks/datasets/wand.py  # plugin: wand_catloc, wand_*
```

## Quick start

```bash
# 1. preprocess on Legion (x86) with --output-spaces fsLR:den-91k MNI152NLin6Asym:res-2

# 2. trait curation (rest, run-level)
python datasets/WAND/scripts/wand_curation_trait.py \
  --bids-root /data/raw/wand --fmriprep-root /data/datasets/fmriprep/wand

# 3. state curation (catloc, trial-locked)
python datasets/WAND/scripts/wand_curation_state.py \
  --bids-root /data/raw/wand --fmriprep-root /data/datasets/fmriprep/wand

# 4. arrow generation (one per space)
python datasets/WAND/scripts/make_wand_rest_arrow.py   --root <deriv> --out-root <arrow> --space schaefer400
python datasets/WAND/scripts/make_wand_catloc_arrow.py --root <deriv> --out-root <arrow> --space schaefer400

# 5. drop wand.py into a Brainmarks fork; run probes
brainmarks eval --dataset wand_catloc --space schaefer400 --probe attention
brainmarks eval --dataset wand_age_bin --space schaefer400 --probe logistic
```

## Caveats

- **Subject-level splits.** Both heads split by subject (no subject appears in
  more than one split) to prevent leakage.
- **State framing is A.** I initially considered run-level state decoding ("B")
  but the paper's HCP-YA Task21 head is already trial-locked, so there is no
  run-level state target to compare against. Run-level remains the trait
  pattern only.
- **HRF shift is a single-scalar approximation** (default 4 s = 2 TRs). For
  a more careful baseline, regress out canonical HRF first; for a probe-focused
  comparison this is fine.
- **n=165 with 2 runs × ~16 blocks/run** → ~5k trial clips. Substantially
  better powered than synaesthesia for multi-class decoding.
