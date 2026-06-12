# brainmarks-syn

Synaesthesia (OpenNeuro ds004466) curation + loader plugin for
[Brainmarks](https://github.com/MedARC-AI/Brainmarks), the open evaluation
suite for fMRI foundation models from the CortexMAE paper
([arXiv 2510.13768](https://arxiv.org/abs/2510.13768)).

## Why

Brainmarks ships with ABIDE / ADHD-200 / ADNI / PPMI / HCP-A as trait-prediction
benchmarks. Synaesthesia (102 synaesthetes + 25 controls, HCP-D protocol) is a
clean novel probe for whether a CortexMAE backbone pretrained on HCP-YA
generalises to a phenotype it has never seen — and whether features beat the
functional-connectivity baseline that is hard to dethrone on trait tasks.

## Layout

```
datasets/SYN/
  README.md                            # workflow + curation criteria
  scripts/syn_curation.py              # fMRIPrep walk → QC → split → metadata
  scripts/make_syn_arrow.py            # ports ABIDE arrow generator
  metadata/                            # generated; gitignored

src/brainmarks/datasets/syn.py         # plugin: syn_dx / syn_sex / syn_age_bin
```

## Quick start

1. Preprocess on x86 (Legion / cluster) with `--output-spaces fsLR:den-91k MNI152NLin6Asym:res-2`.
2. `python datasets/SYN/scripts/syn_curation.py --bids-root <bids> --fmriprep-root <deriv>`
3. `python datasets/SYN/scripts/make_syn_arrow.py --root <deriv> --out-root <arrow> --space schaefer400`
4. Drop `src/brainmarks/datasets/syn.py` into a Brainmarks fork; run
   `brainmarks eval --dataset syn_dx --space schaefer400 --probe logistic`.

See [`datasets/SYN/README.md`](datasets/SYN/README.md) for the full pipeline.

## Caveats

- n=127 is small for a 70/15/15 split; prefer 5-fold CV for the headline number.
- HCP-D TR=0.8 s; the 150-TR truncation matches ABIDE for comparability but
  discards most of each run — a `_full` variant may be needed for power.
