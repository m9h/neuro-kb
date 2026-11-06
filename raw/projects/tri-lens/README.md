# tri-lens — do three interpretability instruments agree about the same activation?

Three 2025–26 instruments each claim to read what a model internally holds. Nobody had run
several on the **same activations of the same model** and asked whether they agree. This does,
against a mismatch null, then makes it causal with SAE steering as ground truth.

**Model:** `google/gemma-3-27b-it`, **layer 41** — the only layer where a Jacobian lens, an NLA
checkpoint, Gemma Scope 2 SAE weights *and* Neuronpedia feature labels all exist.

## Results

**Phase 1 — agreement vs a cross-prompt mismatch null**

| pair | matched | mismatched | excess | z | beats **all** ~40 mispairings |
|---|---|---|---|---|---|
| **J-lens vs NLA** | **0.298** | 0.007 | **+0.291** | **+8.05** | **0.79** (chance 0.024) |
| NLA vs SAE | 0.121 | 0.032 | +0.089 | +1.05 | 0.14 |
| J-lens vs SAE | 0.011 | 0.001 | +0.010 | +0.35 | 0.05 |

A **linear** readout and a **trained RL'd verbaliser**, sharing no machinery, recover the same
content 42× above the null — convergent validity for "verbalizable content" as a property of the
*representation*. The SAE rows are **inconclusive, not negative**: Neuronpedia publishes 2,048
labels for a 16,384-feature SAE, so most SAE readouts are mostly empty.

**Phase 2 — steering as ground truth**

| condition | J-lens | NLA |
|---|---|---|
| **steer α=0.5** | **0.188** | **0.118** |
| negate / random / unsteered | 0.000 | 0.000–0.006 |

Inject a *known* feature and both instruments report it; inject nothing, a random direction, or
the negation and neither does.

## Also here — the corrections

- **`results/RECON_NULL_FINDING.md`** — a retraction. We first reported that NLA round-trip
  fidelity "does not survive a mismatch null". It was two bugs of ours (sampling below the AV's
  `min_position=50` training regime, and a √d scaling correction applied in the wrong
  direction). Verified against the authors' published worked example the pipeline scores
  `fve_nrm` **0.839** vs their **0.821**. **A null-only design cannot distinguish "the method
  fails" from "my reproduction is broken."**
- **`results/MOE_SPARSITY_THRESHOLD.md`** — revised twice. An apparent 4–6× MoE sparsity penalty
  turned out to be mostly *model family*; a within-family control reversed the conclusion.
- **`tests/test_nla_reference.py`** — the red-green harness, anchored to published numbers, with
  red cases that must fail.


## Related

Part of one program — a controls-first, open-weights attempt to make the 2025–26 interpretability
claims checkable:

| repo | what |
|---|---|
| [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp) | the **curriculum** — 8 rungs, 6 runnable on a laptop, each ending in its own null |
| [jacobian-lens](https://github.com/m9h/jacobian-lens) | the **research** — OLMo post-training ladder, metacognition, the Consciousness-Indicator Scorecard |
| [tri-lens](https://github.com/m9h/tri-lens) | do **three instruments agree** about the same activation? |
| [societies-of-thought](https://github.com/m9h/societies-of-thought) | the **adversarial replication** — rebuild a no-code/no-data paper, then try to break it |
| [controls-and-trajectories](https://github.com/m9h/controls-and-trajectories) | the **published datasets** — nulls and developmental trajectories |

Datasets: [induction-emergence-pythia](https://huggingface.co/datasets/mhough/induction-emergence-pythia) ·
[inceptionv1-tuning-atlas](https://huggingface.co/datasets/mhough/inceptionv1-tuning-atlas) ·
[trilens-instrument-agreement](https://huggingface.co/datasets/mhough/trilens-instrument-agreement) ·
[olmo3-jacobian-lenses](https://huggingface.co/mhough/olmo3-jacobian-lenses)

Data and full logs: [huggingface.co/datasets/mhough/trilens-instrument-agreement](https://huggingface.co/datasets/mhough/trilens-instrument-agreement).
Curriculum: [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp).

Apache-2.0. Morgan Hough, Orthogonal Research and Education Lab (OREL).
