---
type: method
timestamp: 2026-06-13T08:55:27-07:00
title: HT-SR Weight-Spectrum Analysis
tags: [method, spectral, foundation-models]
description: "Heavy-Tailed Self-Regularization (HT-SR) is a data-free diagnostic that scores the quality of a trained neural-network layer from its weights alone — no training or test data, no forward pass."
category: spectral
implementations: [wwj:core, wwj:bayes, WeightWatcher, emeg-fm:eeg_fm]
related: [foundation-models.md, method-spectral-analysis.md, method-variational-inference.md, jax-ecosystem.md]
---

# HT-SR Weight-Spectrum Analysis

Heavy-Tailed Self-Regularization (HT-SR) is a **data-free** diagnostic that scores
the quality of a trained neural-network layer from its weights alone — no training
or test data, no forward pass. For each weight matrix `W` it forms the correlation
matrix `X = WᵀW`, takes the empirical spectral density (ESD) of its eigenvalues `λ`,
and fits a power law `ρ(λ) ∝ λ^(−α)` to the tail. The exponent **α** is the layer's
quality score; aggregated across layers it predicts generalization without ever
touching the data the model was trained on [@martin2021predicting].

This is the spectral axis of the project family's two-axis foundation-model
comparison (the other axis is task decoding / linear-probe accuracy). See
[foundation-models.md](foundation-models.md).

## The α scale

| α range | Regime | Interpretation |
|---|---|---|
| α < 2 | Over-correlated / **correlation trap** | Spectrum dominated by a few directions; layer memorizes, poor self-averaging |
| **α ≈ 2** | **Critical / optimal** | Self-averaging boundary = RG scale-balance point; best-trained dense layers sit here |
| 2 ≲ α ≲ 6 | Well-trained | Heavy-tailed, implicitly self-regularized |
| α ≳ 6 | Under-trained | Tail too light; layer is closer to random initialization |

Lower α (down to 2) means a more strongly correlated, better-trained layer. The
classic HT-SR result is that the **mean α across layers** anti-correlates with test
error across hundreds of pretrained models [@martin2021predicting; @martin2021implicit].

## Renormalization Group reading (Martin 2026)

Charles Martin's *Renormalization Group Theory of Learning* unifies the earlier
HT-SR and SETOL theories into a Wilsonian RG account of layer convergence
[@martin2026rg; @martin2025setol]. The central claim: **α ≈ 2 is simultaneously
the self-averaging boundary** (where the trace-law gain is spread across many
eigencomponents rather than dominated by a few) **and the RG scale-balance point**.
Below α = 2, a "correlation trap" forms — a dominant-tail takeover where the
representation is carried by a handful of directions and the layer overfits.

This refines the weights-only toolkit with three diagnostics beyond the bare α:

- **Fitted α** — power-law tail exponent (as above).
- **Dominant-tail burden** — how much of the trace is carried by the leading
  eigenvalue(s); a trap indicator.
- **Participation count** `M_tr = (Σλ)² / Σλ²` — effective number of eigen-directions
  carrying the spectrum. The same `(Σλ)²/Σλ²` statistic applied to *activations* (the
  participation ratio) predicts how much structure a sparse dictionary can recover —
  see the activation-side note below.

## Estimators

Two tail-fitting modes, trading exactness against differentiability:

- **`csn`** — Clauset–Shalizi–Newman MLE with a Kolmogorov–Smirnov scaling-window
  (`xmin`) search [@clauset2009powerlaw; @alstott2014powerlaw]. Matches the reference
  WeightWatcher. Non-differentiable. `wwj` does the *exact* adaptive `xmin` search
  (every eigenvalue a candidate, vmap-parallel KS distance) rather than the
  log-spaced grid the Python implementation falls back to.
- **`hill`** — closed-form Hill estimator on a fixed window. Fully differentiable;
  this is what powers the training-time regularizer.

Supporting statistics:

- **`bootstrap_alpha_ci`** — vmap-parallel bootstrap confidence interval on α.
- **`fit_distributions`** — power-law vs exponential vs log-normal MLEs in one jit
  pass, returning Vuong's likelihood-ratio test [@vuong1989likelihood] (positive ⇒
  power-law preferred). This is the "is this layer *actually* a power law?" check
  that any α claim requires.

## Differentiable α→2 regularizer (and why steering α is a Goodhart trap)

`alpha_loss(model, target=2.0)` (Hill estimator, fully differentiable) drops into any
optax pipeline as `task_loss + weight · alpha_loss(model)`, and was proposed as the
**explicit** alternative to Muon's *indirect* spectral shaping — directly penalizing
each layer's α toward the RG-optimal value of 2 [@martin2026rg] rather than nudging
the spectrum implicitly through the optimizer geometry.

**But the causal test says don't use it as a control knob.** The wwj RG-robustness
experiments ran a seed-matched controlled comparison — two from-scratch GPT-2-124M
with *identical* initialization and data order, differing only by the regularizer
(`λ Σ_ℓ (α_ℓ − 2)²`, λ=1, Hill surrogate) — and found a clean **Goodhart failure**:

- The regularizer drove its *own* (fixed-window Hill) target to ≈2 (penalty <10⁻⁵),
  yet the rigorous `wwjd` posterior α moved the **wrong way** — ending *further* from 2
  than the unregularized baseline (≈2.71 vs ≈2.68; worst mid-training, ≈3.17 vs ≈2.82).
  The spectrum became *less* concentrated (mean stable rank 51 vs 47). The model found
  weights that satisfy the single-window metric without a genuinely heavier tail; the
  window-marginalizing Bayesian estimator is exactly what exposes the hack.
- It bought **no circuit**: language-modeling loss unchanged (≈2.38 vs ≈2.37) and the
  induction circuit formed at the same time and strength (8 induction heads either way).

The reading: **α is a *diagnostic readout* of training quality, not a control knob.**
The heavy tail that emerges under good training is *downstream* of the computation;
optimizing the exponent directly is a metric-hack that the point estimate rewards and
the posterior catches. This is also the strongest argument for the Bayesian layer
below over the bare Hill point estimate.

## Bayesian layer (wwjd)

Where the frequentist core reports a point estimate of α, the Bayesian extension
(`wwjd`, *"what would Jaynes do"* [@jaynes2003probability]) reports a calibrated
posterior — and most of it is closed-form. The tail above `xmin` is Pareto; the
substitution `t = log(λ/xmin)` makes it Exponential with rate `β = α − 1`, so a
conjugate `Gamma(a₀, b₀)` prior on β gives an exact `Gamma(a₀+n, b₀+Σt)` posterior.
Consequences:

- Credible intervals on α are Gamma quantiles; `P(α < 2)` is a Gamma CDF — and
  differentiable.
- **`alpha_posterior_bma`** — evidence-weighted Bayesian model averaging over the
  `xmin` choice, instead of committing to a single KS window.
- **`model_posterior`** — analytic Bayes factors + posterior model probabilities over
  {power-law, exponential, log-normal}, the Bayesian counterpart of the Vuong test.
- **`hierarchical_analyze`** — a NumPyro cross-layer hierarchical posterior on the
  population-mean α (the only part that samples).
- **`bayes_alpha_loss`** — regularizer that penalizes the posterior *mean* toward 2
  and optionally its *variance*, differentiating through the Gamma posterior.

See [method-variational-inference.md](method-variational-inference.md) and
[method-sbi.md](method-sbi.md) for the broader Bayesian-inference context.

## Application: ranking foundation-model pretraining quality

Because HT-SR needs no data, it ranks frozen foundation models before any probe is
trained. From the E/MEG FM audit in `emeg-fm` (per-block weight α; lower-and-near-2
is better-trained, but a large fraction of layers with α < 2 signals trap-dominated
under-training):

| E/MEG FM | weight α-mean | % layers α<2 | verdict |
|---|---:|---:|---|
| REVE | 3.61 | 5.6% | well-trained |
| LaBraM | 3.76 | 4.2% | well-trained |
| LUNA | 3.93 | 10.4% | well-trained, heaviest tails |
| BENDR | 2.10 | 68.0% | severely under-trained (largest model in suite) |

**Activation-side companion.** The participation ratio `(Σλ)²/Σλ²` of a model's
*activations* directly predicts sparse-autoencoder (SAE) yield [@gao2024scaling]: a
near-rank-2 activation spectrum has almost nothing for a wide dictionary to
decompose (the activation-space analogue of the non-self-averaging / dominant-tail
regime). In `emeg-fm`, REVE's activation participation ratio ≈ 4.9 (viable SAE
substrate) versus ZUNA's ≈ 1.7 (≈99% dead features is *correct*, not a bug — a 32-d
masked-diffusion bottleneck).

## Tooling

- **wwj** — JAX/Equinox port targeting the Kidger SciML stack; frequentist `core` +
  Bayesian `bayes`/`hierarchical`. Operates on Equinox-native weights or numpy-bridged
  from PyTorch/TF/Flax. Validated against the CalculatedContent WeightWatcher fork as
  numerical oracle.
- **WeightWatcher** — the reference Python implementation (CalculatedContent).
- **wwj-spectral / wwj-compare** — the bounded, reproducible analysis skills wrapping
  this library (single-model and cross-model respectively).

## Relevant Projects

- **wwj**: the library — `core` (frequentist α, traps, trace-log alignment),
  `bayes`/`wwjd` (closed-form posteriors, BMA, hierarchical), `alpha_loss` regularizer.
- **WeightWatcher**: reference oracle for numerical validation.
- **emeg-fm**: applies per-block α + participation diagnostics to E/MEG foundation
  models; couples the weight spectrum to SAE-yield on activations.
- **smri-fm**, **nanopath**, **hippy-feat**: additional weight-matrix sources used to
  validate `wwj` against the reference implementation.

## Citations
- **martin2021predicting**: Martin, Peng & Mahoney (2021). Predicting trends in the
  quality of state-of-the-art neural networks without access to training or test data.
  Nature Communications 12:4122. *(WeightWatcher / HT-SR foundational paper.)*
- **martin2026rg**: Martin (2026). A Renormalization Group Theory of Learning.
- **martin2025setol**: Martin (2025). SETOL: a semi-empirical theory of learning.
- **martin2021implicit**: Martin & Mahoney (2021). Implicit self-regularization in deep
  neural networks. JMLR 22.
- **clauset2009powerlaw**: Clauset, Shalizi & Newman (2009). Power-law distributions in
  empirical data. SIAM Review 51:661-703.
- **alstott2014powerlaw**: Alstott, Bullmore & Plenz (2014). powerlaw: a Python package
  for analysis of heavy-tailed distributions. PLoS ONE 9:e85777.
- **vuong1989likelihood**: Vuong (1989). Likelihood ratio tests for model selection.
  Econometrica 57:307-333.
- **jaynes2003probability**: Jaynes (2003). Probability Theory: The Logic of Science.
- **gao2024scaling**: Gao et al. (2024). Scaling and evaluating sparse autoencoders.

## See Also

- [foundation-models.md](foundation-models.md) — the models this scores; two-axis benchmarking
- [method-spectral-analysis.md](method-spectral-analysis.md) — spectral analysis of *signals* (vs. weights here)
- [method-variational-inference.md](method-variational-inference.md) — Bayesian inference machinery behind wwjd
- [jax-ecosystem.md](jax-ecosystem.md) — the Kidger/Equinox stack wwj targets
