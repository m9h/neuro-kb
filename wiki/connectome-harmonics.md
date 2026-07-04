---
type: method
title: Connectome Harmonics
tags: [method, spectral, connectomics]
description: "A structure-function basis in which brain activity is expressed as a weighted sum of the eigenvectors of the graph Laplacian of the structural connectome."
timestamp: 2026-07-03T00:00:00-07:00
category: spectral
implementations: ["connectome_harmonic_core:construct_harmonics.py", "connectome_harmonic_core:compute_spectra.py", "HADES:utils", "anatomical-compiler:src"]
related: [connectomics.md, structural-mri.md, diffusion-mri.md, fmri.md, method-spectral-analysis.md, method-hypergraph.md]
---

# Connectome Harmonics

Connectome harmonics are the **eigenvectors of the graph Laplacian** of the structural connectome — the network analogue of Fourier modes or the vibrational modes of a drum. Low-order harmonics are smooth, spatially extended patterns; high-order harmonics are fine-grained and localized. Any brain map (an fMRI frame, an EEG topography projected to cortex) can be decomposed into this basis, yielding a **harmonic power spectrum** that summarizes how structured-vs-fragmented the state is.

## Construction

1. Build the adjacency `A` from a structural connectome (white-matter tractography + local cortical connectivity).
2. Form the graph Laplacian `L = D − A` (or symmetric-normalized `L = I − D^{-1/2} A D^{-1/2}`).
3. Eigendecompose: `L ψ_k = λ_k ψ_k`. The `ψ_k` are the harmonics; `λ_k` are spatial frequencies.
4. Project data `x` onto the basis: `c_k = ⟨x, ψ_k⟩`; the power spectrum is `|c_k|²`.

The **Hodge Laplacian** generalizes this to higher-order (edge/triangle) structure — used in `anatomical-compiler` for module identifiability on regulatory hypergraphs (see [method-hypergraph.md](method-hypergraph.md)).

## Uses

- Compact spectral fingerprint of brain states (e.g. flattening of the harmonic hierarchy under psychedelics — `HADES`, DMT dataset)
- Structure-constrained basis for connectivity and dynamics
- Harmonic-mode reduced models of whole-brain activity

## Implementations

- **connectome_harmonic_core (CHAP)** — Laplacian eigendecomposition of the structural connectome + harmonic power spectra of fMRI.
- **HADES** — harmonic decomposition of spacetime; time-resolved functional-harmonic modes.
- **anatomical-compiler** — Hodge-Laplacian module identifiability on regulome hypergraphs.

## Citations

[1] Atasoy et al. (2016). Human brain networks function in connectome-specific harmonic waves. Nature Communications.
[2] Atasoy et al. (2017). Connectome-harmonic decomposition of human brain activity reveals dynamical repertoire re-organization under LSD.

## See Also

- [connectomics.md](connectomics.md) - structural connectome construction
- [method-spectral-analysis.md](method-spectral-analysis.md) - spectral decomposition methods
- [diffusion-mri.md](diffusion-mri.md) - tractography source of the connectome
- [method-hypergraph.md](method-hypergraph.md) - higher-order (Hodge) generalization
