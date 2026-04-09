---
title: Change Log
description: Chronological audit trail of wiki changes
---

# Change Log

## 2026-04-07
- Initialized wiki structure with CLAUDE.md schema
- Created index.md, log.md
- Seeded SCI head model page from /data/datasets/sci_head_model/

## 2026-04-07
- Distilled 8 wiki pages: tissue-scalp.md, tissue-skull.md, tissue-csf.md, tissue-gray-matter.md, tissue-white-matter.md, tissue-optical-properties.md, tissue-electrical-properties.md, tissue-acoustic-properties.md

## 2026-04-07
- Distilled 5 wiki pages: physics-electromagnetic.md, physics-diffusion-equation.md, physics-acoustic.md, physics-bloch.md, physics-hemodynamic.md

## 2026-04-07
- Distilled 9 wiki pages: eeg.md, meg.md, fnirs.md, structural-mri.md, diffusion-mri.md, fmri.md, mrs.md, tus.md, tms.md

## 2026-04-07
- Distilled 2 wiki pages: jax-ecosystem.md, data-formats.md

## 2026-04-07
- Distilled 10 wiki pages: method-fem.md, method-bem.md, method-monte-carlo.md, method-sbi.md, method-neural-ode.md, method-source-imaging.md, method-spectral-analysis.md, method-hmm-dynamics.md, method-active-inference.md, method-hypergraph.md

## 2026-04-07
- Created connectomics.md: concept page covering SC/FC, parcellation atlases, tractography, graph measures, higher-order structure (hypergraphs/simplicial complexes), SC-FC coupling, conduction delays, standard datasets, and network controllability. Cross-linked to diffusion-mri.md, method-hypergraph.md, jax-ecosystem.md, and project pages (hgx, vbjax, jaxctrl).

## 2026-04-07
- Created quantitative-mri.md: modality page covering T1/T2/T2*/PD mapping methods (IR, MP2RAGE, VFA/DESPOT1, Look-Locker, MESE, GraSE, multi-echo GRE), magnetization transfer (MTR, qMT two-pool model), myelin-sensitive metrics (T1w/T2w ratio, MTV, R1-myelin correlation), relaxation time tables at 3T and 7T, conductivity-from-T1 mapping for neurojax forward models, SBI connection for multi-compartment relaxometry, PINN/NODE relaxometry, BIDS structure, and QC. Cross-linked to structural-mri.md, physics-bloch.md, diffusion-mri.md, method-sbi.md, method-fem.md, tissue pages.
- Created coordinate-systems.md: coordinate-system page covering MNI152/305, Talairach, scanner/voxel coordinates, RAS vs LPS, FreeSurfer spaces (tkRAS, scanner RAS, MNI305), CTF/Neuromag/BESA device coordinates, 10-20/10-10/10-5 electrode systems, fiducials and coregistration, rigid/affine/nonlinear transforms, SPM vs FSL conventions. Updated index.md.

## 2026-04-07
- Created head-model-mida.md: MIDA head model (IT'IS Foundation), 115+ tissues, 0.5 mm iso, with comparison table vs SCI/Colin27/ICBM152/SimNIBS/BrainWeb, tissue property assignments (electrical, acoustic, optical), meshing guidance for FEM and pseudospectral solvers, and project cross-references (neurojax, brain-fwi, vbjax, dot-jax). Updated index.md.

## 2026-04-07
- Citation sweep: added Key References sections with bib keys from references.bib to 31 wiki pages that previously lacked citations. Priority pages (eeg, meg, fmri, fnirs, tms, tus, method-fem, method-bem, method-source-imaging, physics-electromagnetic, physics-acoustic, physics-hemodynamic, tissue-electrical-properties, tissue-acoustic-properties, method-hypergraph, method-neural-ode, method-active-inference) completed first, then remaining pages (diffusion-mri, structural-mri, mrs, method-sbi, method-monte-carlo, method-hmm-dynamics, method-spectral-analysis, jax-ecosystem, physics-bloch, physics-diffusion-equation, sci-head-model, tissue-csf, tissue-gray-matter, tissue-optical-properties, tissue-scalp, tissue-skull, tissue-white-matter, data-formats). All citation keys verified against references.bib (261 entries).

## 2026-04-07
- Created method-variational-inference.md: VI as optimization-based Bayesian inference, covering ELBO/KL formulation, mean-field vs structured approximations, Variational Laplace for DCM, VAEs in neuroimaging, amortized VI (bridge to SBI), active inference connection (alf), JAX implementation patterns (reparameterization trick, vmap batching), and VI vs MCMC vs SBI comparison table. Cross-linked to method-sbi.md, method-active-inference.md, method-neural-ode.md, physics-hemodynamic.md, method-source-imaging.md.
