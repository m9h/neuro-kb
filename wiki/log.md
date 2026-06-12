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

## 2026-06-12
- **Fleet refresh + new themes.** Refreshed raw sources from the m9h fleet after a ~7.5-week gap (last refresh 2026-04-20). Re-pulled 17 changed already-ingested repos, fetched 18 new public repos, and added 6 previously-missing PRIVATE repos (the four `brainmarks-*` plugins + `cpjax`, `betse-unified`) that `fetch_raw.py`'s `users/m9h/repos` listing had silently skipped (public-only endpoint).
- **references.bib**: 366 → 424 entries. 54 auto-ingested (notably the wwj HT-SR theory stack: martin2021predicting, martin2026rg, martin2025setol, clauset2009powerlaw, alstott2014powerlaw, vuong1989likelihood; CortexMAE=lane2025scaling; MarS-FM=kapusniak2025mars) + 4 hand-curated (lin2026identity, fisch2024deepmriprep, gao2024scaling, jaynes2003probability).
- **3 new wiki pages**:
  - `htsr-weight-analysis.md` (method/spectral) — Heavy-Tailed Self-Regularization: power-law α of the weight ESD, α≈2 RG critical point, participation count, CSN/Hill estimators, differentiable α→2 regularizer, wwjd Bayesian layer, FM pretraining-quality ranking. Sources: wwj, WeightWatcher, emeg-fm.
  - `foundation-models.md` (concept) — the two-axis (spectral + decoding) + identity-audit benchmark; MedARC FM family (fmri-fm/CortexMAE, smri-fm, emeg-fm) + sibling non-brain FMs (nanopath, mars-fm).
  - `benchmark-datasets.md` (concept) — Brainmarks fMRI-FM evaluation cohorts: HBN, WAND, DLBS, SYN plugins with state/trait probe heads.
- **structural-mri.md**: fixed malformed double-frontmatter + broken BIDS code fence; added T1Prep/deepmriprep DL-preprocessing and the smri-fm morphometry-baseline benchmark; cross-linked foundation-models/benchmark-datasets.
- **distill.py**: registered a `foundation-models` topic cluster and added the new projects (smri-fm, emeg-fm, ephys-tokenizer-jax, T1Prep, brainmarks-*) to existing modality/method clusters so the pipeline maintains the new pages.
- **Pruned** 9 stale renamed/deleted raw dirs (Thinking_Higher, quantum-cognition, QuantumComp4Neuro, quantum-deeponet, jsPsych2, stac-mjx, track-mjx, organoid-hgx-benchmark, organoid_regulomes); kept LAYNII + neurotech-primer-book (declared connected projects).
- NOTE: did not mass-re-run distill.py over existing pages — its MAX_TOKENS=4096 cap truncates longer curated pages. New pages hand-authored; existing pages refreshed selectively.
