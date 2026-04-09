---
type: concept
title: Connectomics
related: [diffusion-mri.md, structural-mri.md, method-hypergraph.md, method-source-imaging.md, method-spectral-analysis.md, jax-ecosystem.md]
---

# Connectomics

Connectomics is the study of the brain's wiring diagram at scales ranging from individual synapses to whole-brain white matter tracts. A **connectome** is a comprehensive map of neural connections, typically represented as a graph or matrix. Two complementary flavors dominate neuroimaging:

- **Structural connectivity (SC)**: physical fiber tracts reconstructed from diffusion MRI tractography.
- **Functional connectivity (FC)**: statistical dependencies between regional time series from fMRI, EEG, or MEG.

## Parcellation Atlases

A connectome requires parcellating the cortex (and optionally subcortical structures) into discrete regions. The choice of atlas determines the matrix dimensions and the biological interpretability of each node.

| Atlas | Regions | Basis | Resolution | Notes |
|-------|---------|-------|------------|-------|
| **Desikan-Killiany** | 68 cortical | Sulcal landmarks | Coarse | FreeSurfer default; widely used in clinical studies |
| **Destrieux** | 148 cortical | Sulcal + gyral | Medium | Used in vbjax examples (164 regions with subcortical) |
| **AAL** (Automated Anatomical Labeling) | 90/116 | MNI template | Coarse | Common in fMRI FC studies; no surface-based definition |
| **Schaefer** | 100-1000 | Resting-state fMRI parcels | Tunable | Aligned to Yeo 7/17 networks; available at multiple granularities |
| **HCP-MMP1.0** (Glasser) | 360 cortical | Multi-modal (T1w, T2w, fMRI, myelin) | Fine | Gold standard for HCP data; requires surface registration |

**Practical note**: Connectome matrices from different atlases are not directly comparable. When coupling SC to simulation (e.g., in vbjax), the atlas must match between the tractography pipeline and the neural mass model parcellation.

## Structural Connectivity from Diffusion Tractography

### Tractography Algorithms

Tractography reconstructs white matter pathways by following the local fiber orientation field estimated from diffusion MRI (see [diffusion-mri.md](diffusion-mri.md)).

| Algorithm class | Method | Strengths | Weaknesses |
|----------------|--------|-----------|------------|
| **Deterministic** | FACT, Euler, RK4 | Fast, reproducible | No uncertainty; fails at crossings with DTI |
| **Probabilistic** | iFOD2 (MRtrix3), ProbtrackX (FSL) | Handles crossing fibers, quantifies uncertainty | Computationally expensive; distance bias |
| **Global** | MITK Global Tractography, COMMIT | Optimizes full tractogram jointly | Very slow; better anatomical plausibility |

Probabilistic tractography with constrained spherical deconvolution (CSD) and anatomically constrained tractography (ACT) is the current best practice for connectome construction.

### Filtering and Weighting

Raw tractography overestimates short connections and underestimates long ones. Filtering methods correct for these biases:

- **SIFT** (Spherical-deconvolution Informed Filtering of Tractograms): removes streamlines so that the streamline density matches the fiber orientation distribution (FOD) lobe integrals. Typically reduces 10M streamlines to ~2M.
- **SIFT2**: assigns a per-streamline weight rather than removing streamlines, preserving more connectivity information.
- **COMMIT** (Convex Optimization Modeling for Microstructure Informed Tractography): optimizes streamline weights by fitting a forward model to the diffusion signal.

### Edge Weighting Strategies

| Weight | Formula | Interpretation |
|--------|---------|----------------|
| **Streamline count** | N_ij | Raw number of streamlines between regions i and j |
| **Streamline density** | N_ij / (V_i + V_j) | Normalized by region volume; reduces size bias |
| **SIFT2 weight** | sum(w_s) for streamlines connecting i,j | Biologically informed weighting |
| **Mean FA along tract** | mean(FA(s)) for streamlines between i,j | Proxy for tract microstructural integrity |
| **Mean tract length** | mean(L(s)) | Important for conduction delay estimation |

## Functional Connectivity

FC matrices are computed from the pairwise statistical dependence between regional time series. Common measures:

- **Pearson correlation**: standard for resting-state fMRI; assumes linearity and stationarity.
- **Partial correlation** (precision matrix): direct connections only, removes indirect effects.
- **Coherence / imaginary coherence**: frequency-domain measure for EEG/MEG; imaginary part removes zero-lag (volume conduction) artifacts.
- **Phase-locking value (PLV)**: phase synchrony for bandpass-filtered EEG/MEG signals.
- **Mutual information**: nonlinear, captures any statistical dependence.

**Dynamic FC**: Sliding-window correlation or hidden Markov models (see [method-hmm-dynamics.md](method-hmm-dynamics.md)) capture time-varying connectivity states.

## Graph-Theoretic Measures

Once a connectome is represented as a weighted graph G = (V, E, W), standard network science measures characterize its topology:

| Measure | Definition | Interpretation |
|---------|-----------|----------------|
| **Degree** (k_i) | sum of edge weights at node i | Hub importance |
| **Strength** | sum(W_ij) for all j connected to i | Weighted degree |
| **Betweenness centrality** | Fraction of shortest paths passing through node i | Information relay importance |
| **Clustering coefficient** | Fraction of triangles around node i | Local segregation |
| **Modularity** (Q) | Quality of partition into communities | Functional segregation; typical Q ~ 0.4-0.6 for brain networks |
| **Rich club coefficient** | Connectivity among high-degree hubs exceeding random expectation | Hub-hub backbone; present in structural connectomes |
| **Small-world index** (sigma) | C/C_rand >> 1 and L/L_rand ~ 1 | High clustering with short path lengths; sigma > 1 for brain networks |
| **Global efficiency** | Mean inverse shortest path length | Integration capacity |
| **Participation coefficient** | Distribution of a node's connections across modules | Between-module connector vs. within-module provincial hub |

**Toolboxes**: Brain Connectivity Toolbox (BCT, MATLAB/Python), NetworkX, graph-tool, igraph. BCT remains the standard reference implementation for neuroscience-specific measures (Rubinov & Sporns 2010).

## Higher-Order Structure

Pairwise graphs cannot represent multi-way interactions. Two formalisms extend classical graph theory:

### Hypergraphs

A hypergraph H = (V, E) allows each hyperedge e in E to connect any number of vertices. Represented by an incidence matrix H in R^(n x m) where H_ij = 1 if vertex i belongs to hyperedge j. The **hgx** library provides JAX-native hypergraph neural networks with 14 convolution architectures, including tensorized layers (THNNConv) that capture higher-order vertex interactions within hyperedges via CP decomposition. See [method-hypergraph.md](method-hypergraph.md).

Applications to connectomics:
- Modeling synaptic cliques and functional modules as hyperedges
- Neural circuit motifs beyond pairwise connectivity
- Continuous-time dynamics on hypergraph topology via HypergraphNeuralODE/SDE (Diffrax integration)

### Simplicial Complexes

A simplicial complex K assigns a hierarchy of simplices: nodes (0-simplices), edges (1-simplices), triangles (2-simplices), etc. Clique complexes of structural connectomes reveal topological features via persistent homology. Hodge Laplacians L_k generalize the graph Laplacian to k-simplices, enabling spectral analysis of higher-order flow. The **hgx** library supports persistent homology and Hodge Laplacians via its topology extra.

## SC-FC Coupling

Structural connectivity constrains but does not determine functional connectivity. The SC-FC relationship is a central question in computational neuroscience:

- **Linear correlation**: Pearson r between SC and FC matrices is typically r ~ 0.3-0.5 across studies.
- **Communication models**: Shortest-path, diffusion, navigation, and search-information models predict FC from SC with varying accuracy.
- **Neural mass models**: Whole-brain simulation bridges SC to FC by propagating dynamics on the structural connectome. In vbjax, the coupling term `c = k * x.sum(axis=1)` (or weighted by the SC matrix) drives neural mass dynamics (MPR, Jansen-Rit, CMC) whose simulated BOLD or EEG signals produce emergent FC. See vbjax for available models: MPR (2D), JR (6D), CMC (8D), Dopa (6D).
- **Optimization target**: Differentiable simulation in JAX allows gradient-based fitting of SC weights or model parameters to minimize the distance between simulated and empirical FC (vbjax supports `jax.grad` through the simulation loop).

Regions of strong SC-FC coupling tend to be unimodal sensory cortices; weak coupling predominates in transmodal association cortex (Baum et al. 2020).

## Distance-Dependent Connectivity and Conduction Delays

White matter tract length introduces **conduction delays** that shape oscillatory dynamics and cross-frequency coupling:

- **Conduction velocity**: 5-20 m/s in myelinated cortical white matter (speed varies with axon diameter and myelination).
- **Delay matrix**: D_ij = L_ij / v, where L_ij is the mean tract length between regions i and j, and v is conduction velocity.
- **Typical delays**: 5-30 ms for cortico-cortical connections; up to 50 ms for long-range (e.g., prefrontal-occipital) tracts.
- **Impact on dynamics**: Delays promote multistability, traveling waves, and frequency-dependent synchronization in neural mass models. In vbjax, delays are incorporated via a delay buffer in the integration loop.

Exponential distance rule: connection probability decays exponentially with inter-regional Euclidean distance, with characteristic length scale ~30 mm in human cortex (Ercsey-Ravasz et al. 2013).

## Standard Connectome Datasets

| Dataset | Source | Subjects | Atlas/Resolution | Modality | Access |
|---------|--------|----------|-----------------|----------|--------|
| **HCP** (Human Connectome Project) | WU-Minn | 1200 | HCP-MMP1.0 (360), multiple | dMRI (multi-shell, 1.25mm) + rfMRI | ConnectomeDB |
| **Hagmann et al. (2008)** | Lausanne | 5 | Desikan (66/998 regions) | DSI tractography | Published matrices |
| **CoCoMac** | Literature | N/A | Macaque cortical areas | Tract tracing | cocomac.g-node.org |
| **Allen Mouse Brain** | Allen Institute | ~100 | Voxel (100 um) | Viral tracer injection | connectivity.brain-map.org |
| **TVB default** | Aggregated | Average | Desikan-Killiany (68) | DTI | Bundled with The Virtual Brain |

**Scale-free approximations**: Some theoretical models approximate connectome degree distributions as scale-free (P(k) ~ k^-gamma, gamma ~ 2-3), though empirical brain networks are better described as heavy-tailed with exponential cutoff rather than true power-law (Gastner & Odor 2016).

## Controllability of Brain Networks

Network control theory, implemented in **jaxctrl**, quantifies how external inputs (e.g., stimulation) can drive the brain's state transitions along the structural connectome:

- **Modal controllability**: ability to drive activity into difficult-to-reach states; high in default mode network hubs.
- **Average controllability**: ability to drive the system to many nearby states; high in highly connected regions.
- **Minimum driver nodes**: smallest set of nodes to control the full network (Liu, Slotine & Barabasi 2011).
- **Hypergraph controllability**: jaxctrl extends control theory to hypergraphs via tensor Kalman rank and algebraic Riccati tensor equations (ARTE), enabling controllability analysis of higher-order brain network structure.

## Key References

- **Margulies2016situated**: Margulies et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. PNAS 113:12574-12579.
- **Sydnor2021axis**: Sydnor et al. (2021). Neurodevelopment of the association cortices: patterns, mechanisms, and implications for psychopathology. Neuron 109:2820-2846.
- **Hansen2022neurotransmitter**: Hansen et al. (2022). Mapping neurotransmitter systems to the structural and functional organization of the human neocortex. Nature Neuroscience 25:1569-1581.
- **Markello2022neuromaps**: Markello et al. (2022). neuromaps: structural and functional interpretation of brain maps. Nature Methods 19:1472-1479.
- **Coifman2006diffusion**: Coifman & Lafon (2006). Diffusion maps. Applied and Computational Harmonic Analysis 21:5-30.
- **dong2024controllability**: Dong et al. (2024). Controllability and Observability of Temporal Hypergraphs. arXiv:2408.12085.

## Relevant Projects

| Project | Role in Connectomics |
|---------|---------------------|
| **hgx** | Hypergraph neural networks for higher-order connectivity; spectral methods, persistent homology, continuous dynamics on hypergraph topology |
| **vbjax** | Connectome-coupled whole-brain simulation; SC matrix drives neural mass models (MPR, JR, CMC); gradient-based SC-FC fitting |
| **jaxctrl** | Network controllability analysis on graphs and hypergraphs; Lyapunov/Riccati solvers; minimum driver node identification |
| **sbi4dwi** | Diffusion MRI microstructure estimation via simulation-based inference; provides tract-level metrics for SC weighting |

## See Also

- [diffusion-mri.md](diffusion-mri.md) -- dMRI physics and tractography fundamentals
- [structural-mri.md](structural-mri.md) -- T1w/T2w imaging for parcellation
- [method-hypergraph.md](method-hypergraph.md) -- Hypergraph convolution architectures and control
- [method-source-imaging.md](method-source-imaging.md) -- EEG/MEG inverse problems for FC estimation
- [method-spectral-analysis.md](method-spectral-analysis.md) -- Frequency-domain FC measures (coherence, PLV)
- [method-hmm-dynamics.md](method-hmm-dynamics.md) -- Dynamic FC via hidden Markov models
- [jax-ecosystem.md](jax-ecosystem.md) -- JAX tools for connectome analysis and simulation
