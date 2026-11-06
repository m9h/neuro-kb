---
type: physics
title: Cerebrovascular, Metabolic and Pharmacokinetic Physiology
governing_equations: "Balloon-Windkessel ODE (s, f, v, q); Riera 8-state multi-compartment NVC ODE; Fick's principle CMRO2 = CBF × OEF × CaO2; flow-dependent extraction E(f) = 1 − (1 − E0)^(1/f); Grubb power law F_out = v^(1/α); qBOLD S(TE) = S0·F(OEF, DBV, TE)·exp(−R2·TE)"
related: [physics-hemodynamic.md, fmri.md, fnirs.md, neural-mass-models.md, quantitative-mri.md, mrs.md, physics-bloch.md, tissue-csf.md, tissue-gray-matter.md, method-pharmaco-eeg.md, method-neural-ode.md, method-sbi.md, method-variational-inference.md, connectomics.md, jax-ecosystem.md, meg.md, eeg.md, tms.md, structural-mri.md]
description: "This page covers the physiological chain that converts neural activity into vascular and metabolic observables: neural drive → metabolic demand (CMRO₂) → vasodilatory signalling (NO, adenosine) → cerebral blood flow (CBF) → blood volume (CB"
timestamp: 2026-11-05T15:59:53-08:00
tags: [physics, hemodynamic, physiology, pharmacokinetics]
---

# Cerebrovascular, Metabolic and Pharmacokinetic Physiology

This page covers the **physiological chain that converts neural activity into vascular and metabolic observables**: neural drive → metabolic demand (CMRO₂) → vasodilatory signalling (NO, adenosine) → cerebral blood flow (CBF) → blood volume (CBV) and deoxyhemoglobin (dHb) → BOLD / ASL / VASO / qBOLD signals. It also covers systemic modulators of that chain (cardiac, autonomic, sleep state) and the current status of pharmacokinetic modelling. The general physics of hemodynamic signal formation (optical and MR contrast mechanisms) lives in [physics-hemodynamic.md](physics-hemodynamic.md); this page is the model-level physiology implemented in `vpjax` on top of neural drive from `vbjax`.

## The causal chain

```
vbjax:  neural mass model → synaptic activity → coupling        [@woodman2024insamuvbjax]
  ↓ u(t)
vpjax:  activity → CMRO₂ → vasodilation (NO, adenosine)
        → CBF (compliance, autoregulation) → CBV, dHb (Balloon)
        → BOLD (T2*) | ASL (kinetic model) | VASO (CBV) | qBOLD (regional OEF)
```
(source: `vpjax:README.md`)

`vbjax.make_bold()` provides a simplified Balloon-Windkessel model adequate for BOLD fitting [@woodman2024insamuvbjax]. `vpjax` adds what it lacks: metabolic intermediaries, compartmental vascular geometry, an ASL forward model, regional OEF from qBOLD, and Local Linearization integration for stiff neurovascular ODEs (`vpjax:README.md`).

## 1. Balloon-Windkessel model (single compartment)

State vector **y** = (s, f, v, q): vasodilatory signal, normalized inflow (CBF), normalized volume (CBV), normalized dHb content (Buxton 1998; Friston 2000; source `vpjax:docs/tutorials/balloon_windkessel.md`):

$$
\begin{aligned}
\dot{s} &= u - \kappa s - \gamma (f - 1) \\
\dot{f} &= s \\
\tau \dot{v} &= f - v^{1/\alpha} \\
\tau \dot{q} &= \frac{f\,E(f)}{E_0} - \frac{v^{1/\alpha} q}{v}, \qquad E(f) = 1 - (1-E_0)^{1/f}
\end{aligned}
$$

- κ: signal decay rate (κ = 0.65 s⁻¹ is the value used in the vpjax gradient example); γ: flow-dependent feedback; τ: venous transit time; α: Grubb exponent (outflow ∝ v^{1/α}); E₀: resting OEF.
- Steady state at rest: s = 0, f = v = q = 1; all observation functions return 0 (they report change from baseline).
- **Worked example:** with E₀ = 0.34, a 50 % flow increase (f = 1.5) lowers extraction to E ≈ 0.24 — flow outpaces oxygen consumption, which is why BOLD is positive.
- Integration: `solve_balloon` uses Diffrax Tsit5 with piecewise-constant stimulus interpolation; `jax.grad` of peak BOLD w.r.t. κ is finite and non-zero, enabling gradient-based fitting.

### Observation signatures

| Modality | Function | Sensitive to | Sign during activation |
|---|---|---|---|
| BOLD | `observe_bold()` | dHb (q, v) | Positive (dHb washout) |
| ASL | `observe_asl()` | CBF (f) | Positive |
| VASO | `observe_vaso()` | CBV (v); S ∝ (1 − CBV) | **Negative** |

ASL peaks before (or with) BOLD, since BOLD depends on the slow venous v and q dynamics while ASL tracks f directly (`vpjax:docs/tutorials/balloon_windkessel.md`). See [fmri.md](fmri.md) for acquisition-side details.

## 2. Riera multi-compartment neurovascular coupling

The Riera model (Riera 2006/2007; Sotero & Trujillo-Barreto 2007) splits the single venous balloon into **arteriolar** (compliant, actively regulated), **capillary** (stiff, oxygen exchange) and **venous** (compliant, passive, BOLD origin) compartments, with two vasodilatory pathways (source `vpjax:docs/tutorials/riera_nvc.md`):

1. **Nitric oxide** — fast, neurally driven (interneuron NO onto arteriolar smooth muscle), τ ≈ 1/κ_NO ≈ 1.5 s.
2. **Adenosine** — slow, metabolically driven (ATP consumption when CMRO₂ rises), τ ≈ 1/κ_ade ≈ 2.5 s.

8-state vector (s_NO, s_ade, f_a, v_a, v_c, v_v, q_v, CMRO₂):

$$
\begin{aligned}
\dot{s}_{NO} &= c_{NO} u - \kappa_{NO} s_{NO} - \gamma_{NO}(f_a - 1) \\
\dot{s}_{ade} &= c_{ade}(\mathrm{CMRO_2} - 1) - \kappa_{ade} s_{ade} - \gamma_{ade}(f_a - 1) \\
\dot{f}_a &= s_{NO} + s_{ade} \\
\tau_a \dot{v}_a &= f_a - v_a^{1/\alpha_a}, \quad
\tau_c \dot{v}_c = v_a^{1/\alpha_a} - v_c^{1/\alpha_c}, \quad
\tau_v \dot{v}_v = v_c^{1/\alpha_c} - v_v^{1/\alpha_v} \\
\tau_v \dot{q}_v &= \frac{f_a E(f_a)}{E_0} - \frac{v_v^{1/\alpha_v} q_v}{v_v}, \qquad
\tau_m \dot{\mathrm{CMRO_2}} = (1 + \phi u) - \mathrm{CMRO_2}
\end{aligned}
$$

### Parameters (vpjax `RieraParams` defaults)

| Parameter | Symbol | Default | Unit / meaning |
|---|---|---|---|
| `kappa_no` | κ_NO | 0.65 | s⁻¹, NO signal decay |
| `kappa_ade` | κ_ade | 0.40 | s⁻¹, adenosine signal decay |
| `c_no` | c_NO | 1.0 | NO coupling gain |
| `c_ade` | c_ade | 0.5 | adenosine coupling gain |
| `tau_a` | τ_a | 0.5 | s, arteriolar transit |
| `tau_c` | τ_c | 1.0 | s, capillary transit |
| `tau_v` | τ_v | 2.0 | s, venous transit |
| `alpha_a` | α_a | 0.20 | arteriolar Grubb exponent |
| `alpha_c` | α_c | 0.10 | capillary Grubb exponent (stiffest) |
| `alpha_v` | α_v | 0.32 | venous Grubb exponent |
| `E0` | E₀ | 0.34 | resting OEF |
| `phi` | φ | 0.5 | metabolic–neural coupling (CMRO₂ → 1.5 at u = 1) |
| `tau_m` | τ_m | 3.0 | s, metabolic response time constant |

Total arterial-to-venous transit at rest is τ_a + τ_c + τ_v = 3.5 s. Resting CBV distribution used by `riera_total_cbv`:

$$\mathrm{CBV_{total}} = 0.20\,v_a + 0.05\,v_c + 0.75\,v_v$$

i.e. ~75 % of cerebral blood volume is venous. `riera_to_balloon` maps the 8-state system back to Balloon-equivalent (v, q) for BOLD observation.

## 3. Oxygen metabolism: Fick's principle and the CMRO₂ hierarchy

$$\mathrm{CMRO_2} = \mathrm{CBF} \times \mathrm{OEF} \times C_aO_2$$
(`vpjax:metabolism/fick.py`). vpjax estimates CMRO₂ at three levels of specificity (`vpjax:README.md`):

| Level | Method | OEF source | Spatial | Temporal | Data |
|---|---|---|---|---|---|
| 1 Global | TRUST + pCASL | Sagittal sinus venous T2 → SvO₂ (Lu 2008) | Global OEF | Static | pCASL + TRUST |
| 2 Regional | qBOLD + pCASL | Per-voxel multi-echo GRE | Voxel | Static | pCASL + 7-echo GRE |
| 3 Dynamic | Riera model | Model state | Regional | Time-varying | MEG + fMRI + ASL |

### qBOLD signal model

$$S(TE) = S_0 \cdot F(\mathrm{OEF}, \mathrm{DBV}, TE) \cdot e^{-R_2 TE}, \qquad R_2' = R_2^* - R_2 \propto \mathrm{OEF} \times \mathrm{DBV} \times B_0$$

F models the intravoxel frequency distribution from randomly oriented deoxygenated vessels (He & Yablonskiy 2007). Short TE is dominated by R₂; long TE by the reversible R₂′ from deoxy-Hb. The WAND 7-echo GRE (TE = 5–35 ms, 0.67 mm isotropic) is the target acquisition. Relaxation physics: [physics-bloch.md](physics-bloch.md), [quantitative-mri.md](quantitative-mri.md). Susceptibility χ = χ_iron (+) + χ_myelin (−) from QSM is used to separate iron from myelin contributions to R₂* (`vpjax:qsm/`).

## 4. Vascular mechanics and layer specificity

- **Compliance:** pressure–volume relations follow the Grubb power law (v ∝ f^α); the smaller α_c = 0.10 encodes capillary stiffness relative to arterioles (0.20) and veins (0.32) (`vpjax:vascular/compliance.py`).
- **Autoregulation:** Lassen-curve plateau of CBF vs perfusion pressure (`vpjax:vascular/autoregulation.py`).
- **Cortical depth:** layer-specific NVC distinguishes superficial (feedback, layers I–III) from deep (feedforward, layers V–VI) cortex in the ~2.5 mm ribbon; at 0.67–0.7 mm, ~2–3 depth bins are resolvable. VASO deveining/BOCO follow LAYNII (`LN_BOCO`, `LN2_DEVEIN`) (`vpjax:README.md`). See [tissue-gray-matter.md](tissue-gray-matter.md), [structural-mri.md](structural-mri.md).

## 5. Systemic modulators

### Cardiac coupling (`vpjax:docs/tutorials/cardiac_coupling.md`)

| Pathway | Model | Quantitative anchors |
|---|---|---|
| Brain → heart | DLPFC → mPFC → vagus ODE (`VagalODE`) | Baseline RR ≈ 857 ms (~70 bpm); iTBS-like 2 s on / 8 s off (0.1 Hz) entrains HR deceleration |
| Heart → brain | Baroreceptor → NTS inhibition | Excitability minimum at ~0.3π rad after R-peak (systole), maximum ~π (diastole) |
| Heart → signal | CBV pulsatility | ~1 Hz CBV/BOLD/ASL confound; arterial pressure 60–140 mmHg; amplitude scales with compliance (0.01–0.05) |

The vagal pathway links to [tms.md](tms.md) (DLPFC stimulation as a heart-rate biomarker).

### Sleep state and glymphatics (`vpjax:docs/tutorials/sleep_physiology.md`)

- NVC gain ordering: Wake > N1 > N2 > N3; REM closer to wake than to N3. N3 reduces Riera c_NO.
- Locus coeruleus norepinephrine oscillates at ~0.02 Hz (period ~50 s), driving arteriolar vasomotion; CBV amplitude ~0.005 (wake) vs ~0.03 (NREM).
- CBV oscillations drive delayed perivascular CSF flow; cumulative flow is the glymphatic clearance proxy (Fultz 2019). See [tissue-csf.md](tissue-csf.md).

## 6. Pharmacokinetics and autonomic state-space identification

Drug PK/PD is a **planned** extension, not yet implemented. The current first step is a reduced joint-autonomic model fitted to D'Amelio's multimodal DMT release [@damelio2026multimodal] (no fMRI in the dataset):

$$y_k(t) = \ell_k\,x(t) + \varepsilon_k, \qquad x(t+dt) = x(t) + \eta, \; \mathrm{Var}(\eta) = \sigma_p^2\,dt$$

Scalar Kalman filtering + Rauch–Tung–Striebel smoothing over the union of sensor timestamps (heart rate, SMNA, RVT, optionally EEG); missing samples are skipped, never interpolated. The latent x is an **uncalibrated shared drive**, not an identified sympathetic, vagal or receptor-occupancy state; uncertainty is conditional on fixed loadings ℓ_k and noise SDs (`vpjax:docs/tutorials/dmt_shared_drive.md`). Not yet modelled: sensor delays, respiratory gas exchange, pressure-generating circulation, drug PK/PD, neural-to-BOLD coupling. For spectral drug effects on EEG see [method-pharmaco-eeg.md](method-pharmaco-eeg.md).

## 7. Numerics and identifiability

- **Stiffness:** neurovascular ODEs mix ~0.5 s arteriolar and ~3 s metabolic time scales; vpjax provides Ozaki (1992) Local Linearization (`vpjax:integrators/local_linearization.py`) alongside Diffrax. See [method-neural-ode.md](method-neural-ode.md).
- **Identifiability:** each Balloon state has an independent observable — f (pCASL), v (VASO, planned), q (qBOLD), with s inferred — giving "zero free hemodynamic parameters" when all are measured. Neural drive is constrained by [meg.md](meg.md); neurotransmitters by [mrs.md](mrs.md) GABA/Glu (`vpjax:README.md`).
- **Inference:** end-to-end differentiability supports gradient fitting, MCMC (NumPyro in vbjax [@woodman2024insamuvbjax]), [method-variational-inference.md](method-variational-inference.md) and [method-sbi.md](method-sbi.md).

## Relevant Projects

- **vpjax** — all models on this page: `hemodynamics/` (balloon, riera, bold, optics), `metabolism/` (cmro2, oef, fick), `vascular/`, `perfusion/` (asl, kinetic, trust), `qbold/`, `qsm/`, `vaso/`, `layers/`, `cardiac/`, `sleep/`, `integrators/`.
- **vbjax** — neural mass drive u(t) (e.g. Montbrió-Pazó-Roxin), connectome coupling, simplified `make_bold()` [@woodman2024insamuvbjax].
- **neurojax** — exports baseline-standardized DMT observations (`examples/dmt_observations.py`) consumed by vpjax; pharmaco-EEG spectral models.
- **LAYNII** — layer definition, VASO BOCO/deveining for layer-resolved NVC.
- **mrs-jax** — GABA/Glu constraints on metabolic coupling.
- **hippy-feat** — fMRI preprocessing/GLM that consumes hemodynamic response models.

## See Also

- [physics-hemodynamic.md](physics-hemodynamic.md) — hemodynamic contrast physics (optical and MR)
- [fmri.md](fmri.md), [fnirs.md](fnirs.md) — modalities observing this chain
- [neural-mass-models.md](neural-mass-models.md) — source of neural drive u(t)
- [quantitative-mri.md](quantitative-mri.md), [physics-bloch.md](physics-bloch.md) — R₂, R₂*, QSM, T1
- [mrs.md](mrs.md) — metabolite constraints
- [tissue-csf.md](tissue-csf.md), [tissue-gray-matter.md](tissue-gray-matter.md) — tissue context
- [method-pharmaco-eeg.md](method-pharmaco-eeg.md) — drug effects on EEG spectra
- [method-neural-ode.md](method-neural-ode.md), [method-sbi.md](method-sbi.md), [method-variational-inference.md](method-variational-inference.md) — fitting and inference
- [connectomics.md](connectomics.md), [jax-ecosystem.md](jax-ecosystem.md)
