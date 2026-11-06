---
type: physics
title: Hemodynamic Response / Neurovascular Coupling
physics: hemodynamic
governing_equations: "Balloon–Windkessel model (flow-inducing signal s, inflow f, venous volume v, deoxyhemoglobin q); BOLD observation y = V0[k1(1-q) + k2(1-q/v) + k3(1-v)]; linear convolution y = x ⊗ h; Fick principle CMRO2 = CBF·OEF·CaO2; Davis calibrated-BOLD model"
related: [fmri.md, physics-cerebrovascular.md, neural-mass-models.md, quantitative-mri.md, fnirs.md, eeg.md, meg.md, physics-bloch.md, physics-electromagnetic.md, method-neural-ode.md, method-sbi.md, method-variational-inference.md, method-neural-fields.md, connectomics.md, fus.md, method-pharmaco-eeg.md, structural-mri.md]
description: "Neurovascular coupling is the chain linking neuronal activity to local changes in cerebral blood flow (CBF), blood volume (CBV) and oxygen metabolism (CMRO₂)."
timestamp: 2026-11-05T15:59:53-08:00
tags: [physics, hemodynamic, neurovascular]
---

# Hemodynamic Response / Neurovascular Coupling

Neurovascular coupling is the chain linking neuronal activity to local changes in cerebral blood flow (CBF), blood volume (CBV) and oxygen metabolism (CMRO₂). The **hemodynamic response function (HRF)** is the system's impulse response as seen by a given sensor: BOLD [fmri.md](fmri.md), HbO/HbR in [fnirs.md](fnirs.md), or CBV-weighted power Doppler in [fus.md](fus.md). It is the forward model connecting fast electrophysiology ([eeg.md](eeg.md), [meg.md](meg.md), [neural-mass-models.md](neural-mass-models.md)) to slow hemodynamic measurements. Vessel-level mechanics (Windkessel compliance, autoregulation, CO₂ reactivity) live in [physics-cerebrovascular.md](physics-cerebrovascular.md); this page covers the neural-to-signal transfer.

## Causal chain

1. **Neural drive** — synaptic activity (glutamate release, astrocytic Ca²⁺, NO/prostaglandin signalling), not spiking per se, is the primary vasoactive driver.
2. **Vasodilation** — arteriolar smooth muscle and capillary pericytes relax within ~0.5–1 s.
3. **Functional hyperemia** — CBF rises ~30–70%, while CMRO₂ rises only ~10–30%; the coupling ratio n = ΔCBF/ΔCMRO₂ ≈ 2–4.
4. **Washout of deoxyhemoglobin** — because flow outpaces consumption, venous O₂ saturation rises and paramagnetic dHb falls. This produces the *positive* BOLD signal (longer T2*, see [physics-bloch.md](physics-bloch.md)).
5. **Venous volume lag** — slow venous compliance ("balloon") delays CBV recovery. Together with persistent CMRO₂ elevation, this contributes to the post-stimulus undershoot.

## Governing equations: Balloon–Windkessel model

This is the hemodynamic model of Friston et al. (2000), built on Buxton et al. (1998), in the parameterisation of Stephan et al. (2007). It is the same model used by TVB-style simulators. With neural input x(t):

```
ds/dt = x − κ·s − γ·(f − 1)                 # vasodilatory (flow-inducing) signal
df/dt = s                                    # normalised inflow f = F/F0
τ0·dv/dt = f − v^(1/α)                       # normalised venous volume (Windkessel outflow)
τ0·dq/dt = f·E(f,E0)/E0 − v^(1/α)·q/v        # normalised deoxyhemoglobin
E(f,E0) = 1 − (1 − E0)^(1/f)                 # flow-dependent O2 extraction
```

The BOLD observation equation separates intravascular and extravascular contributions:

```
y = V0 · [ k1·(1 − q) + k2·(1 − q/v) + k3·(1 − v) ]
k1 = 4.3·ϑ0·E0·TE,   k2 = ε·r0·E0·TE,   k3 = 1 − ε      (Stephan 2007)
```

In the original Friston 2000 form, the coefficients are k1 = 7E0, k2 = 2, k3 = 2E0 − 0.2.

### Parameters

| Parameter | Symbol | Value | Units | Source |
|---|---|---|---|---|
| Rate constant of vasodilatory signal decay | κ (1/τ_s) | 0.65 (0.64 in Stephan 2007) | s⁻¹ | Friston 2000 |
| Rate constant of flow-dependent autoregulation | γ (1/τ_f) | 0.41 (0.32 in Stephan 2007) | s⁻¹ | Friston 2000 |
| Mean venous transit time | τ0 | 0.98 (2.0 in Stephan 2007) | s | Friston 2000 |
| Grubb exponent (outflow–volume) | α | 0.32 | – | Friston 2000 |
| Grubb exponent, total CBV–CBF (original) | α_Grubb | 0.38 | – | Grubb 1974 |
| Grubb exponent, venous CBV only | α_v | ~0.2–0.23 | – | Chen & Pike 2010 |
| Resting O₂ extraction fraction | E0 | 0.34 (Friston 2000); 0.4 (Buxton 1998, Stephan 2007) | – | — |
| Resting venous blood volume fraction | V0 | 0.02 (Friston 2000); 0.04 (Stephan 2007) | – | — |
| Frequency offset of magnetised vessel surface | ϑ0 | 40.3 at 1.5 T; scales ∝ B0 (≈80.6 at 3 T) | s⁻¹ | Obata 2004 |
| Intravascular relaxation slope | r0 | 25 at 1.5 T (field-dependent, larger at 3 T) | s⁻¹ | Obata 2004 |
| Intra/extravascular signal ratio | ε | ~0.5–1.4 (field- and TE-dependent) | – | Obata 2004 |
| Echo time | TE | 0.04 (typical 3 T GRE-EPI default) | s | — |

> **Correction note.** A previous version of this page listed τ_v = 35 s and τ_q = 20 s. These are not balloon-model values; the transit time τ0 is ≈1–2 s. It also omitted the 1/E0 normalisation in the dq/dt equation. Both errors are fixed above.

## Canonical HRF shapes

For linear GLM analysis, the balloon dynamics are summarised by a fixed kernel h(t), and the predicted signal is y(t) = (x ⊗ h)(t) + drift + ε.

| Model | Form | Key numbers |
|---|---|---|
| SPM canonical double-gamma | Γ(t; 6, 1) − (1/6)·Γ(t; 16, 1) | Peak ≈ 5 s, undershoot minimum ≈ 15 s, return to baseline ≈ 25–30 s |
| Glover (1999) | (t/d1)^a1·e^{−(t−d1)/b1} − c·(t/d2)^a2·e^{−(t−d2)/b2}, with d_i = a_i·b_i | a1 = 6, a2 = 12, b1 = b2 = 0.9 s, c = 0.35 → peak ≈ 5.4 s, undershoot ≈ 10.8 s |
| Canonical + temporal/dispersion derivatives | 3 regressors | Absorbs ±1 s latency and width variation |
| FLOBS (Woolrich et al. 2004) | 3-basis optimal set | Used by hippy-feat Variant B |
| Per-voxel library (GLMsingle-style) | 20 candidate HRFs, one selected per voxel | Used by hippy-feat Variant C |
| FIR | 1 regressor per post-stimulus bin | Unconstrained; typically spans 0–30 s |

The previous version of this page gave Glover parameters a2 = 16 and c = 0.167. Those values mix the SPM gamma-pdf convention with the Glover formula. Use one convention or the other, not both.

An **initial dip** (~1–2 s, a transient dHb increase) is reported mainly at high field and in optical imaging. It is small and inconsistently observed, so treat it as optional in forward models.

### Variability and nonlinearity

- **Latency variability:** HRF peak latency varies across regions and subjects by roughly ±1–2 s. This is why fixed-HRF real-time pipelines lose accuracy relative to GLMsingle ([@hough2026hippyfeat]).
- **Refractory nonlinearity:** responses to closely spaced events (ISI < ~2 s) are sub-additive. This is captured by the nonlinear balloon equations, not by convolution.
- **Draining-vein bias:** gradient-echo BOLD is weighted toward pial veins, which displaces and delays signal at the cortical surface. This motivates laminar analysis (LAYNII) and spin-echo or VASO contrasts at 7 T.

## Measuring the coupling quantities directly

These measurements constrain balloon-model parameters (E0, V0, CBF0) per subject instead of relying on literature priors. See [quantitative-mri.md](quantitative-mri.md).

| Quantity | MRI method | Protocol detail (WAND, [@mcnabb2025wand]) | Model |
|---|---|---|---|
| Venous O₂ saturation Yv → global OEF | TRUST | `ep2d_TRUST_AsymShTE`, 24 volumes. Standard eTEs are 0/40/80/160 ms; the asymmetric variant is unconfirmed (`neurojax:docs/CUBRIC_QUESTIONS.md`) | Blood T2 → Yv calibration; OEF = (Ya − Yv)/Ya |
| CBF | pCASL | PLD 2.0 s, 110 volumes (≈55 control–label pairs), background suppression (`SAT2`) | Single-PLD kinetic model; needs blood T1 |
| Blood T1 | Inversion recovery | `mg_IR_T1_VE11c`, 960 timepoints (probably Look-Locker; needs T1* → T1 correction) | Literature ~1.6–1.7 s at 3 T |
| Local OEF, deoxygenated blood volume (DBV) | qBOLD (multi-echo GRE, ses-06) | T2*/R2* maps in `derivatives/qmri/` | R2′ = DBV·(4/3)πγ·Δχ0·Hct·(1−Y)·B0 (He & Yablonskiy 2007) |
| CMRO₂ | TRUST + pCASL | — | Fick principle: CMRO₂ = CBF · OEF · CaO₂ |

**Calibrated BOLD (Davis model).** The relative BOLD change is

ΔS/S0 = M·[1 − (CBF/CBF0)^(α−β)·(CMRO₂/CMRO₂,0)^β]

with α ≈ 0.2 (venous Grubb exponent) and β ≈ 1.3–1.5. M is the maximum BOLD change, obtained from hypercapnia or hyperoxia calibration; dual-calibrated fMRI (Germuska et al. 2019) uses both. The earlier version of this page stated a different, incorrect formula.

## Multi-modal forward models

The generic simulation path from electrophysiology to BOLD is:

```
neural mass state (e.g. RWW S_E, Jansen-Rit PSP)  ──►  x(t)  ──►  balloon ODE  ──►  y_BOLD (sampled at TR)
            └──►  dipoles × leadfield  ──►  EEG/MEG
```

- **Timescale separation.** Neural dynamics are integrated at dt ≈ 0.1–1 ms, the balloon ODE at ~1–10 ms, and BOLD is sampled at TR ≈ 0.5–2 s. In JAX, the stiff balloon ODE can be integrated with Diffrax and differentiated end-to-end ([method-neural-ode.md](method-neural-ode.md)).
- **Units matter.** neurojax found that the reduced Wong–Wang model had γ_E = 0.641/1000 where it should have been 0.641 s⁻¹. The error drove the single-node fixed point to S_E ≈ 0.0002 instead of the Deco 2013 value of ≈0.164, which would silently corrupt any simulated BOLD derived from it (`neurojax:docs/BUG_REPORTS.md`, fixed in commit `49acac2`).
- **Connectome-scale BOLD.** Simulated BOLD functional connectivity is the standard fit target for whole-brain models on structural connectomes ([connectomics.md](connectomics.md), [method-neural-fields.md](method-neural-fields.md)). vbjax provides network SDE integrators and a BOLD monitor for this ([@woodman2024insamuvbjax]); neurojax wraps it via `neurojax:src/neurojax/bench/` (vbjax adapter).
- **EEG–fMRI fusion.** hippy-feat's `jaxoccoli/fusion.py` drives a balloon model from EEG features. It is marked experimental (`hippy-feat:README.md`).

## Inference, identifiability, and real-time use

**Identifiability.** κ, γ, τ0, α and E0 are strongly correlated given BOLD alone. E0 and V0 in particular trade off almost perfectly. Practical remedies:

- Fix them from TRUST/qBOLD/ASL measurements.
- Use informative priors with variational Laplace ([method-variational-inference.md](method-variational-inference.md)).
- Check posterior calibration with simulation-based inference ([method-sbi.md](method-sbi.md)).

vpjax targets this structural and practical identifiability analysis for cerebrovascular and metabolic state-space models. Drug effects on vascular tone confound pharmaco-fMRI and pharmaco-EEG comparisons ([method-pharmaco-eeg.md](method-pharmaco-eeg.md)).

**Beta-series estimation.** Single-trial betas depend on the HRF assumed, and least-squares-separate estimation reduces collinearity for fast event designs ([@mumford2012deconvolving]). Correlating OLS point estimates discards variance and biases beta-series connectivity ([@rissman2004measuring]). hippy-feat addresses this with conjugate and AR(1)-prewhitened Bayesian GLMs following BROCCOLI ([@eklund2014broccoli]), and propagates (β mean, β variance) through differentiable parcellation ([@ciric2022differentiable]).

**Real-time HRF trade-offs.** hippy-feat benchmarks HRF strategies for real-time MindEye decoding ([@scotti2024mindeye2]). Each variant outputs 8627 z-scored betas, with TR = 1.5 s; the core per-volume preprocessing takes ~54 ms, giving 28× headroom.

| Variant | HRF treatment | Per-TR time |
|---|---|---|
| A | Glover canonical | 163 ms |
| A+N | Glover + CSF/WM nuisance regressors | ~170 ms |
| B | FLOBS 3-basis | 155 ms |
| C | Per-voxel HRF (20-HRF library, preselected) | 33 ms |
| D | Bayesian shrinkage (conjugate Gaussian) | 237 ms |
| C+D | Per-voxel HRF + Bayesian | 39 ms |

Source: `hippy-feat:README.md`.

## Relevant Projects

- **vbjax** — network SDE simulation of neural mass models with BOLD monitoring; differentiable, so balloon parameters can be fit by gradient descent or NumPyro MCMC.
- **neurojax** — qBOLD OEF and multi-echo T2* fitting (`src/neurojax/qmri/`). Also covers WAND TRUST/pCASL/blood-T1 processing, reduced Wong–Wang neural drive, and the vbjax adapter for BOLD-level simulation.
- **hippy-feat** — HRF libraries (Glover, FLOBS, per-voxel), Bayesian and AR(1) GLMs, variance-propagating beta series, and experimental EEG–fMRI balloon fusion.
- **vpjax** — virtual physiology: cerebrovascular and metabolic models, pharmacokinetic modulation of vascular tone, and state-space identifiability of hemodynamic parameters.
- **LAYNII** — laminar fMRI analysis where draining-vein HRF bias is central.
- **open-fus-plan** — CBV-weighted power Doppler, an alternative hemodynamic readout.

## See Also

- [fmri.md](fmri.md) — BOLD acquisition and contrast mechanisms
- [physics-cerebrovascular.md](physics-cerebrovascular.md) — vessel mechanics, autoregulation, CBF/CBV physiology
- [physics-bloch.md](physics-bloch.md) — T2/T2* relaxation underlying BOLD
- [quantitative-mri.md](quantitative-mri.md) — qBOLD, T2*, relaxometry
- [fnirs.md](fnirs.md) and [tissue-optical-properties.md](tissue-optical-properties.md) — optical hemodynamic measurement
- [fus.md](fus.md) — functional ultrasound CBV imaging
- [neural-mass-models.md](neural-mass-models.md) — neural drive x(t)
- [method-neural-fields.md](method-neural-fields.md), [connectomics.md](connectomics.md) — whole-brain BOLD simulation
- [method-neural-ode.md](method-neural-ode.md), [method-sbi.md](method-sbi.md), [method-variational-inference.md](method-variational-inference.md) — fitting hemodynamic models
- [eeg.md](eeg.md), [meg.md](meg.md), [physics-electromagnetic.md](physics-electromagnetic.md) — electrophysiological counterpart in multimodal fusion

## Literature (not yet in bibliography; add BibTeX keys before citing)

- Buxton, Wong & Frank (1998). Dynamics of blood flow and oxygenation changes during brain activation: the balloon model. *MRM* 39:855–864.
- Friston, Mechelli, Turner & Price (2000). Nonlinear responses in fMRI: the Balloon model, Volterra kernels, and other hemodynamics. *NeuroImage* 12:466–477.
- Stephan et al. (2007). Comparing hemodynamic models with DCM. *NeuroImage* 38:387–401.
- Obata et al. (2004). Discrepancies between BOLD and flow dynamics in primary and supplementary motor areas. *NeuroImage* 21:144–153.
- Glover (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage* 9:416–429.
- Grubb et al. (1974). *Stroke* 5:630–639. Chen & Pike (2010). *JCBFM* 30:1094–1099.
- Riera et al. (2004). A state-space model of the hemodynamic approach. *NeuroImage* 21:547–567. (The previous version of this page misdated this paper as 2006.)
- He & Yablonskiy (2007). *MRM* 57:115–126. Lu & Ge (2008). *MRM* 60:357–363. Germuska et al. (2019). *NeuroImage* 184:717–728.
- Deco et al. (2013). *J Neurosci* 33:11239–11252.
