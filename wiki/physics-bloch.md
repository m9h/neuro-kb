---
type: physics
title: Bloch Equations / MR Physics
physics: electromagnetic
governing_equations: dM/dt = γ(M × B) - R(M - M₀)
related: [physics-nmr.md, modality-fmri.md, modality-dti.md, coordinate-system-scanner.md, tissue-brain.md, tissue-csf.md]
---

# Bloch Equations / MR Physics

The Bloch equations describe the quantum mechanical evolution of nuclear magnetic moments in external magnetic fields, forming the fundamental physics underlying all MRI modalities including functional MRI, diffusion MRI, and MR spectroscopy.

## Governing Equations

The Bloch equations govern magnetization dynamics in the presence of static field **B₀**, RF pulses **B₁(t)**, and relaxation:

```
dMₓ/dt = γ(MyBz - MzBy) - Mₓ/T₂
dMy/dt = γ(MzBₓ - MₓBz) - My/T₂  
dMz/dt = γ(MₓBy - MyBₓ) - (Mz - M₀)/T₁
```

Where:
- **M** = (Mₓ, My, Mz) is the magnetization vector
- **γ** = gyromagnetic ratio (42.58 MHz/T for ¹H)
- **T₁** = longitudinal relaxation time
- **T₂** = transverse relaxation time  
- **M₀** = equilibrium magnetization

## Physical Parameters

### Gyromagnetic Ratios
| Nucleus | γ/2π (MHz/T) | Clinical Use |
|---------|--------------|--------------|
| ¹H | 42.58 | Standard MRI, MRS |
| ¹³C | 10.71 | Hyperpolarized imaging, metabolism |
| ³¹P | 17.25 | Energy metabolism, ATP |
| ²³Na | 11.26 | Sodium imaging |

### Tissue Relaxation Times (3T)

| Tissue | T₁ (ms) | T₂ (ms) | T₂* (ms) | Source |
|--------|---------|---------|----------|---------|
| Gray matter | 1331 | 110 | 69 | Stanisz 2005 |
| White matter | 832 | 79.6 | 53 | Stanisz 2005 |
| CSF | 4163 | 2569 | 503 | Stanisz 2005 |
| Blood | 1441 | 290 | 50 | Lu 2004 |

### Field Strength Effects

Relaxation times are field-dependent:
- **T₁ increases** with B₀ (longer recovery)
- **T₂ approximately constant** 
- **T₂* decreases** with B₀ (increased susceptibility effects)

| Field | GM T₁ (ms) | WM T₁ (ms) | Source |
|-------|------------|------------|---------|
| 1.5T | 1124 | 786 | Wansapura 1999 |
| 3T | 1331 | 832 | Stanisz 2005 |
| 7T | 1939 | 1220 | Rooney 2007 |

## MRI Signal Formation

### RF Excitation
RF pulse with flip angle α rotates magnetization:
```
Mz' = M₀ cos(α)
Mxy' = M₀ sin(α)
```

### Free Induction Decay (FID)
After excitation, transverse magnetization precesses and decays:
```
S(t) = M₀ sin(α) exp(-t/T₂*) exp(iγB₀t)
```

### Echo Formation
- **Spin echo**: 90° - τ - 180° - τ - echo refocuses B₀ inhomogeneity
- **Gradient echo**: flip - read - spoil allows rapid imaging

## Sequence-Specific Physics

### Diffusion Weighting
Diffusion encoding gradients create phase dispersion proportional to molecular motion:
```
S(b) = S₀ exp(-b·D)
```
Where **b** = γ²G²δ²(Δ - δ/3) encodes diffusion weighting.

### BOLD Contrast
Blood oxygenation level dependent contrast arises from deoxyhemoglobin susceptibility:
```
ΔR₂* ∝ [dHb] ∝ (1 - Y)·CBV
```
Where Y = blood oxygenation, CBV = cerebral blood volume.

### Perfusion (ASL)
Arterial spin labeling uses magnetically labeled water as endogenous tracer:
```
ΔS = 2M₀f·T₁eff·exp(-TI/T₁eff)
```
Where f = perfusion, TI = inversion time.

## Spectroscopy Extensions

### Chemical Shift
Different molecular environments shift resonance frequency:
```
ω = γB₀(1 + σ)
```
Where σ is the shielding constant (ppm relative to reference).

### J-Coupling
Scalar coupling between nuclear spins creates multiplet patterns:
```
H(J) = 2πJ₁₂ I₁·I₂
```

### Metabolite Relaxation
| Metabolite | T₁ (ms) | T₂ (ms) | Conc (mM) |
|------------|---------|---------|-----------|
| NAA | 1470 | 344 | 12.2 |
| Creatine | 1570 | 186 | 5.5 |
| GABA | 1310 | 88 | 1.3 |

## Simulation Implementation

### Analytical Solutions
For simple pulses, closed-form rotation matrices:
```python
def rotation_x(angle):
    return np.array([[1, 0, 0],
                     [0, cos(angle), -sin(angle)],
                     [0, sin(angle), cos(angle)]])
```

### Numerical Integration
Complex RF shapes require ODE solving:
```python
def bloch_ode(t, M, B, gamma, T1, T2):
    Bx, By, Bz = B(t)
    dMdt = gamma * np.cross(M, [Bx, By, Bz])
    dMdt[0] -= M[0] / T2
    dMdt[1] -= M[1] / T2  
    dMdt[2] -= (M[2] - M0) / T1
    return dMdt
```

### EPG (Extended Phase Graphs)
Multi-echo sequences with complex refocusing:
```python
def epg_rf(F, alpha):
    """RF pulse mixing"""
    F_new = F.copy()
    F_new[0] = F[0] * cos(alpha/2)**2 + F[1] * sin(alpha/2)**2
    F_new[1] = F[1] * cos(alpha/2)**2 + F[0] * sin(alpha/2)**2  
    return F_new
```

## Relevant Projects

- **sbi4dwi**: Diffusion MRI forward modeling using analytical Bloch solutions for PGSE sequences
- **mrs-jax**: MR spectroscopy with chemical shift evolution and J-coupling Hamiltonians
- **MCMRSimulator.jl**: Multi-modal Monte Carlo including T₁, T₂ weighting and BOLD effects

## See Also

- [physics-nmr.md](physics-nmr.md) - Nuclear magnetic resonance fundamentals
- [modality-fmri.md](modality-fmri.md) - BOLD contrast mechanisms
- [modality-dti.md](modality-dti.md) - Diffusion encoding gradients
- [tissue-brain.md](tissue-brain.md) - Brain tissue relaxation properties
- [coordinate-system-scanner.md](coordinate-system-scanner.md) - Scanner coordinate reference frames

## Key References

- **stanisz2005t1**: Stanisz et al. (2005). T1, T2 relaxation and magnetization transfer in tissue at 3T. MRM 54:507-512. doi:10.1002/mrm.20605
- **wansapura1999nmr**: Wansapura et al. (1999). NMR relaxation times in the human brain at 3.0 Tesla. JMRI 9:531-538.
- **degraafinvivo**: de Graaf (2019). In Vivo NMR Spectroscopy: Principles and Techniques. 3rd ed. Wiley. Comprehensive MRS physics textbook.
- **Deoni2003despot**: Deoni et al. (2003). Rapid combined T1 and T2 mapping using gradient recalled acquisition in the steady state (DESPOT). MRM 49:515-526.
- **Alsop2015asl**: Alsop et al. (2015). Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications. MRM 73:102-116.

## References

- Haacke EM, Brown RW, Thompson MR, Venkatesan R. *Magnetic Resonance Imaging: Physical Principles and Sequence Design*. Wiley, 1999.
- Stanisz GJ, et al. T₁, T₂ relaxation and magnetization transfer in tissue at 3T. *Magn Reson Med* 2005;54:507-512.
- Bernstein MA, King KF, Zhou XJ. *Handbook of MRI Pulse Sequences*. Elsevier, 2004.
- de Graaf RA. *In Vivo NMR Spectroscopy: Principles and Techniques*. 3rd ed. Wiley, 2019.