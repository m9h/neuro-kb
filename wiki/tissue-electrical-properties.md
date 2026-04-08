---
type: tissue
title: "Tissue Electrical Conductivity (cross-tissue)"
properties:
  conductivity_S_m:
    scalp: "0.33 ± 0.05 (10 Hz - 1 kHz)"
    skull_compact: "0.0064 ± 0.0022 (10 Hz)"
    skull_spongy: "0.025 ± 0.008 (10 Hz)"
    csf: "1.79 ± 0.18 (10 Hz)"
    gray_matter: "0.276 ± 0.035 (10 Hz)"
    white_matter_isotropic: "0.126 ± 0.024 (10 Hz)"
    white_matter_longitudinal: "0.412 ± 0.067 (10 Hz)"
    white_matter_transverse: "0.083 ± 0.015 (10 Hz)"
    dura: "0.5 ± 0.2"
    blood: "0.7 ± 0.1"
    muscle: "0.354 ± 0.074"
    fat: "0.025 ± 0.008"
    air_cavities: "1e-15"
    eyeball: "1.5 ± 0.3"
sources: [gabriel1996, mccann2019, huang2017]
related: [head-model-charm.md, tissue-anisotropy.md, qmri-conductivity-mapping.md, bem-forward-modeling.md]
---

# Tissue Electrical Conductivity (cross-tissue)

Electrical conductivity values for brain and head tissues across neuroimaging modalities (EEG, MEG, ECoG, tDCS, TMS). These values are fundamental for accurate forward modeling in source imaging and stimulation planning.

## Properties Summary

### Core Brain Tissues

| Tissue | Conductivity (S/m) | Frequency | Notes |
|--------|-------------------|-----------|-------|
| **Gray matter** | 0.276 ± 0.035 | 10 Hz | Isotropic assumption |
| **White matter (iso)** | 0.126 ± 0.024 | 10 Hz | Legacy isotropic value |
| **White matter (∥)** | 0.412 ± 0.067 | 10 Hz | Along fiber direction |
| **White matter (⊥)** | 0.083 ± 0.015 | 10 Hz | Across fiber bundles |
| **CSF** | 1.79 ± 0.18 | 10 Hz | Highly conductive |

### Head Tissues

| Tissue | Conductivity (S/m) | Notes |
|--------|-------------------|-------|
| **Scalp** | 0.33 ± 0.05 | Frequency-dependent |
| **Skull (compact)** | 0.0064 ± 0.0022 | Dense cortical bone |
| **Skull (spongy)** | 0.025 ± 0.008 | Diploe layer |
| **Dura mater** | 0.5 ± 0.2 | Tough meningeal layer |
| **Blood** | 0.7 ± 0.1 | Vascular compartment |

### Other Tissues

| Tissue | Conductivity (S/m) | Usage |
|--------|-------------------|--------|
| **Muscle** | 0.354 ± 0.074 | Facial/neck muscles |
| **Fat** | 0.025 ± 0.008 | Subcutaneous layer |
| **Air cavities** | 1e-15 | Sinuses, ear canals |
| **Eyeball** | 1.5 ± 0.3 | Vitreous humor |

## Frequency Dependence

Tissue conductivity exhibits frequency-dependent behavior following the Cole-Cole dispersion model:

```
σ(f) = σ_∞ + (σ_s - σ_∞) / (1 + (jωτ)^(1-α))
```

Where:
- `σ_∞`: high-frequency conductivity
- `σ_s`: static (DC) conductivity  
- `τ`: relaxation time constant
- `α`: distribution parameter (0-1)

### Typical Frequency Scaling
- **10 Hz - 1 kHz**: Values in the table above
- **1-10 kHz**: ~10-20% increase for most tissues
- **>100 kHz**: Approaches high-frequency asymptote

## Anisotropy in White Matter

White matter conductivity is highly anisotropic due to myelinated fiber organization:

```
Conductivity ratio = σ_∥ / σ_⊥ ≈ 5:1
```

This anisotropy significantly affects current flow patterns in:
- [bem-forward-modeling.md](bem-forward-modeling.md)
- Source localization accuracy
- tDCS/TMS current distribution

## qMRI-Based Conductivity Mapping

Modern approaches derive tissue-specific conductivity from quantitative MRI:

### T1-Based Mapping (Tuch et al. 2001)
```
σ = a × BPF + b × (1 - BPF) + c
```
Where BPF is bound pool fraction from QMT, related to myelin content.

### Multi-Modal Mapping
- **DTI**: Provides fiber orientation for anisotropy
- **QMT**: Myelin content → conductivity scaling
- **T1 mapping**: Tissue water content correlation

## Head Model Integration

### CHARM 60-Tissue Segmentation
The CHARM atlas provides detailed tissue classification enabling:
- Tissue-specific conductivity assignment
- Partial volume correction
- Anatomically accurate forward models

### Implementation in Forward Models
```python
# Example: neurojax conductivity assignment
sigma = sigma_from_qmri(t1_values, bpf_values, labels, params)
K = assemble_stiffness(vertices, elements, sigma)  # FEM
```

## Validation Studies

### Cross-Modal Validation
- **EIT measurements**: Direct in-vivo conductivity
- **Phantom studies**: Controlled conductivity validation
- **Multi-frequency EEG**: Frequency response validation

### Literature Consensus
Values above represent consensus from multiple studies:
- Gabriel et al. (1996): Comprehensive dielectric database
- McCann et al. (2019): Modern head modeling review  
- Huang et al. (2017): Anisotropic white matter measurements

## Clinical Considerations

### Individual Variability
- Age-related changes: ±20% variation with age
- Pathology effects: Lesions, edema alter local conductivity
- Hydration status: Affects tissue water content

### Safety Margins
For stimulation applications, conductivity uncertainty requires:
- Conservative current density limits
- Safety factor incorporation
- Individual head model validation where possible

## Relevant Projects

- **neurojax**: `geometry/fem_forward.py` - differentiable conductivity assignment
- **vbjax**: Forward model integration for whole-brain simulation
- **setae**: Tissue property modeling in bio-inspired mechanics
- **libspm**: Classical SPM tissue classification integration
- **neurotech-primer-book**: Educational coverage of tissue properties across modalities

## See Also

- [head-model-charm.md](head-model-charm.md) - Multi-tissue head model implementation
- [tissue-anisotropy.md](tissue-anisotropy.md) - Detailed anisotropic conductivity modeling
- [qmri-conductivity-mapping.md](qmri-conductivity-mapping.md) - MRI-derived conductivity methods
- [bem-forward-modeling.md](bem-forward-modeling.md) - Boundary element method applications
- [coordinate-system-mni152.md](coordinate-system-mni152.md) - Standard space tissue atlases