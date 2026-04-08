---
type: tissue
title: Scalp
properties:
  conductivity_S_m: 0.435
  relative_permittivity: 1106
  density_kg_m3: 1100
  thickness_mm: 5-7
  optical_mua_1_mm: 0.019
  optical_musp_1_mm: 0.86
  optical_n: 1.37
sources: ["Gabriel1996", "Prahl1999", "MCX_Colin27"]
related: ["tissue-skull.md", "tissue-csf.md", "tissue-gray-matter.md", "head-model-colin27.md", "modality-fnirs.md", "modality-eeg.md"]
---

# Scalp

The scalp is the outermost tissue layer of the head, consisting of skin, connective tissue, aponeurosis, loose areolar tissue, and periosteum. In neuroimaging forward models, scalp properties significantly influence signal propagation for EEG, MEG, fNIRS, and transcranial stimulation.

## Electrical Properties

### Conductivity
Scalp conductivity varies with frequency and measurement conditions:

- **Low frequency (DC-100 Hz)**: 0.435 S/m (Gabriel et al., 1996)
- **EEG frequencies (0.1-100 Hz)**: 0.33-0.43 S/m
- **Anisotropy**: Approximately isotropic (ratio ~1.1:1)

The relatively high conductivity compared to skull (0.01 S/m) makes scalp a significant current pathway in EEG forward modeling.

### Permittivity
At body temperature (37°C):
- **Relative permittivity (εr)**: 1106 at 100 Hz
- **Frequency dependence**: Decreases with increasing frequency following Cole-Cole dispersion

## Optical Properties (NIR)

For fNIRS and DOT applications at typical wavelengths (650-950 nm):

| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Absorption coefficient (μa) | 0.019 | mm⁻¹ | At ~800 nm |
| Reduced scattering (μs') | 0.86 | mm⁻¹ | Similar to skull |
| Refractive index (n) | 1.37 | - | Standard tissue value |
| Anisotropy factor (g) | 0.9 | - | Highly forward-scattering |

These values are used in the MCX Colin27 benchmark and dot-jax default tissue property tables.

## Physical Properties

- **Thickness**: 5-7 mm (varies with location and individual)
- **Density**: ~1100 kg/m³
- **Composition**: ~80% water, varies with hydration state
- **Temperature**: 32-34°C (surface), 37°C (deeper layers)

## Modeling Considerations

### Multi-layer Structure
Detailed scalp models may include:
1. **Epidermis**: Very thin (~0.1 mm), higher resistance
2. **Dermis**: Main conductive layer (3-5 mm)
3. **Subcutaneous tissue**: Variable thickness, lower conductivity
4. **Muscle/fascia**: Anisotropic, higher conductivity along fibers

### Individual Variation
Scalp properties vary significantly between subjects:
- **Age effects**: Decreased hydration and increased resistance in elderly
- **Skin condition**: Dry skin increases contact resistance
- **Hair**: Minimal effect on bulk tissue properties but affects electrode contact
- **Temperature**: 2-3% conductivity increase per °C

## Simulation Parameters

### EEG Forward Modeling
Standard values used in neurojax and similar packages:
- **Conductivity**: 0.33 S/m (conservative estimate)
- **Thickness**: 6 mm (population average)
- **Boundary condition**: Grounded at infinity or reference electrode

### fNIRS/DOT Forward Modeling
From dot-jax property tables:
```python
# Scalp optical properties at 800 nm
scalp_props = {
    'mua': 0.019,      # 1/mm
    'musp': 0.86,      # 1/mm
    'n': 1.37,         # refractive index
    'g': 0.9           # anisotropy factor
}
```

### Transcranial Stimulation
For TMS/tDCS modeling:
- **Conductivity**: 0.465 S/m (higher estimate for active stimulation)
- **Anisotropy**: Often neglected (assumed isotropic)
- **Nonlinear effects**: May exhibit conductivity changes with current density

## Measurement Methods

### In Vivo Techniques
- **Electrical impedance tomography (EIT)**: Bulk conductivity estimation
- **Four-electrode impedance**: Local conductivity measurement
- **Time-domain reflectometry**: Dielectric properties
- **Diffuse reflectance spectroscopy**: Optical properties

### Ex Vivo Validation
- **Tissue dielectric probe**: Gold standard for electrical properties
- **Integrating sphere spectrophotometry**: Optical absorption and scattering
- **Histological analysis**: Structural composition verification

## Relevant Projects

- **dot-jax**: Uses scalp as outermost layer in FEM mesh assembly and optical forward modeling
- **neurojax**: Incorporates scalp conductivity in BEM/FEM head models for source localization
- **sbi4dwi**: May include scalp in diffusion-informed tissue property estimation

## Key References

- **Wolters2004fem**: Wolters et al. (2006). Influence of tissue conductivity anisotropy on EEG/MEG forward modeling. NeuroImage 30:813-826.
- **Brigadoi2015short**: Brigadoi & Cooper (2015). How short is short? Optimum source-detector distance for short-separation channels in fNIRS. Neurophotonics 2(2):025005. Impact of scalp hemodynamics on fNIRS.
- **Saager2005short**: Saager & Berger (2005). Direct characterization and removal of interfering absorption trends in two-layered turbid media. JOSA A 22:1874-1882.

## See Also

- [tissue-skull.md](tissue-skull.md) - Next inward tissue layer
- [tissue-gray-matter.md](tissue-gray-matter.md) - Primary imaging target beneath skull
- [head-model-colin27.md](head-model-colin27.md) - Reference head model with 4-layer tissue segmentation
- [modality-eeg.md](modality-eeg.md) - Electrical neuroimaging sensitive to scalp conductivity
- [modality-fnirs.md](modality-fnirs.md) - Optical neuroimaging through scalp tissue
- [physics-bioelectric.md](physics-bioelectric.md) - Electrical field propagation through tissues
- [physics-diffusion.md](physics-diffusion.md) - Light propagation in scattering media