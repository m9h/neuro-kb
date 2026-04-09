```yaml
type: tissue
title: Skull / Cortical Bone
properties:
  conductivity_S_m: ~0.02
  relative_permittivity: ~20
  acoustic_impedance_MRayl: 5.18
  sound_speed_m_s: 2800
  density_kg_m3: 1850
  attenuation_dB_cm_MHz: 4.0
sources: ["aubry2022itrusst", "gabriel1996", "iacono2015mida"]
related: [tissue-csf.md, tissue-brain.md, head-model-mida.md, head-model-sci.md, physics-acoustic-wave-equation.md, modality-tfus.md]
```

# Skull / Cortical Bone

The skull (cortical bone) is the primary challenge in transcranial neuroimaging and neuromodulation. Its high acoustic impedance, electrical resistivity, and complex multilayer structure create strong reflections, phase aberrations, and signal attenuation across all neuroimaging modalities.

## Acoustic Properties

Cortical bone has dramatically different acoustic properties compared to soft tissue:

| Property | Value | Units | Notes |
|----------|-------|-------|-------|
| Sound speed | 2800 | m/s | ITRUSST benchmark BM3 |
| Density | 1850 | kg/m³ | 1.85x water |
| Acoustic impedance | 5.18 | MRayl | 3.5x soft tissue |
| Attenuation | 4.0 | dB/cm/MHz | High frequency-dependent loss |

The large acoustic impedance mismatch between soft tissue (~1.6 MRayl) and skull (~5.2 MRayl) creates **93% attenuation** at transcranial focused ultrasound frequencies (400 kHz), requiring sophisticated phase correction and amplitude compensation for therapeutic applications.

## Electrical Properties

| Property | Frequency | Value | Units |
|----------|-----------|-------|-------|
| Conductivity | 10 Hz | 0.020 | S/m |
| Conductivity | 100 Hz | 0.022 | S/m |
| Conductivity | 1 kHz | 0.025 | S/m |
| Relative permittivity | 10 Hz | 20 | - |
| Relative permittivity | 100 Hz | 18 | - |
| Relative permittivity | 1 kHz | 15 | - |

The low conductivity (50x lower than CSF) makes skull the dominant factor in EEG forward modeling and current flow during tDCS/tES stimulation.

## Anatomical Structure

Real skull has a complex trilayer structure not captured in simple models:

1. **Outer cortical bone** (compacta externa): ~1-2 mm thick, highest sound speed
2. **Diploe** (trabecular bone + marrow): ~2-8 mm thick, intermediate properties
3. **Inner cortical bone** (compacta interna): ~0.5-1 mm thick

| Layer | Sound speed (m/s) | Density (kg/m³) |
|-------|-------------------|------------------|
| Cortical | 2800 | 1850 |
| Trabecular | 2300 | 1700 |

The **MIDA head model** represents this trilayer structure with 153 anatomical regions, while **BrainWeb** and synthetic models use a single homogeneous skull layer.

## Frequency-Dependent Effects

### Ultrasound Attenuation
Skull attenuation increases approximately linearly with frequency:
- **180 kHz**: 32% attenuation (optimal for deep brain TUS)
- **400 kHz**: 55% attenuation (balance of penetration/focusing)  
- **1 MHz**: 75% attenuation (superficial targets only)

This frequency dependence drives the selection of 180-400 kHz for transcranial focused ultrasound systems.

### EEG/MEG Forward Modeling
Skull conductivity shows weak frequency dependence but remains 2-3 orders of magnitude lower than brain tissue across the physiological frequency range (1-100 Hz), making it the bottleneck for current flow in EEG source localization.

## Heterogeneity and Individual Variation

Skull thickness varies dramatically across the head:
- **Temporal bone**: 2-4 mm (acoustic window for TUS)
- **Frontal/parietal**: 6-12 mm (requires stronger correction)
- **Occipital**: 8-15 mm (most challenging for transcranial access)

This heterogeneity necessitates **subject-specific modeling** using CT-derived skull segmentation for clinical TUS applications.

## Head Model Implementation

### SCI Institute Head Model
The **SCI head model** (Warner et al. 2019) represents skull as a single tissue class with ITRUSST properties. Used extensively in `sbi4dwi` TUS optimization experiments:
- 208×256×256 at 1mm resolution
- 8 tissue classes including homogeneous skull
- CC-BY 4.0 license

### MIDA Head Model  
The **MIDA model** (Iacono et al. 2015) provides the most detailed skull representation:
- 500 μm isotropic resolution
- Separate cortical/trabecular bone regions
- IT'IS Foundation license required
- Used in `brain-fwi` full waveform inversion

## Multi-Physics Coupling

Skull properties couple across physics domains:
- **Acoustic → Mechanical**: Radiation force F = 2αI/c drives tissue displacement
- **Mechanical → Electrical**: Strain modifies conductivity tensor  
- **Electrical → Thermal**: Joule heating from tDCS current flow
- **Thermal → Acoustic**: Temperature-dependent sound speed (PRF thermometry)

## Clinical Relevance

Skull modeling accuracy directly impacts:
- **TUS focal targeting**: 1 mm skull thickness error → 2-3 mm focal shift
- **EEG source localization**: 20% conductivity error → 10-15 mm localization error
- **tDCS dose modeling**: Skull thinning (aging) increases brain current by 2-3x

## Relevant Projects

- **sbi4dwi**: TUS skull correction, heterogeneous skull segmentation, OpenLIFU integration
- **brain-fwi**: Full waveform inversion for skull velocity mapping, ITRUSST benchmarks
- **jwave**: j-Wave acoustic simulation wrapper with skull attenuation validation

## Key References

- **aubry2022itrusst**: Aubry et al. (2022). Benchmark problems for transcranial ultrasound simulation: intercomparison of compressional wave models. JASA. Consensus acoustic property values.
- **Wolters2004fem**: Wolters et al. (2006). Influence of tissue conductivity anisotropy on EEG/MEG forward modeling. NeuroImage 30:813-826. Skull conductivity impact on source localization.
- **Wharton_2012**: Wharton & Bowtell (2012). Fiber orientation-dependent white matter contrast in gradient echo MRI. PNAS 109:18559-18564. Myelin susceptibility and skull bone modeling.
- **schenck96_role_magnet_suscep_magnet_reson_imagin**: Schenck (1996). The role of magnetic susceptibility in magnetic resonance imaging. Med Phys 23:815-850.
- **martin2025tfus**: Martin et al. (2025). MRI-guided transcranial focused ultrasound neuromodulation with a 256-element helmet array. Nature Communications.

## See Also

- [tissue-csf.md](tissue-csf.md) - Cerebrospinal fluid properties
- [tissue-brain.md](tissue-brain.md) - Gray and white matter  
- [head-model-mida.md](head-model-mida.md) - MIDA 153-structure model
- [head-model-sci.md](head-model-sci.md) - SCI Institute head model
- [physics-acoustic-wave-equation.md](physics-acoustic-wave-equation.md) - Wave propagation physics
- [modality-tfus.md](modality-tfus.md) - Transcranial focused ultrasound