---
type: head-model
title: MIDA Head Model
source: IT'IS Foundation (Zurich)
tissues: 115+
resolution: 0.5 mm isotropic
formats: [NIfTI, STL, Sim4Life]
local_path: /data/datasets/MIDA/
related: [sci-head-model.md, tissue-gray-matter.md, tissue-white-matter.md, tissue-skull.md, tissue-scalp.md, tissue-csf.md, tissue-electrical-properties.md, tissue-acoustic-properties.md, method-fem.md, coordinate-systems.md]
---

# MIDA Head Model

MIDA (Multimodal Imaging-Based Detailed Anatomical model) is the most anatomically detailed publicly available head model, with 115+ distinct tissue structures segmented at 0.5 mm isotropic resolution. It was developed by the IT'IS Foundation (Zurich) and published by Iacono et al. (2015, PLOS ONE).

## Source Data

The model is based on multi-contrast MRI of a single healthy 29-year-old female volunteer:

- **T1-weighted** (MPRAGE) -- primary contrast for gray/white matter, skull
- **T2-weighted** -- CSF spaces, edema differentiation
- **Proton density** -- soft tissue boundaries
- **MR angiography** -- vascular structures (arteries, veins, sinuses)
- **Diffusion tensor imaging** -- white matter fiber orientation

All images were acquired at 3T with 0.5 mm isotropic resolution and co-registered to a common anatomical space.

## Tissue Classes

The 115+ segmented structures span the following categories:

### Brain parenchyma (~15 regions)
Cerebral gray matter (cortical, deep nuclei), cerebral white matter, cerebellum (gray and white), brainstem, hippocampus, amygdala, thalamus, caudate, putamen, globus pallidus, hypothalamus, substantia nigra, red nucleus.

### Meninges and CSF spaces
Dura mater, arachnoid, pia mater, lateral ventricles, third ventricle, fourth ventricle, subarachnoid CSF, cisterns.

### Skull layers
- **Outer cortical bone** (compact, high density)
- **Diploe / cancellous bone** (trabecular, lower density)
- **Inner cortical bone** (compact)
- **Sutures** (fibrous connective tissue, acoustically and electrically distinct from bone)

### Extracranial structures
Scalp (skin, subcutaneous fat, muscle), eyes (cornea, lens, vitreous humor, retina, optic nerve), paranasal sinuses (frontal, maxillary, ethmoid, sphenoid), temporal muscles, masseter, blood vessels (carotid, vertebral arteries; jugular veins; dural sinuses), cranial nerves (trigeminal, facial, vestibulocochlear), cartilage, air cavities.

## Comparison with Other Head Models

| Model | Tissues | Resolution | Skull Layers | Subject | Mesh Available | License |
|-------|---------|-----------|--------------|---------|---------------|---------|
| **MIDA** | 115+ | 0.5 mm iso | 3 (cortical, diploe, cortical) + sutures | 1 female, 29 yr | STL surfaces | IT'IS (free for academic) |
| **SCI** | 8 | tet mesh (~9.9M elem) | 1 (homogeneous) | 1 subject | Tetrahedral | Open |
| **Colin27** | 3-9 | 1 mm iso | 1 (homogeneous) | 1 male, 27 yr | Via FreeSurfer | Open |
| **ICBM152** | 3-6 | 1 mm iso | 1 (homogeneous) | 152-subject average | Via SPM/FSL | Open |
| **SimNIBS (CHARM)** | 10-60 | ~1 mm | 2 (compact + spongy) | Per-subject from T1/T2 | Surface + tet | GPL |
| **BrainWeb** | 12 | 1 mm iso | 1 (homogeneous) | 20 synthetic | Volumetric | Open |

Key differentiators for MIDA:

- **3-layer skull** with separate cortical and diploe compartments is critical for transcranial ultrasound simulation, where the trabecular layer causes significant beam aberration (speed of sound: cortical ~2800 m/s vs diploe ~2300 m/s; Aubry et al. 2022)
- **Suture segmentation** matters for neonatal/young adult TUS and tDCS -- sutures have lower acoustic impedance and higher electrical conductivity than cortical bone
- **Vascular detail** enables realistic modeling of blood flow artifacts in fNIRS and realistic thermal modeling for safety assessment
- **0.5 mm resolution** resolves thin structures (dura ~0.5-1 mm, cortical bone ~1-3 mm) that 1 mm models alias

## Tissue Property Assignment

Each tissue label must be mapped to physical properties for simulation. The IT'IS Foundation maintains the Database of Tissue Properties (IT'IS v4.1) with frequency-dependent values.

### Electrical properties (EEG/MEG/tDCS)
Assign conductivity (S/m) and relative permittivity from Gabriel et al. (1996) or IT'IS v4.1. See [tissue-electrical-properties.md](tissue-electrical-properties.md) for cross-tissue values.

Critical assignments:
- Gray matter: 0.126 S/m (isotropic approximation)
- White matter: 0.062 S/m (isotropic) or anisotropic tensor from DTI
- CSF: 1.79 S/m
- Skull compact: 0.01 S/m
- Skull diploe: 0.025 S/m
- Scalp: 0.465 S/m

### Acoustic properties (TUS/FWI)
Assign speed of sound (m/s), density (kg/m^3), and attenuation (dB/cm/MHz). See [tissue-acoustic-properties.md](tissue-acoustic-properties.md). ITRUSST benchmark values (Aubry et al. 2022):

| Tissue | Speed of Sound (m/s) | Density (kg/m^3) | Attenuation (dB/cm/MHz) |
|--------|---------------------|-------------------|------------------------|
| Brain (avg) | 1546 | 1046 | 0.6 |
| CSF | 1515 | 1007 | 0.002 |
| Skull cortical | 2800 | 1900 | 4.0 |
| Skull diploe | 2300 | 1500 | 8.0 |
| Scalp / soft tissue | 1540 | 1100 | 0.6 |

### Optical properties (fNIRS/DOT)
Assign absorption (mu_a) and reduced scattering (mu_s') coefficients. See [tissue-optical-properties.md](tissue-optical-properties.md) and [sci-head-model.md](sci-head-model.md) for reference values at 690/830 nm.

## Meshing Considerations

### Surface extraction
- Marching cubes on the 0.5 mm label volume yields initial triangulated surfaces
- Surface smoothing (Laplacian or Taubin) is needed to remove staircase artifacts
- Decimate to target triangle edge length of 0.5-1.0 mm for skull, 1-2 mm for brain
- STL files distributed with MIDA provide pre-extracted surfaces for each tissue

### Tetrahedral meshing for FEM
- **iso2mesh** (Fang & Boas 2009): MATLAB/Octave toolbox, CGAL backend
- **Gmsh**: open-source, scriptable, supports adaptive refinement
- **TetGen**: Delaunay-based, quality guarantees (dihedral angle bounds)

Resolution requirements by application:

| Application | Min element size | Typical element count | Rationale |
|------------|-----------------|----------------------|-----------|
| EEG forward | 1 mm | 1-5M tet | Skull conductivity gradients |
| tDCS optimization | 0.5 mm | 5-15M tet | Electrode-skull interface |
| TUS beam simulation | 0.1-0.3 mm | 10-50M tet | >6 elements per wavelength at 500 kHz |
| fNIRS (MMC) | 1 mm | 5-10M tet | Photon transport in scattering media |

### Volume-based simulation
For pseudospectral solvers (j-Wave in brain-fwi), the 0.5 mm voxel grid can be used directly as a structured grid, avoiding meshing entirely. Grid spacing must satisfy >10 points per wavelength (at 500 kHz in brain: wavelength ~3 mm, requiring dx <= 0.3 mm -- may need interpolation to finer grid).

## Typical Applications

1. **EEG/MEG forward modeling** -- leadfield computation via FEM with 115+ tissue conductivities; the 3-layer skull improves accuracy for temporal lobe sources by 10-20% vs single-layer skull (Dannhauer et al. 2011)
2. **TMS dose planning** -- E-field modeling with anisotropic conductivity from DTI-informed white matter
3. **TUS beam simulation** -- full-wave acoustic propagation through the 3-layer skull using j-Wave or k-Wave; critical for predicting focal pressure and thermal dose
4. **tDCS electrode optimization** -- current flow modeling to maximize cortical target intensity while respecting safety limits
5. **Full waveform inversion** -- brain-fwi uses MIDA as ground truth phantom for validating FWI reconstruction of brain speed-of-sound from transcranial ultrasound data

## Limitations

- **Single subject**: no population variability captured; cannot assess inter-individual anatomical differences in skull thickness, sinus volume, or cortical folding
- **Female anatomy**: skull thickness and geometry differ systematically from male anatomy (thinner parietal bone, different frontal sinus morphology); results may not generalize
- **Age-specific**: 29-year-old adult; not representative of pediatric (open sutures, thinner skull) or elderly (atrophy, enlarged ventricles) populations
- **Static anatomy**: no cardiac or respiratory pulsation; CSF and vascular volumes are fixed
- **License restrictions**: requires IT'IS Foundation license agreement; free for academic use but not fully open-source like SCI or BrainWeb
- **No DTI tensor field**: while the source MRI included DTI, the distributed model provides scalar labels only -- users must estimate anisotropic conductivity separately

## Data Access

- **IT'IS Foundation**: https://itis.swiss/virtual-population/tissue-models/mida-model/
- **Dryad TUS dataset**: Pre-computed transcranial ultrasound simulations through MIDA skull (doi:10.5061/dryad.nzs7h44n7, CC0 license)
- **Local path**: `/data/datasets/MIDA/`

## Relevant Projects

- **neurojax** -- differentiable FEM head modeling; MIDA provides a high-fidelity validation target for leadfield computation with 115+ tissue conductivities (`neurojax:geometry/fem_forward.py`)
- **brain-fwi** -- full waveform inversion for transcranial ultrasound; MIDA is a primary phantom with 3-layer skull (`brain_fwi/phantoms/` loads MIDA label maps and assigns ITRUSST acoustic properties)
- **vbjax** -- whole-brain simulation; MIDA brain parcellation can provide region boundaries for neural mass models on the cortical surface
- **dot-jax** -- diffuse optical tomography; MIDA tissue segmentation supports FEM mesh generation for photon transport simulation with anatomically realistic geometry

## Key References

1. Iacono MI, Neufeld E, Akinnagbe E, et al. (2015). MIDA: a multimodal imaging-based detailed anatomical model of the human head and neck. *PLOS ONE* 10(4):e0124126.
2. Aubry JF, Bhatt O, et al. (2022). Benchmark problems for transcranial ultrasound simulation. *JASA* 152(2):1003-1019.
3. Gabriel S, Lau RW, Gabriel C (1996). The dielectric properties of biological tissues. *Phys Med Biol* 41:2231-2249.
4. Dannhauer M, Lanfer B, Wolters CH, Kn\"osche TR (2011). Modeling of the human skull in EEG source analysis. *Hum Brain Mapp* 32(9):1383-1399.

## See Also

- [sci-head-model.md](sci-head-model.md) -- 8-layer open-source head model for photon transport (simpler but fully open)
- [tissue-gray-matter.md](tissue-gray-matter.md) -- gray matter properties across modalities
- [tissue-white-matter.md](tissue-white-matter.md) -- white matter properties and anisotropy
- [tissue-skull.md](tissue-skull.md) -- skull layer properties (conductivity, acoustic impedance)
- [tissue-scalp.md](tissue-scalp.md) -- scalp tissue properties
- [tissue-csf.md](tissue-csf.md) -- cerebrospinal fluid properties
- [tissue-electrical-properties.md](tissue-electrical-properties.md) -- cross-tissue conductivity table
- [tissue-acoustic-properties.md](tissue-acoustic-properties.md) -- cross-tissue acoustic property table
- [method-fem.md](method-fem.md) -- finite element method for head modeling
