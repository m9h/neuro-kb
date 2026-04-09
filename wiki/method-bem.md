```yaml
type: method
title: Boundary Element Method
category: BEM
implementations: [neurojax:geometry/bem_forward.py]
related: [method-fem.md, physics-bioelectromagnetics.md, head-model-mida.md, coordinate-system-mni152.md]
```

# Boundary Element Method

The Boundary Element Method (BEM) is a computational technique for solving partial differential equations in unbounded domains by reformulating volume integrals as surface integrals. In neuroimaging, BEM is the dominant approach for computing electromagnetic forward models from source currents to M/EEG sensors.

## Theory

BEM transforms Poisson's equation ∇·(σ∇φ) = -∇·J into a surface integral equation using Green's reciprocal theorem. For a dipole source J at position r₀, the potential φ at sensor position r is:

φ(r) = (1/4π) ∫∂Ω [φ(r')∂G/∂n' - G(r',r)∂φ/∂n'] dS'

where G(r',r) = 1/|r-r'| is the free-space Green's function and ∂Ω represents tissue boundaries.

## Implementation Details

### Surface Discretization
- **Mesh requirements**: Closed, non-intersecting triangulated surfaces
- **Typical resolution**: 5000-10000 triangles per boundary
- **Quality metrics**: Aspect ratio < 5, solid angle close to 4π for closed surfaces
- **Standard topology**: Brain-CSF-Skull-Scalp (3-layer) or Brain-CSF-Skull-Scalp-Air (4-layer)

### Conductivity Values
Standard tissue conductivities at 1 kHz [@gabriel1996]:

| Tissue | Conductivity (S/m) |
|--------|-------------------|
| Brain (GM/WM) | 0.33 / 0.14 |
| CSF | 1.79 |
| Skull | 0.0042 |
| Scalp | 0.43 |

### Numerical Methods
- **Integration**: Analytical for self-interactions, numerical quadrature for far-field
- **Singularity treatment**: Duffy transformation for near-singular integrals
- **Linear system**: Dense, typically solved with LU decomposition
- **Memory scaling**: O(N²) where N = total surface vertices

## Properties/Parameters

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| **Surface vertex count** | 1000-20000 per boundary | Accuracy vs computational cost |
| **Skull conductivity** | 0.003-0.015 S/m | High sensitivity - 3x change affects source depth |
| **Brain-to-skull ratio** | 20-80:1 | Primary determinant of EEG forward model |
| **CSF thickness** | 1-5 mm | Strong effect on superficial source visibility |
| **Integration order** | 3-7 Gaussian points | Accuracy of near-field calculations |

### Accuracy Benchmarks
- **Relative error vs analytical**: < 1% for spherical head models
- **Leadfield correlation**: > 0.95 with high-resolution FEM
- **Computational time**: ~10 minutes for 3-layer model (10k vertices) on modern CPU

## Advantages and Limitations

### Advantages
- **Unbounded domains**: Natural handling of open boundaries to infinity
- **Surface-only meshing**: No volumetric discretization required
- **High accuracy**: Exact treatment of tissue boundaries
- **Established workflow**: Mature tools (OpenMEEG, MNE, FieldTrip BEM)

### Limitations
- **Piecewise constant conductivity**: Cannot model gradual conductivity transitions
- **Dense linear systems**: Memory scales as O(N²)
- **Surface quality dependent**: Requires high-quality closed meshes
- **Limited anisotropy**: Difficult to incorporate DTI-derived conductivity tensors

## Head Model Integration

### Preprocessing Pipeline
1. **Segmentation**: T1w → tissue labels (FreeSurfer, FSL FAST, SPM)
2. **Surface extraction**: Marching cubes or deformable models
3. **Mesh repair**: Fill holes, remove self-intersections (MeshLab)
4. **Decimation**: Reduce vertex count while preserving geometry
5. **Quality check**: Verify closed surfaces, normal orientations

### CHARM Integration
For high-resolution CHARM (60-tissue) head models:
- **Tissue grouping**: Combine tissues into BEM-compatible layers
- **Interface detection**: Extract boundaries between conductivity regions
- **Mesh fusion**: Merge adjacent tissue surfaces when conductivities are similar

## Key References

- **GraveDePeralta2001laura**: Grave de Peralta Menendez et al. (2001). Noninvasive localization of electromagnetic epileptic activity: method descriptions and simulations. Brain Topography 14:131-137.
- **GraveDePeralta2004laura**: Grave de Peralta Menendez et al. (2004). Electrical neuroimaging based on biophysical constraints. NeuroImage 21:527-539.
- **Hamalainen1994mne**: Hamalainen & Ilmoniemi (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Med & Biol Eng & Comp 32:35-42.
- **Michel2004eegsi**: Michel et al. (2004). EEG source imaging. Clinical Neurophysiology 115:2195-2222.
- **Wolters2004fem**: Wolters et al. (2006). Influence of tissue conductivity anisotropy on EEG/MEG field computation using FEM. NeuroImage 30:813-826.

## Relevant Projects

- **neurojax**: `geometry/bem_forward.py` - JAX-accelerated BEM with OpenMEEG backend
- **vbjax**: BEM leadfields for whole-brain simulation validation
- **setae**: Surface mechanics inform mesh quality requirements
- **libspm**: C implementation with MATLAB interface

## Software Implementations

### OpenMEEG
- **Language**: C++ with Python/MATLAB bindings
- **Features**: Symmetric BEM, dipole singularity subtraction
- **Performance**: Optimized BLAS/LAPACK, OpenMP parallelization
- **Integration**: Native support in MNE-Python, FieldTrip

### MNE-Python BEM
- **Method**: Single-layer with conductor model
- **Surfaces**: Typically brain-only (watershed algorithm)
- **Advantages**: Fast, automatic mesh generation
- **Limitations**: Less accurate for deep sources than 3-layer

### FieldTrip BEM
- **Implementation**: Dipoli, OpenMEEG, or native MATLAB
- **Features**: Multi-layer support, extensive validation tools
- **Integration**: Seamless with FieldTrip source analysis pipeline

## Validation and Quality Control

### Numerical Validation
- **Sphere models**: Compare against analytical solutions (Berg, de Munck)
- **Reciprocity**: Verify Maxwell reciprocity principle
- **Conservation**: Check current conservation at boundaries

### Geometric Validation
- **Surface closure**: Solid angle = 4π for points inside, 0 for outside
- **Intersection testing**: Ray-triangle intersection algorithms
- **Mesh quality**: Aspect ratios, edge lengths, triangle areas

## See Also

- [method-fem.md](method-fem.md) - Finite Element Method comparison
- [physics-bioelectromagnetics.md](physics-bioelectromagnetics.md) - Underlying electromagnetic theory
- [head-model-mida.md](head-model-mida.md) - MIDA head model with BEM meshes
- [tissue-csf.md](tissue-csf.md) - CSF conductivity and BEM modeling
- [coordinate-system-mni152.md](coordinate-system-mni152.md) - Standard space for BEM templates