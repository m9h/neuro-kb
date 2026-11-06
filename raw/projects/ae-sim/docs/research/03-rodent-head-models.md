# Open rodent head templates and datasets for AE volume conductors

*2026-10-23. Prompted by Morgan: "There are open source rat brain templates and datasets."*

A volume conductor for the rodent experiments needs the **skull**, not just the brain. Brain
atlases are plentiful; whole-head datasets with a labelled skull are the scarce resource, and
they are listed first.

## The toolchain already supports this

SimNIBS 4.6 on the build host exposes

```python
simnibs.mesh_tools.meshing.create_mesh(
    label_img, affine,
    elem_sizes={"standard": {"range": [1, 5], "slope": 1.0}},   # mm; shrink to ~0.1-0.5 for rodents
    smooth_size_field=2, skin_facet_size=2.0, facet_distances=...,
    optimize=False, remove_spikes=True, skin_tag=1005, ...)
```

which meshes *any* labelled volume with an affine, so rodent head FEM lead fields come out of
the same `ae_sim.head` path as the human head, with no `charm` segmentation step.
BabelViscoFDTD takes an arbitrary material grid, so the same labels drive the skull acoustics,
which also delivers the skull step that was deferred on the human side.

## Datasets

### Rat

| dataset | what it has | skull | licence / access |
|---|---|---|---|
| **SIGMA v2 (Wistar)** [Zenodo 10635831](https://zenodo.org/records/10635831), NITRC `sigma_template` | whole-head T2w / T1w / T2*w templates, probabilistic GM / WM / CSF / **skull** / out-of-brain maps, brain mask, CT and 18F-FDG reference spaces, diffusion templates, Paxinos-Watson coordinates | yes | CC BY 4.0, one 1.5 GB zip |
| Waxholm Space SD atlas, [NITRC whs-sd-atlas](https://www.nitrc.org/projects/whs-sd-atlas) / EBRAINS | 39 µm MRI, 78 µm DTI, detailed brain labels | no | open |

SIGMA v2 is a complete Wistar head model. Waxholm gives finer brain parcellation if a
compartment-specific source model is needed later.

### Mouse

| dataset | what it has | skull | licence / access |
|---|---|---|---|
| **Digimouse** [neuroimage.usc.edu/neuro/Digimouse](https://neuroimage.usc.edu/neuro/Digimouse) | CT + cryosection, 0.1 mm voxels, 380 × 992 × 208; labels for skeleton (skull), skin, whole brain / cerebrum / cerebellum / olfactory bulbs / striatum / medulla, masseter, eyes, organs; surface and volume tessellations for BEM / FEM | yes | free with citation: Dogdas et al. 2007, Phys Med Biol, DOI [10.1088/0031-9155/52/3/003](https://doi.org/10.1088/0031-9155/52/3/003) |
| Toronto NeuroMouse surgical atlases, [db.phm.utoronto.ca/surgical.htm](http://db.phm.utoronto.ca/surgical.htm) | MRI / CT adult mouse head, C57Bl/6J and 129S1/SvImJ, brain + labels + dorsal / complete skull (MINC) | yes | free download, no stated licence |

Digimouse is the best mouse head for both FEM and skull acoustics. The Toronto atlases add a
complete C57Bl/6J skull, the strain used in the Rintoul in vivo work.

### Not found

No open rat head volume-conductor or ultrasound model exists on GitHub; the rat tDCS FEM
papers (2011 onward) did not publish their models.

## What this makes replicable

1. **Rintoul 2026 in vivo mouse** (see [02](02-rintoul-heterodyning-and-in-vivo.md)):
   Digimouse mesh, electrodes at bregma 0 / ML −2 mm and V1 at AP −3.5 / ML +2.25 mm, loop
   electrodes in the mouth and over the head, 500 kHz at 1 MPa. Reproduce the standing-wave
   pattern of her Sim4Life supplement with BabelViscoFDTD, then the ~9 mV AE potential with
   a Poisson solve. A stronger test than the tank because the applied field is nothing like
   uniform.
2. **Tianjin rat DBS mapping** (Zhou 2024, 2026) on the SIGMA head: a DBS contact in the rat
   brain, transcranial focus, 167.74 µV V⁻¹ MPa⁻¹ and < 2 mm resolution. Order of magnitude
   until the paywalled methods are in hand.
3. **Rat SSVEP** (Chen 2023) needs a neural source model in visual cortex, so it follows the
   LFPykit source block.

## Practical points

- A mouse skull is ~0.3 mm thick, so the acoustic grid must be ~0.1 mm rather than our usual
  6 cells per wavelength (λ ≈ 3 mm at 500 kHz). That is still only a few million cells for a
  mouse head and fine on the DGX Spark.
- Element sizes for `create_mesh` must drop to ~0.1–0.5 mm for rodents.
- SIGMA and Digimouse are candidates for NeuroFedora data packages alongside `simnibs-data`,
  so a rodent model would be reproducible from the COPR rather than from a scratch directory.

## Next step

Download SIGMA v2 (rat) and Digimouse (mouse) into fresh directories, verify the label
contents, build meshes with `create_mesh` at ~0.2 mm, then add rodent cases to `ae_sim.head`.
Recommended order: Digimouse + the Rintoul in vivo configuration first (open data, author
reachable), then SIGMA for the DBS case once the Tianjin methods are available. Both exist now:
`ae_sim.mousehead` (tutorial 1, stages 24/39/41/42/43) and `ae_sim.sigma_rat` (tutorial 2,
stages 44/45; the rat case uses generic electrode and beam parameters until the Tianjin
methods are in hand).
