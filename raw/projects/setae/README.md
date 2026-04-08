---
category: research
section: introduction
weight: 10
title: "Setae: Bio-Inspired Surface Mechanics in JAX"
status: draft
slide_summary: "Differentiable simulation of biological surface contacts — gecko adhesion, shark drag, lotus self-cleaning — filling the gap left by existing physics engines that lack JKR/DMT adhesive contact mechanics."
tags: [jax, bio-inspired, contact-mechanics, adhesion, gecko, differentiable-physics, surface-mechanics, hierarchical-structures]
---

# setae

Bio-inspired surface mechanics in JAX. Differentiable simulation of
how organisms use micro/nanoscale surface contacts -- and how to optimize
synthetic designs using gradient-based and evolutionary methods.

## Why setae?

Organisms solve surface contact problems that engineers still struggle
with. Geckos climb glass. Sharks reduce drag by 8%. Lotus leaves
self-clean. Tree frogs grip in rain. These aren't accidents -- they're
the result of ~500 million years of evolutionary optimization on
hierarchical surface structures.

**setae** makes these phenomena computationally accessible:
biology -> physics -> JAX -> design.

| What | How | Physics |
|------|-----|---------|
| Gecko adhesion | van der Waals on hierarchical setae | JKR contact |
| Tree frog grip | Wet adhesion + drainage channels | Capillary forces |
| Shark drag reduction | Denticle riblets | Boundary layer |
| Lotus self-cleaning | Superhydrophobic papillae | Cassie-Baxter |
| Snake locomotion | Anisotropic ventral scales | Directional friction |
| Nacre toughness | Brick-and-mortar hierarchy | Fracture mechanics |

**No existing differentiable physics framework handles adhesion contact
mechanics.** Engines like DiffDART (rigid-body), DiffTaichi (MPM), and
MJX/Brax (penalty contact) all lack JKR/DMT adhesive interactions. setae
fills this gap: fully differentiable adhesion contact in JAX, composable
with the rest of the JAX simulation ecosystem.

---

## Table of contents

- [Physics foundations](#physics-foundations)
  - [Contact mechanics](#contact-mechanics)
  - [Nanoscale forces](#nanoscale-forces)
  - [Rough surface contact](#rough-surface-contact)
- [Biological systems](#biological-systems)
- [Constitutive models for biological materials](#constitutive-models-for-biological-materials)
- [Computational landscape](#computational-landscape)
  - [Differentiable physics engines](#differentiable-physics-engines)
  - [JAX simulation ecosystem](#jax-simulation-ecosystem)
  - [ML-accelerated approaches](#ml-accelerated-approaches)
- [Bio-inspired design applications](#bio-inspired-design-applications)
- [The multi-scale challenge](#the-multi-scale-challenge)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Notebooks](#notebooks-planned)
- [Key references](#key-references)
- [License](#license)

---

## Physics foundations

### Contact mechanics

The theory of how elastic bodies deform and adhere when pressed together
is the backbone of biological surface mechanics.

**Hertz (1882)** solved the foundational problem: two elastic spheres
pressed together with force F form a circular contact of radius
a = (3FR/4E*)^(1/3), where R is the combined radius and E* is the
combined modulus. No adhesion -- purely elastic deformation. This works
well for stiff, large-scale contacts but fails badly at small scales
where surface energy dominates.

**JKR -- Johnson, Kendall & Roberts (1971)** extended Hertz by adding
surface energy. The contact radius becomes larger than Hertz predicts
(surfaces "snap in"), and separation requires a finite pull-off force:
F_pull-off = -(3/2) pi W R, where W is the work of adhesion. JKR assumes
adhesion acts only inside the contact zone. This is accurate for soft,
large, high-energy contacts -- precisely the regime of biological
adhesion pads.

**DMT -- Derjaguin, Muller & Tabor (1975)** took the opposite limit:
adhesion acts only outside the contact zone as long-range surface forces,
while the contact profile remains Hertzian. Pull-off force:
F_pull-off = -2 pi W R. This works for stiff, small, low-energy contacts.

**Maugis-Dugdale (1992)** unified JKR and DMT through a transition
parameter lambda (the Tabor parameter):

    lambda = (R W^2 / E*^2 z_0^3)^(1/3)

where z_0 is the equilibrium separation (~0.2-0.4 nm). When lambda >> 1,
JKR dominates (soft, compliant); when lambda << 1, DMT dominates (stiff,
small). The Johnson-Greenwood adhesion map plots contact behavior across
this parameter space. Most biological adhesion pads operate firmly in the
JKR regime (lambda ~ 1-100).

setae implements both JKR and DMT models with automatic selection based
on the Tabor parameter, and all contact calculations are end-to-end
differentiable through `jax.grad`.

### Nanoscale forces

At the scales relevant to biological adhesion (1 nm - 10 um), several
forces compete:

**van der Waals forces** arise from fluctuating electromagnetic dipoles
between all matter. The Hamaker approach treats this pairwise; the
Lifshitz theory (more rigorous) derives interaction from bulk dielectric
spectra. For a sphere near a flat surface:
F_vdW = -A R / (6 D^2), where A is the Hamaker constant (typically
0.4-4 x 10^-19 J for biological/polymer systems) and D is the separation.
This is the dominant force in gecko adhesion -- Autumn et al. (2002)
confirmed this definitively by showing gecko setae adhere equally well
to hydrophobic and hydrophilic surfaces.

**Casimir forces** become significant below ~20 nm separation, where
retardation of the electromagnetic interaction matters. The crossover
from non-retarded van der Waals (F ~ D^-2) to retarded Casimir
(F ~ D^-3) occurs at D ~ lambda_char / (2 pi), typically 10-20 nm.

**Capillary forces** arise from liquid bridges between surfaces. At
ambient humidity, water condenses in the contact region, creating a
meniscus that generates adhesive forces via Laplace pressure:
F_cap = 4 pi R gamma cos(theta), where gamma is surface tension and
theta is the contact angle. These dominate in tree frog adhesion and
insect wet pads, and can exceed van der Waals forces by 10-100x in
humid conditions.

**DLVO electrostatics** (Derjaguin-Landau-Verwey-Overbeek) describe
the interplay of van der Waals attraction and electric double-layer
repulsion in aqueous/ionic environments. Relevant for marine organisms
(octopus suckers, mussel adhesion) and any system operating in wet or
ionic conditions.

In real biological systems, these forces don't act in isolation. A gecko
spatula experiences van der Waals attraction modified by capillary
condensation at ambient humidity, with contact geometry set by JKR
deformation of the beta-keratin tip. setae composes these force
contributions within a unified differentiable framework.

### Rough surface contact

Real surfaces are never smooth. Understanding this is critical because
roughness can reduce adhesion by orders of magnitude compared to ideal
predictions.

**Greenwood & Williamson (1966)** modeled rough surfaces as a collection
of independent hemispherical asperities with Gaussian height
distributions. Each asperity makes Hertzian contact independently. This
captures the basic observation that real contact area is a tiny fraction
of apparent area, but assumes asperities don't interact and ignores
long-range waviness.

**Persson's theory (2001-)** takes a fundamentally different approach:
rough surfaces are characterized by their power spectral density (PSD)
across all length scales. Many natural and engineered surfaces have
approximately self-affine fractal PSDs, meaning roughness exists at every
scale. Persson's contact mechanics theory uses a diffusion equation in
magnification space to compute the real contact area and pressure
distribution, naturally handling the multi-scale nature of real contacts.

For biological adhesion, roughness is both enemy and friend. Random
roughness on a substrate devastates adhesion -- gecko spatulae must be
compliant enough to conform to surface asperities. But controlled
roughness on the adhesive pad itself (the hierarchical seta/spatula
structure) dramatically enhances adhesion by increasing compliance,
enabling crack trapping, and providing redundancy against defects.

---

## Biological systems

Nature has evolved remarkably diverse solutions to surface contact
problems. Each system below includes mechanism, key dimensions, force
measurements, and material properties from the experimental literature.

### Gecko (Gekko gecko)

The canonical example of dry adhesion. The gecko toe pad is a
hierarchical structure spanning four orders of magnitude:

| Level | Structure | Dimensions | Count |
|-------|-----------|------------|-------|
| Toe pad | Lamellae (scansors) | ~1-2 mm wide | ~20 per toe |
| Lamella | Setae (hair-like) | ~110 um long, 4-5 um dia | ~14,000 per mm^2 |
| Seta tip | Branches (spatulae stalks) | ~2-5 um long | 100-1000 per seta |
| Terminal | Spatulae (flat pads) | 200-300 nm wide, ~5 nm thick | ~10^9 per foot |

**Forces**: A single spatula generates ~10 nN adhesion (Huber et al.
2005). A single seta produces ~200 uN (Autumn et al. 2000). A single
toe can support ~4 N. Total for all four feet: ~20 N -- giving a ~50 kg
gecko a safety factor of ~40x body weight.

**Key physics**: Adhesion is purely van der Waals (Autumn et al. 2002),
not suction, capillary, or electrostatic. The setae achieve directional
adhesion: pulling parallel to the surface (shear) activates adhesion;
pulling perpendicular (peeling at >30 deg) releases it. This is the
"frictional adhesion" model: shear load drives the contact into a
JKR-like state with enhanced pull-off.

**Materials**: Beta-keratin, E ~ 1-2 GPa. The effective modulus of the
setal array is much lower (~100 kPa) due to the hierarchical
architecture -- this satisfies the Dahlquist criterion for pressure-
sensitive adhesion (E_eff < 300 kPa) despite being made of a stiff
material.

**Self-cleaning**: Gecko setae self-clean by energetic disequilibrium --
dirt particles adhere more strongly to the substrate than to the
spatulae, and are left behind during locomotion. No grooming required.

### Tree frog (Litoria caerulea and relatives)

Wet adhesion on flooded surfaces -- a regime where gecko-style dry
adhesion fails completely.

**Morphology**: Toe pads are soft (E ~ 5-15 kPa), covered with
hexagonal epithelial cells (5-10 um diameter) separated by channels
(0.5-1.5 um wide). Each cell surface has an array of nanopillars: 326 nm
diameter, ~350 nm height, ~430 nm spacing (Scholz et al. 2009).

**Mechanism**: Three components act in concert:
1. **Stefan adhesion**: Viscous resistance to separation of two plates
   with a thin fluid film between them. F ~ eta A^2 / (2 pi h^3 dh/dt),
   where eta is fluid viscosity. The hexagonal channels allow fluid
   drainage to reduce the film thickness rapidly (within ~5 ms), bringing
   surfaces close enough for intimate contact.
2. **Capillary adhesion**: Once the fluid film is thin enough,
   meniscus bridges form at the pad boundary, generating Laplace pressure.
3. **Hydrodynamic**: Close contact enables hydrodynamic friction during
   sliding.

**Forces**: Adhesion stress ~1-2 kPa on smooth wet glass. On rough
surfaces, adhesion drops dramatically -- the soft pad must conform to
asperities. Nanopillars may function to increase conformability or to
break the fluid film into separate menisci.

### Insects (diverse orders)

Insect adhesion pads come in two forms:

**Smooth pads** (arolia in Hymenoptera, pulvilli in Diptera): Soft,
deformable structures that maximize contact area through material
compliance. E ~ 10-100 kPa.

**Hairy pads** (setae in Coleoptera, Chrysomelidae): Fibrillar
structures similar in concept to gecko setae but operating at larger
scales (2-5 um diameter fibers vs. 200 nm spatulae). Contact splitting
principle: dividing a contact into N sub-contacts multiplies adhesion
by sqrt(N) (Arzt et al. 2003).

**Two-phase secretions**: Most insects use a thin liquid secretion
(emulsion of hydrophobic lipids in aqueous phase) on their pads.
This generates both capillary adhesion and viscous forces, and
enables adhesion on surfaces where dry vdW alone would be insufficient.

### Octopus (Octopus vulgaris)

Active suction + chemical sensing in a single structure.

**Morphology**: Each sucker has an infundibulum (inner cup, ~2-4 mm
diameter depending on arm position) with annular grooves and a central
dome (acetabulum) backed by muscular sphincters.

**Mechanism**: Suction by muscular expansion of the acetabulum chamber,
generating pressure differentials up to 0.27 MPa (~2.7 atm). The
infundibulum grooves increase conformability to rough surfaces. Unlike
gecko adhesion, this is pressure-based, not molecular adhesion.

**Chemotactile sensing**: Suckers contain chemoreceptors that can taste
on contact, integrating grip with chemical perception. This dual
sensing-gripping function has no analog in synthetic adhesive systems.

### Shark skin (Squalus, Isurus, Carcharodon)

Passive drag reduction through surface texture.

**Denticles**: Shark skin is covered with tooth-like scales (placoid
denticles), 200-500 um in size, each with 3-7 longitudinal riblets.
The riblet spacing, characterized by the non-dimensional wall unit
s+ = s u_tau / nu (where u_tau is friction velocity, nu is kinematic
viscosity), determines drag reduction effectiveness. Optimal s+ ~ 15-20.

**Drag reduction**: Riblets reduce turbulent skin friction by lifting
and constraining streamwise vortices in the viscous sublayer, preventing
cross-stream momentum transfer. Experimental measurements show up to
8-10% local drag reduction, with typical whole-body reductions of 3-8%.
The mechanism is well-understood via DNS (direct numerical simulation)
of turbulent boundary layers over riblet geometries.

**Fouling resistance**: The denticle geometry also reduces biofouling
settlement. The combination of drag reduction and fouling resistance is
the basis for biomimetic surface coatings.

### Lotus leaf (Nelumbo nucifera)

The archetype of biological self-cleaning surfaces.

**Dual-scale roughness**: Epidermal cells form papillae (10-20 um
diameter, 10-15 um height) covered with epicuticular wax crystals
(tubular, 80-120 nm diameter). This two-level hierarchy is critical --
single-scale roughness alone is insufficient for robust
superhydrophobicity.

**Wetting**: Static water contact angle: 162 +/- 2 deg. Roll-off
angle: < 2 deg. The surface operates in the Cassie-Baxter state: water
droplets sit on top of the roughness features with air trapped in the
interstices. The transition to the Wenzel state (liquid filling the
roughness) destroys superhydrophobicity and self-cleaning.

**Self-cleaning**: Water droplets rolling off the surface collect and
remove contamination particles. The particles adhere more strongly to
the water meniscus than to the waxy nanocrystal tips, which have
minimal contact area. First characterized quantitatively by Barthlott &
Neinhuis (1997).

### Snake ventral scales (various Serpentes)

Friction anisotropy enabling limbless locomotion.

**Surface texture**: Ventral scales have oriented micro-denticulations
(fibril-like protrusions), ~2.5 um long x 0.6 um wide, tilted in the
caudal (tail) direction. The denticulation geometry varies between
species and correlates with locomotion mode and habitat.

**Friction anisotropy**: Coefficient of friction in the forward
(cranial) direction is ~33% lower than in the backward (caudal) direction
(Hazel et al. 1999). This anisotropy provides the asymmetric thrust
that enables serpentine, concertina, and rectilinear locomotion without
limbs. The mechanism is primarily geometric interlocking of the tilted
denticulations with surface asperities, not differential adhesion.

### Nacre / mother-of-pearl (Haliotis, Pinctada)

Hierarchical architecture for extreme fracture toughness.

**Brick-and-mortar structure**: Aragonite (CaCO3) tablets ~500 nm thick,
5-10 um diameter, stacked in layers separated by ~20-30 nm organic
matrix (chitin + silk-like proteins). The tablets are 95-97% of the
volume; the organic phase is 3-5%.

**Mechanical properties**: Fracture toughness: 3-8 MPa sqrt(m) -- this
is roughly 3000x the fracture work of pure aragonite mineral. The
organic matrix enables multiple toughening mechanisms:
1. **Crack deflection**: Cracks are forced to navigate around tablets
   rather than cutting through them, increasing the crack path length.
2. **Tablet pullout**: Frictional sliding of tablets against the organic
   matrix dissipates energy. The tablets have nanoscale asperities
   (mineral bridges, nanoasperities) that create resistance.
3. **Mineral bridging**: Nanoscale mineral connections between adjacent
   tablets provide direct load transfer and must be broken before
   pullout.
4. **Organic matrix viscoelasticity**: The thin organic layers deform
   and dissipate energy under load, acting as sacrificial bonds.

### Spider silk attachment discs (Araneae)

How spiders anchor dragline silk to surfaces.

**Architecture**: Attachment discs are made from piriform silk, laid
down in a "staple-pin" pattern: the spider presses the spinneret against
the surface while pulling the dragline at an angle, creating a broad
base of cemented fibers anchoring a central load-bearing thread.

**Material properties**: Piriform silk has a tensile strength of ~511 MPa
and extensibility of ~40-60%. The staple-pin geometry distributes peel
loads across many attachment points, preventing catastrophic failure from
a single crack front -- an engineering principle analogous to composite
laminate design.

---

## Constitutive models for biological materials

Biological tissues are mechanically complex: nonlinear, rate-dependent,
often fluid-infiltrated, and structured across multiple length scales.
setae uses constitutive models appropriate to each system.

### Hyperelastic models

For large-deformation elasticity without time dependence (setal
bending, pad deformation):

| Model | Strain energy W | Best for |
|-------|----------------|----------|
| Neo-Hookean | W = (mu/2)(I_1 - 3) | Small-moderate strains, stiff bio-materials |
| Mooney-Rivlin | W = C_10(I_1-3) + C_01(I_2-3) | Moderate strains, rubber-like tissues |
| Ogden | W = sum mu_k/alpha_k (lambda_i^alpha_k - 1) | Large strains, complex response curves |
| Arruda-Boyce | 8-chain network model | Polymers, keratin networks |

Gecko beta-keratin operates at small strains (< 5%) and is well-served
by neo-Hookean or linear elasticity. Tree frog pads undergo large
deformations requiring Ogden or Mooney-Rivlin. Nacre organic matrix
undergoes moderate strains with significant hysteresis.

### Viscoelastic models

For rate-dependent behavior (dynamic adhesion, peeling):

- **Maxwell model**: Spring + dashpot in series. Captures stress
  relaxation but not creep recovery. Useful for modeling the rate-
  dependent pull-off of gecko setae (adhesion increases with pull-off
  speed up to a critical velocity).
- **Kelvin-Voigt model**: Spring + dashpot in parallel. Captures
  creep with full recovery. Approximates tree frog pad deformation
  under sustained loads.
- **Generalized Maxwell (Prony series)**: Multiple Maxwell elements
  in parallel, capturing relaxation across a spectrum of timescales.
  This is the standard for biological soft tissues and is what setae
  will implement for dynamic simulations.

### Poroelastic models

For fluid-infiltrated tissues where drainage affects mechanical response:

**Biot consolidation** models the coupled deformation-diffusion problem:
a porous elastic solid saturated with fluid, where mechanical loading
drives fluid flow and fluid pressure affects the stress state. This is
directly relevant to tree frog toe pads, where the epithelial channels
drain interstitial fluid to achieve close contact. The timescale of
drainage (determined by permeability, elastic modulus, and channel
geometry) sets the "speed limit" of tree frog adhesion.

### Fracture mechanics

For nacre and other tough biological composites:

- **Griffith energy balance**: Crack extends when strain energy release
  rate G exceeds twice the surface energy (2 gamma). Sets the baseline.
- **Cohesive zone models**: Traction-separation laws at crack tips that
  capture process zone behavior -- tablet pullout, bridging fiber
  stretch, organic matrix deformation. These are essential for modeling
  nacre's R-curve behavior (toughness that increases with crack length
  as more toughening mechanisms activate in the wake).

---

## Computational landscape

### Differentiable physics engines

The last five years have seen an explosion of differentiable simulators
for robotics, design, and control. setae occupies a specific gap in
this landscape.

| Engine | Method | Strengths | Adhesion? |
|--------|--------|-----------|-----------|
| **DiffDART/Nimble** (Liu et al.) | LCP rigid-body, complementarity-aware gradients | Accurate contact, 87x faster than finite diff | No -- rigid-body only |
| **DiffTaichi** (Hu et al.) | MPM soft body via Taichi compiler | Soft materials, 188x faster than TF | No -- penalty contact |
| **DiffPD** (Du et al.) | Projective dynamics FEM | Deformable solids, 4-19x speedup via Cholesky backprop | No |
| **MJX/Brax** (Google) | Penalty contact in MuJoCo/JAX | Millions of SPS on GPU, naturally smooth gradients | No -- penalty only |
| **ChainQueen** (Hu et al.) | MPM predecessor to DiffTaichi | Soft body robotics | No |
| **PlasticineLab** (Huang et al.) | DiffTaichi-based | Soft-body manipulation tasks | No |
| **setae** | JKR/DMT contact in JAX | Adhesion + hierarchy + bio-materials | **Yes** |

The gap is clear: existing differentiable simulators model rigid-body
contact or penalty-based soft-body contact, but none implement adhesive
contact mechanics (JKR/DMT). This means you cannot currently
backpropagate through a gecko-foot simulation in any other framework.

### JAX simulation ecosystem

setae is designed to compose with the broader JAX scientific computing
ecosystem:

- **jax-md** (Schoenholz & Cubuk): Molecular dynamics with
  Lennard-Jones, Morse, and learned neural potentials. Relevant for
  atomistic modeling of van der Waals interactions and surface energy.
- **JAX-FEM** (Xue et al.): Finite element method with hyperelasticity,
  plasticity, and topology optimization. Could model detailed setal
  deformation at single-seta resolution.
- **JAX-MPM**: Material point method for large deformations, 2.7M
  particles in ~22 seconds on GPU. Relevant for modeling soft-body
  contact and material flow in insect adhesive secretions.
- **Diffrax** (Kidger): ODE/PDE integration with adaptive stepping
  and adjoint methods. Backbone for any time-dependent simulation
  (viscoelastic relaxation, poroelastic drainage).
- **HydraxMPM**: MPM for large deformation solid mechanics.
- **Optimistix** (Kidger): Root-finding and least-squares for implicit
  solvers (Maugis-Dugdale requires iterating to self-consistency).

The composability advantage: because everything is JAX, you can chain
contact mechanics -> beam deformation -> hierarchical homogenization
-> design optimization end-to-end and differentiate through the entire
pipeline with `jax.grad`. No glue code, no finite differences, no
surrogate models needed.

### ML-accelerated approaches

Machine learning is increasingly used to accelerate or replace physics
simulation. These methods complement setae for different use cases:

**Graph neural network simulators**: MeshGraphNets (Pfaff et al.),
Graph Network Simulator (Sanchez-Gonzalez et al.) learn to simulate
mesh-based physics from data. Fast at inference but require extensive
training data and generalize poorly outside the training distribution.

**Neural operators**: DeepONet (Lu et al.), Fourier Neural Operator
(Li et al.) learn mappings between function spaces -- effectively
PDE surrogate models. Promising for replacing expensive multi-scale
simulations with learned approximations once the underlying physics is
well-characterized.

**ML interatomic potentials**: MACE (Batatia et al.), NequIP (Batzner
et al.) provide near-DFT-accuracy atomistic force fields at a fraction
of the cost. Relevant for computing surface energies and adhesion
parameters from first principles.

**Generative materials discovery**: MatterGen (Zeni et al.), CDVAE
(Xie et al.) generate novel crystal structures with target properties.
Could be used to discover new materials for synthetic adhesive surfaces.

---

## Bio-inspired design applications

The physics modeled in setae has already produced real engineering
applications:

**AeroSHARK** (Lufthansa + BASF): Riblet film applied to aircraft
fuselage, directly inspired by shark denticle geometry. Applied to the
entire Boeing 777 fleet starting 2022. Measured 1-2% fuel savings,
translating to ~3,700 tonnes CO2 reduction per aircraft per year.

**Sto Lotusan** paint: Facade paint with engineered micro/nanostructure
mimicking lotus leaf dual-scale roughness. Water contact angle > 150 deg,
self-cleaning in rain. Commercial since 1999.

**Geckskin** (UMass Amherst): Macro-scale gecko-inspired adhesive using
woven fabric integrated with a stiff tendon and soft elastomer pad.
Supports ~700 lb on smooth glass. Key insight: it's not the van der
Waals nanostructure that matters most at human scales, but the load
distribution and compliance matching.

**Stanford gecko gripper**: Gecko-inspired adhesive pads used on the
International Space Station for grasping non-cooperative objects in
microgravity. Demonstrates directional adhesion: load in shear to grip,
peel to release.

**Mushroom-shaped micropillars**: Engineered fibrillar adhesives with
mushroom-shaped tips achieve adhesion strengths up to ~300 kPa on smooth
surfaces (Gorb, del Campo) -- far exceeding flat-punch adhesion for the
same materials. The tip geometry maximizes contact area and eliminates
edge stress concentrations.

**Biomimetic drag reduction coatings**: Riblet surfaces on ship hulls,
wind turbine blades, and pipeline interiors. Typical drag reductions of
3-8% depending on geometry optimization and fouling state.

---

## The multi-scale challenge

Biological surface mechanics presents a unique computational challenge:
**features at every scale from nm to cm, with no clean scale separation.**

A gecko toe pad spans:
- **~5 nm**: Spatula-surface gap (van der Waals interaction range)
- **200 nm**: Spatula width (contact mechanics)
- **5 um**: Seta diameter (beam mechanics)
- **100 um**: Seta length (cantilever deflection)
- **1 mm**: Lamella width (structural compliance)
- **1 cm**: Toe pad (load distribution)

Traditional multi-scale methods assume scale separation: you solve a
representative volume element (RVE) at the fine scale, homogenize
properties, and pass them up. This works beautifully for periodic
composites but breaks down when:

1. **Scales interact bidirectionally**: The macroscopic loading angle on
   the toe pad determines which setae engage, which determines the
   spatula contact geometry, which determines the adhesion force, which
   feeds back into the macroscopic load balance.
2. **Statistical variability matters**: Not all setae are identical. The
   distribution of setal lengths, orientations, and defects affects the
   collective adhesion in ways that can't be captured by homogenizing
   a single "average seta."
3. **The hierarchy IS the mechanism**: Nacre's toughness isn't a property
   of aragonite or chitin -- it emerges from their hierarchical
   arrangement. Homogenizing away the hierarchy homogenizes away the
   phenomenon of interest.

setae addresses this with a layered approach: contact-level physics
(JKR/DMT) feeds into structural mechanics (beam bending, hierarchical
arrays), which feeds into system-level models (gecko toe, tree frog pad).
Each layer is differentiable, so gradients flow across scales. This
isn't concurrent multi-scale simulation (that remains computationally
prohibitive for real-time optimization), but it captures the essential
cross-scale couplings that produce emergent behavior.

---

## Installation

```bash
uv pip install -e .               # core (contact + structures)
uv pip install -e ".[dev]"        # everything for development
```

## Quick start

```python
import setae

# Single gecko spatula: JKR adhesion on glass
keratin = setae.beta_keratin()
glass = setae.glass()
contact = setae.jkr_contact(keratin, glass, R=200e-9)
print(f"Spatula pull-off: {abs(contact.pull_off_force)*1e9:.1f} nN")

# Full gecko toe pad
F_total = setae.gecko_adhesion_force()
sf = setae.gecko_safety_factor()
print(f"Total adhesion: {float(F_total):.1f} N")
print(f"Safety factor: {float(sf):.0f}x body weight")

# Differentiable: optimize spatula radius
import jax
grad = jax.grad(lambda R: float(setae.gecko_adhesion_force(spatula_radius=R)))
```

## Architecture

| Layer | Modules | Purpose |
|-------|---------|---------|
| 0 | `_contact`, `_surface_energy`, `_friction` | Contact/surface primitives |
| 1 | `_materials`, `_beam`, `_shell`, `_hierarchical` | Structural mechanics |
| 2 | `_capillary`, `_wetting`, `_drag` | Fluid-surface interaction |
| 3 | `_gecko`, `_tree_frog`, `_shark`, ... | Bio-system models |
| 4 | `_optimize`, `_evolutionary` | Optimization and design |

## Notebooks (planned)

1. Contact Mechanics: When Surfaces Touch
2. Surface Energy and Wetting
3. Soft Beams and Biological Hairs
4. Hierarchical Structures: Nature's Trick
5. Gecko Adhesion: Walking on Ceilings
6. Wet Grip: Tree Frogs and Insects
7. Octopus Suckers: Grip + Sense
8. Shark Skin: Drag Reduction by Design
9. Lotus Effect: Self-Cleaning Surfaces
10. Snake Scales: Friction for Locomotion
11. Nacre: Toughness Through Hierarchy
12. Differentiable Design: Backprop Through Physics
13. Evolutionary Textures: Evolving Surfaces
14. Design Challenge: Your Bio-Inspired Surface

## Key references

- Hertz (1882). Ueber die Beruehrung fester elastischer Koerper. J. reine angew. Math.
- Johnson, Kendall & Roberts (1971). Surface energy and contact of elastic solids. Proc. R. Soc. Lond. A.
- Derjaguin, Muller & Tabor (1975). Effect of contact deformations on the adhesion of particles. J. Colloid Interface Sci.
- Maugis (1992). Adhesion of spheres: the JKR-DMT transition. J. Colloid Interface Sci.
- Greenwood & Williamson (1966). Contact of nominally flat surfaces. Proc. R. Soc. Lond. A.
- Persson (2001). Theory of rubber friction and contact mechanics. J. Chem. Phys.
- Autumn et al. (2000). Adhesive force of a single gecko foot-hair. Nature.
- Autumn et al. (2002). Evidence for van der Waals adhesion in gecko setae. PNAS.
- Arzt, Gorb & Spolenak (2003). From micro to nano contacts in biological attachment devices. PNAS.
- Huber et al. (2005). Evidence for capillarity contributions to gecko adhesion. PNAS.
- Barthlott & Neinhuis (1997). Purity of the sacred lotus. Annals of Botany.
- Scholz et al. (2009). Ultrastructure and physical properties of an adhesive surface, the toe pad epithelium of the tree frog. J. Exp. Biol.
- Hazel et al. (1999). Nanoscale design of snake skin for full-scale friction. J. Tribology.
- Jackson, Milliron & Bhatt (2010). On the hierarchy of aragonite in nacre. J. Struct. Biol.
- Hu et al. (2019). DiffTaichi: Differentiable programming for physical simulation. ICLR.
- Schoenholz & Cubuk (2020). JAX, M.D.: A framework for differentiable physics. NeurIPS.

## License

Apache 2.0
