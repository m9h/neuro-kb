# Learning microscope physics and optical design

Verified July 2026. Prices USD. `[?]` marks anything that could not be confirmed.

**Calibrated to this lab.** If you can write a differentiable forward model, you
already own the wave-optics and image-formation half of this field. What you lack
is **geometric optics as an engineering discipline, real glass, opto-mechanics,
and alignment craft.** This list is weighted accordingly: skip the maths you'd
find easy, spend the time on the bench.

## The single highest-leverage move

Get to **one** hands-on course, and build **one** real relay on your own bench.
Everything below is support for those two acts.

## Books, in reading order

**Start here**

1. **Mertz, *Introduction to Optical Microscopy*, 2nd ed., Cambridge 2019.**
   ISBN 9781108428309, ~$105. The spine. Physics-grade image formation, coherence,
   PSF/OTF, confocal, light-sheet, super-res. No newer edition.
2. **Bhandari, Kadambi & Raskar, *Computational Imaging*, MIT Press 2022.**
   **Free official PDF: <https://imagingtext.github.io/cibook.pdf>.** Your native
   language — read in parallel with Mertz.
3. **Goodman, *Introduction to Fourier Optics*, 4th ed. 2017.** Treat as review;
   mine it for the transfer-function formalism.

**The hardware gap — spend your time here**

4. **Moore, Davis, Coplan & Greer, *Building Scientific Apparatus*, 4th ed.,
   Cambridge 2009.** ISBN 9780521878586, ~$70–90. *The* hardware-education book:
   opto-mechanics, mounts, detectors. Highest value on this list after Mertz.
   Read it **while** building something; it lands completely differently.
5. **Murphy & Davidson, *Fundamentals of Light Microscopy and Electronic Imaging*,
   2nd ed.** The concrete complement to Mertz's theory.
6. **Tkaczyk, *Field Guide to Microscopy*, SPIE FG13.** ~$42. Desk reference.

**When you design and mount optics**

7. **Fischer, Tadic-Galeb & Yoder, *Optical System Design*, 2nd ed.** Read this
   before you drive Zemax — it teaches what a lens-design program is doing.
   Directly relevant to designing your scan lens and relay.
8. **Yoder & Vukobratovich, *Opto-Mechanical Systems Design*, 4th ed.** Overkill
   until you are physically mounting optics; then indispensable. `[?]` year/price.

**Reference shelf**

9. **Pawley (ed.), *Handbook of Biological Confocal Microscopy*, 3rd ed., 2006.**
   Still the latest — **no 4th edition exists**. (A 2024 "Biological Confocal
   Microscopy Handbook" from Educohack is an unrelated derivative. Ignore it.)
10. Hecht *Optics* 5th; Born & Wolf 60th-anniversary ed.; Lakowicz *Principles of
    Fluorescence Spectroscopy* 3rd (no 4th).

**For this profile specifically:** Voelz, *Computational Fourier Optics: A MATLAB
Tutorial* (SPIE TT89) — port the MATLAB to JAX as an exercise.

## Free resources (all live, July 2026)

- **Nikon MicroscopyU** — <https://www.microscopyu.com> — best starting portal.
- **Evident (ex-Olympus)** — <https://evidentscientific.com/en/learn>
- **ZEISS Microscopy Online Campus** — <https://zeiss-campus.magnet.fsu.edu/>
  (hosted at FSU, *not* on zeiss.com)
- **iBiology Microscopy Series** —
  <https://www.ibiology.org/online-biology-courses/microscopy-series/> — the best
  free structured course. Kurt Thorn on optical sectioning and confocal.
- **Thorlabs Insights** and **Edmund Optics Knowledge Center** — read the MTF and
  aberration notes *before* speccing parts.
- **QUAREP-LiMi** — <https://quarep.org/> — how to *measure whether your
  instrument actually works*. Protocols on protocols.io.

**Open-instrument builders — the most relevant reading you will do:**

- **Andrew York** — <https://andrewgyork.github.io/> — interactive open-science
  papers. Read *Remote refocus* and the high-NA single-objective light-sheet.
- **Alfred Millett-Sikking** — <https://amsikking.github.io/> — his
  `lab_resources`, `calculations` and `parts` pages are genuine build references.

## Courses — what books cannot teach

Most 2026 iterations have already run. Plan for 2027 and apply early; these are
selective.

| Course | Where | Next | Cost | Deadline |
|---|---|---|---|---|
| **AQLM** (Analytical & Quantitative Light Microscopy) | MBL, Woods Hole | spring 2027 | ~$4.5–5k `[?]` | ~Mar 2027 |
| **OMIBS** (Optical Microscopy & Imaging) | MBL | **Aug 11–21, 2026** | $3,875 + $990 | passed |
| **Quantitative Imaging** (biennial) | CSHL | **Apr 5–20, 2027** | `[?]` | **Jan 22, 2027** |
| **"Hack your Microscope"** (EMBO + openUC2) | ITQB NOVA, Portugal | watch for next | — | ran Apr 2026 |
| **Optomechanical Design / Alignment** | U. Rochester summer short courses | Jun 2027 | `[?]` | — |

**"Hack your Microscope"** is the closest direct match for building open-source
microscopes (openFlexure, openUC2, openFrame).

**Janelia Advanced Imaging Center** — <https://www.aicjanelia.org/> — **free**
access to pre-commercial instruments for approved proposals. Not a build course,
but unmatched access.

**mesoSPIM** runs "build & use a mesoSPIM" workshops — <https://mesospim.org/meetings_and_workshops/>
(last confirmed Oct 2025 Zurich; `[?]` no 2026/27 date posted).

## Optical design software

**Framing that matters:** a scan lens + tube lens relay is a **geometric ray-design,
real-glass, tolerancing** problem — f-θ linearity, telecentricity, field flatness,
lateral colour across the scan field. It is **not** a wave-propagation problem, so
your JAX instinct points at the wrong tool for the *design* step. Keep wave optics
(Chromatix) for modelling the system afterwards.

| Tool | Backend | Differentiable | Role |
|---|---|---|---|
| **Optiland** ([HarrisonKramer/optiland](https://github.com/HarrisonKramer/optiland)) | NumPy + PyTorch | **yes** | **Your workhorse.** Real ray trace, PSF/MTF, autograd into curvatures/thicknesses/glasses, refractiveindex.info catalog, Monte-Carlo tolerancing. |
| **DeepLens / AutoLens** ([singer-yang/DeepLens](https://github.com/singer-yang/DeepLens)) | PyTorch | yes | Differentiable ray-based lens designer; end-to-end optics+network co-design. |
| **rayoptics** ([mjhoptics](https://github.com/mjhoptics/ray-optics)) | NumPy | no | Paraxial layout; **imports/exports .zmx** — your bridge to Zemax. |
| **prysm** | numpy/cupy/torch | via torch | Physical-optics PSF/MTF, phase retrieval. |
| **Chromatix** | **JAX** | yes | Differentiable wave-optics forward modelling. **Not a lens designer.** |
| **Zemax OpticStudio** | commercial | — | ~$4.9–14.9k `[?]`. **Free 30-day trial; free student version** in Ansys Student. |

**Workflow:** prototype the relay in **Optiland** (autograd merit function: spot vs
field, f-θ, telecentricity, chromatic). Then rebuild it in the free **Zemax** trial
using **stock Thorlabs/Edmund achromats** — stock-lens catalog matching is the one
thing open tools still can't do, and it is exactly what you need to build a relay
from parts you can buy. Use **rayoptics** as the `.zmx` bridge.

## Communities

- **image.sc** — <https://forum.image.sc> — official support venue for
  Micro-Manager and mesoSPIM. Use the `hardware` tag.
- **Microforum** — <https://forum.microlist.org> — the hardware/optics counterpart.
  (There is no "microforum.forum" domain.)
- **Confocal Microscopy Listserv** — alive, at **University of Minnesota**:
  <https://lists.umn.edu/cgi-bin/wa?A0=confocalmicroscopy>
- **openUC2** <https://openuc2.discourse.group>; **OpenFlexure**
  <https://openflexure.discourse.group>

The field consolidated onto **Discourse forums, not Slack/Discord**. Bluesky is
the de facto successor to science Twitter.

## The 16-week path

**Weeks 1–4 — theory you'll find easy.** Mertz cover to cover. Skim Goodman.
Read the free Bhandari PDF. Watch the whole iBiology Microscopy Series.

**Weeks 2–8 — the actual gap.** Fischer alongside driving Optiland. Design a 4f
relay, then a scan lens + tube lens pair as a differentiable merit function. Then
rebuild it in the Zemax trial with **stock achromats** and feel where real glass
bites. Read Millett-Sikking's `calculations`/`parts` pages.

**Weeks 6–16 — hands. Non-negotiable.** Buy a Thorlabs cage kit, two achromats, a
galvo pair, a laser diode, a camera. Build the relay you designed. Read *Building
Scientific Apparatus* while doing it. Align the pupil onto the galvo, get a flat
scan field, then **measure it against a QUAREP-LiMi protocol** (co-registration,
uniform field, PSF). That measure-your-own-instrument loop is where understanding
forms.

**In parallel, now:** apply to AQLM 2027 and watch for the next EMBO "Hack your
Microscope".

## What cannot be learned from books

- **Alignment as motor skill.** Walking a beam with two mirrors. Centring a pupil.
  Recognising coma and astigmatism by eye in a defocused spot. Getting a conjugate
  plane actually conjugate.
- **Tolerance intuition.** Which misalignment matters at micron scale and which you
  can ignore. How a real achromat differs from its catalogue spec.
- **Failure-mode pattern recognition.** Ghost reflections, back-reflection into the
  laser, scatter, vignetting, "why is my image dim" for a boring mechanical reason.
  Experienced builders diagnose these in seconds. Books do not transfer it.
- **Vendor fluency.** What is actually buyable, lead times, and designing around
  stock parts.

Two things I could not verify: a Kurt Thorn course titled "Practical Optics", and
"Bhattacharya, Building a Microscope from Scratch". Treat both as nonexistent
unless you have a source.
