# Open-source and open-hardware microscopy: a catalogue

Compiled July 2026. Every entry was checked against a primary source. **"Open" is
a spectrum**, and most of this field sits at the wrong end of it: a paper, a dead
repo, and no bill of materials. Where that is true, it says so.

Two patterns worth internalising before you read the tables:

1. **Open *hardware licences* are nearly absent.** Even projects that ship CAD and
   Zemax files usually ship them as documentation under a *code* licence, not
   under CERN-OHL or CC-BY. A repo with no LICENSE file is all-rights-reserved,
   however public it looks.
2. **Almost nobody publishes a BOM total.** Costs below are marked `[S]` sourced,
   `[A]` author's own estimate, or `[I]` inferred. Treat `[I]` as an order of
   magnitude and nothing more.

---

## 1. Confocal and point-scanning

| Project | What | Open? | Cost | Notes |
|---|---|---|---|---|
| **OpenScan** ([openscan-lsm](https://github.com/openscan-lsm/OpenScan), LOCI/UW-Madison) | The real open laser-scanning acquisition stack. Micro-Manager device adapter presents the scanner as a camera. | **BSD-2**, code only | free | Galvo/NI-DAQ. **Windows only.** No paper. The only credible open point-scan control stack. |
| **smfBox** ([craggslab/smfBox](https://github.com/craggslab/smfBox)) | Single-molecule confocal for solution FRET. Not an imaging microscope. | CC BY 4.0 | **£40k** `[S]` | Excelitas SPCM-AQRH SPADs, 20 µm physical pinhole, NI PCIe-6353. *Nat. Commun.* 11:5641 (2020). |
| **UW/Vaughan open spinning disk** | Microlens-free Nipkow disk on a hard-drive spindle. | Figshare CAD + BOM | disk **$400–1,000** `[S]`; module **~$7k** `[S]` | *Biomed. Opt. Express* 13:1102 (2022). **Bolts onto an existing laser-epi stand.** ~1–4% excitation throughput. |
| **BrightEyes-ISM / MCS / TTM** (Vicidomini, IIT) | Image scanning microscopy with a 5×5 SPAD array. | GPL-3, **software only** | 5-figure `[I]` | *Nat. Photonics* 17:457 (2023). Hardware not open; needs SPAD array + NI FPGA. |
| **ScanImage** | Galvo/resonant 2P engine. Now MBF Bioscience. | **Free tier = frozen 5.7R1, Apache-2.0** (unsupported, as-is). Basic/Premium paid, quote-only. | free (frozen) or paid | What most DIY builders actually use. MATLAB + Windows + NI. |

**Nothing here is a turnkey open confocal.** OpenScan is the control software; the
optics are yours to build.

## 2. Light sheet (LSFM / SPIM)

| Project | Open? | Cost | Resolution | Sample class |
|---|---|---|---|---|
| **OpenSPIM** ([openspim](https://github.com/openspim)) | CC BY-SA 3.0 hardware | ~$30–50k `[I]` | ~1–2 µm | Live embryos. *Nat. Methods* 10:598 (2013). Maintained, low tempo. |
| **Benchtop mesoSPIM** ([mesoSPIM](https://github.com/mesoSPIM)) | **GPL-3.0 hardware** | **~€95k / $100–125k** `[S]` | 1.5 µm lat / 3.3 µm ax | **Cleared tissue.** 1 cm³ in ~13 min. *Nat. Commun.* 15:2679 (2024). The credible turnkey-open cleared-tissue answer. |
| **mesoSPIM V5** | GPL-3.0 | **$169.6k–239.6k** `[S]` | ~6.5 µm iso | Whole cleared CNS. *Nat. Methods* 16:1105 (2019). |
| **DaXi** ([royerlab/daxi](https://github.com/royerlab/daxi)) | **BSD-3**, CAD + code | ~$130k `[A]` | 450 nm / 2 µm | Embryos, organoids, cleared. Single-objective OPM. *Nat. Methods* 19:461 (2022). |
| **Altair-LSFM** ([TheDeanLab/altair](https://github.com/TheDeanLab/altair)) | CC BY 4.0 sw; hw free academic | **~$150k incl. table + laser** `[S]` | **235 nm / 350 nm** | Adherent cells, subcellular. *Explicitly not* for cleared tissue. *eLife* 14:RP106910 (2026). |
| **Snouty / high-NA SOLS** ([AndrewGYork](https://github.com/AndrewGYork/high_na_single_objective_lightsheet)) | CC BY 4.0 | 6-figure `[I]` | ~200 nm | Live cells on coverslips. The bespoke AMS-AGY tertiary objective is the bottleneck. |
| **openUC2 light-sheet** | CERN-OHL | €400 DIY floor `[S]`; realistically several k€ | unstated | **Repo archived (read-only, 2022).** Teaching grade. ECi-cleared brain demo exists. |
| **Flamingo** (Huisken) | **No — lending programme** | n/a | — | "There is no single Flamingo we could give you the plans for." Non-reproducible by design. |
| **SCAPE** | No — Columbia patent → Leica | 6-figure `[I]` | ~1–2 µm | 100+ vol/s. Fast behaving organisms. |
| **Lattice light-sheet** | Disseminated under HHMI terms, not OSHW | $350–700k `[I]` | 230/370 nm | *Science* 346:1257998 (2014). Months to self-build. |

**Open concept:** OPM (Dunsby, *Opt. Express* 16:20306, 2008) is the underlying
single-objective geometry. SCAPE, DaXi, Snouty and Altair are all implementations.

## 3. Light field and snapshot-volumetric

**Read the caveat first.** LFM trades lateral resolution for volume, hyperbolically.
Classic LFM also has a **native-focal-plane artifact**: at z≈0 all angular views
coincide, so there is no depth diversity and reconstruction degrades to
lenslet-shaped blocks. Prevedel et al. literally positioned specimens *off* focus
to dodge it. Report resolution as a function of z; never as one number.

| Project | Open? | Performance | Notes |
|---|---|---|---|
| **LFDisplay** (Levoy, Stanford) | **GPL-2.0** | — | The founding paper: *ACM ToG* 25(3):924 (2006). |
| **Prevedel 2014 LFM** ([lightfieldscope.org](http://www.lightfieldscope.org/)) | MATLAB, licence **not stated** | 1.4 µm lat / 2.6 µm ax, 50 Hz | *Nat. Methods* 11:727. Whole-animal neuronal activity. |
| **XLFM** (Cong, eLife 2017) | eLife source files, CC-BY | 3.4 / 5 µm, 77 Hz, Ø800 µm | Fourier-plane MLA. ION Shanghai + USTC (**not** Janelia). |
| **DAOSLIMIT / sLFM** ([THU-IBCS](https://github.com/THU-IBCS/DAOSLIMIT_Protocol)) | Protocol repo **GPL-2.0**; main repo **unlicensed** | 220 nm / 400 nm | *Cell* 184:3318 (2021). Digital adaptive optics, no wavefront sensor. Protocol: *Nat. Protoc.* 17:1953 (2022), includes Zemax files. |
| **Miniscope3D** ([Waller-Lab](https://github.com/Waller-Lab/Miniscope3D)) | **BSD-3** | 2.76 µm lat / 15 µm ax, 2.5 g | *Light Sci. Appl.* 9 (2020). Reconstruction 8–24 min on a 2080 Ti. |
| **DiffuserCam** ([Waller-Lab](https://github.com/Waller-Lab/DiffuserCam-Tutorial)) | **BSD-3** | — | **~$80** DIY 2D build (Pi HQ camera + Scotch-tape diffuser). |
| **VCD-Net** ([feilab-hust](https://github.com/feilab-hust/VCD-Net)) | MIT | ms reconstruction | *Nat. Methods* 18 (2021). |
| **HyLFM-Net** ([kreshuklab](https://github.com/kreshuklab/hylfm-net)) | MIT | ~26.7 Hz 3D inference | *Nat. Methods* 18 (2021). Pairs LFM with a **light-sheet validation arm** — the honest answer to DL hallucination. |
| **LFMNet** ([pvjosue](https://github.com/pvjosue/LFMNet)) | Apache-2.0 | 20–50 ms/volume | Confocal-equivalent volume from one LF image. |

**Deep-learning LFM hallucinates.** Networks infer 3D detail the ambiguous
measurement does not contain. Only HyLFM (validation arm) and CWFA (*Biomed. Opt.
Express*, 2024 — conditional normalizing flows with an out-of-distribution flag)
address it directly. Treat DL "resolution" as learned, not optical.

**Unlicensed-but-public** (legally all-rights-reserved): `bbncWLG/DAOSLIMIT`,
`freemercury/Aberration_correction_metasensor`, `yuanlong-o/RUSH3D`.
`vazirilab/SID` and `MesoLF` use custom non-commercial academic terms.

## 4. Platforms, bodies, and super-resolution

| Project | What | Licence | Cost |
|---|---|---|---|
| **Squid / Cephla** ([Cephla-Lab/Squid](https://github.com/Cephla-Lab/Squid)) | Modular open widefield platform. **Confocal is not a Squid modality** — treat any "Squid confocal" claim as false. | BSD-3 software, CERN-OHL hardware | **$500–$8k** `[S]` vs $50–150k commercial | 50+ units in 25+ labs; Cephla gives turnkey continuity. Flagship is still a **2020 bioRxiv preprint**. |
| **openFrame** (French & Dunsby, Imperial + Cairn Research) | Modular open microscope **body** — stackable aluminium layers on a common axis. The closest open analogue to a commercial stand. Includes `openAF` single-shot autofocus. | **CERN-OHL-P-2.0** hardware (permissive), BSD-2 software | not published; Cairn sells DIY kits (open STEP files) or turnkey, quote-only | *J. Microscopy* (2023), [10.1111/jmi.13219](https://doi.org/10.1111/jmi.13219). [repo](https://github.com/ImperialCollegeLondon/openFrame). Demonstrated: epifluorescence, easySTORM, pDPC. Small community; leans on Cairn or a machine shop. |
| **openUC2** | 50 mm 3D-printed optical cubes. Widefield, holography, SIM, ISM, light-sheet. **No confocal pinhole modality.** | CERN-OHL + MIT | €100–400/instrument `[S]` |
| **OpenFlexure** | 3D-printed flexure-stage microscope | **CERN-OHL-S v2** (hardware) | <$200 |
| **UCLA Miniscope v4** ([Aharoni-Lab](https://github.com/Aharoni-Lab/Miniscope-v4)) | Head-mounted 2D widefield. The base most 3D miniatures build on. | GPL-3.0 | ~$400–1,000/scope `[S]` |
| **ASI RAMM** | Open modular commercial stand | commercial | $15–40k `[I]` |
| **fairSIM**, **SMAP** (Ries), **storm-control** (Zhuang) | SIM / SMLM software | open | free |

## 4b. Super-resolution, open

**The barrier is detectors and optics, not code.** Open analysis software is
plentiful and good; open super-resolution *hardware* that is both real and recent
is rare.

| Project | What | Licence | Status |
|---|---|---|---|
| **Ries lab 3DSMLM** ([ries-lab/3DSMLM](https://github.com/ries-lab/3DSMLM)) | Full open protocol + CAD + board files for an automated 3D multicolour high-throughput SMLM scope. **The most credible open SMLM hardware.** | open | *Nat. Protoc.* 19:2467 (2024). Recent, peer-reviewed. Not cheap. |
| **openSIM (EPFL/LBNI)** ([EstherRaeth/openSIM_LBNI](https://github.com/EstherRaeth/openSIM_LBNI)) | Open **hardware** add-on converting a fluorescence scope to SR-SIM (LED + SLM). | **CC0-1.0** | *Nat. Commun.* 15:1741 (2024). Genuinely open. |
| **fairSIM** ([fairSIM](https://github.com/fairSIM/fairSIM)) | SIM reconstruction, Fiji plugin, ~100–130 nm | GPL-2.0 | *Nat. Commun.* 7:10980 (2016). Active. |
| **Picasso** ([jungmannlab/picasso](https://github.com/jungmannlab/picasso)) | DNA-PAINT / SMLM analysis | MIT | Active (2026). |
| **BrightEyes-ISM** ([VicidominiLab](https://github.com/VicidominiLab/BrightEyes-ISM)) | ISM analysis: pixel reassignment, Focus-ISM, deconvolution | GPL-3 | *Nat. Photonics* (2023). Very active. Analysis only. |
| **BrightEyes-TTM** | Open-hardware FPGA time-tagger, ~30 ps, on a Xilinx KC705 | **CC-BY-NC 4.0** (non-commercial!) | ~**$3,000** `[S]`. Real cheap open hardware; a solder-and-VHDL project. |
| **easySTORM / easySIM** (Imperial) | Low-cost STORM using cheap multimode diode lasers; runs on openFrame | open | ~25 nm, ~$12–26k `[paywall-derived]` |

**The ISM barrier is the SPAD array**, not the software: a small array in the
image plane is quote-only at **tens of k€** (Pi Imaging SPAD23, PicoQuant PDA-23,
Genoa Instruments). `BrightEyes-MCS` also needs a commercial NI FPGA.

**Not open, despite appearing in "open / low-cost" searches:**
`miniSTORM` (patented, US 11,067,510 B2, commercial), **MINFLUX** (Abberior-only;
`pyflux` has *no LICENSE file* and is therefore not legally open), **open STED
does not exist**, and **MoNaLISA**'s RESOLFT hardware was never released — only
its software (ImSwitch).

**Dead or dormant:** `ZhuangLab/storm-control` (last release 2019), Ries's `SMAP`
GitHub (quiet since 2020, MATLAB), the original `zitmen/thunderstorm`
(**archived Sep 2025** — use `HohlbeinLab/prj-thunderstorm`), `UC2-GIT` and
`UC2_openSIM` (frozen/archived).

**Expansion microscopy** is a wet-lab sample-prep protocol, not an instrument.

## 4c. Two-photon, open — and why it does not fit a $30k budget

| Project | Licence | Cost | Notes |
|---|---|---|---|
| **TWINKLE** ([BrainCOGS/Microscope](https://github.com/BrainCOGS/Microscope)) | **CC-BY-4.0** | **~$110k excl. laser; ~$200k incl.** `[S]` | Best-documented recent open benchtop 2P. Fusion360 CAD + BOM + Zemax. *PLoS ONE* (2025). |
| **Janelia MIMMS** | open | — | The reference design most others descend from. Runs on ScanImage. |
| **MINI2P** ([kavli-ntnu](https://github.com/kavli-ntnu/MINI2P_toolbox)) | MIT | ~$50–60k + laser | *Cell* 185:1240 (2022). 2.4 g head-mounted. **Dormant** since 2023. Genuinely hard build. |
| **UCLA 2P Miniscope** ([golshanilab](https://github.com/golshanilab/UCLA_2P_Miniscope)) | GPL-3 | headpiece **$7,245**; **full system $109,644** `[S]` | *Nat. Commun.* (2025). Two on-board SiPMs. Active. |
| **TIMAHC** | — | ~$65k excl. laser | *PLoS ONE* 9:e110475 (2014). **No repo** — CAD lives only as PLOS supplements. |

**The femtosecond laser is the floor.** Ti:Sapphire (tunable): **$120–200k**,
quote-only. Fixed-wavelength fibre lasers (Coherent Axon, Spark ALCOR, Toptica
FemtoFiber): **from ~$40k**. The ~3–5× drop is exactly what enabled the miniscope
generation — but **there is no sub-$10k femtosecond source**, and the "$7k
miniscope" figures are the *headpiece only*. A complete freely-moving 2P system is
~$100–110k. This is why two-photon is out of reach at $30k, whatever the samples
demand.

## 5. Control and acquisition software

| Stack | Licence | Linux? | Point-scanning? |
|---|---|---|---|
| **Micro-Manager** | LGPL-2.1 | yes | drives stages/cameras/shutters/CSU — **not** a proprietary scan head |
| **pycro-manager** | **BSD-3** | yes | scripted hardware-triggered acquisition |
| **OpenScan** | BSD-2 | **no (Windows + NI-DAQmx)** | **yes** — galvo raster, the real thing |
| **ImSwitch** | **GPL-3.0** | yes | yes — `PointScanTTLCycleDesigner`, per-pixel dwell, APD readout |
| **Cockpit / python-microscope** | GPL-3.0 | yes | no demonstrated point-scanning |
| **ScanImage** | free frozen 5.7R1 (Apache-2.0), current versions paid | no | yes |
| **napari** | BSD-3 | yes | viewer |

**The Linux problem is real, and worse than it first looks.**

No mature, turnkey, open, Linux-native laser-scanning acquisition stack exists.
`OpenScanLib`'s README states plainly: *"Only Windows/MSVC is currently
supported."* Every established open scanning package is Windows-bound because
scan-waveform generation plus synchronized point detection goes through NI-DAQmx.

The non-obvious nuance: **the full NI-DAQmx driver now runs on Linux Desktop**
(`ni/nidaqmx-python`, actively maintained), so the driver barrier is largely gone
— but NI's supported distributions are RHEL, openSUSE Leap and Ubuntu LTS.
**Fedora is not among them.** And no acquisition application is packaged to
exploit Linux NI-DAQmx anyway. On Linux you will be *integrating*, not installing.

Four concrete routes, none of them turnkey:

- **A — `pymmcore-plus` + a DAQ.** `pymmcore-plus` (BSD-3) is genuinely
  Linux-native, no Java. The missing piece is a Micro-Manager NIDAQ device adapter
  that builds and runs on Linux doing galvo AO plus APD counter-in. That is a C++
  port nobody has published. *Unverified that anyone has done it.*
- **B — ImSwitch galvo scanning on Linux (most promising).** ImSwitch *does*
  implement point scanning (`GalvoScanDesigner`, `PointScanTTLCycleDesigner`, APD
  counter read), built for a Testa-lab STED rig on an NI PCI-6371, and the
  framework installs on Linux. The scanning path uses `nidaqmx`, so it needs
  NI-DAQmx-for-Linux and a supported card. **Not a documented or tested config** —
  every real scanning deployment is Windows. And ImSwitch is **GPL-3**, which
  relicenses anything that links it.
- **C — Red Pitaya FPGA scan engine.** The STEMlab is itself a Linux FPGA SBC, so
  this sidesteps NI and Windows entirely; Pavel Demin's `red-pitaya-notes` has a
  scanning FPGA project that can drive galvos. **No maintained, ImSwitch-integrated
  Red-Pitaya confocal module was found.** A firmware project, not a package.
- **D — Port OpenScan.** A real MSVC → Linux port of `OpenScanLib` and its NI
  module. Largest payoff (BSD-2, keeps this project BSD-3), largest effort.

Tutorial-grade prior art worth reading before starting: **SimplePyScanner**
(`SWC-Advanced-Microscopy/SimplePyScanner`, GPL-3, linear galvo scanning via
`nidaqmx`, explicitly "for demo purposes") and **lsmaq** (`danionella/lsmaq`, MIT,
MATLAB + NI). There is **no project named "PyScanner."**

The pragmatic default remains: **Windows acquisition box, Linux reconstruction.**

## 6. Micro-Manager drives a Yokogawa CSU

Adapters exist: `Yokogawa` (CSU22), `YokogawaCSUX` (CSU-X1), `YokogawaCSUW1`.
A CSU is fundamentally a motor, a filter wheel and a shutter. Caveat: CSU serial
commands are slow, so per-frame channel switching must go through the AOTF or the
laser, never the CSU filter wheel.

---

## Verification flags

- OpenSPIM's $30–50k is community folklore, not a published figure.
- All commercial light-sheet prices are estimates; no vendor publishes list prices.
- DaXi's ~$130k and Squid's "10–50× cheaper" are author claims, not validated totals.
- openUC2's €400 is a real 2020 DIY BOM floor; the current commercial add-on is
  €1,699–2,294 ex-VAT and needs separately-bought base modules.
- MiniLFM has no dedicated open-hardware repo; treat as not openly released.
- Author-reported volume rates and speedups are self-benchmarked.
