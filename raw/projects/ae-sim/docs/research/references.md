# References

Experimental acoustoelectric imaging and neuromodulation papers consulted for this project,
with the role each plays. Searched via PubMed and Consensus, 2026-10-22/23.

## Replication targets (absolute sensitivities)

- Preston C, Kasoff WS, Witte RS. Selective mapping of deep brain stimulation lead currents
  with acoustoelectric imaging. *Ultrasound Med Biol* 2018.
  DOI [10.1016/j.ultrasmedbio.2018.06.021](https://doi.org/10.1016/j.ultrasmedbio.2018.06.021)
- Barragan A, Preston C, Alvarez A, Bera T, Qin Y, Weinand M, Kasoff W, Witte RS. Acoustoelectric
  imaging of deep dipoles in a human head phantom for guiding treatment of epilepsy.
  *J Neural Eng* 2020. DOI [10.1088/1741-2552/abb63a](https://doi.org/10.1088/1741-2552/abb63a)
- Kang Y et al. Multichannel acoustoelectric brain imaging (simulation + saline cube). *IEEE
  Trans Med Imaging* 2023. DOI [10.1109/TMI.2022.3215748](https://doi.org/10.1109/TMI.2022.3215748)
- Allard A et al. Transcranial acoustoelectric imaging through a human skull with a depth
  electrode. *IEEE Trans Biomed Eng* 2025. DOI
  [10.1109/TBME.2024.3514553](https://doi.org/10.1109/TBME.2024.3514553), PMC12153511
- Cheong et al. k-Wave acoustic fields for the 0.6 MHz 18 × 7 array. *Ultrasonics* 2026.
  DOI [10.1016/j.ultras.2026.108229](https://doi.org/10.1016/j.ultras.2026.108229)
- Abu Farha et al. Detection of an ictal waveform with Wiener filtering. *J Neural Eng* 2026.
  DOI [10.1088/1741-2552/ae7764](https://doi.org/10.1088/1741-2552/ae7764)

## Coupling constant

- Wang et al. Measurement of the acoustoelectric interaction constant in saline and porcine
  brain. *IEEE Trans Biomed Eng* 2026.
  DOI [10.1109/TBME.2026.3682610](https://doi.org/10.1109/TBME.2026.3682610)

## Rintoul (Imperial)

- Rintoul J et al. Acoustoelectric heterodyning. *Commun Phys* 2023.
  DOI [10.1038/s42005-023-01198-w](https://doi.org/10.1038/s42005-023-01198-w).
  Code: <https://github.com/Acoustoelectric/Acoustoelectric> (CC BY-NC-SA 4.0). Data:
  Figshare c.6365098 (CC BY 4.0). Later collections: c.7909283 (2026 neuromodulation figure
  data, CC BY 4.0, code Apache-2.0), c.7754537 (2026 recording example, CC BY 4.0).
- Rintoul J et al. Acoustoelectric neuromodulation and its contribution to ultrasound
  stimulation. *Nat Commun* 2026.
  DOI [10.1038/s41467-026-73826-2](https://doi.org/10.1038/s41467-026-73826-2), PMC13276402
- Rintoul J et al. Acoustoelectric neural recording. *Commun Eng* 2026.
  DOI [10.1038/s44172-026-00598-4](https://doi.org/10.1038/s44172-026-00598-4), PMC12920806

## Tianjin in vivo rat line (parameters from abstracts only)

- Chen et al. 2023, SSVEP acoustoelectric brain imaging in rat.
- Zhou et al. 2024 and 2026, transcranial acoustoelectric mapping of DBS lead currents in rat.
- Zhou et al. 2023, *Front Physiol*.
- Zhang et al. 2023 and 2026, skull-aberration-corrected phased-array acoustoelectric imaging.
- Guo et al. 2026, *Sensors*, acoustoelectric simulation platform.

## Head models and datasets

- SIGMA v2 Wistar rat template: Zenodo [10635831](https://zenodo.org/records/10635831), CC BY 4.0.
  Paper: Barrière DA et al. The SIGMA rat brain templates and atlases for multimodal MRI data
  analysis and visualization. *Nat Commun* 2019;10:5699.
  DOI [10.1038/s41467-019-13575-7](https://doi.org/10.1038/s41467-019-13575-7) (PMID 31836716).
- Waxholm Space Sprague Dawley atlas: <https://www.nitrc.org/projects/whs-sd-atlas>.
- Digimouse: Dogdas B, Stout D, Chatziioannou AF, Leahy RM. *Phys Med Biol* 2007.
  DOI [10.1088/0031-9155/52/3/003](https://doi.org/10.1088/0031-9155/52/3/003);
  <https://neuroimage.usc.edu/neuro/Digimouse>. CC BY-SA 4.0; `atlas.zip` (0.1 mm labels) and
  `tesselated_atlas.zip` (58 k-node / 307 k-tet whole-body mesh with organ labels) at
  <https://neuroimage.usc.edu/mouse_atlas/>.
- Toronto NeuroMouse surgical atlases: <http://db.phm.utoronto.ca/surgical.htm>.
- SimNIBS 4.6 (`ernie` example head, `create_mesh`): <https://simnibs.github.io/simnibs/>.

## Neural readout (weak-field thresholds, activating function)

- Hodgkin AL, Huxley AF. A quantitative description of membrane current and its application to
  conduction and excitation in nerve. *J Physiol* 1952;117:500-44.
  DOI [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)
  (PMID 12991237, PMC1392413): the point-neuron model of `ae_sim.neuron` (1952 squid-axon
  parameters, rest at -65 mV).

- Rattay F. Analysis of models for external stimulation of axons. *IEEE Trans Biomed Eng* 1986;33:974-7.
  DOI [10.1109/TBME.1986.325670](https://doi.org/10.1109/TBME.1986.325670) (PMID 3770787): the
  activating function d2V_e/dx2.
- Radman T, Ramos RL, Brumberg JC, Bikson M. Role of cortical cell type and morphology in
  subthreshold and suprathreshold uniform electric field stimulation in vitro. *Brain Stimul*
  2009;2:215-28. DOI [10.1016/j.brs.2009.03.007](https://doi.org/10.1016/j.brs.2009.03.007)
  (PMID 20161507): somatic polarisation ~0.1-0.3 mV per V/m, layer V pyramidal cells the most
  sensitive; direct action-potential thresholds of tens of V/m.
- Reato D, Rahman A, Bikson M, Parra LC. Low-intensity electrical stimulation affects network
  dynamics by modulating population rate and spike timing. *J Neurosci* 2010;30:15067-79.
  DOI [10.1523/JNEUROSCI.2059-10.2010](https://doi.org/10.1523/JNEUROSCI.2059-10.2010)
  (PMID 21068312): weak (< 10 V/m) low-frequency fields modulate gamma oscillations in slices;
  spike-phase entrainment resonance at 0.2 V/m.

## Rodent skull acoustics

- Estrada H, Rebling J, Turner J, Razansky D. Broadband acoustic properties of a murine skull.
  *Phys Med Biol* 2016;61:1932-46. DOI [10.1088/0031-9155/61/5/1932](https://doi.org/10.1088/0031-9155/61/5/1932)
  (PMID 26878583): insertion loss and elastic constants of ex vivo adult mouse skulls.
- O'Reilly MA, Muller A, Hynynen K. Ultrasound insertion loss of rat parietal bone appears to be
  proportional to animal mass at submegahertz frequencies. *Ultrasound Med Biol* 2011;37:1930-7.
  DOI [10.1016/j.ultrasmedbio.2011.08.001](https://doi.org/10.1016/j.ultrasmedbio.2011.08.001)
  (PMID 21925788): mouse skull aberration negligible at 2.5 MHz.
- Gerstenmayer M, Fellah B, Magnin R, Selingue E, Larrat B. Acoustic transmission factor through
  the rat skull as a function of body mass, frequency and position. *Ultrasound Med Biol*
  2018;44:2336-44. DOI [10.1016/j.ultrasmedbio.2018.06.005](https://doi.org/10.1016/j.ultrasmedbio.2018.06.005)
  (PMID 30076032): reflection dominates the insertion loss.

The skull material in `examples/digimouse_skull.py` (density 1900 kg/m^3, 2900 m/s, 40 Np/m at
500 kHz, no shear) is generic cortical bone; replacing it with the mouse values from Estrada
2016 is on the roadmap.

## Solvers in the stack

- BabelViscoFDTD / BabelBrain (Pichardo): <https://github.com/ProteusMRIgHIFU/BabelViscoFDTD>.
- scikit-fem: <https://github.com/kinnala/scikit-fem>.
