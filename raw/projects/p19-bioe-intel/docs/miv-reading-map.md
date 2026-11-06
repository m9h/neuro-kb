# Mind in Vitro reading map and review status

Updated 1 October 2026. The [official publication list](https://mindinvitro.illinois.edu/publications) spans wetware, interfacing, neural dynamics, computing, robotics and theory. The complete set has not been read. The selective review below is not a complete review of all methods, supplements or code.

| Source | Inspection completed | Roadmap relevance and remaining work |
| --- | --- | --- |
| [Zhang et al., MiV platforms, 2024](https://doi.org/10.1002/advs.202306826) | Targeted primary text on platform, deployment, 2D preparations and calcium imaging; [PMC source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10953569/) and indexed excerpts | Main instrumentation anchor. It includes concurrent calcium/electrical measurements and optical interfacing. Review full supplement, component lists and optical setup before selecting a build. |
| [MiV open-hardware documentation](https://gazzolalab.github.io/MiV-OH/) | Overview and previously inspected system documentation | Choose exact MEA, PCB, acquisition/stimulation configuration and optical access. Configuration-specific instructions and local calibration remain necessary. |
| [Kim et al., chaos-controlled RC and knowledge transplant, 2026 preprint](https://arxiv.org/html/2604.02552v1) | Primary main-text task, culture/interface, diagnostic and timing sections inspected | Later optical-input/electrical-readout reference. Supplement and implementation audit remain incomplete; neither P19 transfer nor a short-session reproduction is established. |
| [Rahman, Joy and Saif, optogenetic neural spheroids, 2025](https://doi.org/10.1088/1741-2552/ade28d) | [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/40490012/) and institutional metadata only | Potential separate actuator-population route. Full methods/supplements and P19 compatibility need review before adoption. |
| [Upadhyay et al., Living Neurosheets, 2025 preprint](https://doi.org/10.64898/2025.11.28.691240) | Abstract and metadata only | Later architecture/handling candidate. Full paper and preparation details still to review. |

The next sources to inspect from the official list concern matrix stiffness/neurotrophic factors and functional connectivity, synaptic tension, astrocyte-associated network development, and neural-dynamics measurements. These may inform culture support and reproducibility before computing. Theory and robotic embodiment papers should be mapped separately to their contribution rather than treated as validation of the P19 preparation.

For each paper, extract cell identity, genetic constructs, differentiation and culture-age clocks, support-cell conditions, interface geometry, electrical/optical hardware, calibration, controls, independent biological sample counts and failure reporting. Record the exact section/figure and review status. A reported demonstration is distinct from a locally reproducible student method.
