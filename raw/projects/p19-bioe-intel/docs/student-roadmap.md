# Student roadmap: record, image, stimulate, compute

Calcium imaging and optogenetics are explicit project goals. Each stage should produce a useful student report even if later stages take another cohort. Dates depend on access, method recovery and measured culture readiness.

| Stage | Student question and skills | Deliverable and advancement evidence |
| --- | --- | --- |
| 1. Establish the preparation | What cells and ASCL1 implementation are we using? Learn provenance, construct interpretation, controls, microscopy and viability scoring. | Verified records and quantitative neuronal yield, composition and survival across preparations |
| 2. Collect baseline electrophysiology | When and how consistently does activity emerge? Learn noise checks, electrode mapping, raw-trace inspection, event detection, raster plots and repeated measurements. | Baseline electrical dataset with silent cultures, cell-free controls, a positive reference and culture-level summaries |
| 3. Measure calcium dynamics | Where are indicator responses and what do they mean? Learn indicator loading/expression, fluorescence optics, segmentation, background correction, bleaching and ΔF/F analysis. | Optical dataset with indicator-negative controls and evidence that illumination preserves culture health and electrical activity |
| 4. Relate optical and electrical activity | Which calcium events coincide with electrical population events? Learn timing alignment, spatial overlap and measurement uncertainty. | Synchronized MEA/imaging recordings with validated offset/drift and an explicit account of indicator kinetics |
| 5. Validate optogenetic input | Does the chosen actuator cause repeatable electrical responses? Learn construct/delivery design, expression checks, cell-plane irradiance calibration and stimulus-response analysis. | Light-locked MEA or patch-clamp responses, response probability/latency/recovery and opsin-negative/light-artifact controls |
| 6. Integrate imaging and stimulation | Can we measure recruitment while delivering controlled optical input? Learn spectral compatibility, channel isolation, filtering and interleaving. | Measured interference matrix, common timing log, single-channel controls and validated combined acquisition |
| 7. Evaluate computation | Can held-out inputs be decoded reliably? Learn balanced task design, image-grouped splits, readout training, software baselines and drift tests. | Temporal-pattern and MNIST reports using qualified electrical or optical input, with separate claims for decoding and biological adaptation |

Stages 3 and 5 can develop as separate modules after electrical qualification. Stage 6 requires both. Stage 7 can initially use electrical input while the optical modules are developed; imaging and optogenetics remain planned goals rather than prerequisites for every student to complete a first report.

## Instrument and construct choices

Use available fluorescence microscopy for initial validation when that reduces integration burden. The bench-mounted UCLA Miniscope is a candidate portable implementation, with field of view, focus, window clearance, filters and timing verified on the actual culture/MEA. Its use is not already validated here.

A calcium dye can allow an initial imaging module without another genetic construct. A genetically encoded indicator adds delivery and expression requirements. Decide with the mentor after checking the actual equipment and culture compatibility.

Design ASCL1 reporter, indicator and actuator together. Fluorescent expression supports localization; it does not prove actuation. Test whether imaging excitation activates the opsin, whether actuator light enters the camera, and whether light affects the MEA. Wavelength separation and acquisition interleaving require measurements on the selected components.

For optical-control experiments, include opsin-negative illuminated cultures, opsin-positive dark recordings, cell-free illumination, viability and recovery measurements. Establish actuator function through electrical recording before relying on calcium readout alone. Optical pulse parameters remain unresolved in the [parameter sheet](../student-materials/parameter-lock.json).

## Reading and assessment

The [MiV reading map](miv-reading-map.md) distinguishes inspected sources from work still needing review. The platform study's concurrent imaging demonstration is the initial methods anchor; the optical reservoir study is a later computational reference. Neither supplies a validated ASCL1/P19 optical protocol.

Students should be assessed on documented controls, reproducible analysis, interpretation of negative findings and successful handoff to the next cohort. Software/replay exercises can begin while cells mature. Maintain separate preparation, acquisition and analysis clocks.
