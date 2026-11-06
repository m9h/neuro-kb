# P19 computing benchmark plan

This is a proposed study sequence. No P19 task performance or shortened maturation schedule has been demonstrated by this project.

## Reading anchors

- Cell-source framework: [Tanveer et al., Starting a synthetic biological intelligence lab from scratch](https://doi.org/10.1016/j.patter.2025.101232), Table 1.
- MNIST comparison: [Loeffler et al., 2026 preprint](https://doi.org/10.64898/2026.08.10.743829), Results 2.5 and Methods 5.3–5.4. Author manuscript inspected through its [deposited copy](https://www.researchgate.net/publication/412314138_Handwritten_Digit_classification_with_neural_cultures_is_influenced_by_neural_architecture_network_dynamics_and_decoding_methods); direct bioRxiv PDF access was rate limited. The [authors' repository](https://github.com/Cortical-Labs/handwritten-digit-neural-cultures) supplies a recording manifest and analysis notebooks.
- MiV comparison: [Kim et al., 2026 preprint](https://arxiv.org/html/2604.02552v1), Figures 1–4 and reconstruction-task section. John Beggs is a coauthor.

The article contains the source-specific parameters and performance distinctions. Both 2026 papers are preprints as checked on 30 September 2026. Archive versions and inspect supplements and code before freezing a reproduction protocol.

## First milestone: baseline electrophysiology

The project lead reports little electrophysiology collected so far. First collect repeat baseline recordings across culture age and independent inductions, with raw traces, noise and waveform checks, active-electrode fraction, firing and bursting, viability, and drift. Include silent cultures, cell-free recordings and a positive neural reference. Add evoked-response testing after acquisition is validated. This initial dataset can be the first project report even if no computing task is reached.

Published RA-derived P19 network recordings and bHLH-conversion studies motivate this work but do not establish the current ASCL1 preparation. Do not describe the small set of reviewed precedents as a comprehensive mature electrophysiology literature.

The [student roadmap](student-roadmap.md) adds calcium imaging and optogenetics as explicit learning goals, with independently assessable datasets before multimodal computing.

## Proposed sequence

| Gate | Measure | Advance when |
| --- | --- | --- |
| Reproducible cell supply | Bank recovery, passage, identity, contamination, cost and hands-on time | Starting cultures have traceable provenance |
| ASCL1 induction | Expression, whole-culture neuronal yield, viability, residual proliferation and composition | Pilot-defined preparation requirements are met |
| Functional coupling | Activity on actual MEA, stimulus response after measured artifact recovery, repeated-input reliability | The apparatus and culture pass frozen readiness limits |
| Short temporal task | Ten classes, balanced blocks, later held-out block, equal-count timing probes | Discrimination exceeds matched artifact controls reproducibly |
| MNIST | Frozen encoder, image-grouped train/test split, single-trial and predefined repeated-image inference | Performance is interpretable with software and no-cell baselines |
| Stability | Frozen readout evaluated later in session and on another day | Drift is quantified rather than hidden by refitting |
| Extensions | Patterned architecture, optical stabilization, transfer, contingent feedback | Each addition has an established baseline and explicit controls |

## Session budgets

The initial temporal pilot proposes 300 balanced presentations. A one-second cycle gives five minutes of nominal presentation time. Rest, calibration, preflight recording and controls increase the total. The culture's measured recovery can require a slower cycle.

The initial MNIST plan proposes 1,000 presentations. A two-second cycle gives roughly 33 minutes of nominal acquisition, conditional on recovery and viability validation. This is scheduling arithmetic, not a published P19 protocol or assurance of comparable accuracy. Repeat presentations and image identities must be defined explicitly.

Start software and replay work while cultures mature. Do not assign an unverified day on which ASCL1 cultures must be task ready. Track bank-to-readiness, induction-to-readiness, acquisition duration and analysis time separately. Count all failed cultures in classroom reliability and cost estimates.

## Analysis contract

- Lock input encoding and task versions before live acquisition. Disclose what the software encoder contributes.
- Group all representations, repetitions and bins from one image within a split; reserve a chronological test block. Fit transforms and tune hyperparameters using training data only.
- Define repeat groups using input identifiers known before inference. Label-based pooling is a retrospective analysis, not an operational recognizer.
- Compare live output, shuffled labels, no-cell stimulation, replay/electronic controls, and encoded-input-only features with the same readout.
- Report ten-class chance level, confusion matrices, per-class performance, independent-culture uncertainty, failures and drift.
- Above-chance decoding establishes discriminability. Temporal/nonlinear probes and input-only comparisons assess the substrate's contribution. Feedback-dependent biological adaptation requires its own experiment.

Fill the unresolved fields in [parameter-lock.json](../student-materials/parameter-lock.json) from records and a pilot before using this as a confirmatory protocol. Short-session success does not establish long-duration stability, cross-culture transfer or a published benchmark match.
