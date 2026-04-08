```yaml
type: modality
title: Transcranial Magnetic Stimulation
physics: electromagnetic
measurement: neural activation via magnetic field induction
spatial_resolution: 0.5-1.0 cm
temporal_resolution: milliseconds
related: [eeg.md, tms-eeg.md, neural-mass-models.md, electromagnetic-forward-modeling.md]
```

# Transcranial Magnetic Stimulation

Transcranial Magnetic Stimulation (TMS) is a non-invasive neurostimulation technique that uses rapidly changing magnetic fields to induce localized electrical currents in brain tissue, causing temporary activation or inhibition of neural circuits.

## Physics

TMS operates on the principle of electromagnetic induction described by Faraday's law:

```
∇ × E = -∂B/∂t
```

A time-varying magnetic field B(t) in the TMS coil induces an electric field E in the brain tissue. The induced electric field creates ionic currents that can depolarize neuronal membranes when the field strength exceeds the neural excitation threshold.

The magnetic field strength decays approximately as 1/r² with distance from the coil, limiting effective stimulation to cortical regions within ~2-3 cm of the scalp surface.

## Technical Parameters

| Parameter | Typical Range | Units |
|-----------|---------------|-------|
| Magnetic field strength | 1.5-4.0 | Tesla |
| Pulse duration | 100-300 | microseconds |
| Repetition rate (rTMS) | 1-50 | Hz |
| Coil-cortex distance | 10-25 | mm |
| Induced E-field | 100-200 | V/m |
| Penetration depth | 15-30 | mm |
| Spatial resolution | 5-10 | mm |

## Coil Types

**Figure-8 Coil**: Two adjacent circular loops with current flowing in opposite directions, creating focal stimulation with peak field at the intersection.

**Circular Coil**: Single loop producing more diffuse stimulation with maximum field at the coil center.

**H-Coil**: Deep TMS coil design for stimulating deeper brain structures up to 4-6 cm from scalp.

## Stimulation Protocols

### Single-Pulse TMS
- Used for measuring cortical excitability
- Motor threshold determination
- Paired-pulse paradigms for intracortical inhibition/facilitation

### Repetitive TMS (rTMS)
- **Low frequency (≤1 Hz)**: Generally inhibitory effects
- **High frequency (>1 Hz)**: Generally excitatory effects
- **Theta burst**: Patterned stimulation mimicking endogenous theta rhythms

### TMS-EEG
Concurrent TMS with EEG recording to measure evoked responses (TEPs) and connectivity changes. Requires specialized EEG amplifiers with rapid recovery from TMS artifacts.

## Forward Modeling

TMS forward modeling predicts the induced electric field distribution from coil geometry and head anatomy:

```python
E_induced = solve_tms_forward(coil_position, head_model, conductivity_tensor)
```

Key components:
- **Coil model**: Wire path and current distribution
- **Head model**: Tissue segmentation with conductivity values
- **Field calculation**: Boundary element method (BEM) or finite element method (FEM)

Typical tissue conductivities for TMS modeling:
- Scalp: 0.465 S/m
- Skull: 0.010 S/m  
- CSF: 1.654 S/m
- Gray matter: 0.275 S/m
- White matter: 0.126 S/m (isotropic) or DTI-derived tensor

## Neural Response Modeling

TMS-induced neural responses can be modeled using neural mass models that incorporate the external stimulation:

```python
def tms_neural_mass_model(state, t, tms_input):
    # Base neural mass dynamics
    dxdt = neural_mass_dfun(state, coupling, params)
    
    # Add TMS perturbation
    if tms_active(t):
        dxdt += tms_coupling * tms_input * spatial_weight
    
    return dxdt
```

The TMS perturbation typically affects excitatory population parameters, with the spatial weight determined by the coil-induced E-field strength.

## Safety Considerations

- **Seizure risk**: Particularly with high-frequency rTMS
- **Heating**: Coil and scalp temperature monitoring
- **Contraindications**: Metallic implants, cardiac pacemakers, pregnancy
- **Dosage limits**: Maximum pulses per session and inter-session intervals

International safety guidelines specify maximum stimulation intensities and pulse counts based on frequency and session duration.

## Applications

### Research
- Causal brain-behavior relationships
- Virtual lesion studies  
- Cortical excitability mapping
- Connectivity measurements via TMS-EEG

### Clinical
- Treatment-resistant depression (FDA-approved)
- Migraine prevention
- Obsessive-compulsive disorder
- Stroke rehabilitation

## Relevant Projects

**neurojax**: Implements TMS forward modeling with differentiable head models, enabling gradient-based optimization of coil placement for targeted stimulation.

**vbjax**: Provides neural mass models (CMC, JR, RWW) that can incorporate TMS perturbations for simulating evoked responses and aftereffects.

## See Also

- [eeg.md](eeg.md) - EEG recording and analysis
- [electromagnetic-forward-modeling.md](electromagnetic-forward-modeling.md) - Forward model computation
- [neural-mass-models.md](neural-mass-models.md) - Neural population dynamics
- [head-models.md](head-models.md) - Anatomical models for field calculation