---
type: method
title: Spectral Analysis (multitaper, wavelets)
category: spectral
implementations: [neurojax:analysis.spectral, coffeine:compute_coffeine]
related: [method-time-frequency-analysis.md, method-hilbert-transform.md, method-fourier-transform.md, method-covariance-estimation.md]
---

# Spectral Analysis (multitaper, wavelets)

Spectral analysis methods for extracting frequency-domain features from neural signals, including multitaper spectral estimation and wavelet transforms. Critical for analyzing oscillatory dynamics in M/EEG and other neuroimaging modalities.

## Overview

Spectral analysis decomposes neural signals into their frequency components, revealing oscillatory patterns that reflect underlying neural processes. Modern approaches use sophisticated windowing (multitaper) or time-frequency decomposition (wavelets) to provide robust, high-resolution spectral estimates.

## Key Methods

### Multitaper Spectral Estimation
- **Principle**: Uses multiple orthogonal tapers (DPSS sequences) to reduce spectral leakage
- **Advantages**: Better bias-variance tradeoff than single-window methods
- **Parameters**: Time-bandwidth product (TW), number of tapers (K = 2×TW - 1)
- **Applications**: Power spectral density, coherence, cross-spectral analysis

### Wavelet Transform
- **Continuous Wavelet Transform (CWT)**: Provides time-frequency decomposition with adaptive resolution
- **Morlet wavelets**: Complex wavelets with Gaussian envelope, optimal for neural oscillations
- **Parameters**: Central frequency, bandwidth parameter, number of cycles
- **Applications**: Time-frequency analysis, event-related spectral perturbation

## Properties/Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| **Time-bandwidth product (TW)** | 2-4 | Controls frequency resolution vs. leakage |
| **Number of tapers (K)** | 2×TW - 1 | Typically 3-7 for neural data |
| **Frequency resolution (df)** | 2×TW / T Hz | T = window duration |
| **Wavelet cycles** | 3-7 cycles | Lower = better time resolution |
| **Frequency bands** | Delta: 1-4 Hz<br>Theta: 4-8 Hz<br>Alpha: 8-13 Hz<br>Beta: 13-30 Hz<br>Gamma: 30-100 Hz | Standard M/EEG bands |

## Implementation Details

### Multitaper Parameters
```python
# Standard parameters for M/EEG
TW = 2.0          # Time-bandwidth product
K = 3             # Number of tapers (2*TW - 1)
df = 2*TW / T     # Frequency resolution
```

### Wavelet Parameters
```python
# Morlet wavelet parameters
central_freq = 1.0    # Hz, central frequency
sigma_f = central_freq / n_cycles  # Frequency bandwidth
sigma_t = 1.0 / (2 * pi * sigma_f) # Time bandwidth
```

## Advantages and Limitations

### Multitaper
**Advantages:**
- Reduced spectral leakage compared to single windows
- Statistical properties well-understood
- Excellent for stationary signals

**Limitations:**
- Fixed time-frequency resolution
- Not optimal for non-stationary signals
- Computational cost scales with number of tapers

### Wavelets
**Advantages:**
- Adaptive time-frequency resolution
- Good for transient, non-stationary signals
- Natural for event-related analysis

**Limitations:**
- Time-frequency uncertainty principle
- Parameter selection affects results
- Edge effects at signal boundaries

## Applications in Neuroimaging

### M/EEG Analysis
- **Resting-state power spectra**: Multitaper for robust PSD estimation
- **Event-related spectral perturbation**: Wavelets for trial-by-trial dynamics
- **Connectivity analysis**: Cross-spectral coherence via multitaper
- **Source-space analysis**: Frequency-specific source reconstruction

### Covariance-Based Pipelines
The [coffeine](coffeine) package uses filter-bank approaches where spectral analysis creates frequency-specific covariance matrices for machine learning pipelines. Each frequency band produces a separate covariance feature set.

## Relevant Projects

- **neurojax**: Implements JAX-accelerated multitaper and wavelet transforms in `analysis/spectral.py`
- **coffeine**: Filter-bank covariance estimation using frequency bands defined by spectral analysis
- **vbjax**: Spectral analysis of simulated neural mass model outputs
- **qcccm**: Quantum-classical correspondence in neural oscillations

## Frequency Band Definitions

### Standard M/EEG Bands
- **Delta**: 1-4 Hz (deep sleep, unconsciousness)
- **Theta**: 4-8 Hz (memory, navigation, REM sleep)
- **Alpha**: 8-13 Hz (relaxed wakefulness, eyes closed)
- **Beta**: 13-30 Hz (active thinking, motor control)
- **Gamma**: 30-100 Hz (binding, consciousness)

### IPEG Consortium Bands
More refined frequency bands used in predictive M/EEG modeling:
- **Delta**: 1-4 Hz
- **Theta**: 4-8 Hz  
- **Alpha1**: 8-10 Hz
- **Alpha2**: 10-13 Hz
- **Beta1**: 13-20 Hz
- **Beta2**: 20-30 Hz

## Mathematical Foundation

### Multitaper PSD Estimate
```
S(f) = (1/K) * Σ_k |X_k(f)|²
```
where X_k(f) is the k-th tapered Fourier transform.

### Wavelet Transform
```
W(a,b) = (1/√a) * ∫ x(t) * ψ*((t-b)/a) dt
```
where ψ is the mother wavelet, a is scale, b is translation.

## Key References

- **thomson1982spectrum**: Thomson (1982). Spectrum estimation and harmonic analysis. Proceedings of the IEEE 70:1055-1096. Foundational paper on multitaper spectral estimation.
- **Donoghue2020fooof**: Donoghue et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience 23:1655-1665.
- **sabbagh2020predictive**: Sabbagh et al. (2020). Predictive regression modeling with MEG/EEG: from source power to signals and cognitive states. NeuroImage 222:116893.
- **ValdesSosa2026xialphanet**: Valdes-Sosa et al. (2026). xi-alphaNet: conduction delays shape alpha oscillations across the lifespan. National Science Review.

## See Also

- [method-hilbert-transform.md](method-hilbert-transform.md) - Analytic signal and instantaneous frequency
- [method-time-frequency-analysis.md](method-time-frequency-analysis.md) - General time-frequency methods
- [method-covariance-estimation.md](method-covariance-estimation.md) - Covariance matrices in frequency bands
- [concept-neural-oscillations.md](concept-neural-oscillations.md) - Biological basis of neural rhythms