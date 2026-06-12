# Moved → m9h/neurocontainers-arm

This has been consolidated into **[neurocontainers-arm/mamba-ssm-gb10](https://github.com/m9h/neurocontainers-arm/tree/main/mamba-ssm-gb10)**,
the home for aarch64 / NVIDIA Grace–Blackwell (GB10 / DGX Spark) neuroimaging container + package support.

- **Generic conda on GB10:** conda-forge already works — `mamba create -c conda-forge "cuda-version=12.9" pytorch-gpu causal-conv1d mamba-ssm`
- **NGC PyTorch container (torch 2.12 / CUDA 13):** prebuilt `causal-conv1d` wheel on the
  [neurocontainers-arm releases](https://github.com/m9h/neurocontainers-arm/releases).

This repo is archived.
