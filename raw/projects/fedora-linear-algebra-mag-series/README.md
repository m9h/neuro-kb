# Fedora Linear Algebra Magazine Series

An 8-part article series covering the scientific linear algebra and GPU computing stack on Fedora, written in the style of [Fedora Magazine](https://fedoramagazine.org/introduction-to-blas/).

Targets **Fedora 44** package versions. Each article includes hands-on code examples, `dnf install` commands, benchmarks, and RPM packaging guidance.

## The Series

| Part | Article | Topic |
|---|---|---|
| 1 | [BLAS](linear-algebra-series-part1-blas-v3.md) | FlexiBLAS, OpenBLAS, BLIS, the Netlib/ORNL history, Goto's cache-blocking insight, Numerical Recipes legacy |
| 2 | [LAPACK](linear-algebra-series-part2-lapack.md) | Dense solvers (dgesv, dgesvd), driver/computational hierarchy, LAPACKE C interface, Python/NumPy integration |
| 3 | [Eigen](linear-algebra-series-part3-eigen.md) | C++ expression templates, fixed-size optimization, BLAS delegation, geometry module, Eigen 5.x migration |
| 4 | [Armadillo](linear-algebra-series-part4-armadillo.md) | MATLAB-like C++ syntax, PCA example, Armadillo vs Eigen comparison, FSL and the neuroimaging ecosystem |
| 5 | [Sparse & Distributed](linear-algebra-series-part5-sparse-and-beyond.md) | SuiteSparse/UMFPACK, SuperLU, MUMPS, ARPACK, PETSc, Hypre, SUNDIALS, ScaLAPACK, full stack diagram |
| 6 | [OpenCL](linear-algebra-series-part6-opencl.md) | ICD architecture, 4 runtimes (pocl/Rusticl/Intel NEO/ROCm), CLBlast, PyOpenCL, SPIR-V, memory model |
| 7 | [GPU Linear Algebra](linear-algebra-series-part7-gpu-linear-algebra.md) | rocBLAS, MAGMA, CLBlast, HIP portability layer, data transfer optimization |
| 8 | [GPU Programming Models](linear-algebra-series-part8-portability.md) | OpenACC, OpenMP target offloading, OpenCL, Kokkos, Vulkan compute, portability comparison |

## Part 1 drafts

Part 1 has three iterations showing the editorial evolution:

- [v1](linear-algebra-series-part1-blas.md) — Original (encyclopedic, lists-first)
- [v2](linear-algebra-series-part1-blas-v2.md) — Technical rewrite (performance hook, FlexiBLAS-first)
- [v3](linear-algebra-series-part1-blas-v3.md) — Final (adds Netlib/ORNL history, Goto story, Numerical Recipes context)

## Key Fedora 44 versions covered

| Package | Version | Role |
|---|---|---|
| FlexiBLAS | 3.5.0 | BLAS/LAPACK runtime switching |
| OpenBLAS | 0.3.29 | Default BLAS backend |
| Eigen3 | 5.0.1 | C++ template linear algebra (breaking change from 3.4) |
| Armadillo | 12.8.1 | MATLAB-like C++ linear algebra |
| SuiteSparse | 7.11.0 | Sparse direct solvers (UMFPACK, CHOLMOD, etc.) |
| PETSc | 3.24.5 | Scientific computing framework |
| ROCm | 7.1.1 | AMD GPU compute platform |
| MAGMA | 2.9.0 | Hybrid CPU+GPU LAPACK |
| GCC | 16.0.1 | Compiler with OpenACC/OpenMP offloading |
| PyTorch | 2.9.1 | ML framework (ROCm-enabled) |

## License

Content is available for reuse. Written by Morgan Hough.
