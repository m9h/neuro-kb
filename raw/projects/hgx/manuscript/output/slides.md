---
title: "Hypergraph neural networks in JAX/Equinox. Topological and geometric deep learning on higher-order domains."
author:
  - Morgan G Hough
date: "2026-03-20"
theme: metropolis
---


# Introduction


## HGX: Hypergraph Neural Networks in JAX/Equinox

HGX is the first JAX-native library for deep learning on hypergraphs and higher-order topological domains, providing 14 convolution layers, continuous-time dynamics via Diffrax, and JIT-compatible dynamic topology -- all built on Equinox.


# Methodology


## Hypergraph Data Structure

The Hypergraph dataclass is the core data structure, representing topology via an incidence matrix with optional geometry and node/edge features, constructed from incidence matrices, edge lists, or adjacency matrices.


## Convolution Layers

HGX provides six hypergraph convolution layers (UniGCN, UniGAT, UniGIN, THNN, and sparse variants) sharing a unified interface, enabling first-order message passing, attention-weighted aggregation, and tensorized higher-order interactions.


## Model Architecture

HGNNStack is a composable multi-layer hypergraph neural network builder supporting configurable convolution types, activation functions, dropout, and optional readout layers.


## Sparse Message Passing Utilities

Index-based sparse message passing via star expansion provides O(nnz) aggregation using segment_sum, replacing dense O(n*m) matrix multiplication for scalable hypergraph convolution.


## Dynamic Topology

JIT-compatible dynamic topology operations (add/remove nodes and hyperedges) use pre-allocated masked arrays so that array shapes remain fixed, enabling topology changes within JAX-compiled functions.


## Continuous-Time Dynamics

HypergraphNeuralODE and HypergraphNeuralSDE evolve node features in continuous time using hypergraph convolution layers as the learned vector field, integrated via Diffrax.


## Hypergraph Transforms

Transforms convert between hypergraph and graph representations, including clique expansion to adjacency matrices and normalized/unnormalized hypergraph Laplacian computation.


## Visualization

Visualization utilities render hypergraphs as bipartite star-expansion layouts, incidence matrix heatmaps, and attention weight diagrams using matplotlib and networkx.
