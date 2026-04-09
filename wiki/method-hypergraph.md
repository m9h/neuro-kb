---
type: method
title: Hypergraph Methods
category: differentiable
implementations: [hgx:core, jaxctrl:hypergraph]
related: [method-neural-ode.md, method-tensor-decomposition.md, concept-controllability.md, concept-higher-order-interactions.md]
---

# Hypergraph Methods

Computational methods for learning and control on hypergraphs, which generalize standard graphs by allowing hyperedges to connect any number of vertices simultaneously. These methods capture multi-way interactions that pairwise edges cannot represent, making them particularly valuable for modeling protein complexes, chemical reactions, gene regulatory networks, and neural circuits.

## Core Principles

Hypergraphs are represented via an incidence matrix H ∈ ℝ^(n×m) where H_ij = 1 if vertex i belongs to hyperedge j. This enables message passing through a two-step process:
1. **Vertex-to-hyperedge**: Aggregate node features to hyperedge representations
2. **Hyperedge-to-vertex**: Aggregate hyperedge features back to nodes

## Key Parameters

| Parameter | Typical Range | Description |
|-----------|--------------|-------------|
| `in_dim` | 8-512 | Input node feature dimension |
| `out_dim` | 8-512 | Output node feature dimension |
| `rank` (THNN) | 16-128 | CP decomposition rank for tensorized layers |
| `dropout_rate` | 0.0-0.3 | Dropout probability during training |
| `num_heads` (GAT) | 1-8 | Number of attention heads |
| `sigma_init` (SDE) | 0.01-0.1 | Initial noise scale for stochastic dynamics |

## Convolution Architectures

### First-Order Methods
- **UniGCNConv**: Mean aggregation message passing, reduces to GCN on pairwise graphs
- **UniGATConv**: Attention-weighted aggregation with learned attention coefficients
- **UniGINConv**: GIN-style MLP aggregation with learnable self-loop parameter

### Higher-Order Methods
- **THNNConv**: Tensorized hypergraph networks via CP decomposition, captures higher-order vertex interactions within hyperedges
- **SheafHypergraphConv**: Sheaf-theoretic message passing with structured data on vertices

### Geometric Methods
- **PoincareHypergraphConv**: Poincaré ball hyperbolic geometry for hierarchical structures
- **SE3HypergraphConv**: SE(3)-equivariant layers for 3D molecular systems

## Sparse Implementation

Sparse variants use `jax.ops.segment_sum` over star expansion indices rather than dense matrix multiplication:
- Complexity: O(nnz) instead of O(n×m)
- Memory: Linear in number of vertex-hyperedge memberships
- Numerical equivalence: Identical results to dense variants

## Dynamic Topology

JIT-compatible topology modification via pre-allocated masked arrays:
```python
hg = hgx.preallocate(hg, max_nodes=100, max_edges=50)
hg = hgx.add_node(hg, features=new_feats, hyperedges=membership_mask)
hg = hgx.add_hyperedge(hg, members=node_membership)
```

## Continuous-Time Dynamics

Neural ODEs/SDEs on hypergraphs where the vector field is a hypergraph convolution:
- **HypergraphNeuralODE**: dx/dt = σ(Conv(H, x(t)))
- **HypergraphNeuralSDE**: dx = σ(Conv(H, x(t)))dt + Σ dW
- Integration via Diffrax with adaptive timestep control

## Control Theory Extensions

Hypergraph controllability analysis via tensor methods (jaxctrl integration):

### Adjacency Tensor
3-mode tensor A ∈ ℝ^(n×n×m) where A_ijk = 1 if vertices i,j both belong to hyperedge k.

### Controllability Metrics
- **Tensor Kalman rank**: Generalization of controllability matrix rank to hypergraphs
- **Control energy**: Minimum energy to drive system between states
- **Minimum driver nodes**: Smallest control input set for full controllability

### Implementation
```python
# Convert hypergraph to tensor representation
A_tensor = jaxctrl.adjacency_tensor(hg)
rank = jaxctrl.tensor_kalman_rank(A_tensor, B)
energy = jaxctrl.control_energy(A_tensor, B, x0, xf, T=1.0)
```

## Performance Characteristics

### Computational Complexity
- Dense convolution: O(n × m × d) where d is feature dimension
- Sparse convolution: O(|E| × d) where |E| is number of memberships
- Dynamic operations: O(1) amortized with pre-allocation

### Memory Requirements
- Incidence matrix: n × m floats
- Node features: n × d floats  
- Hyperedge features: m × d floats (optional)
- Sparse indices: 2 × |E| integers

### Scalability
Tested on hypergraphs up to 10^4 nodes and 10^3 hyperedges. Performance bottlenecks:
1. Dense matrix multiplication at O(n × m)
2. Attention computation at O(n × m × num_heads)  
3. Memory bandwidth for large feature dimensions

## Applications

### Gene Regulatory Networks
- Nodes: genes, transcription factors
- Hyperedges: regulatory complexes, pathways
- Features: expression levels, chromatin accessibility

### Neural Circuits  
- Nodes: neurons
- Hyperedges: synaptic cliques, functional modules
- Dynamics: continuous-time neural mass models

### Chemical Reaction Networks
- Nodes: molecular species
- Hyperedges: reactions (multiple reactants/products)
- Control: reaction rate optimization

## Key References

- **dong2024controllability**: Dong et al. (2024). Controllability and Observability of Temporal Hypergraphs. arXiv:2408.12085.
- **sharf2022functional**: Sharf et al. (2022). Functional neuronal circuitry and oscillatory dynamics in human brain organoids. Nature Communications 13:4403.
- **Kidger2021equinox**: Kidger & Garcia (2021). Equinox: neural networks in JAX via callable PyTrees and filtered transformations. Differentiable Programming workshop at NeurIPS.
- **Kidger2022neuralDE**: Kidger (2022). On Neural Differential Equations. PhD thesis, University of Oxford. Theoretical foundations for continuous-time dynamics on graphs.
- **delathauwer2000multilinear**: De Lathauwer et al. (2000). A multilinear singular value decomposition. SIAM J Matrix Analysis 21:1253-1278.

## Relevant Projects

- **hgx**: Core hypergraph neural network implementations, 14 convolution layers, dynamic topology, continuous dynamics via Diffrax
- **jaxctrl**: Hypergraph controllability analysis, tensor control methods, driver node identification

## See Also

- [method-tensor-decomposition.md](method-tensor-decomposition.md) - CP decomposition for THNN layers
- [method-neural-ode.md](method-neural-ode.md) - Continuous-time dynamics on hypergraphs  
- [concept-controllability.md](concept-controllability.md) - Network control theory fundamentals
- [concept-higher-order-interactions.md](concept-higher-order-interactions.md) - Multi-way interaction modeling