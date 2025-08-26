# Attention as linear transformations - embedding weights change based on context
- [source](https://www.youtube.com/watch?v=UPtG_38Oq8o&list=PLCip3d1iHEMXcAZPhPSb6Br0dykmPKcji&t=2175s)

*   Attention **resolves word ambiguity** by **contextually modifying embeddings**, intuitively visualized as words **"pulling" each other** to new, precise locations in space.
*   These contextual modifications are achieved through **linear transformations (matrices)**, which **rotate, stretch, or shear** embeddings to create better-separated, context-aware representations.
*   The **Query (Q) and Key (K)** matrices transform embeddings to **optimize for similarity calculations** (via dot product), quantifying how words relate to each other.
*   While Q and K are for **finding relationships**, the **Value (V) matrix** creates embeddings **optimized for the Transformer's primary task of next-word prediction**.