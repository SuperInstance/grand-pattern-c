# Grand Pattern - C Implementation

Fibonacci Dual-Direction Architecture — a cellular graph system where each cell (room) maintains dual perception/prediction databases with JEPA-style cross-comparison and double-entry bookkeeping.

## Architecture

- **Perception DB (Z_in)**: stores incoming sensor embeddings
- **Prediction DB (Z_out)**: stores predicted future embeddings
- **JEPA mapping**: cross-DB comparison, computes prediction error (surprise)
- **Double-entry bookkeeping**: every tick updates BOTH databases, must balance
- **Vibe**: (position, velocity, acceleration) tuple on the embedding manifold
- **GC**: 3-phase (merge similar → decay old → prune weak)
- **Cellular graph**: rooms as nodes, algorithms as edges, murmur as gossip protocol

## Build & Test

```bash
make test
```

## Dependencies

- C11 compiler (gcc/clang)
- Standard library only (math.h, stdlib.h, string.h, stdio.h)
