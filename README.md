# grand-pattern-c

C Grand Pattern toolkit — standalone cellular graph intelligence in pure C99.

## Building

```bash
make test
```

## Structure

- `include/grand_pattern.h` — Public API header
- `src/vibe.c` — 16-dimensional vibe operations
- `src/jepa.c` — JEPA prediction engine
- `src/murmur.c` — Gossip protocol
- `src/tick.c` — Temporal coordination
- `src/signal.c` — Signal routing
- `src/room.c` — Room cell
- `src/graph.c` — CellGraph composition
- `test/test_all.c` — 30 tests
