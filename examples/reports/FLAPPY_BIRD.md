# Flappy Bird

## Conclusion

For the current 100-generation, 3-seed sweep:

```text
linear-gate > node-GRU > Hebbian ~= plain
```

Linear-gate won this short sweep. Treat that as a timing-smoother win, not as
proof that it is a richer memory model than GRU.

## Current Sweep

Source:

```text
logs/NEAT/sample_memory_sweep_100g_current/summary.txt
```

Setup:

```text
generations: 100
seeds: 701, 702, 703
metric: report fitness
```

Result by average report fitness:

```text
linear-gate  avg 653.73   best seed 701   1063.40   67 pipes
node-GRU     avg 492.71   best seed 702    950.19   60 pipes
Hebbian      avg 460.88   best seed 703    672.41   47 pipes
plain        avg 451.29   best seed 703    631.12   42 pipes
```

## Read

Flappy Bird mostly needs timing and smoothing. A simple linear-gated recurrent
state can be easier for NEAT to discover than a full gate topology.

Node-GRU still produced a strong best seed, so it remains worth testing at
longer generations.

## Legacy Note

Older 500-generation single-seed records had node-GRU/cell-tree as the best
report result:

```text
node-GRU/cell 610.40
connection-GRU 544.31
memory-gate    544.20
plain          462.28
```

That legacy result says richer node memory can work here, but the new 100g
multi-seed result says simple linear gating is easier early.

## Run

```powershell
cd C:\k_flower_card\neat-rust
cargo run --release --example flappy_bird -- 100 0 linear-gate 701
```

