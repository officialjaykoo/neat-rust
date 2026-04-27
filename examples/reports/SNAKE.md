# Snake

## Conclusion

For the current 100-generation, 3-seed sweep:

```text
Hebbian > plain > node-GRU
```

Hebbian is the clear current winner. Node-GRU v2 performed poorly in this short
Snake sweep.

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
Hebbian      avg 167.27   best seed 702   231.94   27 apples
plain        avg  85.61   best seed 702   145.90   19 apples
node-GRU     avg  15.12   best seed 702    22.87    4 apples
```

## Read

Snake is the strongest current argument for keeping Hebbian. The fast-weight
memory seems to help short route/planning behavior without making NEAT search
too hard.

Node-GRU may need more generations, different initialization, or stronger
protection from bad gate topologies. At 100 generations it is not competitive.

## Legacy Note

Older 500-generation single-seed records had node-GRU/cell-tree as the best
Snake result:

```text
node-GRU/cell 265.71 report fitness, 29 apples
memory-gate   111.49 report fitness, 14 apples
plain          92.95 report fitness, 12 apples
```

That was the older cell-tree version, so do not compare it directly with the
new v2 sweep. It does show that complex node memory can become useful when the
search finds the right structure.

## Run

```powershell
cd C:\k_flower_card\neat-rust
cargo run --release --example snake_cli -- 100 0 hebbian 702
```

