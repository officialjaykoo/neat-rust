# MountainCar

## Conclusion

For the current 100-generation, 3-seed sweep:

```text
node-GRU > plain > Hebbian
```

The margin between node-GRU and plain is small. MountainCar confirms that
memory can help with hidden momentum, but it does not strongly separate the
architectures once policies solve the task.

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
node-GRU     avg 1246.61   best seed 703   1248.05   avg steps 102
plain        avg 1223.47   best seed 702   1248.61   avg steps 102
Hebbian      avg 1178.90   best seed 702   1187.05   avg steps 133
```

## Read

Node-GRU is the best current average, but plain also solves strong seeds. This
means MountainCar is a useful smoke test for memory, not a decisive architecture
test.

Hebbian is weaker here. That makes sense: the task mostly needs smooth momentum
tracking, not fast per-episode association.

## Legacy Note

Older 1000-generation records with removed profiles reached almost the same
solved behavior:

```text
connection-GRU 1248.25
node-GRU/cell  1248.18
memory-gate    1247.02
plain          1184.80
```

The historical lesson still holds: long training makes most explicit memory
variants converge on this task.

## Run

```powershell
cd C:\k_flower_card\neat-rust
cargo run --release --example mountain_car -- 100 0 node-gru 703
```

