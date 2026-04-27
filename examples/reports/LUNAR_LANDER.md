# Lunar Lander

## Conclusion

The latest Lunar Lander record is still from the older memory comparison:

```text
node-GRU / cell-tree > connection-GRU > plain bootstrap > memory-gate > plain
```

There is no new `node_memory_kind` sweep for Lunar Lander yet. The old result
are tested under the new system.

## Legacy 1000-Generation Record

Setup:

```text
generations: 1000
seed: fixed single seed
metric: final fitness
```

Best records:

```text
node-GRU/cell   600.72   genome 55909
connection-GRU  529.71   genome 58030
plain bootstrap 391.98   genome 74
memory-gate     354.54   genome 43495
plain           252.63   genome 19189
```

## Read

Lunar Lander rewards richer recurrent state more than the simpler arcade
samples. The old node-GRU/cell result beat connection-GRU by about 13.4%.

But this result predates the current cleanup. The next useful Lunar test is:

```text
plain / node-GRU v2 / Hebbian
1000 generations each
same seed set if practical
```

## Run

Current example profiles are `plain` and `node-gru`:

```powershell
cd C:\k_flower_card\neat-rust
cargo run --release --example lunar_lander -- 1000 0 node-gru
```

