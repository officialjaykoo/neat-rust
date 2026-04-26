# Blackjack

## Conclusion

For the current 100-generation, 3-seed sweep:

```text
Hebbian > plain > node-GRU ~= linear-gate
```

Hebbian is the only current memory profile that clearly improved the Blackjack
report result. Node-GRU v2 did not show an advantage here at 100 generations.

## Current Sweep

Source:

```text
logs/NEAT/sample_memory_sweep_100g_current/summary.txt
```

Setup:

```text
generations: 100
seeds: 701, 702, 703
report: about 1000 blackjack rounds per winner
metric: EV per round, closer to zero is better
```

Result by average EV:

```text
Hebbian      avg EV -0.0046   best seed 703   EV -0.0014   win rate 43.3%
plain        avg EV -0.0072   best seed 702   EV -0.0024   win rate 43.0%
node-GRU     avg EV -0.0096   best seed 701   EV -0.0096   win rate 37.5%
linear-gate  avg EV -0.0096   best seed 701   EV -0.0096   win rate 37.5%
```

## Read

Hebbian is interesting here because Blackjack is partial-observation and
sequence dependent. The scalar fast weight is simple, but it may be enough to
bias short in-shoe behavior.

Node-GRU v2 needs either more generations or better pressure to use its gates.
At this length it is not beating the baseline.

## Legacy Note

Older 50-generation records included removed profiles:

```text
memory-gate had the best old 50-generation average
connection-GRU was not better by EV
node-GRU/cell-tree had one good seed but high variance
```

Those records are useful as history, but the active comparison is now
`plain / node-gru / hebbian / linear-gate`.

## Run

```powershell
cd C:\k_flower_card\neat-rust
cargo run --release --example blackjack -- 100 0 hebbian 1000 ..\logs\NEAT\blackjack_hebbian_100g_seed703 703
```

