# Sample Game Reports

This folder keeps the result notes for the `neat-rust/examples` sample games.
The notes are written for quick human reading: conclusion first, then the few
numbers that explain it.

## Current Read

The newest comparable sweep is:

```text
logs/NEAT/sample_memory_sweep_100g_current/
```

It ran:

```text
games: blackjack, mountain_car, flappy_bird, snake_cli
profiles: plain, node-gru, hebbian, linear-gate
seeds: 701, 702, 703
generations: 100
parallelism: 6
```

## One-Screen Summary

Blackjack:

```text
winner: Hebbian
reason: best average EV and best seed EV
read: Hebbian is worth a longer partial-observation test
```

MountainCar:

```text
winner: node-GRU by a small margin
reason: best average report fitness among current profiles
read: explicit memory helps, but the task saturates quickly
```

Flappy Bird:

```text
winner: linear-gate in the 100-generation sweep
reason: best average and best single seed
read: likely a simple timing smoother, not proof that it is a better memory model
```

Snake:

```text
winner: Hebbian
reason: best average report fitness and best apple count
read: fast-weight style memory is promising for short path/planning behavior
```

Lunar Lander:

```text
winner so far: legacy node-GRU / cell-tree
reason: best 1000-generation single-seed fitness
read: needs a new node_memory_kind sweep before comparing with v2
```

## Reports

- [Blackjack](BLACKJACK.md)
- [MountainCar](MOUNTAIN_CAR.md)
- [Flappy Bird](FLAPPY_BIRD.md)
- [Snake](SNAKE.md)
- [Lunar Lander](LUNAR_LANDER.md)

