# neat-rust

Generic Rust NEAT core used by K-Flower training and evaluation.

The crate is intentionally split by public use case rather than by every
internal file:

- `algorithm`: population, genomes, genes, species, reproduction, statistics.
- `network`: feed-forward, recurrent, CTRNN, and Izhikevich execution.
- `io`: typed config loading, checkpoint restore/save, genome export.
- `runtime`: compiled policy inference, CPU batch execution, native CUDA policy bridge.
- `bridge`: DTOs for external evaluator process boundaries.

Prefer importing from `neat_rust::prelude::*` for application code, or from the
named public modules when building tooling.

## Config Boundary

NEAT config is TOML-only. The parser expects `[neat]`, `[genome]`,
`[species_set]`, `[stagnation]`, and `[reproduction]` tables, with nested
attribute tables such as `[genome.activation]` and `[genome.weight]`. After
parsing, the core does not keep stringly typed config: activation, aggregation,
structural mutation policy, species policy, spawn policy, and probability-like
rates are represented by enums/newtypes.

Probability-like values use `Probability`, not raw `f64`. This keeps mutation
rates, replacement rates, survival thresholds, connection/node mutation
probabilities, and interspecies crossover probability in the `0.0..=1.0`
domain at the parser boundary.

## Checkpoints

Checkpoints are serde JSON documents using `neat_rust_checkpoint_v3`. Restore
expects the v3 JSON document and a `config_path`; older text checkpoints are not
part of the supported Rust format.

