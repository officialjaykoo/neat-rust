# neat-rust Porting Plan

## 1. Goal

- Port the core behavior of `neat-python` 2.1 into Rust as faithfully as practical.
- Keep the NEAT core Rust-native, and push Python/JS compatibility to boundary layers.
- Preserve compatibility with the current `k-flower_card` training/eval pipeline.

This document is intentionally short. It tracks the current port status and the next useful work only.

## 2. Current Status

```text
1. Cargo project skeleton        done
2. Config parser                 done
3. Attribute system              done
4. Gene structs                  done
5. Genome configure_new/export   done
6. Activation/aggregation        done
7. FeedForwardNetwork            done
8. RecurrentNetwork              done
9. Genome mutation               done
10. Genome crossover             done
11. Innovation registry          done
12. Species set                  done
13. Stagnation                   done
14. Reproduction                 done
15. Population run loop          done
16. Reporting/statistics         done
17. Checkpoint/resume            done
18. JSON export compatibility    done
19. JS eval bridge               done
20. k-flower train runner        done
21. Smoke training               done
22. Python vs Rust parity        partial
```

Practical state:

- CPU training works.
- Rust checkpoint save/restore works.
- Rust genome JSON is readable by the current JS eval path.
- Winner playoff path works.
- CTRNN / IZNN are implemented.
- Rust/CUDA inference offload exists for NEAT forward batches.
- Full duel simulation is still driven by JS workers.

## 3. Current Public Shape

```text
core
  Rust-native NEAT logic
  config, genome, networks, species, reproduction, population, reporting, GPU evaluators

compat
  neat-python / JS compatibility boundary
  config loading, JSON export, checkpoint adapter, JS bridge helpers

runtime
  project-facing runners
  neat-train-rs, runtime config parsing, worker orchestration, playoff
```

Recent refactor direction:

- public API is being split into `core`, `compat`, and `runtime`
- core config options for fitness/speciation/mutation modes now use typed enums/newtypes
- `Population` now depends on a checkpoint sink trait, not a concrete checkpoint format
- neat-python-style checkpoint IO lives under `compat::neat_python`

## 4. Implemented Scope

Core:

- config and attribute mutation rules
- node/connection genes
- genome creation, mutation, crossover, distance
- feed-forward and recurrent execution
- species, stagnation, reproduction, population loop
- reporting/statistics
- Rust checkpoint format
- CTRNN / IZNN
- GPU CTRNN / IZNN evaluators with CPU fallback and native CUDA path

Compatibility / runtime:

- neat-python-style config loading
- genome JSON export for current JS loaders
- JS eval worker bridge
- `neat-train-rs`
- runtime config `extends`
- winner playoff
- resume from Rust checkpoint

## 5. Remaining Work

High-value remaining work:

1. Add more parity tests from `neat-python`
2. Keep tightening the Rust-native core surface
3. Harden edge cases where parity risk is still higher than average

Concrete parity areas still worth adding:

- XOR / simple_run style end-to-end checks
- more species/reproduction edge cases
- more reporter/checkpoint edge cases
- a few more population-level parity fixtures

Only do this if it becomes necessary:

- full Rust-native k-flower duel evaluator
- full train/playoff migration away from JS workers

## 6. Not Planned

These are intentionally not goals right now:

- Python pickle checkpoint compatibility
- Python RNG bit-for-bit stream matching
- Python dynamic class loading / API 1:1 cloning
- broad API mimicry that does not improve training correctness

## 7. Verification

Core verification:

```powershell
cd neat-rust
.\run-cargo.ps1 check
.\run-cargo.ps1 test
```

Inspect/export verification:

```powershell
cd neat-rust
.\run-cargo.ps1 run --bin neat-rust-inspect -- --config ../scripts/configs/neat_recurrent_memory8.ini
```

Training smoke verification:

```powershell
cd neat-rust
.\run-cargo.ps1 run --bin neat-train-rs -- `
  --config ../scripts/configs/neat_recurrent_memory8.ini `
  --runtime-config ../scripts/configs/runtime_phase1_smoke.json `
  --output-dir ../logs/NEAT/neat_rust_smoke
```

## 8. Important Files

- `scripts/configs/neat_recurrent_memory8.ini`
- `scripts/configs/runtime_phase1.json`
- `scripts/configs/runtime_phase1_dev_i5_12400f.json`
- `scripts/neat_eval_worker.mjs`
- `neat-rust/src/train_runner.rs`

## 9. Rule For This Document

- Keep only current status, scope, remaining work, and verification.
- Do not accumulate long historical logs here.
- Detailed change history belongs in commits and diffs, not in this file.
