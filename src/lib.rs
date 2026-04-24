//! Generic NEAT engine.
//!
//! The public API is intentionally grouped by use case:
//! - [`algorithm`] for genomes, populations, species, and evolution.
//! - [`network`] for executable neural networks.
//! - [`io`] for config, checkpoint, and model export.
//! - [`runtime`] for compiled policy inference and batch/GPU helpers.
//! - [`bridge`] for process-boundary DTOs used by external evaluators.
//!
//! Internal implementation modules stay private so the crate reads like a
//! library surface instead of a directory dump.

mod activation;
mod aggregation;
mod attributes;
mod checkpoint;
mod config;
mod eval_bridge;
mod evolution;
mod export;
mod gene;
mod genome;
mod gpu;
mod graph;
mod ids;
mod innovation;
mod io_boundary;
mod native;
#[path = "network/mod.rs"]
mod network_impl;
mod policy_bridge;
mod population;
mod reporting;
mod reproduction;
mod species;
mod stagnation;
mod statistics;

pub mod prelude;

/// Evolution-domain types: genomes, genes, populations, species, and reporters.
pub mod algorithm {
    pub use crate::attributes::{
        AttributeError, BoolAttribute, ChoiceAttribute, FloatAttribute, RandomSource, XorShiftRng,
    };
    pub use crate::evolution::{
        sync_species_members, PopulationCheckpointError, PopulationCheckpointSink,
        PopulationFitnessSummary, PopulationFitnessSummaryError, SpawnPlan, SpawnPlanEntry,
        SpeciesAssignment,
    };
    pub use crate::gene::{
        ConnectionKey, DefaultConnectionGene, DefaultNodeGene, GeneError, NodeKey,
    };
    pub use crate::genome::{input_keys, output_keys, DefaultGenome, GenomeError};
    pub use crate::graph::{creates_cycle, feed_forward_layers, required_for_output};
    pub use crate::ids::{GenomeId, SpeciesId};
    pub use crate::innovation::{InnovationTracker, MutationType};
    pub use crate::population::{FitnessError, FitnessResult, Population, PopulationError};
    pub use crate::reporting::{mean, median2, stdev, Reporter, ReporterSet, StdOutReporter};
    pub use crate::reproduction::{
        adjust_spawn_exact, compute_spawn, compute_spawn_proportional, ReproductionError,
        ReproductionState,
    };
    pub use crate::species::{GenomeDistanceCache, Species, SpeciesSet};
    pub use crate::stagnation::{
        is_better_fitness, species_fitness, worst_fitness, DefaultStagnation, StagnationUpdate,
    };
    pub use crate::statistics::{SpeciesFitnessSnapshot, StatisticsReporter};
}

/// Executable neural network implementations.
pub mod network {
    pub use crate::network_impl::{
        Ctrnn, CtrnnError, CtrnnNodeEval, FeedForwardError, FeedForwardNetwork, IzNeuron, IzParams,
        Iznn, IznnError, NodeEval, RecurrentError, RecurrentNetwork, RecurrentNodeEval,
        CHATTERING_PARAMS, FAST_SPIKING_PARAMS, INTRINSICALLY_BURSTING_PARAMS,
        LOW_THRESHOLD_SPIKING_PARAMS, REGULAR_SPIKING_PARAMS, RESONATOR_PARAMS,
        THALAMO_CORTICAL_PARAMS,
    };
}

/// Config, checkpoint, and JSON model export boundaries.
pub mod io {
    pub use crate::checkpoint::{CheckpointError, Checkpointer};
    pub use crate::config::{
        ActivationConfig, AggregationConfig, BoolAttributeConfig, ChoiceAttributeConfig,
        ChoiceAttributeDefault, CompatibilityExcessCoefficient, Config, ConfigChoice, ConfigError,
        ConnectionGeneConfig, FitnessCriterion, FitnessSharingMode, FloatAttributeConfig,
        FloatInitType, GenomeConfig, InitialConnection, InitialConnectionMode, NeatConfig,
        Probability, ReproductionConfig, SpawnMethod, SpeciesFitnessFunction, SpeciesSetConfig,
        StagnationConfig, StructuralMutationSurer, TargetNumSpecies,
    };
    pub use crate::export::{export_genome_json, GenomeJsonOptions, NEAT_GENOME_FORMAT};
    pub use crate::io_boundary::{
        export_neat_genome_json, load_neat_config, new_rust_checkpoint_sink, new_rust_checkpointer,
        restore_rust_checkpoint, save_rust_checkpoint, RustCheckpointSink,
    };
}

/// In-process policy execution and GPU-capable batch helpers.
pub mod runtime {
    pub use crate::gpu::{
        evaluate_ctrnn_batch_cpu, evaluate_iznn_batch_cpu, native_cuda_available,
        pack_ctrnn_population, pack_iznn_population, GPUCTRNNEvaluator, GPUIZNNEvaluator,
        GpuEvaluatorBackend, GpuEvaluatorError, GpuInputBatch, OutputTrajectory,
        PackedCTRNNPopulation, PackedIZNNPopulation,
    };
    pub use crate::policy_bridge::{
        evaluate_policy_batch, native_policy_cuda_available, AutoPolicyEvaluator,
        CompiledPolicyNodeEval, CompiledPolicyRequest, CompiledPolicyResult,
        CompiledPolicySnapshot, CompiledPolicySpec, CpuPolicyEvaluator, CudaNativePolicyEvaluator,
        PolicyActivation, PolicyAggregation, PolicyBatchEvaluator, PolicyBridgeBackend,
        PolicyBridgeError, PolicyIncomingEdge, PolicyIncomingSource, PolicyNativeError,
        PolicyNetworkType, PolicyRuntimeBackend,
    };
}

/// DTOs and helpers for external evaluator processes.
pub mod bridge {
    pub use crate::eval_bridge::{
        default_external_eval_command, run_external_eval_worker, BridgeEarlyStopConfig,
        BridgeGameCount, BridgeJsonArrayArg, BridgeNativeInferenceBackend, BridgeOpponent,
        BridgeSeat, BridgeStepCount, BridgeTurnPolicy, EvalBridgeError, EvalBridgeOptions,
        EvalBridgeOutput, EvalSeed, ExternalEvalCommand,
    };
}

pub use algorithm::{
    DefaultGenome, FitnessError, FitnessResult, GenomeId, Population, PopulationError, SpeciesId,
};
pub use bridge::{
    default_external_eval_command, run_external_eval_worker, EvalBridgeError, ExternalEvalCommand,
};
pub use io::{
    export_neat_genome_json, load_neat_config, restore_rust_checkpoint, Config, ConfigError,
};
pub use network::{FeedForwardNetwork, RecurrentNetwork};
pub use runtime::{
    evaluate_policy_batch, AutoPolicyEvaluator, CompiledPolicyRequest, CompiledPolicyResult,
    CompiledPolicySnapshot, CompiledPolicySpec, CpuPolicyEvaluator, CudaNativePolicyEvaluator,
    PolicyActivation, PolicyAggregation, PolicyBatchEvaluator, PolicyBridgeBackend,
    PolicyBridgeError, PolicyIncomingSource, PolicyNativeError, PolicyNetworkType,
    PolicyRuntimeBackend,
};
