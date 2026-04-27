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
mod bootstrap;
mod checkpoint;
mod codec;
mod config;
mod engine;
mod epoch;
#[cfg(feature = "external-bridge")]
mod eval_bridge;
mod evaluator;
mod evolution;
mod export;
mod fitness;
mod gene;
mod genome;
#[cfg(feature = "gpu")]
mod gpu;
mod graph;
mod ids;
mod innovation;
mod io_boundary;
#[cfg(any(feature = "gpu", feature = "policy-bridge"))]
mod native;
#[path = "network/mod.rs"]
mod network_impl;
mod operators;
#[cfg(feature = "policy-bridge")]
mod policy_bridge;
mod population;
mod problem;
mod reporting;
mod reproduction;
mod selection;
mod species;
mod stagnation;
mod statistics;

pub mod prelude;

/// Evolution-domain types: genomes, genes, populations, species, and reporters.
pub mod algorithm {
    pub use crate::attributes::{
        AttributeError, BoolAttribute, ChoiceAttribute, FloatAttribute, RandomSource, XorShiftRng,
    };
    pub use crate::bootstrap::{BootstrapError, BootstrapStrategy, BootstrapSummary};
    pub use crate::engine::{Engine, EvolutionEngine};
    pub use crate::epoch::{Epoch, EpochStopReason, GenerationStats};
    pub use crate::evaluator::{BatchEvaluator, FitnessEvaluator};
    pub use crate::evolution::{
        sync_species_members, PopulationCheckpointError, PopulationCheckpointSink,
        PopulationFitnessSummary, PopulationFitnessSummaryError, SpawnPlan, SpawnPlanEntry,
        SpeciesAssignment,
    };
    pub use crate::fitness::{FitnessScore, FitnessScoreError};
    pub use crate::gene::{
        ConnectionKey, DefaultConnectionGene, DefaultNodeGene, GeneError, NodeKey,
    };
    pub use crate::genome::{
        input_keys, output_keys, DefaultGenome, GenomeError, InnovationConnectionSpec,
    };
    pub use crate::graph::{creates_cycle, feed_forward_layers, required_for_output};
    pub use crate::ids::{GenomeId, SpeciesId};
    pub use crate::innovation::{InnovationTracker, MutationType};
    pub use crate::operators::{
        CrossoverOperator, DefaultCrossoverOperator, DefaultMutationOperator, MutationOperator,
    };
    pub use crate::population::{FitnessError, FitnessResult, Population, PopulationError};
    pub use crate::problem::{
        BatchProblemEvaluator, GenomeProblem, PopulationProblem, ProblemEvaluator,
    };
    pub use crate::reporting::{mean, median2, stdev, Reporter, ReporterSet, StdOutReporter};
    pub use crate::reproduction::{
        adjust_spawn_exact, compute_spawn, compute_spawn_proportional, ReproductionError,
        ReproductionState,
    };
    pub use crate::selection::{
        sorted_members, ParentSelector, SurvivalSelector, TruncationSurvivalSelector,
        UniformParentSelector,
    };
    pub use crate::species::{GenomeDistanceCache, Species, SpeciesSet};
    pub use crate::stagnation::{
        is_better_fitness, species_fitness, worst_fitness, DefaultStagnation, StagnationUpdate,
    };
    pub use crate::statistics::{SpeciesFitnessSnapshot, StatisticsReporter};
}

/// Executable neural network implementations.
pub mod network {
    pub use crate::codec::{
        DecodedNetwork, GenomeCodec, NetworkCodec, NetworkCodecError, NetworkKind,
    };
    pub use crate::network_impl::{
        Ctrnn, CtrnnError, CtrnnNodeEval, FeedForwardError, FeedForwardNetwork, IzNeuron, IzParams,
        Iznn, IznnError, NodeEval, NodeGruMemory, NodeHebbianMemory, RecurrentConnectionEval,
        RecurrentError, RecurrentNetwork, RecurrentNodeEval, RecurrentNodeMemory,
        CHATTERING_PARAMS, FAST_SPIKING_PARAMS, INTRINSICALLY_BURSTING_PARAMS,
        LOW_THRESHOLD_SPIKING_PARAMS, REGULAR_SPIKING_PARAMS, RESONATOR_PARAMS,
        THALAMO_CORTICAL_PARAMS,
    };
}

/// Config, checkpoint, and JSON model export boundaries.
pub mod io {
    pub use crate::checkpoint::{CheckpointError, Checkpointer};
    pub use crate::config::{
        ActivationConfig, AdaptiveMutationConfig, AggregationConfig, BoolAttributeConfig,
        ChoiceAttributeConfig, ChoiceAttributeDefault, CompatibilityExcessCoefficient, Config,
        ConfigChoice, ConfigError, ConnectionGeneConfig, FitnessCriterion, FitnessSharingMode,
        FloatAttributeConfig, FloatInitType, GenomeConfig, InitialConnection,
        InitialConnectionMode, MutationRateCaps, NeatConfig, NodeGruTopology, NodeHebbianRule,
        NodeMemoryKind, Probability, ReproductionConfig, SpawnMethod, SpeciesFitnessFunction,
        SpeciesSetConfig, StagnationConfig, StructuralMutationSurer, TargetNumSpecies,
    };
    pub use crate::export::{export_genome_json, GenomeJsonOptions, NEAT_GENOME_FORMAT};
    pub use crate::io_boundary::{
        export_neat_genome_json, load_neat_config, new_rust_checkpoint_sink, new_rust_checkpointer,
        restore_rust_checkpoint, save_rust_checkpoint, RustCheckpointSink,
    };
}

/// In-process policy execution and GPU-capable batch helpers.
pub mod runtime {
    #[cfg(feature = "gpu")]
    pub use crate::gpu::{
        evaluate_ctrnn_batch_cpu, evaluate_iznn_batch_cpu, native_cuda_available,
        pack_ctrnn_population, pack_iznn_population, GPUCTRNNEvaluator, GPUIZNNEvaluator,
        GpuEvaluatorBackend, GpuEvaluatorError, GpuInputBatch, OutputTrajectory,
        PackedCTRNNPopulation, PackedIZNNPopulation,
    };
    #[cfg(feature = "policy-bridge")]
    pub use crate::policy_bridge::{
        evaluate_policy_batch, native_policy_cuda_available, AutoPolicyEvaluator,
        CompiledPolicyNodeEval, CompiledPolicyNodeMemory, CompiledPolicyRequest,
        CompiledPolicyResult, CompiledPolicySnapshot, CompiledPolicySpec, CpuPolicyEvaluator,
        CudaNativePolicyEvaluator, PolicyActivation, PolicyAggregation, PolicyBatchEvaluator,
        PolicyBridgeBackend, PolicyBridgeError, PolicyIncomingEdge, PolicyIncomingSource,
        PolicyNativeError, PolicyNetworkType, PolicyNodeGruTopology, PolicyNodeHebbianRule,
        PolicyNodeMemoryKind, PolicyRuntimeBackend,
    };
}

/// DTOs and helpers for external evaluator processes.
#[cfg(feature = "external-bridge")]
pub mod bridge {
    pub use crate::eval_bridge::{
        default_external_eval_command, run_external_eval_worker, BridgeEarlyStopConfig,
        BridgeGameCount, BridgeJsonArrayArg, BridgeNativeInferenceBackend, BridgeOpponent,
        BridgeSeat, BridgeStepCount, BridgeTurnPolicy, EvalBridgeError, EvalBridgeOptions,
        EvalBridgeOutput, EvalSeed, ExternalEvalCommand,
    };
}

pub use algorithm::{
    BatchEvaluator, BootstrapStrategy, DefaultGenome, Engine, Epoch, EvolutionEngine, FitnessError,
    FitnessEvaluator, FitnessResult, FitnessScore, FitnessScoreError, GenerationStats, GenomeId,
    GenomeProblem, InnovationConnectionSpec, Population, PopulationError, PopulationProblem,
    ProblemEvaluator, SpeciesId,
};
#[cfg(feature = "external-bridge")]
pub use bridge::{
    default_external_eval_command, run_external_eval_worker, EvalBridgeError, ExternalEvalCommand,
};
pub use io::{
    export_neat_genome_json, load_neat_config, restore_rust_checkpoint, Config, ConfigError,
};
pub use network::{FeedForwardNetwork, RecurrentNetwork};
#[cfg(feature = "policy-bridge")]
pub use runtime::{
    evaluate_policy_batch, AutoPolicyEvaluator, CompiledPolicyRequest, CompiledPolicyResult,
    CompiledPolicySnapshot, CompiledPolicySpec, CpuPolicyEvaluator, CudaNativePolicyEvaluator,
    PolicyActivation, PolicyAggregation, PolicyBatchEvaluator, PolicyBridgeBackend,
    PolicyBridgeError, PolicyIncomingSource, PolicyNativeError, PolicyNetworkType,
    PolicyRuntimeBackend,
};
