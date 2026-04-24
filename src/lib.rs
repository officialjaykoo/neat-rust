//! `neat-rust` exposes three public layers:
//! - `core`: Rust-native NEAT building blocks
//! - `compat`: neat-python / JS compatibility adapters
//! - `runtime`: application-facing runners built on top of the first two

pub mod activation;
pub mod aggregation;
pub mod attributes;
pub mod checkpoint;
pub mod compat;
pub mod config;
pub mod core;
pub mod eval_bridge;
mod evolution;
pub mod export_json;
pub mod gene;
pub mod genome;
pub mod gpu;
mod gpu_native;
pub mod graph;
pub mod ids;
pub mod innovation;
pub mod network;
pub mod policy_bridge;
mod policy_gpu_native;
pub mod population;
pub mod reporting;
pub mod reproduction;
pub mod runtime;
pub mod species;
pub mod stagnation;
pub mod statistics;
pub mod train_runner;

pub use compat::{
    js::{
        default_node_bin, evaluate_policy_batch, native_policy_cuda_available,
        run_neat_eval_worker, CompiledPolicyNodeEval, CompiledPolicyRequest, CompiledPolicyResult,
        CompiledPolicySnapshot, CompiledPolicySpec, EvalBridgeError, EvalBridgeOptions,
        EvalBridgeOutput, PolicyBridgeBackend, PolicyBridgeError, PolicyIncomingEdge,
    },
    neat_python::{
        export_genome_json, export_neat_python_genome_json, load_neat_python_config,
        new_rust_checkpoint_sink, new_rust_checkpointer, restore_rust_checkpoint,
        save_rust_checkpoint, ActivationConfig, AggregationConfig, BoolAttributeConfig,
        CheckpointError, Checkpointer, CompatibilityExcessCoefficient, Config, ConfigError,
        ConnectionGeneConfig, FitnessCriterion, FitnessSharingMode, FloatAttributeConfig,
        FloatInitType, GenomeConfig, GenomeJsonOptions, InitialConnection, InitialConnectionMode,
        NeatConfig, ReproductionConfig, RustCheckpointSink, SpawnMethod, SpeciesFitnessFunction,
        SpeciesSetConfig, StagnationConfig, StringAttributeConfig, StructuralMutationSurer,
        TargetNumSpecies, NEAT_PYTHON_GENOME_FORMAT,
    },
};
pub use core::{
    attributes::{
        AttributeError, BoolAttribute, FloatAttribute, RandomSource, StringAttribute, XorShiftRng,
    },
    evolution::{
        adjust_spawn_exact, compute_spawn, compute_spawn_proportional, is_better_fitness,
        species_fitness, sync_species_members, worst_fitness, DefaultStagnation, FitnessResult,
        GenomeDistanceCache, Population, PopulationCheckpointError, PopulationCheckpointSink,
        PopulationError, PopulationFitnessSummary, PopulationFitnessSummaryError,
        ReproductionError, ReproductionState, SpawnPlan, SpawnPlanEntry, Species,
        SpeciesAssignment, SpeciesSet, StagnationUpdate,
    },
    genes::{ConnectionKey, DefaultConnectionGene, DefaultNodeGene, GeneError, NodeKey},
    genomes::{
        creates_cycle, feed_forward_layers, input_keys, output_keys, required_for_output,
        DefaultGenome, GenomeError,
    },
    gpu::{
        evaluate_ctrnn_batch_cpu, evaluate_iznn_batch_cpu, native_cuda_available,
        pack_ctrnn_population, pack_iznn_population, GPUCTRNNEvaluator, GPUIZNNEvaluator,
        GpuEvaluatorBackend, GpuEvaluatorError, GpuInputBatch, OutputTrajectory,
        PackedCTRNNPopulation, PackedIZNNPopulation,
    },
    innovation::{InnovationTracker, MutationType},
    math::{
        abs_activation, activate, aggregate, clamped_activation, cube_activation, elu_activation,
        exp_activation, gauss_activation, hat_activation, identity_activation, inv_activation,
        is_valid_activation, is_valid_aggregation, lelu_activation, log_activation,
        max_aggregation, maxabs_aggregation, mean_aggregation, median_aggregation, min_aggregation,
        product_aggregation, relu_activation, selu_activation, sigmoid_activation, sin_activation,
        softplus_activation, square_activation, sum_aggregation, tanh_activation, ActivationError,
        ActivationFunction, AggregationError, AggregationFunction, BUILTIN_ACTIVATIONS,
        BUILTIN_AGGREGATIONS,
    },
    network::{
        Ctrnn, CtrnnError, CtrnnNodeEval, FeedForwardError, FeedForwardNetwork, IzNeuron, IzParams,
        Iznn, IznnError, NodeEval, RecurrentError, RecurrentNetwork, RecurrentNodeEval,
        CHATTERING_PARAMS, FAST_SPIKING_PARAMS, INTRINSICALLY_BURSTING_PARAMS,
        LOW_THRESHOLD_SPIKING_PARAMS, REGULAR_SPIKING_PARAMS, RESONATOR_PARAMS,
        THALAMO_CORTICAL_PARAMS,
    },
    reporting::{
        mean, median2, stdev, Reporter, ReporterSet, SpeciesFitnessSnapshot, StatisticsReporter,
        StdOutReporter,
    },
};
pub use ids::{GenomeId, SpeciesId};
pub use runtime::kflower::{
    run_kflower_training, TrainEvalBackend, TrainRunSummary, TrainRunnerError, TrainRunnerOptions,
    TrainRuntimeConfig,
};
