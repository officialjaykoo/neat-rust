//! Rust-native NEAT core surface.
//!
//! This facade keeps the evolutionary/network internals grouped separately from
//! neat-python compatibility and k-flower runtime glue.

pub mod math {
    pub use crate::activation::{
        abs_activation, activate, clamped_activation, cube_activation, elu_activation,
        exp_activation, gauss_activation, hat_activation, identity_activation, inv_activation,
        is_valid_activation, lelu_activation, log_activation, relu_activation, selu_activation,
        sigmoid_activation, sin_activation, softplus_activation, square_activation,
        tanh_activation, ActivationError, ActivationFunction, BUILTIN_ACTIVATIONS,
    };
    pub use crate::aggregation::{
        aggregate, is_valid_aggregation, max_aggregation, maxabs_aggregation, mean_aggregation,
        median_aggregation, min_aggregation, product_aggregation, sum_aggregation,
        AggregationError, AggregationFunction, BUILTIN_AGGREGATIONS,
    };
}

pub mod attributes {
    pub use crate::attributes::{
        AttributeError, BoolAttribute, FloatAttribute, RandomSource, StringAttribute, XorShiftRng,
    };
}

pub mod ids {
    pub use crate::ids::{GenomeId, SpeciesId};
}

pub mod genes {
    pub use crate::gene::{
        ConnectionKey, DefaultConnectionGene, DefaultNodeGene, GeneError, NodeKey,
    };
}

pub mod genomes {
    pub use crate::genome::{input_keys, output_keys, DefaultGenome, GenomeError};
    pub use crate::graph::{creates_cycle, feed_forward_layers, required_for_output};
}

pub mod innovation {
    pub use crate::innovation::{InnovationTracker, MutationType};
}

pub mod evolution {
    pub use crate::evolution::{
        sync_species_members, PopulationCheckpointError, PopulationCheckpointSink,
        PopulationFitnessSummary, PopulationFitnessSummaryError, SpawnPlan, SpawnPlanEntry,
        SpeciesAssignment,
    };
    pub use crate::population::{FitnessResult, Population, PopulationError};
    pub use crate::reproduction::{
        adjust_spawn_exact, compute_spawn, compute_spawn_proportional, ReproductionError,
        ReproductionState,
    };
    pub use crate::species::{GenomeDistanceCache, Species, SpeciesSet};
    pub use crate::stagnation::{
        is_better_fitness, species_fitness, worst_fitness, DefaultStagnation, StagnationUpdate,
    };
}

pub mod network {
    pub use crate::network::{
        Ctrnn, CtrnnError, CtrnnNodeEval, FeedForwardError, FeedForwardNetwork, IzNeuron, IzParams,
        Iznn, IznnError, NodeEval, RecurrentError, RecurrentNetwork, RecurrentNodeEval,
        CHATTERING_PARAMS, FAST_SPIKING_PARAMS, INTRINSICALLY_BURSTING_PARAMS,
        LOW_THRESHOLD_SPIKING_PARAMS, REGULAR_SPIKING_PARAMS, RESONATOR_PARAMS,
        THALAMO_CORTICAL_PARAMS,
    };
}

pub mod reporting {
    pub use crate::reporting::{mean, median2, stdev, Reporter, ReporterSet, StdOutReporter};
    pub use crate::statistics::{SpeciesFitnessSnapshot, StatisticsReporter};
}

pub mod gpu {
    pub use crate::gpu::{
        evaluate_ctrnn_batch_cpu, evaluate_iznn_batch_cpu, native_cuda_available,
        pack_ctrnn_population, pack_iznn_population, GPUCTRNNEvaluator, GPUIZNNEvaluator,
        GpuEvaluatorBackend, GpuEvaluatorError, GpuInputBatch, OutputTrajectory,
        PackedCTRNNPopulation, PackedIZNNPopulation,
    };
}
