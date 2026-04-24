use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::activation::ActivationFunction;
use crate::aggregation::AggregationFunction;
use crate::config::{Config, GenomeConfig};
use crate::gene::{ConnectionKey, DefaultConnectionGene, NodeKey};
use crate::genome::{input_keys, output_keys, DefaultGenome};
use crate::graph::required_for_output;
use crate::ids::GenomeId;
use crate::native::gpu::{
    ctrnn_native_supported, evaluate_ctrnn_batch_native, evaluate_iznn_batch_native,
    iznn_native_supported,
};

pub type OutputTrajectory = Vec<Vec<f64>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuEvaluatorBackend {
    Auto,
    CpuFallback,
    NativeRequired,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuInputBatch {
    Shared(Vec<f64>),
    PerGenome(Vec<Vec<f64>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedCTRNNPopulation {
    pub genome_keys: Vec<GenomeId>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub max_nodes: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub response: Vec<f64>,
    pub tau: Vec<f64>,
    pub activation: Vec<ActivationFunction>,
    pub node_mask: Vec<bool>,
    pub node_key_maps: Vec<BTreeMap<NodeKey, usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedIZNNPopulation {
    pub genome_keys: Vec<GenomeId>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub max_nodes: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
    pub node_mask: Vec<bool>,
    pub node_key_maps: Vec<BTreeMap<NodeKey, usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuEvaluatorError {
    NativeBackendUnavailable,
    NativeUnsupported(String),
    NativeDriver(String),
    InvalidTimeConfig,
    EmptyPopulation,
    MissingNodeGene {
        genome_key: GenomeId,
        node_key: NodeKey,
    },
    UnsupportedActivation {
        genome_key: GenomeId,
        node_key: NodeKey,
        name: String,
    },
    UnsupportedAggregation {
        genome_key: GenomeId,
        node_key: NodeKey,
        name: String,
    },
    InputCountMismatch {
        expected: usize,
        actual: usize,
    },
    InputBatchSizeMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidNodeParameter {
        genome_key: GenomeId,
        node_key: NodeKey,
        name: &'static str,
        value: f64,
    },
}

pub struct GPUCTRNNEvaluator<I, F> {
    pub dt: f64,
    pub t_max: f64,
    pub input_fn: I,
    pub fitness_fn: F,
    pub backend: GpuEvaluatorBackend,
}

pub struct GPUIZNNEvaluator<I, F> {
    pub dt: f64,
    pub t_max: f64,
    pub input_fn: I,
    pub fitness_fn: F,
    pub backend: GpuEvaluatorBackend,
}

impl GpuInputBatch {
    pub fn shared(values: Vec<f64>) -> Self {
        Self::Shared(values)
    }

    pub fn per_genome(values: Vec<Vec<f64>>) -> Self {
        Self::PerGenome(values)
    }

    pub(crate) fn expand(
        self,
        population_size: usize,
        num_inputs: usize,
    ) -> Result<Vec<Vec<f64>>, GpuEvaluatorError> {
        match self {
            Self::Shared(values) => {
                validate_input_count(values.len(), num_inputs)?;
                Ok(vec![values; population_size])
            }
            Self::PerGenome(values) => {
                if values.len() != population_size {
                    return Err(GpuEvaluatorError::InputBatchSizeMismatch {
                        expected: population_size,
                        actual: values.len(),
                    });
                }
                for row in &values {
                    validate_input_count(row.len(), num_inputs)?;
                }
                Ok(values)
            }
        }
    }
}

impl<I, F> GPUCTRNNEvaluator<I, F>
where
    I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    F: FnMut(&OutputTrajectory) -> f64,
{
    pub fn new(dt: f64, t_max: f64, input_fn: I, fitness_fn: F) -> Self {
        Self {
            dt,
            t_max,
            input_fn,
            fitness_fn,
            backend: GpuEvaluatorBackend::Auto,
        }
    }

    pub fn with_backend(mut self, backend: GpuEvaluatorBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn evaluate(
        &mut self,
        genomes: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> Result<(), GpuEvaluatorError> {
        validate_time_config(self.dt, self.t_max)?;

        let packed = pack_ctrnn_population(genomes, &config.genome)?;
        let trajectories = match select_ctrnn_backend(self.backend, &packed)? {
            NativeExecution::Cpu => {
                evaluate_ctrnn_batch_cpu(&packed, self.dt, self.t_max, &mut self.input_fn)?
            }
            NativeExecution::Cuda => {
                match evaluate_ctrnn_batch_native(&packed, self.dt, self.t_max, &mut self.input_fn)
                {
                    Ok(trajectories) => trajectories,
                    Err(err) if can_auto_fallback(self.backend, &err) => {
                        evaluate_ctrnn_batch_cpu(&packed, self.dt, self.t_max, &mut self.input_fn)?
                    }
                    Err(err) => return Err(err),
                }
            }
        };
        for (idx, genome_key) in packed.genome_keys.iter().enumerate() {
            if let Some(genome) = genomes.get_mut(genome_key) {
                genome.fitness = Some((self.fitness_fn)(&trajectories[idx]));
            }
        }
        Ok(())
    }
}

impl<I, F> GPUIZNNEvaluator<I, F>
where
    I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
    F: FnMut(&OutputTrajectory) -> f64,
{
    pub fn new(dt: f64, t_max: f64, input_fn: I, fitness_fn: F) -> Self {
        Self {
            dt,
            t_max,
            input_fn,
            fitness_fn,
            backend: GpuEvaluatorBackend::Auto,
        }
    }

    pub fn with_backend(mut self, backend: GpuEvaluatorBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn evaluate(
        &mut self,
        genomes: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> Result<(), GpuEvaluatorError> {
        validate_time_config(self.dt, self.t_max)?;

        let packed = pack_iznn_population(genomes, &config.genome)?;
        let trajectories = match select_iznn_backend(self.backend, &packed)? {
            NativeExecution::Cpu => {
                evaluate_iznn_batch_cpu(&packed, self.dt, self.t_max, &mut self.input_fn)?
            }
            NativeExecution::Cuda => {
                match evaluate_iznn_batch_native(&packed, self.dt, self.t_max, &mut self.input_fn) {
                    Ok(trajectories) => trajectories,
                    Err(err) if can_auto_fallback(self.backend, &err) => {
                        evaluate_iznn_batch_cpu(&packed, self.dt, self.t_max, &mut self.input_fn)?
                    }
                    Err(err) => return Err(err),
                }
            }
        };
        for (idx, genome_key) in packed.genome_keys.iter().enumerate() {
            if let Some(genome) = genomes.get_mut(genome_key) {
                genome.fitness = Some((self.fitness_fn)(&trajectories[idx]));
            }
        }
        Ok(())
    }
}

pub fn pack_ctrnn_population(
    genomes: &BTreeMap<GenomeId, DefaultGenome>,
    config: &GenomeConfig,
) -> Result<PackedCTRNNPopulation, GpuEvaluatorError> {
    if genomes.is_empty() {
        return Err(GpuEvaluatorError::EmptyPopulation);
    }

    let config_inputs = input_keys(config);
    let config_outputs = output_keys(config);
    let num_inputs = config_inputs.len();
    let num_outputs = config_outputs.len();
    let per_genome = collect_packing_info(genomes, config)?;
    let max_nodes = per_genome
        .iter()
        .map(|info| info.num_nodes)
        .max()
        .unwrap_or(num_inputs + num_outputs);

    let mut packed = PackedCTRNNPopulation {
        genome_keys: per_genome.iter().map(|info| info.genome_key).collect(),
        num_inputs,
        num_outputs,
        max_nodes,
        weights: vec![0.0; genomes.len() * max_nodes * max_nodes],
        bias: vec![0.0; genomes.len() * max_nodes],
        response: vec![1.0; genomes.len() * max_nodes],
        tau: vec![1.0; genomes.len() * max_nodes],
        activation: vec![ActivationFunction::Sigmoid; genomes.len() * max_nodes],
        node_mask: vec![false; genomes.len() * max_nodes],
        node_key_maps: per_genome.iter().map(|info| info.key_map.clone()).collect(),
    };

    for genome_idx in 0..genomes.len() {
        for idx in 0..num_inputs + num_outputs {
            packed.node_mask[node_idx(genome_idx, idx, max_nodes)] = true;
        }
    }

    for (genome_idx, info) in per_genome.iter().enumerate() {
        for node_key in &info.required {
            let dense_idx = info.key_map[node_key];
            if dense_idx >= num_inputs {
                packed.node_mask[node_idx(genome_idx, dense_idx, max_nodes)] = true;
            }
        }

        for node_key in &info.required {
            let dense_idx = info.key_map[node_key];
            let node =
                info.genome
                    .nodes
                    .get(node_key)
                    .ok_or(GpuEvaluatorError::MissingNodeGene {
                        genome_key: info.genome_key,
                        node_key: *node_key,
                    })?;
            let idx = node_idx(genome_idx, dense_idx, max_nodes);
            packed.bias[idx] = node.bias;
            packed.response[idx] = node.response;
            packed.tau[idx] = node.time_constant;
            packed.activation[idx] =
                supported_gpu_activation(node.activation).ok_or_else(|| {
                    GpuEvaluatorError::UnsupportedActivation {
                        genome_key: info.genome_key,
                        node_key: *node_key,
                        name: node.activation.to_string(),
                    }
                })?;

            if node.aggregation != AggregationFunction::Sum {
                return Err(GpuEvaluatorError::UnsupportedAggregation {
                    genome_key: info.genome_key,
                    node_key: *node_key,
                    name: node.aggregation.to_string(),
                });
            }
        }

        fill_weights(
            &mut packed.weights,
            genome_idx,
            max_nodes,
            &info.key_map,
            &info.required,
            &info.genome.connections,
        );
    }

    Ok(packed)
}

pub fn pack_iznn_population(
    genomes: &BTreeMap<GenomeId, DefaultGenome>,
    config: &GenomeConfig,
) -> Result<PackedIZNNPopulation, GpuEvaluatorError> {
    if genomes.is_empty() {
        return Err(GpuEvaluatorError::EmptyPopulation);
    }

    let config_inputs = input_keys(config);
    let config_outputs = output_keys(config);
    let num_inputs = config_inputs.len();
    let num_outputs = config_outputs.len();
    let per_genome = collect_packing_info(genomes, config)?;
    let max_nodes = per_genome
        .iter()
        .map(|info| info.num_nodes)
        .max()
        .unwrap_or(num_inputs + num_outputs);

    let mut packed = PackedIZNNPopulation {
        genome_keys: per_genome.iter().map(|info| info.genome_key).collect(),
        num_inputs,
        num_outputs,
        max_nodes,
        weights: vec![0.0; genomes.len() * max_nodes * max_nodes],
        bias: vec![0.0; genomes.len() * max_nodes],
        a: vec![0.0; genomes.len() * max_nodes],
        b: vec![0.0; genomes.len() * max_nodes],
        c: vec![-65.0; genomes.len() * max_nodes],
        d: vec![0.0; genomes.len() * max_nodes],
        node_mask: vec![false; genomes.len() * max_nodes],
        node_key_maps: per_genome.iter().map(|info| info.key_map.clone()).collect(),
    };

    for genome_idx in 0..genomes.len() {
        for idx in 0..num_inputs + num_outputs {
            packed.node_mask[node_idx(genome_idx, idx, max_nodes)] = true;
        }
    }

    for (genome_idx, info) in per_genome.iter().enumerate() {
        for node_key in &info.required {
            let dense_idx = info.key_map[node_key];
            if dense_idx >= num_inputs {
                packed.node_mask[node_idx(genome_idx, dense_idx, max_nodes)] = true;
            }
        }

        for node_key in &info.required {
            let dense_idx = info.key_map[node_key];
            let node =
                info.genome
                    .nodes
                    .get(node_key)
                    .ok_or(GpuEvaluatorError::MissingNodeGene {
                        genome_key: info.genome_key,
                        node_key: *node_key,
                    })?;
            let idx = node_idx(genome_idx, dense_idx, max_nodes);
            packed.bias[idx] = node.bias;
            packed.a[idx] = node.iz_a;
            packed.b[idx] = node.iz_b;
            packed.c[idx] = node.iz_c;
            packed.d[idx] = node.iz_d;
        }

        fill_weights(
            &mut packed.weights,
            genome_idx,
            max_nodes,
            &info.key_map,
            &info.required,
            &info.genome.connections,
        );
    }

    Ok(packed)
}

pub fn evaluate_ctrnn_batch_cpu<I>(
    packed: &PackedCTRNNPopulation,
    dt: f64,
    t_max: f64,
    input_fn: &mut I,
) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
where
    I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
{
    validate_time_config(dt, t_max)?;
    let population_size = packed.genome_keys.len();
    let num_steps = (t_max / dt) as usize;
    let mut state = vec![0.0; population_size * packed.max_nodes];
    let mut next_state = vec![0.0; population_size * packed.max_nodes];
    let mut trajectories = vec![vec![vec![0.0; packed.num_outputs]; num_steps]; population_size];

    for step in 0..num_steps {
        let time = step as f64 * dt;
        let inputs = input_fn(time, dt, population_size, packed.num_inputs)
            .expand(population_size, packed.num_inputs)?;

        for (genome_idx, row) in inputs.iter().enumerate() {
            for (input_idx, value) in row.iter().enumerate() {
                state[node_idx(genome_idx, input_idx, packed.max_nodes)] = *value;
            }
        }

        for genome_idx in 0..population_size {
            for dst_idx in 0..packed.max_nodes {
                let flat_idx = node_idx(genome_idx, dst_idx, packed.max_nodes);
                if !packed.node_mask[flat_idx] {
                    next_state[flat_idx] = 0.0;
                    continue;
                }

                if dst_idx < packed.num_inputs {
                    next_state[flat_idx] = inputs[genome_idx][dst_idx];
                    continue;
                }

                let tau = packed.tau[flat_idx];
                if !tau.is_finite() || tau <= 0.0 {
                    let node_key = dense_key(&packed.node_key_maps[genome_idx], dst_idx);
                    return Err(GpuEvaluatorError::InvalidNodeParameter {
                        genome_key: packed.genome_keys[genome_idx],
                        node_key,
                        name: "time_constant",
                        value: tau,
                    });
                }

                let mut weighted_sum = 0.0;
                for src_idx in 0..packed.max_nodes {
                    weighted_sum += packed.weights
                        [weight_idx(genome_idx, dst_idx, src_idx, packed.max_nodes)]
                        * state[node_idx(genome_idx, src_idx, packed.max_nodes)];
                }
                let activation_input =
                    packed.bias[flat_idx] + packed.response[flat_idx] * weighted_sum;
                let z = packed.activation[flat_idx].apply(activation_input);
                let decay = (-dt / tau).exp();
                next_state[flat_idx] = decay * state[flat_idx] + (1.0 - decay) * z;
            }

            for input_idx in 0..packed.num_inputs {
                next_state[node_idx(genome_idx, input_idx, packed.max_nodes)] =
                    inputs[genome_idx][input_idx];
            }
        }

        std::mem::swap(&mut state, &mut next_state);

        for genome_idx in 0..population_size {
            for output_idx in 0..packed.num_outputs {
                trajectories[genome_idx][step][output_idx] =
                    state[node_idx(genome_idx, packed.num_inputs + output_idx, packed.max_nodes)];
            }
        }
    }

    Ok(trajectories)
}

pub fn evaluate_iznn_batch_cpu<I>(
    packed: &PackedIZNNPopulation,
    dt: f64,
    t_max: f64,
    input_fn: &mut I,
) -> Result<Vec<OutputTrajectory>, GpuEvaluatorError>
where
    I: FnMut(f64, f64, usize, usize) -> GpuInputBatch,
{
    validate_time_config(dt, t_max)?;
    let population_size = packed.genome_keys.len();
    let num_steps = (t_max / dt) as usize;
    let mut v = packed.c.clone();
    let mut u: Vec<f64> = packed.b.iter().zip(v.iter()).map(|(b, v)| b * v).collect();
    let mut fired = vec![0.0; population_size * packed.max_nodes];
    let mut source = vec![0.0; population_size * packed.max_nodes];
    let mut trajectories = vec![vec![vec![0.0; packed.num_outputs]; num_steps]; population_size];

    for step in 0..num_steps {
        let time = step as f64 * dt;
        let inputs = input_fn(time, dt, population_size, packed.num_inputs)
            .expand(population_size, packed.num_inputs)?;

        source.copy_from_slice(&fired);
        for (genome_idx, row) in inputs.iter().enumerate() {
            for (input_idx, value) in row.iter().enumerate() {
                source[node_idx(genome_idx, input_idx, packed.max_nodes)] = *value;
            }
        }

        for genome_idx in 0..population_size {
            for dst_idx in packed.num_inputs..packed.max_nodes {
                let flat_idx = node_idx(genome_idx, dst_idx, packed.max_nodes);
                if !packed.node_mask[flat_idx] {
                    fired[flat_idx] = 0.0;
                    continue;
                }

                let mut current = packed.bias[flat_idx];
                for src_idx in 0..packed.max_nodes {
                    current += packed.weights
                        [weight_idx(genome_idx, dst_idx, src_idx, packed.max_nodes)]
                        * source[node_idx(genome_idx, src_idx, packed.max_nodes)];
                }

                v[flat_idx] += 0.5
                    * dt
                    * (0.04 * v[flat_idx] * v[flat_idx] + 5.0 * v[flat_idx] + 140.0 - u[flat_idx]
                        + current);
                v[flat_idx] += 0.5
                    * dt
                    * (0.04 * v[flat_idx] * v[flat_idx] + 5.0 * v[flat_idx] + 140.0 - u[flat_idx]
                        + current);
                u[flat_idx] +=
                    dt * packed.a[flat_idx] * (packed.b[flat_idx] * v[flat_idx] - u[flat_idx]);

                fired[flat_idx] = 0.0;
                if !v[flat_idx].is_finite() || !u[flat_idx].is_finite() {
                    v[flat_idx] = packed.c[flat_idx];
                    u[flat_idx] = packed.b[flat_idx] * v[flat_idx];
                    continue;
                }
                if v[flat_idx] > 30.0 {
                    fired[flat_idx] = 1.0;
                    v[flat_idx] = packed.c[flat_idx];
                    u[flat_idx] += packed.d[flat_idx];
                }
            }
        }

        for genome_idx in 0..population_size {
            for output_idx in 0..packed.num_outputs {
                trajectories[genome_idx][step][output_idx] =
                    fired[node_idx(genome_idx, packed.num_inputs + output_idx, packed.max_nodes)];
            }
        }
    }

    Ok(trajectories)
}

pub fn native_cuda_available() -> bool {
    crate::native::gpu::native_cuda_available()
}

struct PackingInfo<'a> {
    genome_key: GenomeId,
    genome: &'a DefaultGenome,
    required: BTreeSet<NodeKey>,
    key_map: BTreeMap<NodeKey, usize>,
    num_nodes: usize,
}

fn collect_packing_info<'a>(
    genomes: &'a BTreeMap<GenomeId, DefaultGenome>,
    config: &GenomeConfig,
) -> Result<Vec<PackingInfo<'a>>, GpuEvaluatorError> {
    let config_inputs = input_keys(config);
    let config_outputs = output_keys(config);
    let mut result = Vec::new();
    for (genome_key, genome) in genomes {
        let connections: Vec<_> = genome.connections.keys().copied().collect();
        let required = required_for_output(&config_inputs, &config_outputs, &connections);
        let (key_map, num_nodes) = build_node_key_map(config, &required);
        result.push(PackingInfo {
            genome_key: *genome_key,
            genome,
            required,
            key_map,
            num_nodes,
        });
    }
    Ok(result)
}

fn build_node_key_map(
    config: &GenomeConfig,
    required_nodes: &BTreeSet<NodeKey>,
) -> (BTreeMap<NodeKey, usize>, usize) {
    let mut key_map = BTreeMap::new();
    let config_inputs = input_keys(config);
    let config_outputs = output_keys(config);
    for (idx, key) in config_inputs.iter().enumerate() {
        key_map.insert(*key, idx);
    }
    let hidden_start = config_inputs.len() + config_outputs.len();
    for (idx, key) in config_outputs.iter().enumerate() {
        key_map.insert(*key, config_inputs.len() + idx);
    }
    let mut hidden_idx = 0;
    for key in required_nodes {
        if !key_map.contains_key(key) {
            key_map.insert(*key, hidden_start + hidden_idx);
            hidden_idx += 1;
        }
    }
    (key_map, hidden_start + hidden_idx)
}

fn fill_weights(
    weights: &mut [f64],
    genome_idx: usize,
    max_nodes: usize,
    key_map: &BTreeMap<NodeKey, usize>,
    required: &BTreeSet<NodeKey>,
    connections: &BTreeMap<ConnectionKey, DefaultConnectionGene>,
) {
    for connection in connections.values() {
        if !connection.enabled {
            continue;
        }
        let src_key = connection.key.input;
        let dst_key = connection.key.output;
        if !required.contains(&dst_key) {
            continue;
        }
        let (Some(src_idx), Some(dst_idx)) = (key_map.get(&src_key), key_map.get(&dst_key)) else {
            continue;
        };
        weights[weight_idx(genome_idx, *dst_idx, *src_idx, max_nodes)] = connection.weight;
    }
}

fn supported_gpu_activation(value: ActivationFunction) -> Option<ActivationFunction> {
    match value {
        ActivationFunction::Sigmoid
        | ActivationFunction::Tanh
        | ActivationFunction::Relu
        | ActivationFunction::Identity
        | ActivationFunction::Clamped
        | ActivationFunction::Elu
        | ActivationFunction::Softplus
        | ActivationFunction::Sin
        | ActivationFunction::Gauss
        | ActivationFunction::Abs
        | ActivationFunction::Square => Some(value),
        _ => None,
    }
}

fn validate_time_config(dt: f64, t_max: f64) -> Result<(), GpuEvaluatorError> {
    if dt.is_finite() && dt > 0.0 && t_max.is_finite() && t_max >= 0.0 {
        Ok(())
    } else {
        Err(GpuEvaluatorError::InvalidTimeConfig)
    }
}

fn validate_input_count(actual: usize, expected: usize) -> Result<(), GpuEvaluatorError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GpuEvaluatorError::InputCountMismatch { expected, actual })
    }
}

fn ensure_backend(backend: GpuEvaluatorBackend) -> Result<(), GpuEvaluatorError> {
    match backend {
        GpuEvaluatorBackend::Auto | GpuEvaluatorBackend::CpuFallback => Ok(()),
        GpuEvaluatorBackend::NativeRequired => {
            if native_cuda_available() {
                Ok(())
            } else {
                Err(GpuEvaluatorError::NativeBackendUnavailable)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeExecution {
    Cpu,
    Cuda,
}

fn select_ctrnn_backend(
    backend: GpuEvaluatorBackend,
    packed: &PackedCTRNNPopulation,
) -> Result<NativeExecution, GpuEvaluatorError> {
    match backend {
        GpuEvaluatorBackend::CpuFallback => Ok(NativeExecution::Cpu),
        GpuEvaluatorBackend::Auto => {
            if native_cuda_available() && ctrnn_native_supported(packed).is_ok() {
                Ok(NativeExecution::Cuda)
            } else {
                Ok(NativeExecution::Cpu)
            }
        }
        GpuEvaluatorBackend::NativeRequired => {
            ensure_backend(backend)?;
            ctrnn_native_supported(packed)?;
            Ok(NativeExecution::Cuda)
        }
    }
}

fn select_iznn_backend(
    backend: GpuEvaluatorBackend,
    packed: &PackedIZNNPopulation,
) -> Result<NativeExecution, GpuEvaluatorError> {
    match backend {
        GpuEvaluatorBackend::CpuFallback => Ok(NativeExecution::Cpu),
        GpuEvaluatorBackend::Auto => {
            if native_cuda_available() && iznn_native_supported(packed).is_ok() {
                Ok(NativeExecution::Cuda)
            } else {
                Ok(NativeExecution::Cpu)
            }
        }
        GpuEvaluatorBackend::NativeRequired => {
            ensure_backend(backend)?;
            iznn_native_supported(packed)?;
            Ok(NativeExecution::Cuda)
        }
    }
}

fn can_auto_fallback(backend: GpuEvaluatorBackend, err: &GpuEvaluatorError) -> bool {
    backend == GpuEvaluatorBackend::Auto
        && matches!(
            err,
            GpuEvaluatorError::NativeBackendUnavailable
                | GpuEvaluatorError::NativeUnsupported(_)
                | GpuEvaluatorError::UnsupportedActivation { .. }
        )
}

fn node_idx(genome_idx: usize, node_idx: usize, max_nodes: usize) -> usize {
    genome_idx * max_nodes + node_idx
}

fn weight_idx(genome_idx: usize, dst_idx: usize, src_idx: usize, max_nodes: usize) -> usize {
    (genome_idx * max_nodes + dst_idx) * max_nodes + src_idx
}

fn dense_key(key_map: &BTreeMap<NodeKey, usize>, dense_idx: usize) -> NodeKey {
    key_map
        .iter()
        .find(|(_, idx)| **idx == dense_idx)
        .map(|(key, _)| *key)
        .unwrap_or(dense_idx as NodeKey)
}

impl fmt::Display for GpuEvaluatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NativeBackendUnavailable => {
                write!(
                    f,
                    "native CUDA GPU backend is unavailable in this build or environment"
                )
            }
            Self::NativeUnsupported(message) => write!(f, "{message}"),
            Self::NativeDriver(message) => write!(f, "{message}"),
            Self::InvalidTimeConfig => {
                write!(f, "GPU evaluator requires finite dt > 0 and finite t_max >= 0")
            }
            Self::EmptyPopulation => write!(f, "cannot evaluate an empty population"),
            Self::MissingNodeGene {
                genome_key,
                node_key,
            } => write!(f, "genome {genome_key} is missing node gene {node_key}"),
            Self::UnsupportedActivation {
                genome_key,
                node_key,
                name,
            } => write!(
                f,
                "genome {genome_key}, node {node_key}: activation {name:?} is not supported by the GPU evaluator"
            ),
            Self::UnsupportedAggregation {
                genome_key,
                node_key,
                name,
            } => write!(
                f,
                "genome {genome_key}, node {node_key}: aggregation {name:?} is not supported by the GPU evaluator; only sum is supported"
            ),
            Self::InputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::InputBatchSizeMismatch { expected, actual } => {
                write!(f, "expected input batch for {expected} genomes, got {actual}")
            }
            Self::InvalidNodeParameter {
                genome_key,
                node_key,
                name,
                value,
            } => write!(
                f,
                "genome {genome_key}, node {node_key}: invalid {name} value {value}"
            ),
        }
    }
}

impl Error for GpuEvaluatorError {}
