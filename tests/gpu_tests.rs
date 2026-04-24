use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use neat_rust::{
    algorithm::{ConnectionKey, DefaultConnectionGene, DefaultGenome, DefaultNodeGene, GenomeId},
    io::Config,
    prelude::{ActivationFunction, AggregationFunction},
    runtime::{
        native_cuda_available, pack_ctrnn_population, GPUCTRNNEvaluator, GPUIZNNEvaluator,
        GpuEvaluatorBackend, GpuEvaluatorError, GpuInputBatch,
    },
};

fn repo_path(relative: &str) -> PathBuf {
    let relative = relative
        .strip_prefix("scripts/configs/")
        .unwrap_or(relative);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("scripts")
        .join("configs")
        .join(relative)
}

fn conn_key(input: i64, output: i64) -> ConnectionKey {
    ConnectionKey::new(input, output)
}

fn memory8_config() -> Config {
    Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse")
}

fn simple_ctrnn_genome(key: i64) -> DefaultGenome {
    let mut genome = DefaultGenome::new(key);
    let mut output0 = DefaultNodeGene::new(0);
    output0.activation = ActivationFunction::Identity;
    output0.aggregation = AggregationFunction::Sum;
    output0.time_constant = 1.0;
    let mut output1 = DefaultNodeGene::new(1);
    output1.activation = ActivationFunction::Identity;
    output1.aggregation = AggregationFunction::Sum;
    output1.time_constant = 1.0;
    genome.nodes.insert(0, output0);
    genome.nodes.insert(1, output1);
    genome
        .connections
        .insert(conn_key(-1, 0), connection(conn_key(-1, 0), 1, 1.0, true));
    genome
}

fn simple_iznn_genome(key: i64) -> DefaultGenome {
    let mut genome = DefaultGenome::new(key);
    genome.nodes.insert(0, DefaultNodeGene::new(0));
    genome.nodes.insert(1, DefaultNodeGene::new(1));
    genome
        .connections
        .insert(conn_key(-1, 0), connection(conn_key(-1, 0), 1, 1.0, true));
    genome
}

fn connection(
    key: ConnectionKey,
    innovation: i64,
    weight: f64,
    enabled: bool,
) -> DefaultConnectionGene {
    let mut connection = DefaultConnectionGene::with_innovation(key, innovation);
    connection.weight = weight;
    connection.enabled = enabled;
    connection
}

#[test]
fn native_cuda_backend_is_detected_when_nvidia_driver_is_present() {
    let has_nvidia = Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if has_nvidia {
        assert!(native_cuda_available());
    }
}

#[test]
fn gpu_ctrnn_packing_matches_canonical_layout() {
    let config = memory8_config();
    let mut genomes = BTreeMap::new();
    genomes.insert(GenomeId::new(1), simple_ctrnn_genome(1));

    let packed = pack_ctrnn_population(&genomes, &config.genome).expect("pack should work");
    let key_map = &packed.node_key_maps[0];
    let src_idx = key_map[&-1];
    let dst_idx = key_map[&0];
    let weight_idx = dst_idx * packed.max_nodes + src_idx;

    assert_eq!(packed.num_inputs, 8);
    assert_eq!(packed.num_outputs, 2);
    assert_eq!(packed.max_nodes, 10);
    assert_eq!(packed.weights[weight_idx], 1.0);
    assert!(packed.node_mask[dst_idx]);
}

#[test]
fn gpu_ctrnn_evaluator_assigns_fitness_with_cpu_fallback() {
    let config = memory8_config();
    let mut genomes = BTreeMap::from([
        (GenomeId::new(1), simple_ctrnn_genome(1)),
        (GenomeId::new(2), simple_ctrnn_genome(2)),
    ]);
    let mut evaluator = GPUCTRNNEvaluator::new(
        0.1,
        0.2,
        |_time, _dt, _population_size, num_inputs| {
            let mut inputs = vec![0.0; num_inputs];
            inputs[0] = 1.0;
            GpuInputBatch::shared(inputs)
        },
        |trajectory| trajectory.last().map(|row| row[0]).unwrap_or(0.0),
    )
    .with_backend(GpuEvaluatorBackend::CpuFallback);

    evaluator
        .evaluate(&mut genomes, &config)
        .expect("gpu ctrnn evaluator should run");

    let expected = 1.0 - (-0.2_f64).exp();
    for genome in genomes.values() {
        assert!((genome.fitness.unwrap() - expected).abs() < 1e-12);
    }
}

#[test]
fn gpu_iznn_evaluator_assigns_fitness_with_cpu_fallback() {
    let config = memory8_config();
    let mut genomes = BTreeMap::from([(GenomeId::new(1), simple_iznn_genome(1))]);
    let mut evaluator = GPUIZNNEvaluator::new(
        0.05,
        0.05,
        |_time, _dt, _population_size, num_inputs| {
            let mut inputs = vec![0.0; num_inputs];
            inputs[0] = 2000.0;
            GpuInputBatch::shared(inputs)
        },
        |trajectory| trajectory.iter().flatten().sum(),
    );

    evaluator
        .evaluate(&mut genomes, &config)
        .expect("gpu iznn evaluator should run");

    assert_eq!(genomes[&GenomeId::new(1)].fitness, Some(1.0));
}

#[test]
fn gpu_ctrnn_packing_rejects_non_sum_aggregation() {
    let config = memory8_config();
    let mut genome = simple_ctrnn_genome(1);
    genome.nodes.get_mut(&0).unwrap().aggregation = AggregationFunction::Product;
    let genomes = BTreeMap::from([(GenomeId::new(1), genome)]);

    let err = pack_ctrnn_population(&genomes, &config.genome).expect_err("must reject product");

    assert!(matches!(
        err,
        GpuEvaluatorError::UnsupportedAggregation { .. }
    ));
}

#[test]
fn gpu_native_required_executes_ctrnn_when_cuda_available() {
    if !native_cuda_available() {
        return;
    }

    let config = memory8_config();
    let mut genomes = BTreeMap::from([(GenomeId::new(1), simple_ctrnn_genome(1))]);
    let mut evaluator = GPUCTRNNEvaluator::new(
        0.1,
        0.2,
        |_time, _dt, _population_size, num_inputs| {
            let mut inputs = vec![0.0; num_inputs];
            inputs[0] = 1.0;
            GpuInputBatch::shared(inputs)
        },
        |trajectory| trajectory.last().map(|row| row[0]).unwrap_or(0.0),
    )
    .with_backend(GpuEvaluatorBackend::NativeRequired);

    evaluator
        .evaluate(&mut genomes, &config)
        .expect("native CUDA backend should run");

    let expected = 1.0 - (-0.2_f64).exp();
    assert!((genomes[&GenomeId::new(1)].fitness.unwrap() - expected).abs() < 5.0e-3);
}

#[test]
fn gpu_native_required_executes_iznn_when_cuda_available() {
    if !native_cuda_available() {
        return;
    }

    let config = memory8_config();
    let mut genomes = BTreeMap::from([(GenomeId::new(1), simple_iznn_genome(1))]);
    let mut evaluator = GPUIZNNEvaluator::new(
        0.05,
        0.05,
        |_time, _dt, _population_size, num_inputs| {
            let mut inputs = vec![0.0; num_inputs];
            inputs[0] = 2000.0;
            GpuInputBatch::shared(inputs)
        },
        |trajectory| trajectory.iter().flatten().sum(),
    )
    .with_backend(GpuEvaluatorBackend::NativeRequired);

    evaluator
        .evaluate(&mut genomes, &config)
        .expect("native CUDA backend should run");

    assert_eq!(genomes[&GenomeId::new(1)].fitness, Some(1.0));
}
