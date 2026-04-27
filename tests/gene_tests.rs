use std::path::PathBuf;

use neat_rust::{
    algorithm::{ConnectionKey, DefaultConnectionGene, DefaultNodeGene, RandomSource},
    io::{Config, GenomeConfig, NodeMemoryKind},
    prelude::{ActivationFunction, AggregationFunction},
};

#[derive(Debug, Clone)]
struct SequenceRng {
    values: Vec<f64>,
    index: usize,
}

impl SequenceRng {
    fn new(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
            index: 0,
        }
    }
}

impl RandomSource for SequenceRng {
    fn next_f64(&mut self) -> f64 {
        let value = *self
            .values
            .get(self.index)
            .expect("test RNG sequence should have enough values");
        self.index += 1;
        value
    }
}

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

fn key(input: i64, output: i64) -> ConnectionKey {
    ConnectionKey::new(input, output)
}

fn memory8_genome_config() -> GenomeConfig {
    Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("memory8 recurrent config should parse")
        .genome
}

#[test]
fn initializes_node_gene_from_config() {
    let config = memory8_genome_config();
    let mut rng = SequenceRng::new(&[
        0.5, 0.0, // bias gaussian: mean value
        0.5, 0.0, // response gaussian: stdev 0
        0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
        0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
    ]);

    let gene =
        DefaultNodeGene::initialized(0, &config, &mut rng).expect("node gene should initialize");

    assert_eq!(gene.key, 0);
    assert_eq!(gene.activation, ActivationFunction::Tanh);
    assert_eq!(gene.aggregation, AggregationFunction::Sum);
    assert_eq!(gene.response, 1.0);
    assert_eq!(gene.time_constant, 1.0);
    assert_eq!(gene.iz_a, 0.02);
    assert_eq!(gene.iz_b, 0.20);
    assert_eq!(gene.iz_c, -65.0);
    assert_eq!(gene.iz_d, 8.0);
    assert_eq!(gene.node_memory_kind, NodeMemoryKind::None);
    assert!(
        gene.node_hebbian_decay >= config.node_hebbian_decay.min_value
            && gene.node_hebbian_decay <= config.node_hebbian_decay.max_value
    );
}

#[test]
fn initializes_connection_gene_from_config() {
    let config = memory8_genome_config();
    let mut rng = SequenceRng::new(&[0.5, 0.0]);

    let gene = DefaultConnectionGene::initialized(key(-1, 0), &config, &mut rng)
        .expect("connection gene should initialize");

    assert_eq!(gene.key, key(-1, 0));
    assert!(gene.weight >= config.weight.min_value && gene.weight <= config.weight.max_value);
    assert!(gene.enabled);
}

#[test]
fn node_gene_crossover_inherits_each_attribute_independently() {
    let left = DefaultNodeGene {
        key: 1,
        bias: 1.0,
        response: 2.0,
        activation: ActivationFunction::Tanh,
        aggregation: AggregationFunction::Sum,
        time_constant: 1.0,
        iz_a: 0.02,
        iz_b: 0.20,
        iz_c: -65.0,
        iz_d: 8.0,
        node_memory_kind: NodeMemoryKind::NodeGru,
        node_hebbian_eta: 3.0,
        ..DefaultNodeGene::new(1)
    };
    let right = DefaultNodeGene {
        key: 1,
        bias: -1.0,
        response: -2.0,
        activation: ActivationFunction::Relu,
        aggregation: AggregationFunction::Max,
        time_constant: 2.0,
        iz_a: 0.10,
        iz_b: 0.25,
        iz_c: -55.0,
        iz_d: 4.0,
        node_memory_kind: NodeMemoryKind::Hebbian,
        node_hebbian_eta: -3.0,
        ..DefaultNodeGene::new(1)
    };
    let mut rng = SequenceRng::new(&[
        0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1,
        0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1,
    ]);

    let child = left
        .crossover(&right, &mut rng)
        .expect("matching node keys should crossover");

    assert_eq!(child.bias, 1.0);
    assert_eq!(child.response, -2.0);
    assert_eq!(child.activation, ActivationFunction::Tanh);
    assert_eq!(child.aggregation, AggregationFunction::Max);
    assert_eq!(child.time_constant, 1.0);
    assert_eq!(child.iz_a, 0.10);
    assert_eq!(child.iz_b, 0.20);
    assert_eq!(child.iz_c, -55.0);
    assert_eq!(child.iz_d, 8.0);
    assert_eq!(child.node_memory_kind, NodeMemoryKind::NodeGru);
    assert_eq!(child.node_hebbian_eta, 3.0);
}

#[test]
fn connection_gene_crossover_applies_disable_rule() {
    let mut left = DefaultConnectionGene::new(key(-1, 0));
    left.weight = 1.5;
    left.enabled = true;
    let mut right = DefaultConnectionGene::new(key(-1, 0));
    right.weight = -1.5;
    right.enabled = false;
    let mut rng = SequenceRng::new(&[0.1, 0.1, 0.1]);

    let child = left
        .crossover(&right, &mut rng)
        .expect("matching connection keys should crossover");

    assert_eq!(child.weight, -1.5);
    assert!(!child.enabled);
}

#[test]
fn gene_distance_uses_config_weight_coefficient() {
    let config = memory8_genome_config();
    let node_left = DefaultNodeGene {
        key: 0,
        bias: 1.0,
        response: 1.0,
        activation: ActivationFunction::Tanh,
        aggregation: AggregationFunction::Sum,
        time_constant: 1.0,
        iz_a: 0.02,
        iz_b: 0.20,
        iz_c: -65.0,
        iz_d: 8.0,
        node_memory_kind: NodeMemoryKind::None,
        ..DefaultNodeGene::new(0)
    };
    let node_right = DefaultNodeGene {
        key: 0,
        bias: 2.0,
        response: 3.0,
        activation: ActivationFunction::Relu,
        aggregation: AggregationFunction::Sum,
        time_constant: 1.0,
        iz_a: 0.02,
        iz_b: 0.20,
        iz_c: -65.0,
        iz_d: 8.0,
        node_memory_kind: NodeMemoryKind::None,
        ..DefaultNodeGene::new(0)
    };
    let mut conn_left = DefaultConnectionGene::new(key(-1, 0));
    conn_left.weight = 0.25;
    conn_left.enabled = true;
    let mut conn_right = DefaultConnectionGene::new(key(-1, 0));
    conn_right.weight = -0.25;
    conn_right.enabled = false;

    assert!((node_left.distance(&node_right, &config).unwrap() - 1.6).abs() < 1e-12);
    assert!((conn_left.distance(&conn_right, &config).unwrap() - 0.6).abs() < 1e-12);
}
