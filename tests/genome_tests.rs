use std::path::PathBuf;

use neat_rust::{
    CompatibilityExcessCoefficient, Config, ConnectionKey, DefaultConnectionGene, DefaultGenome,
    DefaultNodeGene, FitnessCriterion, InitialConnection, RandomSource, XorShiftRng,
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
        let value = self.values.get(self.index).copied().unwrap_or(0.0);
        self.index += 1;
        value
    }
}

fn repo_path(relative: &str) -> PathBuf {
    let relative = relative
        .strip_prefix("scripts/configs/")
        .unwrap_or(relative);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("configs")
        .join(relative)
}

fn key(input: i64, output: i64) -> ConnectionKey {
    ConnectionKey::new(input, output)
}

#[test]
fn creates_full_direct_memory8_recurrent_genome() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
        .expect("config should parse");
    let mut rng = XorShiftRng::seed_from_u64(1);
    let mut genome = DefaultGenome::new(0);

    genome
        .configure_new(&config.genome, &mut rng)
        .expect("genome should configure");

    assert_eq!(genome.nodes.len(), 2);
    assert_eq!(genome.connections.len(), 18);
    assert!(genome.connections.contains_key(&key(-1, 0)));
    assert!(genome.connections.contains_key(&key(0, 0)));
    assert!(genome.connections.contains_key(&key(1, 1)));
}

#[test]
fn creates_partial_direct_feedforward_genome() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_feedforward_memory8.ini"))
        .expect("config should parse");
    config.genome.initial_connection = InitialConnection::partial_direct(0.5);
    let mut rng = XorShiftRng::seed_from_u64(2);
    let mut genome = DefaultGenome::new(0);

    genome
        .configure_new(&config.genome, &mut rng)
        .expect("genome should configure");

    assert_eq!(genome.nodes.len(), 2);
    assert_eq!(genome.connections.len(), 8);
    assert!(genome
        .connections
        .keys()
        .all(|key| key.input < 0 && key.output >= 0));
}

#[test]
fn mutate_add_node_keeps_split_connection_neutral() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
        .expect("config should parse");
    let mut genome = DefaultGenome::new(1);
    genome.nodes.insert(0, DefaultNodeGene::new(0));
    genome.nodes.insert(1, DefaultNodeGene::new(1));
    genome
        .connections
        .insert(key(-1, 0), connection(-1, 0, 1, 2.5, true));
    let mut rng = SequenceRng::new(&[0.0; 64]);

    let new_node = genome
        .mutate_add_node(&config.genome, &mut rng)
        .expect("mutation should work")
        .expect("node should be added");

    assert_eq!(new_node, 2);
    assert_eq!(genome.nodes[&2].bias, 0.0);
    assert!(!genome.connections[&key(-1, 0)].enabled);
    assert_eq!(genome.connections[&key(-1, 2)].weight, 1.0);
    assert!(genome.connections[&key(-1, 2)].enabled);
    assert_eq!(genome.connections[&key(2, 0)].weight, 2.5);
    assert!(genome.connections[&key(2, 0)].enabled);
}

#[test]
fn prune_dangling_nodes_removes_hidden_nodes_that_cannot_reach_outputs() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
        .expect("config should parse");
    let mut genome = DefaultGenome::new(1);
    genome.nodes.insert(0, DefaultNodeGene::new(0));
    genome.nodes.insert(1, DefaultNodeGene::new(1));
    genome.nodes.insert(2, DefaultNodeGene::new(2));
    genome
        .connections
        .insert(key(-1, 2), connection(-1, 2, 1, 1.0, true));

    genome.prune_dangling_nodes(&config.genome);

    assert!(!genome.nodes.contains_key(&2));
    assert!(!genome.connections.contains_key(&key(-1, 2)));
    assert!(genome.nodes.contains_key(&0));
    assert!(genome.nodes.contains_key(&1));
}

#[test]
fn crossover_honors_min_fitness_direction_for_excess_genes() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
        .expect("config should parse");
    config.neat.fitness_criterion = FitnessCriterion::Min;
    let mut high_fitness_parent = DefaultGenome::new(1);
    high_fitness_parent.fitness = Some(10.0);
    high_fitness_parent.nodes.insert(0, DefaultNodeGene::new(0));
    high_fitness_parent.nodes.insert(1, DefaultNodeGene::new(1));
    high_fitness_parent
        .connections
        .insert(key(-1, 0), connection(-1, 0, 1, 1.0, true));
    let mut low_fitness_parent = DefaultGenome::new(2);
    low_fitness_parent.fitness = Some(1.0);
    low_fitness_parent.nodes.insert(0, DefaultNodeGene::new(0));
    low_fitness_parent.nodes.insert(1, DefaultNodeGene::new(1));
    low_fitness_parent
        .connections
        .insert(key(-2, 0), connection(-2, 0, 2, 2.0, true));
    let mut rng = XorShiftRng::seed_from_u64(11);
    let mut child = DefaultGenome::new(3);

    child
        .configure_crossover(
            &high_fitness_parent,
            &low_fitness_parent,
            &config.genome,
            Some(&config.neat.fitness_criterion),
            &mut rng,
        )
        .expect("crossover should work");

    assert!(child.connections.contains_key(&key(-2, 0)));
    assert!(!child.connections.contains_key(&key(-1, 0)));
}

#[test]
fn distance_uses_innovation_excess_and_enable_penalty() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
        .expect("config should parse");
    config.genome.compatibility_disjoint_coefficient = 1.0;
    config.genome.compatibility_excess_coefficient = CompatibilityExcessCoefficient::Value(2.0);
    config.genome.compatibility_weight_coefficient = 1.0;
    config.genome.compatibility_include_node_genes = false;
    config.genome.compatibility_enable_penalty = 0.5;

    let mut left = DefaultGenome::new(1);
    let mut right = DefaultGenome::new(2);
    left.connections
        .insert(key(-1, 0), connection(-1, 0, 1, 0.0, true));
    left.connections
        .insert(key(-2, 0), connection(-2, 0, 3, 0.0, true));
    left.connections
        .insert(key(-3, 0), connection(-3, 0, 5, 0.0, true));
    right
        .connections
        .insert(key(-1, 0), connection(-1, 0, 1, 1.0, false));
    right
        .connections
        .insert(key(-4, 0), connection(-4, 0, 2, 0.0, true));
    right
        .connections
        .insert(key(-5, 0), connection(-5, 0, 4, 0.0, true));

    let distance = left
        .distance(&right, &config.genome)
        .expect("distance should compute");

    assert!((distance - ((1.5 + 3.0 + 2.0) / 3.0)).abs() < 1e-12);
}

fn connection(
    input: i64,
    output: i64,
    innovation: i64,
    weight: f64,
    enabled: bool,
) -> DefaultConnectionGene {
    let mut connection = DefaultConnectionGene::with_innovation(key(input, output), innovation);
    connection.weight = weight;
    connection.enabled = enabled;
    connection
}
