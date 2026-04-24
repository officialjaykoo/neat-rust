use std::path::PathBuf;

use neat_rust::io::{
    CompatibilityExcessCoefficient, Config, FitnessSharingMode, InitialConnection, Probability,
    SpawnMethod, TargetNumSpecies,
};
use neat_rust::prelude::ActivationFunction;

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

#[test]
fn parses_recurrent_memory8_config() {
    let path = repo_path("scripts/configs/neat_recurrent_memory8.toml");
    let config = Config::from_file(path).expect("memory8 recurrent config should parse");

    assert_eq!(config.neat.pop_size, 72);
    assert_eq!(config.genome.num_inputs, 8);
    assert_eq!(config.genome.num_outputs, 2);
    assert!(!config.genome.feed_forward);
    assert_eq!(
        config.genome.initial_connection,
        InitialConnection::full_direct()
    );
    assert_eq!(config.input_keys(), vec![-1, -2, -3, -4, -5, -6, -7, -8]);
    assert_eq!(config.output_keys(), vec![0, 1]);
    assert!(config
        .genome
        .activation
        .options
        .iter()
        .any(|x| *x == ActivationFunction::Identity));
    assert_eq!(config.genome.memory_gate_enabled.default, false);
    assert_eq!(
        config.genome.memory_gate_enabled.mutate_rate,
        Probability::new(0.02)
    );
    assert_eq!(config.neat.seed, None);
    assert_eq!(
        config.genome.compatibility_excess_coefficient,
        CompatibilityExcessCoefficient::Auto
    );
    assert!(config.genome.compatibility_include_node_genes);
    assert_eq!(config.genome.compatibility_enable_penalty, 1.0);
    assert_eq!(
        config.species_set.target_num_species,
        TargetNumSpecies::Disabled
    );
    assert_eq!(config.species_set.threshold_adjust_rate, 0.1);
    assert_eq!(config.species_set.threshold_min, 0.1);
    assert_eq!(config.species_set.threshold_max, 100.0);
    assert_eq!(
        config.reproduction.fitness_sharing,
        FitnessSharingMode::Normalized
    );
    assert_eq!(config.reproduction.spawn_method, SpawnMethod::Smoothed);
    assert_eq!(
        config.reproduction.interspecies_crossover_prob,
        Probability::zero()
    );
    assert_eq!(config.reproduction.elitism, 2);
    assert!(config.reproduction.adaptive_mutation.enabled);
    assert_eq!(config.reproduction.adaptive_mutation.start_after, 4);
    assert_eq!(config.reproduction.adaptive_mutation.full_after, 18);
    assert_eq!(config.reproduction.adaptive_mutation.max_multiplier, 4.0);
    assert_eq!(
        config.reproduction.adaptive_mutation.caps.conn_add_prob,
        Probability::new(0.30)
    );
}

#[test]
fn parses_recurrent_hand7_config() {
    let path = repo_path("scripts/configs/neat_recurrent_hand7.toml");
    let config = Config::from_file(path).expect("hand7 recurrent config should parse");

    assert_eq!(config.genome.num_inputs, 7);
    assert_eq!(config.genome.num_outputs, 2);
    assert!(!config.genome.feed_forward);
    assert_eq!(config.input_keys(), vec![-1, -2, -3, -4, -5, -6, -7]);
}

#[test]
fn parses_feedforward_memory8_config_without_memory_gate_keys() {
    let path = repo_path("scripts/configs/neat_feedforward_memory8.toml");
    let config = Config::from_file(path).expect("memory8 feed-forward config should parse");

    assert_eq!(config.genome.num_inputs, 8);
    assert_eq!(config.genome.num_outputs, 2);
    assert!(config.genome.feed_forward);
    assert_eq!(config.genome.memory_gate_enabled.default, false);
    assert_eq!(
        config.genome.memory_gate_enabled.mutate_rate,
        Probability::zero()
    );
}
