use std::path::PathBuf;

use neat_rust::{
    algorithm::{adjust_spawn_exact, compute_spawn_proportional, ReproductionError},
    io::{AdaptiveMutationConfig, Config, MutationRateCaps, Probability},
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

#[test]
fn proportional_spawn_matches_canonical_allocation() {
    let spawn = compute_spawn_proportional(&[1.0, 3.0, 6.0], 10, 1);

    assert_eq!(spawn, vec![1, 3, 6]);
}

#[test]
fn proportional_spawn_preserves_species_minimum_when_fitness_is_zero() {
    let spawn = compute_spawn_proportional(&[0.0, 0.0, 0.0], 10, 2);

    assert_eq!(spawn, vec![2, 2, 2]);
}

#[test]
fn adjust_spawn_exact_rejects_impossible_species_minimum() {
    let err = adjust_spawn_exact(vec![2, 2, 2], 5, 2)
        .expect_err("species minimum larger than pop size must fail");

    assert_eq!(err, ReproductionError::SpawnConflict);
}

#[test]
fn adjust_spawn_exact_preserves_population_size_after_rounding() {
    let spawn = adjust_spawn_exact(vec![1, 6, 6], 10, 1)
        .expect("spawn counts should be normalized exactly");

    assert_eq!(spawn.iter().sum::<usize>(), 10);
    assert!(spawn.iter().all(|value| *value >= 1));
}

#[test]
fn adaptive_mutation_scales_structural_rates_after_stagnation() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let adaptive = AdaptiveMutationConfig {
        enabled: true,
        start_after: 2,
        full_after: 6,
        max_multiplier: 3.0,
        caps: MutationRateCaps {
            conn_add_prob: Probability::new(0.20),
            conn_delete_prob: Probability::new(0.05),
            node_add_prob: Probability::new(0.05),
            node_delete_prob: Probability::new(0.02),
        },
    };

    assert_eq!(adaptive.multiplier(1), 1.0);
    assert_eq!(adaptive.multiplier(6), 3.0);

    let adapted = adaptive.adapted_genome_config(&config.genome, 6);
    assert_eq!(adapted.conn_add_prob, Probability::new(0.20));
    assert_eq!(adapted.conn_delete_prob, Probability::new(0.05));
    assert_eq!(adapted.node_add_prob, Probability::new(0.05));
    assert_eq!(adapted.node_delete_prob, Probability::new(0.02));
}
