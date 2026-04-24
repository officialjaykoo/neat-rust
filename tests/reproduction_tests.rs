use neat_rust::{adjust_spawn_exact, compute_spawn_proportional, ReproductionError};

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
