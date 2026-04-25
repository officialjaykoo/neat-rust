use std::path::PathBuf;

use neat_rust::{
    algorithm::{
        BootstrapStrategy, ConnectionKey, DefaultConnectionGene, DefaultGenome, GenomeError,
        InnovationTracker, Population, PopulationError, XorShiftRng,
    },
    io::{new_rust_checkpoint_sink, Checkpointer, Config},
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
fn champion_bootstrap_seeds_half_of_initial_population() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut champion_rng = XorShiftRng::seed_from_u64(99);
    let mut champion_tracker = InnovationTracker::default();
    let mut champion = DefaultGenome::new(999);
    champion
        .configure_new_with_innovation(&config.genome, &mut champion_tracker, &mut champion_rng)
        .expect("champion should configure");
    let champion_connection_count = champion.connections.len();

    let population = Population::new_with_bootstrap(
        config,
        123,
        BootstrapStrategy::from_champion(champion, 0.5),
    )
    .expect("bootstrap population should initialize");

    assert_eq!(population.population.len(), 72);
    assert_eq!(population.species.genome_to_species.len(), 72);
    assert!(population.population.values().take(36).all(|genome| {
        genome.fitness.is_none() && genome.connections.len() >= champion_connection_count - 1
    }));
}

#[test]
fn population_rejects_non_finite_fitness() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut population = Population::new(config, 24).expect("population should initialize");

    let err = population
        .run(
            |genomes, _config| {
                for genome in genomes.values_mut() {
                    genome.fitness = Some(f64::NAN);
                }
                Ok(())
            },
            Some(1),
        )
        .expect_err("non-finite fitness must fail");

    assert!(matches!(
        err,
        PopulationError::Genome(GenomeError::InvalidFitness { .. })
    ));
}

#[test]
fn genome_validation_detects_unknown_connection_source() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut genome = DefaultGenome::new(7);
    genome
        .nodes
        .insert(0, neat_rust::algorithm::DefaultNodeGene::new(0));
    genome
        .nodes
        .insert(1, neat_rust::algorithm::DefaultNodeGene::new(1));
    genome.connections.insert(
        ConnectionKey::new(99, 0),
        DefaultConnectionGene::new(ConnectionKey::new(99, 0)),
    );

    let err = genome
        .validate(&config.genome)
        .expect_err("unknown source node must fail validation");

    assert!(matches!(
        err,
        GenomeError::ConnectionFromUnknownNode { node_key: 99, .. }
    ));
}

#[test]
fn repeated_mutation_preserves_genome_invariants() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");

    for seed in 1..=16 {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let mut tracker = InnovationTracker::default();
        let mut genome = DefaultGenome::new(seed as i64);
        genome
            .configure_new_with_innovation(&config.genome, &mut tracker, &mut rng)
            .expect("genome should configure");

        for _ in 0..128 {
            genome
                .mutate_with_innovation(&config.genome, &mut tracker, &mut rng)
                .expect("mutation should preserve invariants");
            genome
                .validate(&config.genome)
                .expect("mutated genome should validate");
        }
    }
}

#[test]
fn config_validation_rejects_zero_population() {
    let path = repo_path("scripts/configs/neat_recurrent_memory8.toml");
    let text = std::fs::read_to_string(path).expect("config should read");
    let text = text.replace("pop_size = 72", "pop_size = 0");

    let err = text
        .parse::<Config>()
        .expect_err("zero population must fail");

    assert!(err.to_string().contains("pop_size"));
}

#[test]
fn checkpoint_restore_rejects_changed_config_hash() {
    let config_text =
        std::fs::read_to_string(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
            .expect("config should read");
    let dir = unique_temp_dir("neat_rust_checkpoint_hash_test");
    std::fs::create_dir_all(&dir).expect("temp dir should be created");
    let config_path = dir.join("config.toml");
    std::fs::write(&config_path, &config_text).expect("temp config should write");

    let config = Config::from_file(&config_path).expect("config should parse");
    let mut population = Population::new(config, 24).expect("population should initialize");
    let prefix = dir
        .join("neat-rust-checkpoint-gen")
        .to_string_lossy()
        .to_string();
    population.checkpoint_sink = Some(new_rust_checkpoint_sink(
        Some(1),
        prefix.clone(),
        config_path.clone(),
    ));
    population
        .run(
            |genomes, _config| {
                for genome in genomes.values_mut() {
                    genome.fitness = Some(genome.key.raw() as f64);
                }
                Ok(())
            },
            Some(1),
        )
        .expect("population should run");

    std::fs::write(
        &config_path,
        config_text.replace(
            "compatibility_threshold = 3.1",
            "compatibility_threshold = 3.2",
        ),
    )
    .expect("config should mutate");
    let checkpoint_path = Checkpointer::new(Some(1), prefix).checkpoint_path(0);
    let err = match Checkpointer::restore_checkpoint(&checkpoint_path) {
        Ok(_) => panic!("changed config hash must fail restore"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("config hash mismatch"));
    let _ = std::fs::remove_dir_all(dir);
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{prefix}_{}", std::process::id()))
}
