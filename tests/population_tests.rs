use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use neat_rust::{
    algorithm::{Population, PopulationError, Reporter},
    io::{new_rust_checkpoint_sink, Checkpointer, Config, TargetNumSpecies},
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
fn runs_one_generation_with_synthetic_fitness() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let pop_size = config.neat.pop_size;
    let mut population = Population::new(config, 24).expect("population should initialize");

    let best = population
        .run(
            |genomes, _config| {
                for genome in genomes.values_mut() {
                    genome.fitness = Some(genome.key.raw() as f64);
                }
                Ok(())
            },
            Some(1),
        )
        .expect("population should run")
        .expect("best genome should exist");

    assert_eq!(population.generation, 1);
    assert_eq!(population.population.len(), pop_size);
    assert_eq!(best.fitness, Some(pop_size as f64));
}

#[test]
fn initial_speciation_adjusts_dynamic_compatibility_threshold() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    config.species_set.compatibility_threshold = 0.5;
    config.species_set.target_num_species = TargetNumSpecies::Count(999);
    config.species_set.threshold_adjust_rate = 0.2;
    config.species_set.threshold_min = 0.1;
    config.species_set.threshold_max = 100.0;

    let population = Population::new(config, 24).expect("population should initialize");

    assert!((population.species.compatibility_threshold.unwrap() - 0.3).abs() < 1e-12);
}

#[test]
fn no_fitness_termination_requires_explicit_generation_limit() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    config.neat.no_fitness_termination = true;
    let mut population = Population::new(config, 24).expect("population should initialize");

    let err = population
        .run(
            |genomes, _config| {
                for genome in genomes.values_mut() {
                    genome.fitness = Some(0.0);
                }
                Ok(())
            },
            None,
        )
        .expect_err("missing generation limit must fail");

    assert!(matches!(err, PopulationError::NoGenerationalLimit));
}

#[test]
fn restored_post_evaluate_checkpoint_skips_duplicate_evaluation() {
    let config_path = repo_path("scripts/configs/neat_recurrent_memory8.toml");
    let config = Config::from_file(&config_path).expect("config should parse");
    let mut population = Population::new(config, 24).expect("population should initialize");
    let dir = std::env::temp_dir().join(format!(
        "neat_rust_population_checkpoint_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir should be created");
    let prefix = dir
        .join("neat-rust-checkpoint-gen")
        .to_string_lossy()
        .to_string();
    population.checkpoint_sink = Some(new_rust_checkpoint_sink(
        Some(1),
        prefix.clone(),
        config_path,
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
    let checkpoint_path = Checkpointer::new(Some(1), prefix).checkpoint_path(0);
    let mut restored =
        Checkpointer::restore_checkpoint(&checkpoint_path).expect("checkpoint should restore");
    let mut eval_calls = 0usize;

    restored
        .run(
            |genomes, _config| {
                eval_calls += 1;
                for genome in genomes.values_mut() {
                    genome.fitness = Some(genome.key.raw() as f64);
                }
                Ok(())
            },
            Some(1),
        )
        .expect("restored population should run");

    assert_eq!(eval_calls, 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn complete_extinction_returns_error_when_reset_is_disabled() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    config.stagnation.max_stagnation = 0;
    config.stagnation.species_elitism = 0;
    config.neat.reset_on_extinction = false;
    let mut population = Population::new(config, 24).expect("population should initialize");

    let err = population
        .run(
            |genomes, _config| {
                for genome in genomes.values_mut() {
                    genome.fitness = Some(0.0);
                }
                Ok(())
            },
            Some(1),
        )
        .expect_err("must raise complete extinction");

    assert!(matches!(err, PopulationError::CompleteExtinction));
}

#[test]
fn complete_extinction_resets_population_when_enabled() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    config.stagnation.max_stagnation = 0;
    config.stagnation.species_elitism = 0;
    config.neat.reset_on_extinction = true;
    let pop_size = config.neat.pop_size;
    let mut population = Population::new(config, 24).expect("population should initialize");
    let extinction_count = Rc::new(RefCell::new(0usize));
    population.add_reporter(Box::new(ExtinctionReporter {
        count: extinction_count.clone(),
    }));

    population
        .run(
            |genomes, _config| {
                for genome in genomes.values_mut() {
                    genome.fitness = Some(0.0);
                }
                Ok(())
            },
            Some(1),
        )
        .expect("population should recover from extinction");

    assert_eq!(population.population.len(), pop_size);
    assert!(!population.species.species.is_empty());
    assert!(*extinction_count.borrow() > 0);
}

struct ExtinctionReporter {
    count: Rc<RefCell<usize>>,
}

impl Reporter for ExtinctionReporter {
    fn complete_extinction(&mut self) {
        *self.count.borrow_mut() += 1;
    }
}
