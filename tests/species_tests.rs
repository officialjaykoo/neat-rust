use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use neat_rust::{
    algorithm::{
        DefaultGenome, DefaultStagnation, GenomeId, Population, Species, SpeciesId, SpeciesSet,
        XorShiftRng,
    },
    io::Config,
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
fn speciation_has_no_orphaned_mappings() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let population = Population::new(config, 24).expect("population should initialize");

    assert_eq!(
        population.species.genome_to_species.len(),
        population.population.len()
    );

    let mut reverse_mapping = BTreeMap::new();
    for (species_id, species) in &population.species.species {
        for genome_id in species.members.keys() {
            reverse_mapping.insert(*genome_id, *species_id);
        }
    }

    assert_eq!(reverse_mapping, population.species.genome_to_species);
}

#[test]
fn identical_genomes_collapse_into_one_species() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    let mut rng = XorShiftRng::seed_from_u64(7);
    let mut template = DefaultGenome::new(1);
    template
        .configure_new(&config.genome, &mut rng)
        .expect("template genome should configure");

    let mut population = BTreeMap::new();
    for key in 1..=5 {
        let mut genome = template.clone();
        let genome_id = GenomeId::new(key);
        genome.key = genome_id;
        population.insert(genome_id, genome);
    }

    let mut species = SpeciesSet::new();
    species
        .speciate(&config, &population, 0)
        .expect("identical genomes should speciate");

    assert_eq!(species.species.len(), 1);
    assert_eq!(species.genome_to_species.len(), population.len());
    let assigned_species: BTreeSet<SpeciesId> =
        species.genome_to_species.values().copied().collect();
    assert_eq!(assigned_species.len(), 1);
}

#[test]
fn stagnation_protects_top_species_elites() {
    let mut config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.toml"))
        .expect("config should parse");
    config.stagnation.max_stagnation = 1;
    config.stagnation.species_elitism = 1;

    let mut weak_species = Species::new(SpeciesId::new(1), 0);
    weak_species.last_improved = 0;
    weak_species.fitness_history = vec![1.0];
    weak_species
        .members
        .insert(GenomeId::new(1), genome_with_fitness(1, 1.0));

    let mut elite_species = Species::new(SpeciesId::new(2), 0);
    elite_species.last_improved = 0;
    elite_species.fitness_history = vec![10.0];
    elite_species
        .members
        .insert(GenomeId::new(2), genome_with_fitness(2, 10.0));

    let species_set = BTreeMap::from([
        (SpeciesId::new(1), weak_species),
        (SpeciesId::new(2), elite_species),
    ]);
    let genome_to_species = BTreeMap::from([
        (GenomeId::new(1), SpeciesId::new(1)),
        (GenomeId::new(2), SpeciesId::new(2)),
    ]);
    let mut species = SpeciesSet::from_parts(
        species_set,
        genome_to_species,
        SpeciesId::new(3),
        Some(config.species_set.compatibility_threshold),
    );

    let updates = DefaultStagnation::update(&mut species, 10, &config);
    let weak = updates
        .iter()
        .find(|update| update.species_id == SpeciesId::new(1))
        .expect("weak species update exists");
    let elite = updates
        .iter()
        .find(|update| update.species_id == SpeciesId::new(2))
        .expect("elite species update exists");

    assert!(weak.stagnant);
    assert!(!weak.protected_by_elitism);
    assert!(!elite.stagnant);
    assert!(elite.protected_by_elitism);
    assert_eq!(elite.stagnant_time, 10);
}

fn genome_with_fitness(key: i64, fitness: f64) -> DefaultGenome {
    let mut genome = DefaultGenome::new(key);
    genome.fitness = Some(fitness);
    genome
}
