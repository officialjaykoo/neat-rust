use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use neat_rust::{Config, DefaultGenome, GenomeId, Population, SpeciesId, SpeciesSet, XorShiftRng};

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

#[test]
fn speciation_has_no_orphaned_mappings() {
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
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
    let config = Config::from_file(repo_path("scripts/configs/neat_recurrent_memory8.ini"))
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
