use std::collections::BTreeMap;
use std::fs;

use neat_rust::{DefaultGenome, FitnessCriterion, StatisticsReporter};

#[test]
fn statistics_reporter_matches_neat_python_species_tables() {
    let mut stats = StatisticsReporter::new();
    stats.generation_statistics = vec![
        BTreeMap::from([
            (1, BTreeMap::from([(10, 1.0), (11, 3.0)])),
            (3, BTreeMap::from([(30, 9.0)])),
        ]),
        BTreeMap::from([(2, BTreeMap::from([(20, 5.0), (21, 7.0)]))]),
    ];

    assert_eq!(stats.get_fitness_mean(), vec![13.0 / 3.0, 6.0]);
    assert_eq!(stats.get_fitness_median(), vec![3.0, 6.0]);
    assert_eq!(
        stats.get_species_sizes(),
        vec![vec![2, 0, 1], vec![0, 2, 0]]
    );
    assert_eq!(
        stats.get_species_fitness("NA"),
        vec![
            vec!["2".to_string(), "NA".to_string(), "9".to_string()],
            vec!["NA".to_string(), "6".to_string(), "NA".to_string()],
        ]
    );
}

#[test]
fn statistics_reporter_saves_original_csv_files() {
    let mut stats = StatisticsReporter::new();
    let mut first = DefaultGenome::new(1);
    first.fitness = Some(1.0);
    let mut second = DefaultGenome::new(2);
    second.fitness = Some(3.0);
    stats.most_fit_genomes = vec![first, second];
    stats.generation_statistics = vec![
        BTreeMap::from([(1, BTreeMap::from([(1, 1.0), (2, 3.0)]))]),
        BTreeMap::from([(1, BTreeMap::from([(2, 3.0)]))]),
    ];

    let dir =
        std::env::temp_dir().join(format!("neat_rust_statistics_test_{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir should be created");
    let fitness_path = dir.join("fitness_history.csv");
    let speciation_path = dir.join("speciation.csv");
    let species_fitness_path = dir.join("species_fitness.csv");

    stats
        .save_genome_fitness(' ', &fitness_path)
        .expect("fitness history should save");
    stats
        .save_species_count(' ', &speciation_path)
        .expect("speciation should save");
    stats
        .save_species_fitness(' ', "NA", &species_fitness_path)
        .expect("species fitness should save");

    assert_eq!(
        fs::read_to_string(fitness_path).expect("fitness csv should read"),
        "1 2\n3 3"
    );
    assert_eq!(
        fs::read_to_string(speciation_path).expect("speciation csv should read"),
        "2\n1"
    );
    assert_eq!(
        fs::read_to_string(species_fitness_path).expect("species fitness csv should read"),
        "2\n3"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn statistics_best_genomes_honor_min_fitness_criterion() {
    let mut stats = StatisticsReporter::new();
    stats.fitness_criterion = Some(FitnessCriterion::Min);
    let mut high = DefaultGenome::new(1);
    high.fitness = Some(10.0);
    let mut low = DefaultGenome::new(2);
    low.fitness = Some(1.0);
    stats.most_fit_genomes = vec![high, low];

    assert_eq!(stats.best_genome().unwrap().key, 2);
}
