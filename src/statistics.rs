use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use crate::config::{Config, FitnessCriterion};
use crate::genome::DefaultGenome;
use crate::reporting::{mean, median2, stdev, Reporter};
use crate::species::SpeciesSet;

pub type SpeciesFitnessSnapshot = BTreeMap<i64, BTreeMap<i64, f64>>;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StatisticsReporter {
    pub most_fit_genomes: Vec<DefaultGenome>,
    pub generation_statistics: Vec<SpeciesFitnessSnapshot>,
    pub fitness_criterion: Option<FitnessCriterion>,
}

impl StatisticsReporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn best_genome(&self) -> Option<&DefaultGenome> {
        self.best_genomes(1).into_iter().next()
    }

    pub fn best_genomes(&self, n: usize) -> Vec<&DefaultGenome> {
        let mut genomes: Vec<&DefaultGenome> = self.most_fit_genomes.iter().collect();
        sort_genomes_by_fitness(&mut genomes, self.is_min_fitness());
        genomes.truncate(n);
        genomes
    }

    pub fn best_unique_genomes(&self, n: usize) -> Vec<&DefaultGenome> {
        let mut by_key = BTreeMap::new();
        for genome in &self.most_fit_genomes {
            by_key.insert(genome.key, genome);
        }
        let mut genomes: Vec<&DefaultGenome> = by_key.values().copied().collect();
        sort_genomes_by_fitness(&mut genomes, self.is_min_fitness());
        genomes.truncate(n);
        genomes
    }

    pub fn get_fitness_mean(&self) -> Vec<f64> {
        self.get_fitness_stat(mean)
    }

    pub fn get_fitness_stdev(&self) -> Vec<f64> {
        self.get_fitness_stat(stdev)
    }

    pub fn get_fitness_median(&self) -> Vec<f64> {
        self.get_fitness_stat(median2)
    }

    pub fn get_species_sizes_by_id(&self) -> Vec<BTreeMap<i64, usize>> {
        self.generation_statistics
            .iter()
            .map(|snapshot| {
                snapshot
                    .iter()
                    .map(|(species_id, members)| (*species_id, members.len()))
                    .collect()
            })
            .collect()
    }

    pub fn get_species_sizes(&self) -> Vec<Vec<usize>> {
        let Some(max_species) = self.max_species_id() else {
            return Vec::new();
        };
        self.generation_statistics
            .iter()
            .map(|snapshot| {
                (1..=max_species)
                    .map(|species_id| snapshot.get(&species_id).map(BTreeMap::len).unwrap_or(0))
                    .collect()
            })
            .collect()
    }

    pub fn get_species_fitness_values(&self) -> Vec<Vec<Option<f64>>> {
        let Some(max_species) = self.max_species_id() else {
            return Vec::new();
        };
        self.generation_statistics
            .iter()
            .map(|snapshot| {
                (1..=max_species)
                    .map(|species_id| {
                        snapshot
                            .get(&species_id)
                            .filter(|members| !members.is_empty())
                            .map(|members| {
                                let values: Vec<f64> = members.values().copied().collect();
                                mean(&values)
                            })
                    })
                    .collect()
            })
            .collect()
    }

    pub fn get_species_fitness(&self, null_value: &str) -> Vec<Vec<String>> {
        self.get_species_fitness_values()
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        value
                            .map(|number| number.to_string())
                            .unwrap_or_else(|| null_value.to_string())
                    })
                    .collect()
            })
            .collect()
    }

    pub fn get_fitness_stat(&self, stat: fn(&[f64]) -> f64) -> Vec<f64> {
        self.generation_statistics
            .iter()
            .map(|snapshot| {
                let mut values = Vec::new();
                for species_stats in snapshot.values() {
                    values.extend(species_stats.values().copied());
                }
                stat(&values)
            })
            .collect()
    }

    pub fn save(&self) -> io::Result<()> {
        self.save_genome_fitness(' ', "fitness_history.csv")?;
        self.save_species_count(' ', "speciation.csv")?;
        self.save_species_fitness(' ', "NA", "species_fitness.csv")
    }

    pub fn save_genome_fitness(
        &self,
        delimiter: char,
        filename: impl AsRef<Path>,
    ) -> io::Result<()> {
        let best_fitness: Vec<f64> = self
            .most_fit_genomes
            .iter()
            .map(|genome| genome.fitness.unwrap_or(0.0))
            .collect();
        let avg_fitness = self.get_fitness_mean();
        let mut rows = Vec::new();
        for (best, avg) in best_fitness.iter().zip(avg_fitness.iter()) {
            rows.push(format!("{best}{delimiter}{avg}"));
        }
        fs::write(filename, rows.join("\n"))
    }

    pub fn save_species_count(
        &self,
        delimiter: char,
        filename: impl AsRef<Path>,
    ) -> io::Result<()> {
        let rows: Vec<String> = self
            .get_species_sizes()
            .into_iter()
            .map(|row| join_display(row, delimiter))
            .collect();
        fs::write(filename, rows.join("\n"))
    }

    pub fn save_species_fitness(
        &self,
        delimiter: char,
        null_value: &str,
        filename: impl AsRef<Path>,
    ) -> io::Result<()> {
        let rows: Vec<String> = self
            .get_species_fitness(null_value)
            .into_iter()
            .map(|row| row.join(&delimiter.to_string()))
            .collect();
        fs::write(filename, rows.join("\n"))
    }

    fn max_species_id(&self) -> Option<i64> {
        let mut all_species = BTreeSet::new();
        for generation in &self.generation_statistics {
            all_species.extend(generation.keys().copied());
        }
        all_species.into_iter().max()
    }

    fn is_min_fitness(&self) -> bool {
        self.fitness_criterion
            .as_ref()
            .map(FitnessCriterion::is_min)
            .unwrap_or(false)
    }
}

impl Reporter for StatisticsReporter {
    fn post_evaluate(
        &mut self,
        config: &Config,
        _population: &BTreeMap<i64, DefaultGenome>,
        species: &SpeciesSet,
        best_genome: &DefaultGenome,
    ) {
        if self.fitness_criterion.is_none() {
            self.fitness_criterion = Some(config.neat.fitness_criterion.clone());
        }
        self.most_fit_genomes.push(best_genome.clone());
        let mut species_stats = BTreeMap::new();
        for (species_id, species) in &species.species {
            let members = species
                .members
                .iter()
                .filter_map(|(genome_id, genome)| genome.fitness.map(|f| (*genome_id, f)))
                .collect();
            species_stats.insert(*species_id, members);
        }
        self.generation_statistics.push(species_stats);
    }
}

fn sort_genomes_by_fitness(genomes: &mut [&DefaultGenome], ascending: bool) {
    genomes.sort_by(|left, right| {
        let missing = if ascending {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        let left_fitness = left.fitness.unwrap_or(missing);
        let right_fitness = right.fitness.unwrap_or(missing);
        if ascending {
            left_fitness
                .total_cmp(&right_fitness)
                .then_with(|| left.key.cmp(&right.key))
        } else {
            right_fitness
                .total_cmp(&left_fitness)
                .then_with(|| left.key.cmp(&right.key))
        }
    });
}

fn join_display(values: Vec<impl ToString>, delimiter: char) -> String {
    values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(&delimiter.to_string())
}
