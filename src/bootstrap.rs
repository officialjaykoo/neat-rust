use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::attributes::RandomSource;
use crate::config::Config;
use crate::genome::DefaultGenome;
use crate::ids::GenomeId;
use crate::reproduction::ReproductionState;

#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapStrategy {
    Random,
    FromChampion {
        genome: DefaultGenome,
        fraction: f64,
        mutate: bool,
    },
    Mixed {
        genomes: Vec<DefaultGenome>,
        fraction: f64,
        mutate: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapError {
    EmptySeedGenomes,
    EmptyPopulation,
    Genome(crate::genome::GenomeError),
}

impl BootstrapStrategy {
    pub fn from_champion(genome: DefaultGenome, fraction: f64) -> Self {
        Self::FromChampion {
            genome,
            fraction,
            mutate: true,
        }
    }

    pub fn apply(
        self,
        population: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
        reproduction: &mut ReproductionState,
        rng: &mut impl RandomSource,
    ) -> Result<BootstrapSummary, BootstrapError> {
        let (seed_genomes, fraction, mutate) = match self {
            Self::Random => {
                return Ok(BootstrapSummary {
                    seeded_count: 0,
                    population_size: population.len(),
                });
            }
            Self::FromChampion {
                genome,
                fraction,
                mutate,
            } => (vec![genome], fraction, mutate),
            Self::Mixed {
                genomes,
                fraction,
                mutate,
            } => (genomes, fraction, mutate),
        };

        if population.is_empty() {
            return Err(BootstrapError::EmptyPopulation);
        }
        if seed_genomes.is_empty() {
            return Err(BootstrapError::EmptySeedGenomes);
        }

        let count = bootstrap_count(population.len(), fraction);
        let ids = population.keys().copied().take(count).collect::<Vec<_>>();
        for (idx, genome_id) in ids.iter().copied().enumerate() {
            let mut genome = seed_genomes[idx % seed_genomes.len()].clone();
            genome.key = genome_id;
            genome.fitness = None;
            if mutate {
                genome.mutate_with_innovation(
                    &config.genome,
                    &mut reproduction.innovation_tracker,
                    rng,
                )?;
            } else {
                genome.validate(&config.genome)?;
            }
            population.insert(genome_id, genome);
            reproduction.ancestors.insert(genome_id, (None, None));
        }

        Ok(BootstrapSummary {
            seeded_count: count,
            population_size: population.len(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapSummary {
    pub seeded_count: usize,
    pub population_size: usize,
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeedGenomes => write!(f, "bootstrap requires at least one seed genome"),
            Self::EmptyPopulation => write!(f, "bootstrap cannot seed an empty population"),
            Self::Genome(err) => write!(f, "{err}"),
        }
    }
}

impl Error for BootstrapError {}

impl From<crate::genome::GenomeError> for BootstrapError {
    fn from(value: crate::genome::GenomeError) -> Self {
        Self::Genome(value)
    }
}

fn bootstrap_count(population_size: usize, fraction: f64) -> usize {
    let fraction = if fraction.is_finite() { fraction } else { 0.5 };
    ((population_size as f64) * fraction.clamp(0.0, 1.0))
        .round()
        .clamp(1.0, population_size.max(1) as f64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_fraction_seeds_half_population() {
        assert_eq!(bootstrap_count(72, 0.5), 36);
    }
}
