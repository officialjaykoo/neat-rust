use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use crate::attributes::RandomSource;
use crate::config::{Config, FitnessCriterion};
use crate::genome::DefaultGenome;
use crate::ids::{GenomeId, SpeciesId};
use crate::population::Population;
use crate::reporting::mean;
use crate::reproduction::{
    adjust_spawn_exact, compute_spawn, compute_spawn_proportional, ReproductionError,
};
use crate::species::{Species, SpeciesSet};
use crate::stagnation::is_better_fitness;

#[derive(Debug, Clone, PartialEq)]
pub enum PopulationFitnessSummaryError {
    FitnessNotAssigned(GenomeId),
    NoBestGenome,
}

impl fmt::Display for PopulationFitnessSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FitnessNotAssigned(key) => write!(f, "fitness not assigned to genome {key}"),
            Self::NoBestGenome => write!(f, "no best genome found"),
        }
    }
}

impl Error for PopulationFitnessSummaryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopulationCheckpointError {
    pub message: String,
}

impl PopulationCheckpointError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PopulationCheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for PopulationCheckpointError {}

pub trait PopulationCheckpointSink: Send + Sync {
    fn should_save(&self, generation: usize) -> bool;
    fn checkpoint_path(&self, generation: usize) -> PathBuf;
    fn save_population(
        &self,
        population: &Population,
    ) -> Result<PathBuf, PopulationCheckpointError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct PopulationFitnessSummary {
    pub best_genome: DefaultGenome,
    pub criterion_value: f64,
}

impl PopulationFitnessSummary {
    pub fn from_population(
        population: &BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> Result<Self, PopulationFitnessSummaryError> {
        let mut best: Option<DefaultGenome> = None;
        let mut values = Vec::new();

        for genome in population.values() {
            let fitness =
                genome
                    .fitness
                    .ok_or(PopulationFitnessSummaryError::FitnessNotAssigned(
                        genome.key,
                    ))?;
            values.push(fitness);
            if best
                .as_ref()
                .map(|current| is_better_fitness(fitness, current.fitness.unwrap_or(0.0), config))
                .unwrap_or(true)
            {
                best = Some(genome.clone());
            }
        }

        let best_genome = best.ok_or(PopulationFitnessSummaryError::NoBestGenome)?;
        let criterion_value =
            population_fitness_criterion_value(&config.neat.fitness_criterion, &values);

        Ok(Self {
            best_genome,
            criterion_value,
        })
    }
}

pub fn sync_species_members(
    species_set: &mut SpeciesSet,
    population: &BTreeMap<GenomeId, DefaultGenome>,
) {
    for species in species_set.species.values_mut() {
        for (key, member) in species.members.iter_mut() {
            if let Some(genome) = population.get(key) {
                *member = genome.clone();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesAssignment {
    pub representative_id: GenomeId,
    pub member_ids: Vec<GenomeId>,
}

impl SpeciesAssignment {
    pub fn staged(representative_id: impl Into<GenomeId>) -> Self {
        Self {
            representative_id: representative_id.into(),
            member_ids: Vec::new(),
        }
    }

    pub fn add_member(&mut self, genome_id: impl Into<GenomeId>) {
        self.member_ids.push(genome_id.into());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpawnPlanEntry {
    pub species_key: SpeciesId,
    pub spawn_quota: usize,
    pub parent_pool: Vec<(GenomeId, DefaultGenome)>,
}

impl SpawnPlanEntry {
    pub fn choose_parent<'a>(
        &'a self,
        rng: &mut impl RandomSource,
    ) -> Result<(&'a GenomeId, &'a DefaultGenome), ReproductionError> {
        let index = rng
            .next_index(self.parent_pool.len())
            .ok_or(ReproductionError::EmptySpecies)?;
        Ok((&self.parent_pool[index].0, &self.parent_pool[index].1))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpawnPlan {
    pub entries: Vec<SpawnPlanEntry>,
}

impl SpawnPlan {
    pub fn build(
        config: &Config,
        species: &[Species],
        pop_size: usize,
    ) -> Result<Self, ReproductionError> {
        let adjusted_fitnesses: Vec<f64> = species
            .iter()
            .map(|species| species.adjusted_fitness.unwrap_or(0.0))
            .collect();
        let previous_sizes: Vec<usize> = species
            .iter()
            .map(|species| species.members.len())
            .collect();
        let min_species_size = config
            .reproduction
            .min_species_size
            .max(config.reproduction.elitism);
        let mut spawn_amounts = if config.reproduction.spawn_method.is_proportional() {
            compute_spawn_proportional(&adjusted_fitnesses, pop_size, min_species_size)
        } else {
            compute_spawn(
                &adjusted_fitnesses,
                &previous_sizes,
                pop_size,
                min_species_size,
            )
        };
        spawn_amounts = adjust_spawn_exact(spawn_amounts, pop_size, min_species_size)?;

        let parent_pools = build_parent_pools(species, config);
        let entries = species
            .iter()
            .zip(spawn_amounts)
            .map(|(species, spawn_quota)| SpawnPlanEntry {
                species_key: species.key,
                spawn_quota,
                parent_pool: parent_pools.get(&species.key).cloned().unwrap_or_default(),
            })
            .collect();

        Ok(Self { entries })
    }

    pub fn entry(&self, species_key: SpeciesId) -> Option<&SpawnPlanEntry> {
        self.entries
            .iter()
            .find(|entry| entry.species_key == species_key)
    }

    pub fn choose_interspecies_parent<'a>(
        &'a self,
        current_species: SpeciesId,
        rng: &mut impl RandomSource,
    ) -> Result<Option<(&'a GenomeId, &'a DefaultGenome)>, ReproductionError> {
        let other_entries: Vec<&SpawnPlanEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.species_key != current_species && !entry.parent_pool.is_empty())
            .collect();
        let Some(entry_index) = rng.next_index(other_entries.len()) else {
            return Ok(None);
        };
        other_entries[entry_index].choose_parent(rng).map(Some)
    }
}

fn build_parent_pools(
    species: &[Species],
    config: &Config,
) -> BTreeMap<SpeciesId, Vec<(GenomeId, DefaultGenome)>> {
    species
        .iter()
        .map(|species| {
            let members = sorted_members(species, config);
            let cutoff = ((config.reproduction.survival_threshold * members.len() as f64).ceil()
                as usize)
                .max(2)
                .min(members.len());
            (species.key, members.into_iter().take(cutoff).collect())
        })
        .collect()
}

fn sorted_members(species: &Species, config: &Config) -> Vec<(GenomeId, DefaultGenome)> {
    let mut members: Vec<(GenomeId, DefaultGenome)> = species
        .members
        .iter()
        .map(|(key, genome)| (*key, genome.clone()))
        .collect();
    let ascending = config.neat.fitness_criterion.is_min();
    members.sort_by(|a, b| {
        let fa = a.1.fitness.unwrap_or(0.0);
        let fb = b.1.fitness.unwrap_or(0.0);
        if ascending {
            fa.total_cmp(&fb).then_with(|| a.0.cmp(&b.0))
        } else {
            fb.total_cmp(&fa).then_with(|| a.0.cmp(&b.0))
        }
    });
    members
}

fn population_fitness_criterion_value(criterion: &FitnessCriterion, values: &[f64]) -> f64 {
    match criterion {
        FitnessCriterion::Min => values.iter().copied().reduce(f64::min).unwrap_or(0.0),
        FitnessCriterion::Mean => mean(values),
        FitnessCriterion::Max | FitnessCriterion::Other(_) => {
            values.iter().copied().reduce(f64::max).unwrap_or(0.0)
        }
    }
}
