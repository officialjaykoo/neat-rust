use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::attributes::RandomSource;
use crate::config::{Config, SpeciesFitnessFunction};
use crate::evolution::SpawnPlan;
use crate::genome::{DefaultGenome, GenomeError};
use crate::ids::GenomeId;
use crate::innovation::InnovationTracker;
use crate::species::{Species, SpeciesSet};
use crate::stagnation::{species_fitness, DefaultStagnation};

#[derive(Debug, Clone, PartialEq)]
pub struct ReproductionState {
    pub genome_indexer: GenomeId,
    pub ancestors: BTreeMap<GenomeId, (Option<GenomeId>, Option<GenomeId>)>,
    pub innovation_tracker: InnovationTracker,
    pub generations_without_improvement: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReproductionError {
    Genome(GenomeError),
    EmptySpecies,
    SpawnConflict,
}

impl ReproductionState {
    pub fn new() -> Self {
        Self {
            genome_indexer: GenomeId::new(1),
            ancestors: BTreeMap::new(),
            innovation_tracker: InnovationTracker::new(),
            generations_without_improvement: 0,
        }
    }

    pub fn record_global_improvement(&mut self, improved: bool) {
        if improved {
            self.generations_without_improvement = 0;
        } else {
            self.generations_without_improvement =
                self.generations_without_improvement.saturating_add(1);
        }
    }

    pub fn create_new(
        &mut self,
        config: &Config,
        num_genomes: usize,
        rng: &mut impl RandomSource,
    ) -> Result<BTreeMap<GenomeId, DefaultGenome>, ReproductionError> {
        let mut population = BTreeMap::new();
        for _ in 0..num_genomes {
            let key = self.next_genome_key();
            let mut genome = DefaultGenome::new(key);
            genome.configure_new_with_innovation(
                &config.genome,
                &mut self.innovation_tracker,
                rng,
            )?;
            population.insert(key, genome);
            self.ancestors.insert(key, (None, None));
        }
        Ok(population)
    }

    pub fn reproduce(
        &mut self,
        config: &Config,
        species_set: &mut SpeciesSet,
        pop_size: usize,
        generation: usize,
        rng: &mut impl RandomSource,
    ) -> Result<BTreeMap<GenomeId, DefaultGenome>, ReproductionError> {
        self.innovation_tracker.reset_generation();
        let stagnation = DefaultStagnation::update(species_set, generation, config);
        let mut all_fitnesses = Vec::new();
        let mut remaining_species = Vec::new();

        for update in stagnation {
            if !update.stagnant {
                all_fitnesses.extend(
                    update
                        .species
                        .members
                        .values()
                        .filter_map(|member| member.fitness),
                );
                remaining_species.push(update.species);
            }
        }

        if remaining_species.is_empty() {
            species_set.species.clear();
            return Ok(BTreeMap::new());
        }

        adjust_fitnesses(&mut remaining_species, &all_fitnesses, config);
        let spawn_plan = SpawnPlan::build(config, &remaining_species, pop_size)?;
        let mutation_config = config
            .reproduction
            .adaptive_mutation
            .adapted_genome_config(&config.genome, self.generations_without_improvement);

        let mut new_population = BTreeMap::new();
        species_set.species.clear();
        for mut species in remaining_species.into_iter() {
            let species_key = species.key;
            let Some(plan_entry) = spawn_plan.entry(species_key) else {
                continue;
            };
            let mut spawn = plan_entry.spawn_quota.max(config.reproduction.elitism);
            let old_members = sorted_members(&species, config);
            species.members.clear();
            species_set.species.insert(species_key, species);

            for (genome_id, genome) in old_members.iter().take(config.reproduction.elitism) {
                new_population.insert(*genome_id, genome.clone());
                spawn = spawn.saturating_sub(1);
            }

            if spawn == 0 {
                continue;
            }

            if plan_entry.parent_pool.is_empty() {
                return Err(ReproductionError::EmptySpecies);
            }

            for _ in 0..spawn {
                let (parent1_id, parent1) = plan_entry.choose_parent(rng)?;
                let interspecies_crossover_prob =
                    config.reproduction.interspecies_crossover_prob.value();
                let (parent2_id, parent2) = if interspecies_crossover_prob > 0.0
                    && rng.next_f64() < interspecies_crossover_prob
                {
                    match spawn_plan.choose_interspecies_parent(species_key, rng)? {
                        Some(parent) => parent,
                        None => plan_entry.choose_parent(rng)?,
                    }
                } else {
                    plan_entry.choose_parent(rng)?
                };
                let child_key = self.next_genome_key();
                let mut child = DefaultGenome::new(child_key);
                child.configure_crossover(
                    parent1,
                    parent2,
                    &config.genome,
                    Some(&config.neat.fitness_criterion),
                    rng,
                )?;
                child.mutate_with_innovation(
                    &mutation_config,
                    &mut self.innovation_tracker,
                    rng,
                )?;
                new_population.insert(child_key, child);
                self.ancestors
                    .insert(child_key, (Some(*parent1_id), Some(*parent2_id)));
            }
        }

        Ok(new_population)
    }

    fn next_genome_key(&mut self) -> GenomeId {
        let key = self.genome_indexer;
        self.genome_indexer = self.genome_indexer.next();
        key
    }
}

impl Default for ReproductionState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ReproductionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Genome(err) => write!(f, "{err}"),
            Self::EmptySpecies => write!(f, "species has no available parents"),
            Self::SpawnConflict => write!(f, "could not adjust spawn counts to population size"),
        }
    }
}

impl Error for ReproductionError {}

impl From<GenomeError> for ReproductionError {
    fn from(value: GenomeError) -> Self {
        Self::Genome(value)
    }
}

pub fn compute_spawn(
    adjusted_fitnesses: &[f64],
    previous_sizes: &[usize],
    pop_size: usize,
    min_species_size: usize,
) -> Vec<usize> {
    let adjusted_sum: f64 = adjusted_fitnesses.iter().sum();
    let mut spawn_amounts = Vec::new();
    for (adjusted, previous_size) in adjusted_fitnesses.iter().zip(previous_sizes.iter()) {
        let target = if adjusted_sum > 0.0 {
            (min_species_size as f64).max((*adjusted / adjusted_sum) * pop_size as f64)
        } else {
            min_species_size as f64
        };
        let delta = (target - *previous_size as f64) * 0.5;
        let rounded = delta.round() as isize;
        let mut spawn = *previous_size as isize;
        if rounded.abs() > 0 {
            spawn += rounded;
        } else if delta > 0.0 {
            spawn += 1;
        } else if delta < 0.0 {
            spawn -= 1;
        }
        spawn_amounts.push(spawn.max(min_species_size as isize) as usize);
    }

    let total_spawn: usize = spawn_amounts.iter().sum();
    if total_spawn == 0 {
        return spawn_amounts;
    }
    let norm = pop_size as f64 / total_spawn as f64;
    spawn_amounts
        .into_iter()
        .map(|spawn| min_species_size.max((spawn as f64 * norm).round() as usize))
        .collect()
}

pub fn compute_spawn_proportional(
    adjusted_fitnesses: &[f64],
    pop_size: usize,
    min_species_size: usize,
) -> Vec<usize> {
    let adjusted_sum: f64 = adjusted_fitnesses.iter().sum();
    adjusted_fitnesses
        .iter()
        .map(|adjusted| {
            if adjusted_sum > 0.0 {
                min_species_size.max((adjusted / adjusted_sum * pop_size as f64).round() as usize)
            } else {
                min_species_size
            }
        })
        .collect()
}

pub fn adjust_spawn_exact(
    mut spawn_amounts: Vec<usize>,
    pop_size: usize,
    min_species_size: usize,
) -> Result<Vec<usize>, ReproductionError> {
    let species_count = spawn_amounts.len();
    if species_count * min_species_size > pop_size {
        return Err(ReproductionError::SpawnConflict);
    }

    while spawn_amounts.iter().sum::<usize>() < pop_size {
        let idx = spawn_amounts
            .iter()
            .enumerate()
            .min_by_key(|(_, value)| *value)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        spawn_amounts[idx] += 1;
    }
    while spawn_amounts.iter().sum::<usize>() > pop_size {
        let Some((idx, _)) = spawn_amounts
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > min_species_size)
            .max_by_key(|(_, value)| *value)
        else {
            return Err(ReproductionError::SpawnConflict);
        };
        spawn_amounts[idx] -= 1;
    }

    Ok(spawn_amounts)
}

fn adjust_fitnesses(species: &mut [Species], all_fitnesses: &[f64], config: &Config) {
    if all_fitnesses.is_empty() {
        for species in species {
            species.adjusted_fitness = Some(0.0);
        }
        return;
    }

    if config.reproduction.fitness_sharing.is_canonical() {
        let mut adjusted_values = Vec::new();
        for species in species.iter_mut() {
            let member_fitnesses: Vec<f64> = species
                .members
                .values()
                .filter_map(|member| member.fitness)
                .collect();
            let adjusted = species_fitness(&SpeciesFitnessFunction::Mean, &member_fitnesses);
            species.adjusted_fitness = Some(adjusted);
            adjusted_values.push(adjusted);
        }
        let min_adjusted = adjusted_values.into_iter().reduce(f64::min).unwrap_or(0.0);
        if min_adjusted < 0.0 {
            let shift = min_adjusted.abs() + 1.0e-6;
            for species in species {
                species.adjusted_fitness = species.adjusted_fitness.map(|value| value + shift);
            }
        }
        return;
    }

    let min_fitness = all_fitnesses
        .iter()
        .copied()
        .reduce(f64::min)
        .unwrap_or(0.0);
    let max_fitness = all_fitnesses
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let fitness_range = 1.0_f64.max(max_fitness - min_fitness);
    for species in species {
        let member_fitnesses: Vec<f64> = species
            .members
            .values()
            .filter_map(|member| member.fitness)
            .collect();
        let mean_fitness = species_fitness(&SpeciesFitnessFunction::Mean, &member_fitnesses);
        let mut adjusted = (mean_fitness - min_fitness) / fitness_range;
        if config.neat.fitness_criterion.is_min() {
            adjusted = 1.0 - adjusted;
        }
        species.adjusted_fitness = Some(adjusted);
    }
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
