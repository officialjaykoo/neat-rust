use crate::config::Config;
use crate::ids::SpeciesId;
use crate::species::{Species, SpeciesSet};

#[derive(Debug, Clone, PartialEq)]
pub struct StagnationUpdate {
    pub species_id: SpeciesId,
    pub species: Species,
    pub stagnant: bool,
    pub stagnant_time: usize,
    pub protected_by_elitism: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DefaultStagnation;

impl DefaultStagnation {
    pub fn update(
        species_set: &mut SpeciesSet,
        generation: usize,
        config: &Config,
    ) -> Vec<StagnationUpdate> {
        let mut species_data = Vec::new();
        for (species_id, species) in &mut species_set.species {
            let previous_fitness = if species.fitness_history.is_empty() {
                worst_fitness(config)
            } else if config.neat.fitness_criterion.is_min() {
                species
                    .fitness_history
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .unwrap_or_else(|| worst_fitness(config))
            } else {
                species
                    .fitness_history
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .unwrap_or_else(|| worst_fitness(config))
            };
            let fitnesses = species.get_fitnesses();
            let fitness = species_fitness(&config.stagnation.species_fitness_func, &fitnesses);
            species.fitness = Some(fitness);
            species.fitness_history.push(fitness);
            species.adjusted_fitness = None;
            if is_better_fitness(fitness, previous_fitness, config) {
                species.last_improved = generation;
            }
            species_data.push((*species_id, species.clone()));
        }

        let reverse = config.neat.fitness_criterion.is_min();
        species_data.sort_by(|a, b| {
            let left = a.1.fitness.unwrap_or_else(|| worst_fitness(config));
            let right = b.1.fitness.unwrap_or_else(|| worst_fitness(config));
            if reverse {
                right.total_cmp(&left)
            } else {
                left.total_cmp(&right)
            }
        });

        let mut result = Vec::new();
        let mut num_non_stagnant = species_data.len();
        for (idx, (species_id, species)) in species_data.into_iter().enumerate() {
            let stagnant_time = generation.saturating_sub(species.last_improved);
            let mut stagnant = false;
            if num_non_stagnant > config.stagnation.species_elitism {
                stagnant = stagnant_time >= config.stagnation.max_stagnation;
            }
            let protected_by_elitism =
                (species_set.species.len() - idx) <= config.stagnation.species_elitism;
            if protected_by_elitism {
                stagnant = false;
            }
            if stagnant {
                num_non_stagnant = num_non_stagnant.saturating_sub(1);
            }
            result.push(StagnationUpdate {
                species_id,
                species,
                stagnant,
                stagnant_time,
                protected_by_elitism,
            });
        }

        result
    }
}

pub fn species_fitness(criterion: &crate::config::SpeciesFitnessFunction, values: &[f64]) -> f64 {
    criterion.evaluate(values)
}

pub fn is_better_fitness(candidate: f64, previous: f64, config: &Config) -> bool {
    config.neat.fitness_criterion.is_better(candidate, previous)
}

pub fn worst_fitness(config: &Config) -> f64 {
    config.neat.fitness_criterion.worst_value()
}
