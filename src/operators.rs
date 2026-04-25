use crate::attributes::RandomSource;
use crate::config::{FitnessCriterion, GenomeConfig};
use crate::genome::{DefaultGenome, GenomeError};
use crate::innovation::InnovationTracker;

pub trait CrossoverOperator {
    fn crossover(
        &self,
        child: &mut DefaultGenome,
        parent1: &DefaultGenome,
        parent2: &DefaultGenome,
        config: &GenomeConfig,
        fitness_criterion: Option<&FitnessCriterion>,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DefaultCrossoverOperator;

impl CrossoverOperator for DefaultCrossoverOperator {
    fn crossover(
        &self,
        child: &mut DefaultGenome,
        parent1: &DefaultGenome,
        parent2: &DefaultGenome,
        config: &GenomeConfig,
        fitness_criterion: Option<&FitnessCriterion>,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        child.configure_crossover(parent1, parent2, config, fitness_criterion, rng)
    }
}

pub trait MutationOperator {
    fn mutate(
        &self,
        genome: &mut DefaultGenome,
        config: &GenomeConfig,
        innovation_tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DefaultMutationOperator;

impl MutationOperator for DefaultMutationOperator {
    fn mutate(
        &self,
        genome: &mut DefaultGenome,
        config: &GenomeConfig,
        innovation_tracker: &mut InnovationTracker,
        rng: &mut impl RandomSource,
    ) -> Result<(), GenomeError> {
        genome.mutate_with_innovation(config, innovation_tracker, rng)
    }
}
