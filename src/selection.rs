use crate::attributes::RandomSource;
use crate::config::Config;
use crate::genome::DefaultGenome;
use crate::ids::GenomeId;
use crate::reproduction::ReproductionError;
use crate::species::Species;

pub trait ParentSelector {
    fn select<'a>(
        &self,
        parent_pool: &'a [(GenomeId, DefaultGenome)],
        rng: &mut impl RandomSource,
    ) -> Result<(&'a GenomeId, &'a DefaultGenome), ReproductionError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UniformParentSelector;

impl ParentSelector for UniformParentSelector {
    fn select<'a>(
        &self,
        parent_pool: &'a [(GenomeId, DefaultGenome)],
        rng: &mut impl RandomSource,
    ) -> Result<(&'a GenomeId, &'a DefaultGenome), ReproductionError> {
        let index = rng
            .next_index(parent_pool.len())
            .ok_or(ReproductionError::EmptySpecies)?;
        Ok((&parent_pool[index].0, &parent_pool[index].1))
    }
}

pub trait SurvivalSelector {
    fn survivors(&self, species: &Species, config: &Config) -> Vec<(GenomeId, DefaultGenome)>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TruncationSurvivalSelector;

impl SurvivalSelector for TruncationSurvivalSelector {
    fn survivors(&self, species: &Species, config: &Config) -> Vec<(GenomeId, DefaultGenome)> {
        let mut members = sorted_members(species, config);
        let cutoff =
            ((config.reproduction.survival_threshold.value() * members.len() as f64).ceil()
                as usize)
                .max(2)
                .min(members.len());
        members.truncate(cutoff);
        members
    }
}

pub fn sorted_members(species: &Species, config: &Config) -> Vec<(GenomeId, DefaultGenome)> {
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
