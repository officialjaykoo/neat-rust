use std::collections::BTreeMap;

use crate::config::Config;
use crate::genome::DefaultGenome;
use crate::ids::GenomeId;
use crate::population::{FitnessError, FitnessResult};

/// Evaluates one genome and returns its fitness.
pub trait FitnessEvaluator {
    fn evaluate_genome(
        &mut self,
        genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError>;
}

/// Evaluates a whole population and writes fitness values onto genomes.
pub trait BatchEvaluator {
    fn evaluate_population(
        &mut self,
        genomes: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> FitnessResult;
}

impl<T> BatchEvaluator for T
where
    T: FitnessEvaluator,
{
    fn evaluate_population(
        &mut self,
        genomes: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> FitnessResult {
        for (genome_id, genome) in genomes {
            let fitness = self.evaluate_genome(*genome_id, genome, config)?;
            genome.fitness = Some(fitness);
        }
        Ok(())
    }
}
