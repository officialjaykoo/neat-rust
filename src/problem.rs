use std::collections::BTreeMap;

use crate::config::Config;
use crate::evaluator::{BatchEvaluator, FitnessEvaluator};
use crate::genome::DefaultGenome;
use crate::ids::GenomeId;
use crate::population::{FitnessError, FitnessResult};

pub trait GenomeProblem {
    type Error: Into<FitnessError>;

    fn evaluate(
        &mut self,
        genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, Self::Error>;
}

pub trait PopulationProblem {
    type Error: Into<FitnessError>;

    fn evaluate_all(
        &mut self,
        genomes: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProblemEvaluator<P> {
    problem: P,
}

impl<P> ProblemEvaluator<P> {
    pub fn new(problem: P) -> Self {
        Self { problem }
    }

    pub fn problem(&self) -> &P {
        &self.problem
    }

    pub fn problem_mut(&mut self) -> &mut P {
        &mut self.problem
    }

    pub fn into_problem(self) -> P {
        self.problem
    }
}

impl<P> FitnessEvaluator for ProblemEvaluator<P>
where
    P: GenomeProblem,
{
    fn evaluate_genome(
        &mut self,
        genome_id: GenomeId,
        genome: &DefaultGenome,
        config: &Config,
    ) -> Result<f64, FitnessError> {
        self.problem
            .evaluate(genome_id, genome, config)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchProblemEvaluator<P> {
    problem: P,
}

impl<P> BatchProblemEvaluator<P> {
    pub fn new(problem: P) -> Self {
        Self { problem }
    }

    pub fn problem(&self) -> &P {
        &self.problem
    }

    pub fn problem_mut(&mut self) -> &mut P {
        &mut self.problem
    }

    pub fn into_problem(self) -> P {
        self.problem
    }
}

impl<P> BatchEvaluator for BatchProblemEvaluator<P>
where
    P: PopulationProblem,
{
    fn evaluate_population(
        &mut self,
        genomes: &mut BTreeMap<GenomeId, DefaultGenome>,
        config: &Config,
    ) -> FitnessResult {
        self.problem
            .evaluate_all(genomes, config)
            .map_err(Into::into)
    }
}
