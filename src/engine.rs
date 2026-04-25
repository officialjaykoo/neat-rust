use crate::epoch::{Epoch, EpochStopReason};
use crate::evaluator::BatchEvaluator;
use crate::genome::DefaultGenome;
use crate::population::{Population, PopulationError};

pub trait Engine {
    type Epoch;
    type Error;

    fn next_epoch(&mut self) -> Result<Option<Self::Epoch>, Self::Error>;

    fn run_until<F>(&mut self, mut should_stop: F) -> Result<Option<Self::Epoch>, Self::Error>
    where
        F: FnMut(&Self::Epoch) -> bool,
    {
        let mut last_epoch = None;
        while let Some(epoch) = self.next_epoch()? {
            let should_stop_now = should_stop(&epoch);
            last_epoch = Some(epoch);
            if should_stop_now {
                break;
            }
        }
        Ok(last_epoch)
    }
}

pub struct EvolutionEngine<E> {
    population: Population,
    evaluator: E,
    generation_limit: Option<usize>,
    completed_generations: usize,
    finished: bool,
}

impl<E> EvolutionEngine<E> {
    pub fn new(population: Population, evaluator: E) -> Self {
        Self {
            population,
            evaluator,
            generation_limit: None,
            completed_generations: 0,
            finished: false,
        }
    }

    pub fn with_generation_limit(mut self, generations: usize) -> Self {
        self.generation_limit = Some(generations);
        self
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn population_mut(&mut self) -> &mut Population {
        &mut self.population
    }

    pub fn evaluator(&self) -> &E {
        &self.evaluator
    }

    pub fn evaluator_mut(&mut self) -> &mut E {
        &mut self.evaluator
    }

    pub fn into_parts(self) -> (Population, E) {
        (self.population, self.evaluator)
    }

    pub fn run(&mut self) -> Result<Option<DefaultGenome>, PopulationError>
    where
        E: BatchEvaluator,
    {
        if self.population.config.neat.no_fitness_termination && self.generation_limit.is_none() {
            return Err(PopulationError::NoGenerationalLimit);
        }

        while self.next_epoch()?.is_some() {}
        if self.population.config.neat.no_fitness_termination {
            if let Some(best) = self.population.best_genome.clone() {
                self.population.reporters.found_solution(
                    &self.population.config,
                    self.population.generation,
                    &best,
                );
            }
        }
        Ok(self.population.best_genome.clone())
    }
}

impl<E> Engine for EvolutionEngine<E>
where
    E: BatchEvaluator,
{
    type Epoch = Epoch;
    type Error = PopulationError;

    fn next_epoch(&mut self) -> Result<Option<Self::Epoch>, Self::Error> {
        if self.finished {
            return Ok(None);
        }
        if self
            .generation_limit
            .map(|limit| self.completed_generations >= limit)
            .unwrap_or(false)
        {
            self.finished = true;
            return Ok(None);
        }

        let mut epoch = self
            .population
            .next_epoch_with_evaluator(&mut self.evaluator)?;
        self.completed_generations += 1;

        if epoch.should_stop() {
            self.finished = true;
        } else if self
            .generation_limit
            .map(|limit| self.completed_generations >= limit)
            .unwrap_or(false)
        {
            epoch.stop_reason = Some(EpochStopReason::GenerationLimit);
            self.finished = true;
        }

        Ok(Some(epoch))
    }
}
