use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::attributes::XorShiftRng;
use crate::config::Config;
use crate::evolution::{
    sync_species_members, PopulationCheckpointError, PopulationCheckpointSink,
    PopulationFitnessSummary, PopulationFitnessSummaryError,
};
use crate::genome::{DefaultGenome, GenomeError};
use crate::ids::GenomeId;
use crate::reporting::{Reporter, ReporterSet};
use crate::reproduction::{ReproductionError, ReproductionState};
use crate::species::SpeciesSet;
use crate::stagnation::is_better_fitness;

pub type FitnessResult = Result<(), String>;

#[derive(Debug, Clone, PartialEq)]
pub enum PopulationError {
    Genome(GenomeError),
    Reproduction(ReproductionError),
    Checkpoint(PopulationCheckpointError),
    Fitness(String),
    FitnessNotAssigned(GenomeId),
    NoGenerationalLimit,
    CompleteExtinction,
    NoBestGenome,
}

pub struct Population {
    pub config: Config,
    pub population: BTreeMap<GenomeId, DefaultGenome>,
    pub species: SpeciesSet,
    pub generation: usize,
    pub best_genome: Option<DefaultGenome>,
    pub reproduction: ReproductionState,
    pub reporters: ReporterSet,
    pub checkpoint_sink: Option<Arc<dyn PopulationCheckpointSink>>,
    pub skip_first_evaluation: bool,
    rng: XorShiftRng,
}

impl Population {
    pub fn new(config: Config, seed: u64) -> Result<Self, PopulationError> {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let mut reproduction = ReproductionState::new();
        let population = reproduction.create_new(&config, config.neat.pop_size, &mut rng)?;
        let mut species = SpeciesSet::new();
        species.speciate(&config, &population, 0)?;

        Ok(Self {
            config,
            population,
            species,
            generation: 0,
            best_genome: None,
            reproduction,
            reporters: ReporterSet::new(),
            checkpoint_sink: None,
            skip_first_evaluation: false,
            rng,
        })
    }

    pub(crate) fn from_checkpoint_parts(
        config: Config,
        population: BTreeMap<GenomeId, DefaultGenome>,
        species: SpeciesSet,
        generation: usize,
        best_genome: Option<DefaultGenome>,
        reproduction: ReproductionState,
        skip_first_evaluation: bool,
        rng: XorShiftRng,
    ) -> Self {
        Self {
            config,
            population,
            species,
            generation,
            best_genome,
            reproduction,
            reporters: ReporterSet::new(),
            checkpoint_sink: None,
            skip_first_evaluation,
            rng,
        }
    }

    pub(crate) fn rng_state(&self) -> u64 {
        self.rng.state()
    }

    pub fn add_reporter(&mut self, reporter: Box<dyn Reporter>) {
        self.reporters.add(reporter);
    }

    pub fn run<F>(
        &mut self,
        mut fitness_function: F,
        generations: Option<usize>,
    ) -> Result<Option<DefaultGenome>, PopulationError>
    where
        F: FnMut(&mut BTreeMap<GenomeId, DefaultGenome>, &Config) -> FitnessResult,
    {
        if self.config.neat.no_fitness_termination && generations.is_none() {
            return Err(PopulationError::NoGenerationalLimit);
        }

        let mut completed = 0;
        while generations.map(|limit| completed < limit).unwrap_or(true) {
            completed += 1;
            self.reporters.start_generation(self.generation);

            if self.skip_first_evaluation {
                self.skip_first_evaluation = false;
            } else {
                fitness_function(&mut self.population, &self.config)
                    .map_err(PopulationError::Fitness)?;
                sync_species_members(&mut self.species, &self.population);
                let summary =
                    PopulationFitnessSummary::from_population(&self.population, &self.config)?;
                let best = summary.best_genome.clone();
                self.reporters
                    .post_evaluate(&self.config, &self.population, &self.species, &best);

                if self
                    .best_genome
                    .as_ref()
                    .map(|current| {
                        is_better_fitness(
                            best.fitness.unwrap_or(0.0),
                            current.fitness.unwrap_or(0.0),
                            &self.config,
                        )
                    })
                    .unwrap_or(true)
                {
                    self.best_genome = Some(best.clone());
                }

                if let Some(checkpoint_sink) = self.checkpoint_sink.clone() {
                    if checkpoint_sink.should_save(self.generation) {
                        let original_skip = self.skip_first_evaluation;
                        self.skip_first_evaluation = true;
                        let result = checkpoint_sink.save_population(self);
                        self.skip_first_evaluation = original_skip;
                        result.map_err(PopulationError::Checkpoint)?;
                    }
                }

                if !self.config.neat.no_fitness_termination {
                    if meets_threshold(summary.criterion_value, &self.config) {
                        self.reporters
                            .found_solution(&self.config, self.generation, &best);
                        break;
                    }
                }
            }

            self.population = self.reproduction.reproduce(
                &self.config,
                &mut self.species,
                self.config.neat.pop_size,
                self.generation,
                &mut self.rng,
            )?;
            self.reporters
                .post_reproduction(&self.config, &self.population, &self.species);

            if self.species.species.is_empty() {
                self.reporters.complete_extinction();
                if self.config.neat.reset_on_extinction {
                    self.population = self.reproduction.create_new(
                        &self.config,
                        self.config.neat.pop_size,
                        &mut self.rng,
                    )?;
                } else {
                    return Err(PopulationError::CompleteExtinction);
                }
            }

            self.species
                .speciate(&self.config, &self.population, self.generation)?;
            self.reporters
                .end_generation(&self.config, &self.population, &self.species);
            self.generation += 1;
        }

        if self.config.neat.no_fitness_termination {
            if let Some(best) = &self.best_genome {
                self.reporters
                    .found_solution(&self.config, self.generation, best);
            }
        }

        Ok(self.best_genome.clone())
    }
}

impl fmt::Display for PopulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Genome(err) => write!(f, "{err}"),
            Self::Reproduction(err) => write!(f, "{err}"),
            Self::Checkpoint(err) => write!(f, "{err}"),
            Self::Fitness(err) => write!(f, "fitness function failed: {err}"),
            Self::FitnessNotAssigned(key) => write!(f, "fitness not assigned to genome {key}"),
            Self::NoGenerationalLimit => {
                write!(
                    f,
                    "cannot run without generation limit and fitness termination"
                )
            }
            Self::CompleteExtinction => write!(f, "complete extinction"),
            Self::NoBestGenome => write!(f, "no best genome found"),
        }
    }
}

impl Error for PopulationError {}

impl From<ReproductionError> for PopulationError {
    fn from(value: ReproductionError) -> Self {
        Self::Reproduction(value)
    }
}

impl From<PopulationCheckpointError> for PopulationError {
    fn from(value: PopulationCheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<GenomeError> for PopulationError {
    fn from(value: GenomeError) -> Self {
        Self::Genome(value)
    }
}

fn meets_threshold(value: f64, config: &Config) -> bool {
    config.meets_threshold(value)
}

impl From<PopulationFitnessSummaryError> for PopulationError {
    fn from(value: PopulationFitnessSummaryError) -> Self {
        match value {
            PopulationFitnessSummaryError::FitnessNotAssigned(key) => Self::FitnessNotAssigned(key),
            PopulationFitnessSummaryError::NoBestGenome => Self::NoBestGenome,
        }
    }
}
