use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::attributes::XorShiftRng;
use crate::bootstrap::{BootstrapError, BootstrapStrategy};
use crate::config::Config;
use crate::epoch::{Epoch, EpochStopReason, GenerationStats};
use crate::evaluator::BatchEvaluator;
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

pub type FitnessResult = Result<(), FitnessError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitnessError {
    message: String,
}

impl FitnessError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for FitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for FitnessError {}

impl From<String> for FitnessError {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for FitnessError {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PopulationError {
    Genome(GenomeError),
    Reproduction(ReproductionError),
    Bootstrap(BootstrapError),
    Checkpoint(PopulationCheckpointError),
    Fitness(FitnessError),
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
    pub last_generation_stats: Option<GenerationStats>,
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
            last_generation_stats: None,
            reproduction,
            reporters: ReporterSet::new(),
            checkpoint_sink: None,
            skip_first_evaluation: false,
            rng,
        })
    }

    pub fn new_with_bootstrap(
        config: Config,
        seed: u64,
        strategy: BootstrapStrategy,
    ) -> Result<Self, PopulationError> {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let mut reproduction = ReproductionState::new();
        let mut population = reproduction.create_new(&config, config.neat.pop_size, &mut rng)?;
        strategy.apply(&mut population, &config, &mut reproduction, &mut rng)?;
        let mut species = SpeciesSet::new();
        species.speciate(&config, &population, 0)?;

        Ok(Self {
            config,
            population,
            species,
            generation: 0,
            best_genome: None,
            last_generation_stats: None,
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
            last_generation_stats: None,
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

    pub fn last_generation_stats(&self) -> Option<&GenerationStats> {
        self.last_generation_stats.as_ref()
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
            if self.next_epoch(&mut fitness_function)?.should_stop() {
                break;
            }
        }

        if self.config.neat.no_fitness_termination {
            if let Some(best) = &self.best_genome {
                self.reporters
                    .found_solution(&self.config, self.generation, best);
            }
        }

        Ok(self.best_genome.clone())
    }

    pub fn run_with_evaluator<E>(
        &mut self,
        evaluator: &mut E,
        generations: Option<usize>,
    ) -> Result<Option<DefaultGenome>, PopulationError>
    where
        E: BatchEvaluator + ?Sized,
    {
        self.run(
            |genomes, config| evaluator.evaluate_population(genomes, config),
            generations,
        )
    }

    pub fn next_epoch_with_evaluator<E>(
        &mut self,
        evaluator: &mut E,
    ) -> Result<Epoch, PopulationError>
    where
        E: BatchEvaluator + ?Sized,
    {
        self.next_epoch(&mut |genomes, config| evaluator.evaluate_population(genomes, config))
    }

    pub fn next_epoch<F>(&mut self, fitness_function: &mut F) -> Result<Epoch, PopulationError>
    where
        F: FnMut(&mut BTreeMap<GenomeId, DefaultGenome>, &Config) -> FitnessResult,
    {
        let generation = self.generation;
        let reached_threshold = self.run_generation(fitness_function)?;
        let stop_reason = reached_threshold.then_some(EpochStopReason::FitnessThreshold);
        Ok(Epoch::new(
            generation,
            self.last_generation_stats.clone(),
            self.best_genome.clone(),
            stop_reason,
        ))
    }

    fn run_generation<F>(&mut self, fitness_function: &mut F) -> Result<bool, PopulationError>
    where
        F: FnMut(&mut BTreeMap<GenomeId, DefaultGenome>, &Config) -> FitnessResult,
    {
        self.last_generation_stats = None;
        self.reporters.start_generation(self.generation);

        if self.skip_first_evaluation {
            self.skip_first_evaluation = false;
        } else if self.evaluate_generation(fitness_function)? {
            return Ok(true);
        }

        self.reproduce_generation()?;
        self.species
            .speciate(&self.config, &self.population, self.generation)?;
        self.reporters
            .end_generation(&self.config, &self.population, &self.species);
        self.generation += 1;
        Ok(false)
    }

    fn evaluate_generation<F>(&mut self, fitness_function: &mut F) -> Result<bool, PopulationError>
    where
        F: FnMut(&mut BTreeMap<GenomeId, DefaultGenome>, &Config) -> FitnessResult,
    {
        fitness_function(&mut self.population, &self.config).map_err(PopulationError::Fitness)?;
        sync_species_members(&mut self.species, &self.population);
        let summary = PopulationFitnessSummary::from_population(&self.population, &self.config)?;
        let best = summary.best_genome.clone();
        self.reporters
            .post_evaluate(&self.config, &self.population, &self.species, &best);
        let stats = self.build_generation_stats(&summary);
        self.reporters.post_generation_stats(&stats);
        self.last_generation_stats = Some(stats);

        let improved = self
            .best_genome
            .as_ref()
            .map(|current| {
                is_better_fitness(
                    best.fitness.unwrap_or(0.0),
                    current.fitness.unwrap_or(0.0),
                    &self.config,
                )
            })
            .unwrap_or(true);
        if improved {
            self.best_genome = Some(best.clone());
        }
        self.reproduction.record_global_improvement(improved);

        self.save_generation_checkpoint()?;

        if !self.config.neat.no_fitness_termination
            && meets_threshold(summary.criterion_value, &self.config)
        {
            self.reporters
                .found_solution(&self.config, self.generation, &best);
            return Ok(true);
        }

        Ok(false)
    }

    fn build_generation_stats(&self, summary: &PopulationFitnessSummary) -> GenerationStats {
        let fitnesses: Vec<f64> = self.population.values().filter_map(|g| g.fitness).collect();
        GenerationStats::new(
            self.generation,
            self.population.len(),
            fitnesses.len(),
            self.species.species.len(),
            summary.best_genome.key,
            self.species.get_species_id(summary.best_genome.key),
            summary.best_genome.fitness.unwrap_or(0.0),
            crate::reporting::mean(&fitnesses),
            crate::reporting::stdev(&fitnesses),
            summary.criterion_value,
        )
    }

    fn save_generation_checkpoint(&mut self) -> Result<(), PopulationError> {
        let Some(checkpoint_sink) = self.checkpoint_sink.clone() else {
            return Ok(());
        };
        if !checkpoint_sink.should_save(self.generation) {
            return Ok(());
        }
        let original_skip = self.skip_first_evaluation;
        self.skip_first_evaluation = true;
        let result = checkpoint_sink.save_population(self);
        self.skip_first_evaluation = original_skip;
        result.map_err(PopulationError::Checkpoint)?;
        Ok(())
    }

    fn reproduce_generation(&mut self) -> Result<(), PopulationError> {
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
        Ok(())
    }
}

impl fmt::Display for PopulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Genome(err) => write!(f, "{err}"),
            Self::Reproduction(err) => write!(f, "{err}"),
            Self::Bootstrap(err) => write!(f, "{err}"),
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

impl From<BootstrapError> for PopulationError {
    fn from(value: BootstrapError) -> Self {
        Self::Bootstrap(value)
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
            PopulationFitnessSummaryError::InvalidFitness { genome_key, value } => {
                Self::Genome(GenomeError::InvalidFitness { genome_key, value })
            }
            PopulationFitnessSummaryError::NoBestGenome => Self::NoBestGenome,
        }
    }
}
