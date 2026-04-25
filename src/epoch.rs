use crate::genome::DefaultGenome;
use crate::ids::{GenomeId, SpeciesId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochStopReason {
    FitnessThreshold,
    GenerationLimit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStats {
    pub generation: usize,
    pub population_size: usize,
    pub evaluated_count: usize,
    pub species_count: usize,
    pub best_genome_id: GenomeId,
    pub best_species_id: Option<SpeciesId>,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub stdev_fitness: f64,
    pub criterion_value: f64,
}

impl GenerationStats {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        generation: usize,
        population_size: usize,
        evaluated_count: usize,
        species_count: usize,
        best_genome_id: GenomeId,
        best_species_id: Option<SpeciesId>,
        best_fitness: f64,
        mean_fitness: f64,
        stdev_fitness: f64,
        criterion_value: f64,
    ) -> Self {
        Self {
            generation,
            population_size,
            evaluated_count,
            species_count,
            best_genome_id,
            best_species_id,
            best_fitness,
            mean_fitness,
            stdev_fitness,
            criterion_value,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Epoch {
    pub generation: usize,
    pub stats: Option<GenerationStats>,
    pub best_genome: Option<DefaultGenome>,
    pub stop_reason: Option<EpochStopReason>,
}

impl Epoch {
    pub fn new(
        generation: usize,
        stats: Option<GenerationStats>,
        best_genome: Option<DefaultGenome>,
        stop_reason: Option<EpochStopReason>,
    ) -> Self {
        Self {
            generation,
            stats,
            best_genome,
            stop_reason,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.stop_reason.is_some()
    }
}
