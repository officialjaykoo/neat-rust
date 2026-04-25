use std::collections::BTreeMap;
use std::time::Instant;

use crate::config::Config;
use crate::epoch::GenerationStats;
use crate::genome::DefaultGenome;
use crate::ids::{GenomeId, SpeciesId};
use crate::species::{Species, SpeciesSet};

pub trait Reporter {
    fn start_generation(&mut self, _generation: usize) {}

    fn end_generation(
        &mut self,
        _config: &Config,
        _population: &BTreeMap<GenomeId, DefaultGenome>,
        _species_set: &SpeciesSet,
    ) {
    }

    fn post_evaluate(
        &mut self,
        _config: &Config,
        _population: &BTreeMap<GenomeId, DefaultGenome>,
        _species: &SpeciesSet,
        _best_genome: &DefaultGenome,
    ) {
    }

    fn post_generation_stats(&mut self, _stats: &GenerationStats) {}

    fn post_reproduction(
        &mut self,
        _config: &Config,
        _population: &BTreeMap<GenomeId, DefaultGenome>,
        _species: &SpeciesSet,
    ) {
    }

    fn complete_extinction(&mut self) {}

    fn found_solution(&mut self, _config: &Config, _generation: usize, _best: &DefaultGenome) {}

    fn species_stagnant(&mut self, _species_id: SpeciesId, _species: &Species) {}

    fn info(&mut self, _message: &str) {}
}

#[derive(Default)]
pub struct ReporterSet {
    reporters: Vec<Box<dyn Reporter>>,
}

impl ReporterSet {
    pub fn new() -> Self {
        Self {
            reporters: Vec::new(),
        }
    }

    pub fn add(&mut self, reporter: Box<dyn Reporter>) {
        self.reporters.push(reporter);
    }

    pub fn clear(&mut self) {
        self.reporters.clear();
    }

    pub fn start_generation(&mut self, generation: usize) {
        for reporter in &mut self.reporters {
            reporter.start_generation(generation);
        }
    }

    pub fn end_generation(
        &mut self,
        config: &Config,
        population: &BTreeMap<GenomeId, DefaultGenome>,
        species_set: &SpeciesSet,
    ) {
        for reporter in &mut self.reporters {
            reporter.end_generation(config, population, species_set);
        }
    }

    pub fn post_evaluate(
        &mut self,
        config: &Config,
        population: &BTreeMap<GenomeId, DefaultGenome>,
        species: &SpeciesSet,
        best_genome: &DefaultGenome,
    ) {
        for reporter in &mut self.reporters {
            reporter.post_evaluate(config, population, species, best_genome);
        }
    }

    pub fn post_generation_stats(&mut self, stats: &GenerationStats) {
        for reporter in &mut self.reporters {
            reporter.post_generation_stats(stats);
        }
    }

    pub fn post_reproduction(
        &mut self,
        config: &Config,
        population: &BTreeMap<GenomeId, DefaultGenome>,
        species: &SpeciesSet,
    ) {
        for reporter in &mut self.reporters {
            reporter.post_reproduction(config, population, species);
        }
    }

    pub fn complete_extinction(&mut self) {
        for reporter in &mut self.reporters {
            reporter.complete_extinction();
        }
    }

    pub fn found_solution(&mut self, config: &Config, generation: usize, best: &DefaultGenome) {
        for reporter in &mut self.reporters {
            reporter.found_solution(config, generation, best);
        }
    }

    pub fn species_stagnant(&mut self, species_id: SpeciesId, species: &Species) {
        for reporter in &mut self.reporters {
            reporter.species_stagnant(species_id, species);
        }
    }

    pub fn info(&mut self, message: &str) {
        for reporter in &mut self.reporters {
            reporter.info(message);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StdOutReporter {
    pub show_species_detail: bool,
    generation: Option<usize>,
    generation_start_time: Option<Instant>,
    generation_times: Vec<f64>,
    num_extinctions: usize,
}

impl StdOutReporter {
    pub fn new(show_species_detail: bool) -> Self {
        Self {
            show_species_detail,
            generation: None,
            generation_start_time: None,
            generation_times: Vec::new(),
            num_extinctions: 0,
        }
    }
}

impl Reporter for StdOutReporter {
    fn start_generation(&mut self, generation: usize) {
        self.generation = Some(generation);
        self.generation_start_time = Some(Instant::now());
        println!();
        println!(" ****** Running generation {generation} ****** ");
        println!();
    }

    fn end_generation(
        &mut self,
        _config: &Config,
        _population: &BTreeMap<GenomeId, DefaultGenome>,
        _species_set: &SpeciesSet,
    ) {
        let elapsed = self
            .generation_start_time
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        self.generation_times.push(elapsed);
        if self.generation_times.len() > 10 {
            let drain_count = self.generation_times.len() - 10;
            self.generation_times.drain(0..drain_count);
        }
        let average = mean(&self.generation_times);
        println!("Total extinctions: {}", self.num_extinctions);
        if self.generation_times.len() > 1 {
            println!("Generation time: {elapsed:.3} sec ({average:.3} average)");
        } else {
            println!("Generation time: {elapsed:.3} sec");
        }
    }

    fn post_evaluate(
        &mut self,
        _config: &Config,
        population: &BTreeMap<GenomeId, DefaultGenome>,
        species: &SpeciesSet,
        best_genome: &DefaultGenome,
    ) {
        let population_count = population.len();
        let species_count = species.species.len();
        if self.show_species_detail {
            println!("Population of {population_count} members in {species_count} species:");
            println!("   ID   age  size   fitness   adj fit  stag");
            println!("  ====  ===  ====  =========  =======  ====");
            for (species_id, species) in &species.species {
                let generation = self.generation.unwrap_or_default();
                let age = generation.saturating_sub(species.created);
                let size = species.members.len();
                let fitness = species
                    .fitness
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "--".to_string());
                let adjusted_fitness = species
                    .adjusted_fitness
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "--".to_string());
                let stagnation = generation.saturating_sub(species.last_improved);
                println!(
                    "  {species_id:>4}  {age:>3}  {size:>4}  {fitness:>9}  {adjusted_fitness:>7}  {stagnation:>4}"
                );
            }
        } else {
            println!("Population of {population_count} members in {species_count} species");
        }

        let fitnesses: Vec<f64> = population.values().filter_map(|g| g.fitness).collect();
        println!(
            "Population's average fitness: {:.5} stdev: {:.5}",
            mean(&fitnesses),
            stdev(&fitnesses)
        );
        println!(
            "Best fitness: {:.5} - size: {:?} - species {} - id {}",
            best_genome.fitness.unwrap_or(0.0),
            best_genome.size(),
            species.get_species_id(best_genome.key).unwrap_or_default(),
            best_genome.key
        );
    }

    fn post_generation_stats(&mut self, stats: &GenerationStats) {
        println!(
            "Generation stats: evaluated {}/{} - best {:.5} - mean {:.5} - criterion {:.5}",
            stats.evaluated_count,
            stats.population_size,
            stats.best_fitness,
            stats.mean_fitness,
            stats.criterion_value
        );
    }

    fn complete_extinction(&mut self) {
        self.num_extinctions += 1;
        println!("All species extinct.");
    }

    fn found_solution(&mut self, _config: &Config, generation: usize, best: &DefaultGenome) {
        println!(
            "Best individual in generation {generation} meets fitness threshold - complexity: {:?}",
            best.size()
        );
    }

    fn species_stagnant(&mut self, species_id: SpeciesId, species: &Species) {
        if self.show_species_detail {
            println!(
                "Species {} with {} members is stagnated: removing it",
                species_id,
                species.members.len()
            );
        }
    }

    fn info(&mut self, message: &str) {
        println!("{message}");
    }
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub fn stdev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

pub fn median2(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() <= 2 {
        return mean(values);
    }

    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        let index = values.len() / 2;
        (values[index - 1] + values[index]) / 2.0
    }
}
