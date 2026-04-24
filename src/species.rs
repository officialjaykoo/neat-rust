use std::collections::BTreeMap;

use crate::config::Config;
use crate::evolution::SpeciesAssignment;
use crate::genome::{DefaultGenome, GenomeError};
use crate::ids::{GenomeId, SpeciesId};

#[derive(Debug, Clone, PartialEq)]
pub struct Species {
    pub key: SpeciesId,
    pub created: usize,
    pub last_improved: usize,
    pub representative: Option<DefaultGenome>,
    pub members: BTreeMap<GenomeId, DefaultGenome>,
    pub fitness: Option<f64>,
    pub adjusted_fitness: Option<f64>,
    pub fitness_history: Vec<f64>,
}

impl Species {
    pub fn new(key: impl Into<SpeciesId>, generation: usize) -> Self {
        Self {
            key: key.into(),
            created: generation,
            last_improved: generation,
            representative: None,
            members: BTreeMap::new(),
            fitness: None,
            adjusted_fitness: None,
            fitness_history: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        representative: DefaultGenome,
        members: BTreeMap<GenomeId, DefaultGenome>,
    ) {
        self.representative = Some(representative);
        self.members = members;
    }

    pub fn get_fitnesses(&self) -> Vec<f64> {
        self.members
            .values()
            .filter_map(|genome| genome.fitness)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenomeDistanceCache<'a> {
    pub distances: BTreeMap<(GenomeId, GenomeId), f64>,
    pub hits: usize,
    pub misses: usize,
    config: &'a Config,
}

impl<'a> GenomeDistanceCache<'a> {
    pub fn new(config: &'a Config) -> Self {
        Self {
            distances: BTreeMap::new(),
            hits: 0,
            misses: 0,
            config,
        }
    }

    pub fn distance(
        &mut self,
        genome0: &DefaultGenome,
        genome1: &DefaultGenome,
    ) -> Result<f64, GenomeError> {
        let key = normalized_distance_key(genome0.key, genome1.key);
        if let Some(distance) = self.distances.get(&key) {
            self.hits += 1;
            return Ok(*distance);
        }

        let distance = genome0.distance(genome1, &self.config.genome)?;
        self.distances.insert(key, distance);
        self.misses += 1;
        Ok(distance)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesSet {
    pub species: BTreeMap<SpeciesId, Species>,
    pub genome_to_species: BTreeMap<GenomeId, SpeciesId>,
    pub compatibility_threshold: Option<f64>,
    next_species_key: SpeciesId,
}

impl SpeciesSet {
    pub fn new() -> Self {
        Self {
            species: BTreeMap::new(),
            genome_to_species: BTreeMap::new(),
            compatibility_threshold: None,
            next_species_key: SpeciesId::new(1),
        }
    }

    pub fn from_parts(
        species: BTreeMap<SpeciesId, Species>,
        genome_to_species: BTreeMap<GenomeId, SpeciesId>,
        next_species_key: SpeciesId,
        compatibility_threshold: Option<f64>,
    ) -> Self {
        Self {
            species,
            genome_to_species,
            compatibility_threshold,
            next_species_key: SpeciesId::new(next_species_key.raw().max(1)),
        }
    }

    pub fn next_species_key(&self) -> SpeciesId {
        self.next_species_key
    }

    pub fn get_species_id(&self, individual_id: impl Into<GenomeId>) -> Option<SpeciesId> {
        let individual_id = individual_id.into();
        self.genome_to_species.get(&individual_id).copied()
    }

    pub fn get_species(&self, individual_id: impl Into<GenomeId>) -> Option<&Species> {
        self.get_species_id(individual_id)
            .and_then(|species_id| self.species.get(&species_id))
    }

    pub fn speciate(
        &mut self,
        config: &Config,
        population: &BTreeMap<GenomeId, DefaultGenome>,
        generation: usize,
    ) -> Result<(), GenomeError> {
        let compatibility_threshold = self
            .compatibility_threshold
            .unwrap_or(config.species_set.compatibility_threshold);
        self.compatibility_threshold = Some(compatibility_threshold);
        let mut unspeciated: Vec<GenomeId> = population.keys().copied().collect();
        let mut distances = GenomeDistanceCache::new(config);
        let mut assignments: BTreeMap<SpeciesId, SpeciesAssignment> = BTreeMap::new();

        for (species_id, species) in &self.species {
            let Some(representative) = &species.representative else {
                continue;
            };
            let mut candidates = Vec::new();
            for genome_id in &unspeciated {
                let genome = &population[genome_id];
                candidates.push((distances.distance(representative, genome)?, *genome_id));
            }
            if let Some((_, representative_id)) =
                candidates.into_iter().min_by(|a, b| a.0.total_cmp(&b.0))
            {
                let mut assignment = SpeciesAssignment::staged(representative_id);
                assignment.add_member(representative_id);
                assignments.insert(*species_id, assignment);
                unspeciated.retain(|genome_id| *genome_id != representative_id);
            }
        }

        while let Some(genome_id) = unspeciated.first().copied() {
            let genome = &population[&genome_id];
            let mut candidates = Vec::new();
            for (species_id, assignment) in &assignments {
                let representative = &population[&assignment.representative_id];
                let distance = distances.distance(representative, genome)?;
                if distance < compatibility_threshold {
                    candidates.push((distance, *species_id));
                }
            }

            let species_id = if let Some((_, species_id)) =
                candidates.into_iter().min_by(|a, b| a.0.total_cmp(&b.0))
            {
                species_id
            } else {
                let species_id = self.next_species_key;
                self.next_species_key = self.next_species_key.next();
                assignments.insert(species_id, SpeciesAssignment::staged(genome_id));
                self.species
                    .entry(species_id)
                    .or_insert_with(|| Species::new(species_id, generation));
                species_id
            };

            assignments
                .entry(species_id)
                .or_insert_with(|| SpeciesAssignment::staged(genome_id))
                .add_member(genome_id);
            unspeciated.remove(0);
        }

        self.genome_to_species.clear();
        let mut updated_species = BTreeMap::new();
        for (species_id, assignment) in assignments {
            let representative = population[&assignment.representative_id].clone();
            let mut members = BTreeMap::new();
            for member_id in assignment.member_ids {
                members.insert(member_id, population[&member_id].clone());
                self.genome_to_species.insert(member_id, species_id);
            }
            let mut species = self
                .species
                .remove(&species_id)
                .unwrap_or_else(|| Species::new(species_id, generation));
            species.update(representative, members);
            updated_species.insert(species_id, species);
        }
        self.species = updated_species;
        self.adjust_dynamic_threshold(config);

        Ok(())
    }

    fn adjust_dynamic_threshold(&mut self, config: &Config) {
        let Some(target) = config.species_set.target_num_species.target_count() else {
            return;
        };
        let mut threshold = self
            .compatibility_threshold
            .unwrap_or(config.species_set.compatibility_threshold);
        let species_count = self.species.len();
        if species_count > target {
            threshold += config.species_set.threshold_adjust_rate;
        } else if species_count < target {
            threshold -= config.species_set.threshold_adjust_rate;
        }
        threshold = threshold.clamp(
            config.species_set.threshold_min,
            config.species_set.threshold_max,
        );
        self.compatibility_threshold = Some(threshold);
    }
}

impl Default for SpeciesSet {
    fn default() -> Self {
        Self::new()
    }
}

fn normalized_distance_key(left: GenomeId, right: GenomeId) -> (GenomeId, GenomeId) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}
