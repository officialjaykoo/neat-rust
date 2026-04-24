use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::activation::ActivationFunction;
use crate::aggregation::AggregationFunction;
use crate::attributes::XorShiftRng;
use crate::config::Config;
use crate::gene::{ConnectionKey, DefaultConnectionGene, DefaultNodeGene};
use crate::genome::DefaultGenome;
use crate::ids::{GenomeId, SpeciesId};
use crate::innovation::InnovationTracker;
use crate::population::Population;
use crate::reproduction::ReproductionState;
use crate::species::{Species, SpeciesSet};

const CHECKPOINT_FORMAT_VERSION: &str = "neat_rust_checkpoint_v3";

#[derive(Debug, Clone, PartialEq)]
pub struct Checkpointer {
    pub generation_interval: Option<usize>,
    pub filename_prefix: String,
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    Io(String),
    Config(String),
    Invalid(String),
    MissingConfigPath,
}

impl Checkpointer {
    pub fn new(generation_interval: Option<usize>, filename_prefix: impl Into<String>) -> Self {
        Self {
            generation_interval,
            filename_prefix: filename_prefix.into(),
            config_path: None,
        }
    }

    pub fn with_config_path(mut self, config_path: impl Into<PathBuf>) -> Self {
        self.config_path = Some(config_path.into());
        self
    }

    pub fn should_save(&self, generation: usize) -> bool {
        self.generation_interval
            .map(|interval| interval > 0 && (generation + 1) % interval == 0)
            .unwrap_or(false)
    }

    pub fn checkpoint_path(&self, generation: usize) -> PathBuf {
        PathBuf::from(format!("{}{}", self.filename_prefix, generation))
    }

    pub fn save_checkpoint(&self, population: &Population) -> Result<PathBuf, CheckpointError> {
        let path = self.checkpoint_path(population.generation);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| CheckpointError::Io(err.to_string()))?;
            }
        }

        let document = CheckpointDocument::from_population(self, population);
        let mut text = serde_json::to_string_pretty(&document)
            .map_err(|err| CheckpointError::Invalid(err.to_string()))?;
        text.push('\n');
        fs::write(&path, text).map_err(|err| CheckpointError::Io(err.to_string()))?;
        Ok(path)
    }

    pub fn restore_checkpoint(path: impl AsRef<Path>) -> Result<Population, CheckpointError> {
        let text = fs::read_to_string(path.as_ref())
            .map_err(|err| CheckpointError::Io(err.to_string()))?;
        CheckpointDocument::from_json_text(&text)?.into_population()
    }
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "checkpoint I/O error: {message}"),
            Self::Config(message) => write!(f, "checkpoint config error: {message}"),
            Self::Invalid(message) => write!(f, "invalid checkpoint: {message}"),
            Self::MissingConfigPath => write!(f, "checkpoint does not include config_path"),
        }
    }
}

impl Error for CheckpointError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointDocument {
    format_version: String,
    #[serde(default = "crate_version")]
    crate_version: String,
    config_path: Option<PathBuf>,
    #[serde(default)]
    config_hash: Option<String>,
    generation: usize,
    skip_first_evaluation: bool,
    rng_state: u64,
    reproduction: ReproductionDocument,
    species_set: SpeciesSetDocument,
    population: Vec<GenomeDocument>,
    best_genome: Option<GenomeDocument>,
}

impl CheckpointDocument {
    fn from_population(checkpointer: &Checkpointer, population: &Population) -> Self {
        Self {
            format_version: CHECKPOINT_FORMAT_VERSION.to_string(),
            crate_version: crate_version(),
            config_path: checkpointer.config_path.clone(),
            config_hash: checkpointer.config_path.as_ref().and_then(config_file_hash),
            generation: population.generation,
            skip_first_evaluation: population.skip_first_evaluation,
            rng_state: population.rng_state(),
            reproduction: ReproductionDocument::from_state(&population.reproduction),
            species_set: SpeciesSetDocument::from_species_set(&population.species),
            population: population
                .population
                .values()
                .map(GenomeDocument::from_genome)
                .collect(),
            best_genome: population
                .best_genome
                .as_ref()
                .map(GenomeDocument::from_genome),
        }
    }

    fn from_json_text(text: &str) -> Result<Self, CheckpointError> {
        serde_json::from_str(text).map_err(|err| {
            CheckpointError::Invalid(format!("expected checkpoint JSON document: {err}"))
        })
    }

    fn into_population(self) -> Result<Population, CheckpointError> {
        if self.format_version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::Invalid(format!(
                "unsupported format_version {:?}; expected {CHECKPOINT_FORMAT_VERSION}",
                self.format_version
            )));
        }

        let config_path = self.config_path.ok_or(CheckpointError::MissingConfigPath)?;
        if let Some(expected_hash) = self.config_hash.as_deref() {
            let actual_hash = config_file_hash(&config_path).ok_or_else(|| {
                CheckpointError::Config(format!(
                    "failed to hash config file {}",
                    config_path.display()
                ))
            })?;
            if actual_hash != expected_hash {
                return Err(CheckpointError::Config(format!(
                    "config hash mismatch for {}: checkpoint={expected_hash}, current={actual_hash}",
                    config_path.display()
                )));
            }
        }
        let config = Config::from_file(&config_path)
            .map_err(|err| CheckpointError::Config(err.to_string()))?;

        let mut population = BTreeMap::new();
        for document in self.population {
            let genome = document.into_genome()?;
            genome
                .validate(&config.genome)
                .map_err(|err| CheckpointError::Invalid(err.to_string()))?;
            population.insert(genome.key, genome);
        }
        let best_genome = self
            .best_genome
            .map(GenomeDocument::into_genome)
            .transpose()?;
        if let Some(best) = &best_genome {
            best.validate(&config.genome)
                .map_err(|err| CheckpointError::Invalid(err.to_string()))?;
        }
        let species_set = self.species_set.into_species_set(&population)?;
        let reproduction = self.reproduction.into_state();

        Ok(Population::from_checkpoint_parts(
            config,
            population,
            species_set,
            self.generation,
            best_genome,
            reproduction,
            self.skip_first_evaluation,
            XorShiftRng::from_state(self.rng_state),
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReproductionDocument {
    genome_indexer: i64,
    innovation: i64,
    #[serde(default)]
    generations_without_improvement: usize,
    ancestors: Vec<AncestorDocument>,
}

impl ReproductionDocument {
    fn from_state(state: &ReproductionState) -> Self {
        Self {
            genome_indexer: state.genome_indexer.raw(),
            innovation: state.innovation_tracker.current_innovation_number(),
            generations_without_improvement: state.generations_without_improvement,
            ancestors: state
                .ancestors
                .iter()
                .map(|(child, parents)| AncestorDocument {
                    child: child.raw(),
                    parent1: parents.0.map(GenomeId::raw),
                    parent2: parents.1.map(GenomeId::raw),
                })
                .collect(),
        }
    }

    fn into_state(self) -> ReproductionState {
        let ancestors = self
            .ancestors
            .into_iter()
            .map(|entry| {
                (
                    GenomeId::new(entry.child),
                    (
                        entry.parent1.map(GenomeId::new),
                        entry.parent2.map(GenomeId::new),
                    ),
                )
            })
            .collect();
        ReproductionState {
            genome_indexer: GenomeId::new(self.genome_indexer),
            ancestors,
            innovation_tracker: InnovationTracker::with_start_number(self.innovation),
            generations_without_improvement: self.generations_without_improvement,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AncestorDocument {
    child: i64,
    parent1: Option<i64>,
    parent2: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesSetDocument {
    next_species_key: i64,
    compatibility_threshold: Option<f64>,
    species: Vec<SpeciesDocument>,
}

impl SpeciesSetDocument {
    fn from_species_set(species_set: &SpeciesSet) -> Self {
        Self {
            next_species_key: species_set.next_species_key().raw(),
            compatibility_threshold: finite_option(species_set.compatibility_threshold),
            species: species_set
                .species
                .iter()
                .map(|(id, species)| SpeciesDocument {
                    id: id.raw(),
                    created: species.created,
                    last_improved: species.last_improved,
                    representative_key: species
                        .representative
                        .as_ref()
                        .map(|genome| genome.key.raw()),
                    fitness: finite_option(species.fitness),
                    adjusted_fitness: finite_option(species.adjusted_fitness),
                    fitness_history: species
                        .fitness_history
                        .iter()
                        .copied()
                        .filter(|value| value.is_finite())
                        .collect(),
                    members: species.members.keys().map(|key| key.raw()).collect(),
                })
                .collect(),
        }
    }

    fn into_species_set(
        self,
        population: &BTreeMap<GenomeId, DefaultGenome>,
    ) -> Result<SpeciesSet, CheckpointError> {
        let mut species_map = BTreeMap::new();
        let mut genome_to_species = BTreeMap::new();

        for document in self.species {
            let species_id = SpeciesId::new(document.id);
            let mut species = Species::new(species_id, document.created);
            species.last_improved = document.last_improved;
            species.fitness = finite_option(document.fitness);
            species.adjusted_fitness = finite_option(document.adjusted_fitness);
            species.fitness_history = document
                .fitness_history
                .into_iter()
                .filter(|value| value.is_finite())
                .collect();
            if let Some(representative_key) = document.representative_key.map(GenomeId::new) {
                species.representative = population.get(&representative_key).cloned();
            }

            let mut members = BTreeMap::new();
            for member_id in document.members.into_iter().map(GenomeId::new) {
                let Some(genome) = population.get(&member_id) else {
                    return Err(CheckpointError::Invalid(format!(
                        "species {species_id} references missing genome {member_id}"
                    )));
                };
                members.insert(member_id, genome.clone());
                genome_to_species.insert(member_id, species_id);
            }
            species.members = members;
            species_map.insert(species_id, species);
        }

        Ok(SpeciesSet::from_parts(
            species_map,
            genome_to_species,
            SpeciesId::new(self.next_species_key),
            finite_option(self.compatibility_threshold),
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesDocument {
    id: i64,
    created: usize,
    last_improved: usize,
    representative_key: Option<i64>,
    fitness: Option<f64>,
    adjusted_fitness: Option<f64>,
    fitness_history: Vec<f64>,
    members: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenomeDocument {
    id: i64,
    fitness: Option<f64>,
    nodes: Vec<NodeGeneDocument>,
    connections: Vec<ConnectionGeneDocument>,
}

impl GenomeDocument {
    fn from_genome(genome: &DefaultGenome) -> Self {
        Self {
            id: genome.key.raw(),
            fitness: finite_option(genome.fitness),
            nodes: genome
                .nodes
                .values()
                .map(NodeGeneDocument::from_gene)
                .collect(),
            connections: genome
                .connections
                .values()
                .map(ConnectionGeneDocument::from_gene)
                .collect(),
        }
    }

    fn into_genome(self) -> Result<DefaultGenome, CheckpointError> {
        let mut genome = DefaultGenome::new(GenomeId::new(self.id));
        genome.fitness = finite_option(self.fitness);
        for node in self.nodes {
            let gene = node.into_gene()?;
            genome.nodes.insert(gene.key, gene);
        }
        for connection in self.connections {
            let gene = connection.into_gene();
            genome.connections.insert(gene.key, gene);
        }
        Ok(genome)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeGeneDocument {
    id: i64,
    bias: f64,
    response: f64,
    activation: String,
    aggregation: String,
    time_constant: f64,
    iz_a: f64,
    iz_b: f64,
    iz_c: f64,
    iz_d: f64,
    memory_gate_enabled: bool,
    memory_gate_bias: f64,
    memory_gate_response: f64,
}

impl NodeGeneDocument {
    fn from_gene(gene: &DefaultNodeGene) -> Self {
        Self {
            id: gene.key,
            bias: gene.bias,
            response: gene.response,
            activation: gene.activation.name().to_string(),
            aggregation: gene.aggregation.name().to_string(),
            time_constant: gene.time_constant,
            iz_a: gene.iz_a,
            iz_b: gene.iz_b,
            iz_c: gene.iz_c,
            iz_d: gene.iz_d,
            memory_gate_enabled: gene.memory_gate_enabled,
            memory_gate_bias: gene.memory_gate_bias,
            memory_gate_response: gene.memory_gate_response,
        }
    }

    fn into_gene(self) -> Result<DefaultNodeGene, CheckpointError> {
        Ok(DefaultNodeGene {
            key: self.id,
            bias: self.bias,
            response: self.response,
            activation: parse_activation(&self.activation)?,
            aggregation: parse_aggregation(&self.aggregation)?,
            time_constant: self.time_constant,
            iz_a: self.iz_a,
            iz_b: self.iz_b,
            iz_c: self.iz_c,
            iz_d: self.iz_d,
            memory_gate_enabled: self.memory_gate_enabled,
            memory_gate_bias: self.memory_gate_bias,
            memory_gate_response: self.memory_gate_response,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConnectionGeneDocument {
    input: i64,
    output: i64,
    innovation: Option<i64>,
    weight: f64,
    #[serde(default)]
    connection_gru_enabled: bool,
    #[serde(default)]
    connection_memory_weight: f64,
    #[serde(default)]
    connection_reset_input_weight: f64,
    #[serde(default)]
    connection_reset_memory_weight: f64,
    #[serde(default)]
    connection_update_input_weight: f64,
    #[serde(default)]
    connection_update_memory_weight: f64,
    enabled: bool,
}

impl ConnectionGeneDocument {
    fn from_gene(gene: &DefaultConnectionGene) -> Self {
        Self {
            input: gene.key.input,
            output: gene.key.output,
            innovation: gene.innovation,
            weight: gene.weight,
            connection_gru_enabled: gene.connection_gru_enabled,
            connection_memory_weight: gene.connection_memory_weight,
            connection_reset_input_weight: gene.connection_reset_input_weight,
            connection_reset_memory_weight: gene.connection_reset_memory_weight,
            connection_update_input_weight: gene.connection_update_input_weight,
            connection_update_memory_weight: gene.connection_update_memory_weight,
            enabled: gene.enabled,
        }
    }

    fn into_gene(self) -> DefaultConnectionGene {
        DefaultConnectionGene {
            key: ConnectionKey::new(self.input, self.output),
            innovation: self.innovation,
            weight: self.weight,
            connection_gru_enabled: self.connection_gru_enabled,
            connection_memory_weight: self.connection_memory_weight,
            connection_reset_input_weight: self.connection_reset_input_weight,
            connection_reset_memory_weight: self.connection_reset_memory_weight,
            connection_update_input_weight: self.connection_update_input_weight,
            connection_update_memory_weight: self.connection_update_memory_weight,
            enabled: self.enabled,
        }
    }
}

fn parse_activation(value: &str) -> Result<ActivationFunction, CheckpointError> {
    ActivationFunction::from_name(value).ok_or_else(|| {
        CheckpointError::Invalid(format!("node activation must be known, got {value:?}"))
    })
}

fn parse_aggregation(value: &str) -> Result<AggregationFunction, CheckpointError> {
    AggregationFunction::from_name(value).ok_or_else(|| {
        CheckpointError::Invalid(format!("node aggregation must be known, got {value:?}"))
    })
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn crate_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn config_file_hash(path: impl AsRef<Path>) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    Some(stable_text_hash(&text))
}

fn stable_text_hash(text: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn saves_and_restores_population_checkpoint() {
        let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("scripts")
            .join("configs")
            .join("neat_recurrent_memory8.toml");
        let config = Config::from_file(&config_path).expect("config should parse");
        let population = Population::new(config, 7).expect("population should initialize");
        let dir =
            std::env::temp_dir().join(format!("neat_rust_checkpoint_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir should be created");

        let checkpointer = Checkpointer::new(
            Some(1),
            dir.join("neat-rust-checkpoint-gen")
                .to_string_lossy()
                .to_string(),
        )
        .with_config_path(config_path);
        let path = checkpointer
            .save_checkpoint(&population)
            .expect("checkpoint should save");
        let restored = Checkpointer::restore_checkpoint(&path).expect("checkpoint should restore");

        assert_eq!(restored.generation, population.generation);
        assert_eq!(restored.population.len(), population.population.len());
        assert_eq!(
            restored.species.species.len(),
            population.species.species.len()
        );
        assert_eq!(
            restored.reproduction.genome_indexer,
            population.reproduction.genome_indexer
        );

        let text = fs::read_to_string(&path).expect("checkpoint should be readable");
        assert!(text.contains("\"format_version\": \"neat_rust_checkpoint_v3\""));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn restore_checkpoint_requires_config_path() {
        let dir = std::env::temp_dir().join(format!(
            "neat_rust_checkpoint_missing_config_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir should be created");
        let path = dir.join("missing_config.chk");
        fs::write(
            &path,
            r#"{
  "format_version": "neat_rust_checkpoint_v3",
  "config_path": null,
  "generation": 0,
  "skip_first_evaluation": false,
  "rng_state": 1,
  "reproduction": {
    "genome_indexer": 1,
    "innovation": 0,
    "ancestors": []
  },
  "species_set": {
    "next_species_key": 1,
    "compatibility_threshold": null,
    "species": []
  },
  "population": [],
  "best_genome": null
}
"#,
        )
        .expect("checkpoint stub should be written");

        let err = match Checkpointer::restore_checkpoint(&path) {
            Ok(_) => panic!("restore must fail without config_path"),
            Err(err) => err,
        };
        assert_eq!(err, CheckpointError::MissingConfigPath);

        let _ = fs::remove_dir_all(&dir);
    }
}
