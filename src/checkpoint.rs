use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::attributes::XorShiftRng;
use crate::config::Config;
use crate::gene::{DefaultConnectionGene, DefaultNodeGene};
use crate::genome::DefaultGenome;
use crate::innovation::InnovationTracker;
use crate::population::Population;
use crate::reproduction::ReproductionState;
use crate::species::{Species, SpeciesSet};

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

#[derive(Debug, Clone)]
struct SpeciesMeta {
    created: usize,
    last_improved: usize,
    representative_key: Option<i64>,
    fitness: Option<f64>,
    adjusted_fitness: Option<f64>,
    fitness_history: Vec<f64>,
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
        fs::write(&path, checkpoint_text(self, population))
            .map_err(|err| CheckpointError::Io(err.to_string()))?;
        Ok(path)
    }

    pub fn restore_checkpoint(path: impl AsRef<Path>) -> Result<Population, CheckpointError> {
        let text = fs::read_to_string(path.as_ref())
            .map_err(|err| CheckpointError::Io(err.to_string()))?;
        restore_checkpoint_text(&text)
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

fn checkpoint_text(checkpointer: &Checkpointer, population: &Population) -> String {
    let mut out = String::new();
    key_value(&mut out, "format_version", "neat_rust_checkpoint_v2");
    if let Some(config_path) = &checkpointer.config_path {
        key_value(&mut out, "config_path", &config_path.to_string_lossy());
    }
    key_value(&mut out, "generation", &population.generation.to_string());
    key_value(
        &mut out,
        "skip_first_evaluation",
        bool_text(population.skip_first_evaluation),
    );
    key_value(&mut out, "rng_state", &population.rng_state().to_string());
    key_value(
        &mut out,
        "genome_indexer",
        &population.reproduction.genome_indexer.to_string(),
    );
    key_value(
        &mut out,
        "innovation",
        &population
            .reproduction
            .innovation_tracker
            .current_innovation_number()
            .to_string(),
    );
    key_value(
        &mut out,
        "species_next_key",
        &population.species.next_species_key().to_string(),
    );
    key_value(
        &mut out,
        "species_compatibility_threshold",
        &option_f64(population.species.compatibility_threshold),
    );

    for genome in population.population.values() {
        push_genome(&mut out, "population", genome);
    }
    if let Some(best) = &population.best_genome {
        push_genome(&mut out, "best", best);
    }

    for (species_id, species) in &population.species.species {
        line(
            &mut out,
            &[
                "species".to_string(),
                species_id.to_string(),
                species.created.to_string(),
                species.last_improved.to_string(),
                species
                    .representative
                    .as_ref()
                    .map(|genome| genome.key.to_string())
                    .unwrap_or_else(|| "null".to_string()),
                option_f64(species.fitness),
                option_f64(species.adjusted_fitness),
                species
                    .fitness_history
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ],
        );
        for member_id in species.members.keys() {
            line(
                &mut out,
                &[
                    "species_member".to_string(),
                    species_id.to_string(),
                    member_id.to_string(),
                ],
            );
        }
    }

    for (child, parents) in &population.reproduction.ancestors {
        line(
            &mut out,
            &[
                "ancestor".to_string(),
                child.to_string(),
                option_i64(parents.0),
                option_i64(parents.1),
            ],
        );
    }

    out
}

fn push_genome(out: &mut String, role: &str, genome: &DefaultGenome) {
    line(
        out,
        &[
            "genome".to_string(),
            role.to_string(),
            genome.key.to_string(),
            option_f64(genome.fitness),
        ],
    );
    for node in genome.nodes.values() {
        line(
            out,
            &[
                "node".to_string(),
                role.to_string(),
                genome.key.to_string(),
                node.key.to_string(),
                node.bias.to_string(),
                node.response.to_string(),
                node.activation.clone(),
                node.aggregation.clone(),
                node.time_constant.to_string(),
                node.iz_a.to_string(),
                node.iz_b.to_string(),
                node.iz_c.to_string(),
                node.iz_d.to_string(),
                bool_text(node.memory_gate_enabled).to_string(),
                node.memory_gate_bias.to_string(),
                node.memory_gate_response.to_string(),
            ],
        );
    }
    for connection in genome.connections.values() {
        line(
            out,
            &[
                "connection".to_string(),
                role.to_string(),
                genome.key.to_string(),
                connection.key.0.to_string(),
                connection.key.1.to_string(),
                option_i64(connection.innovation),
                connection.weight.to_string(),
                bool_text(connection.enabled).to_string(),
            ],
        );
    }
}

fn restore_checkpoint_text(text: &str) -> Result<Population, CheckpointError> {
    let mut format_version = String::new();
    let mut config_path: Option<PathBuf> = None;
    let mut generation = 0usize;
    let mut skip_first_evaluation = false;
    let mut rng_state = 1u64;
    let mut genome_indexer = 1i64;
    let mut innovation = 0i64;
    let mut species_next_key = 1i64;
    let mut species_compatibility_threshold: Option<f64> = None;
    let mut population = BTreeMap::new();
    let mut best_genome: Option<DefaultGenome> = None;
    let mut species_meta: BTreeMap<i64, SpeciesMeta> = BTreeMap::new();
    let mut species_members: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut ancestors: BTreeMap<i64, (Option<i64>, Option<i64>)> = BTreeMap::new();

    for raw_line in text.lines() {
        if raw_line.is_empty() {
            continue;
        }
        if let Some((key, value)) = raw_line.split_once('=') {
            match key {
                "format_version" => format_version = unescape_field(value),
                "config_path" => config_path = Some(PathBuf::from(unescape_field(value))),
                "generation" => generation = parse_usize(value, "generation")?,
                "skip_first_evaluation" => {
                    skip_first_evaluation = parse_bool(value, "skip_first_evaluation")?
                }
                "rng_state" => rng_state = parse_u64(value, "rng_state")?,
                "genome_indexer" => genome_indexer = parse_i64(value, "genome_indexer")?,
                "innovation" => innovation = parse_i64(value, "innovation")?,
                "species_next_key" => species_next_key = parse_i64(value, "species_next_key")?,
                "species_compatibility_threshold" => {
                    species_compatibility_threshold =
                        parse_option_f64(value, "species_compatibility_threshold")?
                }
                _ => {}
            }
            continue;
        }

        let parts = split_line(raw_line);
        match parts.first().map(String::as_str) {
            Some("genome") => {
                require_len(&parts, 4, "genome")?;
                let role = &parts[1];
                let key = parse_i64(&parts[2], "genome key")?;
                let mut genome = DefaultGenome::new(key);
                genome.fitness = parse_option_f64(&parts[3], "genome fitness")?;
                insert_genome(role, genome, &mut population, &mut best_genome)?;
            }
            Some("node") => {
                if parts.len() != 11 && parts.len() != 16 {
                    return Err(CheckpointError::Invalid(format!(
                        "node must have 11 or 16 fields, got {}",
                        parts.len()
                    )));
                }
                let role = &parts[1];
                let genome_key = parse_i64(&parts[2], "node genome key")?;
                let extended = parts.len() == 16;
                let memory_offset = if extended { 13 } else { 8 };
                let node = DefaultNodeGene {
                    key: parse_i64(&parts[3], "node key")?,
                    bias: parse_f64(&parts[4], "node bias")?,
                    response: parse_f64(&parts[5], "node response")?,
                    activation: parts[6].clone(),
                    aggregation: parts[7].clone(),
                    time_constant: if extended {
                        parse_f64(&parts[8], "node time_constant")?
                    } else {
                        1.0
                    },
                    iz_a: if extended {
                        parse_f64(&parts[9], "node iz_a")?
                    } else {
                        0.02
                    },
                    iz_b: if extended {
                        parse_f64(&parts[10], "node iz_b")?
                    } else {
                        0.20
                    },
                    iz_c: if extended {
                        parse_f64(&parts[11], "node iz_c")?
                    } else {
                        -65.0
                    },
                    iz_d: if extended {
                        parse_f64(&parts[12], "node iz_d")?
                    } else {
                        8.0
                    },
                    memory_gate_enabled: parse_bool(
                        &parts[memory_offset],
                        "node memory_gate_enabled",
                    )?,
                    memory_gate_bias: parse_f64(
                        &parts[memory_offset + 1],
                        "node memory_gate_bias",
                    )?,
                    memory_gate_response: parse_f64(
                        &parts[memory_offset + 2],
                        "node memory_gate_response",
                    )?,
                };
                genome_mut(role, genome_key, &mut population, &mut best_genome)?
                    .nodes
                    .insert(node.key, node);
            }
            Some("connection") => {
                require_len(&parts, 8, "connection")?;
                let role = &parts[1];
                let genome_key = parse_i64(&parts[2], "connection genome key")?;
                let input = parse_i64(&parts[3], "connection input")?;
                let output = parse_i64(&parts[4], "connection output")?;
                let connection = DefaultConnectionGene {
                    key: (input, output),
                    innovation: parse_option_i64(&parts[5], "connection innovation")?,
                    weight: parse_f64(&parts[6], "connection weight")?,
                    enabled: parse_bool(&parts[7], "connection enabled")?,
                };
                genome_mut(role, genome_key, &mut population, &mut best_genome)?
                    .connections
                    .insert(connection.key, connection);
            }
            Some("species") => {
                require_len(&parts, 8, "species")?;
                let key = parse_i64(&parts[1], "species key")?;
                species_meta.insert(
                    key,
                    SpeciesMeta {
                        created: parse_usize(&parts[2], "species created")?,
                        last_improved: parse_usize(&parts[3], "species last_improved")?,
                        representative_key: parse_option_i64(&parts[4], "species representative")?,
                        fitness: parse_option_f64(&parts[5], "species fitness")?,
                        adjusted_fitness: parse_option_f64(&parts[6], "species adjusted_fitness")?,
                        fitness_history: parse_f64_list(&parts[7], "species fitness_history")?,
                    },
                );
            }
            Some("species_member") => {
                require_len(&parts, 3, "species_member")?;
                let species_id = parse_i64(&parts[1], "species_member species")?;
                let genome_id = parse_i64(&parts[2], "species_member genome")?;
                species_members
                    .entry(species_id)
                    .or_default()
                    .push(genome_id);
            }
            Some("ancestor") => {
                require_len(&parts, 4, "ancestor")?;
                ancestors.insert(
                    parse_i64(&parts[1], "ancestor child")?,
                    (
                        parse_option_i64(&parts[2], "ancestor parent1")?,
                        parse_option_i64(&parts[3], "ancestor parent2")?,
                    ),
                );
            }
            Some(other) => {
                return Err(CheckpointError::Invalid(format!(
                    "unknown checkpoint line type {other}"
                )))
            }
            None => {}
        }
    }

    if format_version != "neat_rust_checkpoint_v2" {
        return Err(CheckpointError::Invalid(format!(
            "unsupported format_version {format_version:?}"
        )));
    }
    let config_path = config_path.ok_or(CheckpointError::MissingConfigPath)?;
    let config =
        Config::from_file(&config_path).map_err(|err| CheckpointError::Config(err.to_string()))?;
    let species_set = build_species_set(
        species_meta,
        species_members,
        &population,
        species_next_key,
        species_compatibility_threshold,
    )?;
    let reproduction = ReproductionState {
        genome_indexer,
        ancestors,
        innovation_tracker: InnovationTracker::with_start_number(innovation),
    };

    Ok(Population::from_checkpoint_parts(
        config,
        population,
        species_set,
        generation,
        best_genome,
        reproduction,
        skip_first_evaluation,
        XorShiftRng::from_state(rng_state),
    ))
}

fn build_species_set(
    species_meta: BTreeMap<i64, SpeciesMeta>,
    species_members: BTreeMap<i64, Vec<i64>>,
    population: &BTreeMap<i64, DefaultGenome>,
    next_species_key: i64,
    compatibility_threshold: Option<f64>,
) -> Result<SpeciesSet, CheckpointError> {
    let mut species_map = BTreeMap::new();
    let mut genome_to_species = BTreeMap::new();

    for (species_id, meta) in species_meta {
        let mut species = Species::new(species_id, meta.created);
        species.last_improved = meta.last_improved;
        species.fitness = meta.fitness;
        species.adjusted_fitness = meta.adjusted_fitness;
        species.fitness_history = meta.fitness_history;
        if let Some(representative_key) = meta.representative_key {
            species.representative = population.get(&representative_key).cloned();
        }
        let mut members = BTreeMap::new();
        for genome_id in species_members.get(&species_id).into_iter().flatten() {
            let Some(genome) = population.get(genome_id) else {
                return Err(CheckpointError::Invalid(format!(
                    "species {species_id} references missing genome {genome_id}"
                )));
            };
            members.insert(*genome_id, genome.clone());
            genome_to_species.insert(*genome_id, species_id);
        }
        species.members = members;
        species_map.insert(species_id, species);
    }

    Ok(SpeciesSet::from_parts(
        species_map,
        genome_to_species,
        next_species_key,
        compatibility_threshold,
    ))
}

fn insert_genome(
    role: &str,
    genome: DefaultGenome,
    population: &mut BTreeMap<i64, DefaultGenome>,
    best_genome: &mut Option<DefaultGenome>,
) -> Result<(), CheckpointError> {
    match role {
        "population" => {
            population.insert(genome.key, genome);
            Ok(())
        }
        "best" => {
            *best_genome = Some(genome);
            Ok(())
        }
        other => Err(CheckpointError::Invalid(format!(
            "unknown genome role {other}"
        ))),
    }
}

fn genome_mut<'a>(
    role: &str,
    key: i64,
    population: &'a mut BTreeMap<i64, DefaultGenome>,
    best_genome: &'a mut Option<DefaultGenome>,
) -> Result<&'a mut DefaultGenome, CheckpointError> {
    match role {
        "population" => population
            .get_mut(&key)
            .ok_or_else(|| CheckpointError::Invalid(format!("missing population genome {key}"))),
        "best" => best_genome
            .as_mut()
            .filter(|genome| genome.key == key)
            .ok_or_else(|| CheckpointError::Invalid(format!("missing best genome {key}"))),
        other => Err(CheckpointError::Invalid(format!(
            "unknown genome role {other}"
        ))),
    }
}

fn key_value(out: &mut String, key: &str, value: &str) {
    out.push_str(key);
    out.push('=');
    out.push_str(&escape_field(value));
    out.push('\n');
}

fn line(out: &mut String, values: &[String]) {
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push('\t');
        }
        out.push_str(&escape_field(value));
    }
    out.push('\n');
}

fn split_line(line: &str) -> Vec<String> {
    line.split('\t').map(unescape_field).collect()
}

fn escape_field(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

fn unescape_field(value: &str) -> String {
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            match ch {
                't' => out.push('\t'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                '\\' => out.push('\\'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        out.push('\\');
    }
    out
}

fn bool_text(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn option_f64(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn option_i64(value: Option<i64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn require_len(parts: &[String], expected: usize, label: &str) -> Result<(), CheckpointError> {
    if parts.len() == expected {
        Ok(())
    } else {
        Err(CheckpointError::Invalid(format!(
            "{label} expected {expected} fields, got {}",
            parts.len()
        )))
    }
}

fn parse_bool(value: &str, label: &str) -> Result<bool, CheckpointError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(CheckpointError::Invalid(format!(
            "{label} must be true or false"
        ))),
    }
}

fn parse_f64(value: &str, label: &str) -> Result<f64, CheckpointError> {
    value
        .parse::<f64>()
        .map_err(|_| CheckpointError::Invalid(format!("{label} must be a floating point number")))
}

fn parse_option_f64(value: &str, label: &str) -> Result<Option<f64>, CheckpointError> {
    if value.trim() == "null" {
        Ok(None)
    } else {
        Ok(Some(parse_f64(value, label)?))
    }
}

fn parse_i64(value: &str, label: &str) -> Result<i64, CheckpointError> {
    value
        .parse::<i64>()
        .map_err(|_| CheckpointError::Invalid(format!("{label} must be an integer")))
}

fn parse_option_i64(value: &str, label: &str) -> Result<Option<i64>, CheckpointError> {
    if value.trim() == "null" {
        Ok(None)
    } else {
        Ok(Some(parse_i64(value, label)?))
    }
}

fn parse_usize(value: &str, label: &str) -> Result<usize, CheckpointError> {
    value
        .parse::<usize>()
        .map_err(|_| CheckpointError::Invalid(format!("{label} must be a non-negative integer")))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, CheckpointError> {
    value
        .parse::<u64>()
        .map_err(|_| CheckpointError::Invalid(format!("{label} must be a non-negative integer")))
}

fn parse_f64_list(value: &str, label: &str) -> Result<Vec<f64>, CheckpointError> {
    if value.trim().is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|item| parse_f64(item, label))
        .collect()
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
            .join("neat_recurrent_memory8.ini");
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
        fs::write(&path, "format_version=neat_rust_checkpoint_v2\n")
            .expect("checkpoint stub should be written");

        let err = match Checkpointer::restore_checkpoint(&path) {
            Ok(_) => panic!("restore must fail without config_path"),
            Err(err) => err,
        };
        assert_eq!(err, CheckpointError::MissingConfigPath);

        let _ = fs::remove_dir_all(&dir);
    }
}
