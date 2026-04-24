use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

mod access;
mod ini;

use access::*;
pub use ini::{parse_ini, IniDocument};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub neat: NeatConfig,
    pub genome: GenomeConfig,
    pub species_set: SpeciesSetConfig,
    pub stagnation: StagnationConfig,
    pub reproduction: ReproductionConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitnessCriterion {
    Max,
    Min,
    Mean,
    Other(String),
}

impl FitnessCriterion {
    pub fn from_raw(value: &str) -> Self {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "max" => Self::Max,
            "min" => Self::Min,
            "mean" => Self::Mean,
            _ => Self::Other(value.trim().to_string()),
        }
    }

    pub fn is_min(&self) -> bool {
        matches!(self, Self::Min)
    }

    pub fn is_better(&self, candidate: f64, previous: f64) -> bool {
        if self.is_min() {
            candidate < previous
        } else {
            candidate > previous
        }
    }

    pub fn meets_threshold(&self, value: f64, threshold: f64) -> bool {
        if self.is_min() {
            value <= threshold
        } else {
            value >= threshold
        }
    }

    pub fn worst_value(&self) -> f64 {
        if self.is_min() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl fmt::Display for FitnessCriterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Mean => write!(f, "mean"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitialConnectionMode {
    Unconnected,
    FullNoDirect,
    FullDirect,
    Full,
    PartialNoDirect,
    PartialDirect,
    Partial,
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialConnection {
    pub mode: InitialConnectionMode,
    pub fraction: f64,
}

impl InitialConnection {
    pub fn from_raw(value: &str) -> Self {
        let trimmed = value.trim();
        let mut parts = trimmed.split_whitespace();
        let Some(mode_raw) = parts.next() else {
            return Self::unconnected();
        };
        let fraction = match parts.next() {
            Some(raw) => match raw.parse::<f64>() {
                Ok(value) => value.clamp(0.0, 1.0),
                Err(_) => return Self::unsupported(trimmed),
            },
            None => 1.0,
        };
        let mode = match mode_raw.trim().to_ascii_lowercase().as_str() {
            "unconnected" => InitialConnectionMode::Unconnected,
            "full_nodirect" => InitialConnectionMode::FullNoDirect,
            "full_direct" => InitialConnectionMode::FullDirect,
            "full" => InitialConnectionMode::Full,
            "partial_nodirect" => InitialConnectionMode::PartialNoDirect,
            "partial_direct" => InitialConnectionMode::PartialDirect,
            "partial" => InitialConnectionMode::Partial,
            _ => return Self::unsupported(trimmed),
        };
        Self { mode, fraction }
    }

    pub fn unconnected() -> Self {
        Self {
            mode: InitialConnectionMode::Unconnected,
            fraction: 1.0,
        }
    }

    pub fn full_direct() -> Self {
        Self {
            mode: InitialConnectionMode::FullDirect,
            fraction: 1.0,
        }
    }

    pub fn partial_direct(fraction: f64) -> Self {
        Self {
            mode: InitialConnectionMode::PartialDirect,
            fraction: fraction.clamp(0.0, 1.0),
        }
    }

    pub fn unsupported(value: impl Into<String>) -> Self {
        Self {
            mode: InitialConnectionMode::Unsupported(value.into()),
            fraction: 1.0,
        }
    }
}

impl fmt::Display for InitialConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.mode {
            InitialConnectionMode::Unconnected => write!(f, "unconnected"),
            InitialConnectionMode::FullNoDirect => write!(f, "full_nodirect"),
            InitialConnectionMode::FullDirect => write!(f, "full_direct"),
            InitialConnectionMode::Full => write!(f, "full"),
            InitialConnectionMode::PartialNoDirect => {
                write!(f, "partial_nodirect {}", self.fraction)
            }
            InitialConnectionMode::PartialDirect => write!(f, "partial_direct {}", self.fraction),
            InitialConnectionMode::Partial => write!(f, "partial {}", self.fraction),
            InitialConnectionMode::Unsupported(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralMutationSurer {
    Default,
    Enabled,
    Disabled,
    Other(String),
}

impl StructuralMutationSurer {
    pub fn from_raw(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "yes" | "true" | "on" => Self::Enabled,
            "0" | "no" | "false" | "off" => Self::Disabled,
            "default" => Self::Default,
            _ => Self::Other(value.trim().to_string()),
        }
    }

    pub fn is_enabled(&self, default: bool) -> bool {
        match self {
            Self::Default => default,
            Self::Enabled => true,
            Self::Disabled | Self::Other(_) => false,
        }
    }
}

impl fmt::Display for StructuralMutationSurer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Enabled => write!(f, "true"),
            Self::Disabled => write!(f, "false"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompatibilityExcessCoefficient {
    Auto,
    Value(f64),
    Other(String),
}

impl CompatibilityExcessCoefficient {
    pub fn from_raw(value: &str) -> Self {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            Self::Auto
        } else if let Ok(value) = trimmed.parse::<f64>() {
            Self::Value(value)
        } else {
            Self::Other(trimmed.to_string())
        }
    }

    pub fn resolve(&self, default: f64) -> f64 {
        match self {
            Self::Auto => default,
            Self::Value(value) => *value,
            Self::Other(_) => default,
        }
    }
}

impl fmt::Display for CompatibilityExcessCoefficient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Value(value) => write!(f, "{value}"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetNumSpecies {
    Disabled,
    Count(usize),
    Other(String),
}

impl TargetNumSpecies {
    pub fn from_raw(value: &str) -> Self {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("none") || trimmed.is_empty() {
            Self::Disabled
        } else if let Ok(value) = trimmed.parse::<usize>() {
            Self::Count(value)
        } else {
            Self::Other(trimmed.to_string())
        }
    }

    pub fn target_count(&self) -> Option<usize> {
        match self {
            Self::Count(value) => Some(*value),
            Self::Disabled | Self::Other(_) => None,
        }
    }
}

impl fmt::Display for TargetNumSpecies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "none"),
            Self::Count(value) => write!(f, "{value}"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeciesFitnessFunction {
    Mean,
    Max,
    Min,
    Median,
    Other(String),
}

impl SpeciesFitnessFunction {
    pub fn from_raw(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" => Self::Mean,
            "max" => Self::Max,
            "min" => Self::Min,
            "median" => Self::Median,
            _ => Self::Other(value.trim().to_string()),
        }
    }

    pub fn evaluate(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        match self {
            Self::Min => values.iter().copied().reduce(f64::min).unwrap_or(0.0),
            Self::Max => values.iter().copied().reduce(f64::max).unwrap_or(0.0),
            Self::Median => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            Self::Mean | Self::Other(_) => values.iter().sum::<f64>() / values.len() as f64,
        }
    }
}

impl fmt::Display for SpeciesFitnessFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Median => write!(f, "median"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitnessSharingMode {
    Normalized,
    Canonical,
    Other(String),
}

impl FitnessSharingMode {
    pub fn from_raw(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "normalized" => Self::Normalized,
            "canonical" => Self::Canonical,
            _ => Self::Other(value.trim().to_string()),
        }
    }

    pub fn is_canonical(&self) -> bool {
        matches!(self, Self::Canonical)
    }
}

impl fmt::Display for FitnessSharingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normalized => write!(f, "normalized"),
            Self::Canonical => write!(f, "canonical"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnMethod {
    Smoothed,
    Proportional,
    Other(String),
}

impl SpawnMethod {
    pub fn from_raw(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "smoothed" => Self::Smoothed,
            "proportional" => Self::Proportional,
            _ => Self::Other(value.trim().to_string()),
        }
    }

    pub fn is_proportional(&self) -> bool {
        matches!(self, Self::Proportional)
    }
}

impl fmt::Display for SpawnMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Smoothed => write!(f, "smoothed"),
            Self::Proportional => write!(f, "proportional"),
            Self::Other(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeatConfig {
    pub fitness_criterion: FitnessCriterion,
    pub fitness_threshold: f64,
    pub pop_size: usize,
    pub reset_on_extinction: bool,
    pub no_fitness_termination: bool,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenomeConfig {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_hidden: usize,
    pub feed_forward: bool,
    pub initial_connection: InitialConnection,
    pub conn_add_prob: f64,
    pub conn_delete_prob: f64,
    pub node_add_prob: f64,
    pub node_delete_prob: f64,
    pub single_structural_mutation: bool,
    pub structural_mutation_surer: StructuralMutationSurer,
    pub activation: StringAttributeConfig,
    pub aggregation: StringAttributeConfig,
    pub bias: FloatAttributeConfig,
    pub response: FloatAttributeConfig,
    pub time_constant: FloatAttributeConfig,
    pub iz_a: FloatAttributeConfig,
    pub iz_b: FloatAttributeConfig,
    pub iz_c: FloatAttributeConfig,
    pub iz_d: FloatAttributeConfig,
    pub memory_gate_enabled: BoolAttributeConfig,
    pub memory_gate_bias: FloatAttributeConfig,
    pub memory_gate_response: FloatAttributeConfig,
    pub enabled: BoolAttributeConfig,
    pub compatibility_disjoint_coefficient: f64,
    pub compatibility_excess_coefficient: CompatibilityExcessCoefficient,
    pub compatibility_include_node_genes: bool,
    pub compatibility_enable_penalty: f64,
    pub compatibility_weight_coefficient: f64,
    pub weight: FloatAttributeConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesSetConfig {
    pub compatibility_threshold: f64,
    pub target_num_species: TargetNumSpecies,
    pub threshold_adjust_rate: f64,
    pub threshold_min: f64,
    pub threshold_max: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StagnationConfig {
    pub species_fitness_func: SpeciesFitnessFunction,
    pub max_stagnation: usize,
    pub species_elitism: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReproductionConfig {
    pub elitism: usize,
    pub survival_threshold: f64,
    pub min_species_size: usize,
    pub fitness_sharing: FitnessSharingMode,
    pub spawn_method: SpawnMethod,
    pub interspecies_crossover_prob: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringAttributeConfig {
    pub default: String,
    pub mutate_rate: f64,
    pub options: Vec<String>,
}

pub type ActivationConfig = StringAttributeConfig;
pub type AggregationConfig = StringAttributeConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatInitType {
    Gaussian,
    Uniform,
    Other(String),
}

impl FloatInitType {
    pub fn from_raw(raw: &str) -> Self {
        let normalized = raw.trim().to_ascii_lowercase();
        if normalized.contains("gauss") || normalized.contains("normal") {
            Self::Gaussian
        } else if normalized.contains("uniform") {
            Self::Uniform
        } else {
            Self::Other(normalized)
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Uniform => "uniform",
            Self::Other(value) => value.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloatAttributeConfig {
    pub init_mean: f64,
    pub init_stdev: f64,
    pub init_type: FloatInitType,
    pub max_value: f64,
    pub min_value: f64,
    pub mutate_power: f64,
    pub mutate_rate: f64,
    pub replace_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoolAttributeConfig {
    pub default: bool,
    pub mutate_rate: f64,
    pub rate_to_true_add: f64,
    pub rate_to_false_add: f64,
}

pub type ConnectionGeneConfig = BoolAttributeConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    Io(String),
    Parse {
        line: usize,
        message: String,
    },
    MissingSection(String),
    MissingKey {
        section: String,
        key: String,
    },
    InvalidValue {
        section: String,
        key: String,
        value: String,
        message: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "I/O error: {message}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::MissingSection(section) => write!(f, "missing required section [{section}]"),
            Self::MissingKey { section, key } => {
                write!(f, "missing required key [{section}] {key}")
            }
            Self::InvalidValue {
                section,
                key,
                value,
                message,
            } => write!(f, "invalid value [{section}] {key}={value:?}: {message}"),
        }
    }
}

impl Error for ConfigError {}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let text =
            fs::read_to_string(path.as_ref()).map_err(|err| ConfigError::Io(err.to_string()))?;
        Self::from_str(&text)
    }

    pub fn from_str(text: &str) -> Result<Self, ConfigError> {
        let ini = parse_ini(text)?;
        Self::from_ini(&ini)
    }

    pub fn from_ini(ini: &IniDocument) -> Result<Self, ConfigError> {
        let neat = require_section(ini, "NEAT")?;
        let genome_section = if ini.contains_key("DefaultGenome") {
            "DefaultGenome"
        } else {
            "IZGenome"
        };
        let genome = require_section(ini, genome_section)?;
        let species_set = require_section(ini, "DefaultSpeciesSet")?;
        let stagnation = require_section(ini, "DefaultStagnation")?;
        let reproduction = require_section(ini, "DefaultReproduction")?;

        Ok(Self {
            neat: NeatConfig {
                fitness_criterion: FitnessCriterion::from_raw(&get_string(
                    neat,
                    "NEAT",
                    "fitness_criterion",
                )?),
                fitness_threshold: get_f64(neat, "NEAT", "fitness_threshold")?,
                pop_size: get_usize(neat, "NEAT", "pop_size")?,
                reset_on_extinction: get_bool(neat, "NEAT", "reset_on_extinction")?,
                no_fitness_termination: get_bool(neat, "NEAT", "no_fitness_termination")?,
                seed: get_optional_u64(neat, "NEAT", "seed")?,
            },
            genome: GenomeConfig {
                num_inputs: get_usize(genome, genome_section, "num_inputs")?,
                num_outputs: get_usize(genome, genome_section, "num_outputs")?,
                num_hidden: get_usize(genome, genome_section, "num_hidden")?,
                feed_forward: get_bool(genome, genome_section, "feed_forward")?,
                initial_connection: InitialConnection::from_raw(&get_string(
                    genome,
                    genome_section,
                    "initial_connection",
                )?),
                conn_add_prob: get_f64(genome, genome_section, "conn_add_prob")?,
                conn_delete_prob: get_f64(genome, genome_section, "conn_delete_prob")?,
                node_add_prob: get_f64(genome, genome_section, "node_add_prob")?,
                node_delete_prob: get_f64(genome, genome_section, "node_delete_prob")?,
                single_structural_mutation: get_bool(
                    genome,
                    genome_section,
                    "single_structural_mutation",
                )?,
                structural_mutation_surer: StructuralMutationSurer::from_raw(&get_string(
                    genome,
                    genome_section,
                    "structural_mutation_surer",
                )?),
                activation: get_optional_string_attribute(
                    genome,
                    genome_section,
                    "activation",
                    StringAttributeConfig {
                        default: "identity".to_string(),
                        mutate_rate: 0.0,
                        options: vec!["identity".to_string()],
                    },
                )?,
                aggregation: get_optional_string_attribute(
                    genome,
                    genome_section,
                    "aggregation",
                    StringAttributeConfig {
                        default: "sum".to_string(),
                        mutate_rate: 0.0,
                        options: vec!["sum".to_string()],
                    },
                )?,
                bias: get_float_attribute(genome, genome_section, "bias")?,
                response: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "response",
                    FloatAttributeConfig {
                        init_mean: 1.0,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 1.0,
                        min_value: 1.0,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                time_constant: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "time_constant",
                    FloatAttributeConfig {
                        init_mean: 1.0,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 10.0,
                        min_value: 0.01,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                iz_a: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "a",
                    FloatAttributeConfig {
                        init_mean: 0.02,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 0.02,
                        min_value: 0.02,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                iz_b: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "b",
                    FloatAttributeConfig {
                        init_mean: 0.20,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 0.20,
                        min_value: 0.20,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                iz_c: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "c",
                    FloatAttributeConfig {
                        init_mean: -65.0,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: -65.0,
                        min_value: -65.0,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                iz_d: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "d",
                    FloatAttributeConfig {
                        init_mean: 8.0,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 8.0,
                        min_value: 8.0,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                memory_gate_enabled: get_optional_bool_attribute(
                    genome,
                    genome_section,
                    "memory_gate_enabled",
                    BoolAttributeConfig {
                        default: false,
                        mutate_rate: 0.0,
                        rate_to_true_add: 0.0,
                        rate_to_false_add: 0.0,
                    },
                )?,
                memory_gate_bias: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "memory_gate_bias",
                    FloatAttributeConfig {
                        init_mean: 0.0,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 0.0,
                        min_value: 0.0,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                memory_gate_response: get_optional_float_attribute(
                    genome,
                    genome_section,
                    "memory_gate_response",
                    FloatAttributeConfig {
                        init_mean: 1.0,
                        init_stdev: 0.0,
                        init_type: FloatInitType::Gaussian,
                        max_value: 1.0,
                        min_value: 1.0,
                        mutate_power: 0.0,
                        mutate_rate: 0.0,
                        replace_rate: 0.0,
                    },
                )?,
                enabled: get_bool_attribute(genome, genome_section, "enabled")?,
                compatibility_disjoint_coefficient: get_f64(
                    genome,
                    genome_section,
                    "compatibility_disjoint_coefficient",
                )?,
                compatibility_excess_coefficient: CompatibilityExcessCoefficient::from_raw(
                    &get_optional_string_default(
                        genome,
                        genome_section,
                        "compatibility_excess_coefficient",
                        "auto",
                    )?,
                ),
                compatibility_include_node_genes: get_optional_bool_default(
                    genome,
                    genome_section,
                    "compatibility_include_node_genes",
                    true,
                )?,
                compatibility_enable_penalty: get_optional_f64_default(
                    genome,
                    genome_section,
                    "compatibility_enable_penalty",
                    1.0,
                )?,
                compatibility_weight_coefficient: get_f64(
                    genome,
                    genome_section,
                    "compatibility_weight_coefficient",
                )?,
                weight: get_float_attribute(genome, genome_section, "weight")?,
            },
            species_set: SpeciesSetConfig {
                compatibility_threshold: get_f64(
                    species_set,
                    "DefaultSpeciesSet",
                    "compatibility_threshold",
                )?,
                target_num_species: TargetNumSpecies::from_raw(&get_optional_string_default(
                    species_set,
                    "DefaultSpeciesSet",
                    "target_num_species",
                    "none",
                )?),
                threshold_adjust_rate: get_optional_f64_default(
                    species_set,
                    "DefaultSpeciesSet",
                    "threshold_adjust_rate",
                    0.1,
                )?,
                threshold_min: get_optional_f64_default(
                    species_set,
                    "DefaultSpeciesSet",
                    "threshold_min",
                    0.1,
                )?,
                threshold_max: get_optional_f64_default(
                    species_set,
                    "DefaultSpeciesSet",
                    "threshold_max",
                    100.0,
                )?,
            },
            stagnation: StagnationConfig {
                species_fitness_func: SpeciesFitnessFunction::from_raw(&get_string(
                    stagnation,
                    "DefaultStagnation",
                    "species_fitness_func",
                )?),
                max_stagnation: get_usize(stagnation, "DefaultStagnation", "max_stagnation")?,
                species_elitism: get_usize(stagnation, "DefaultStagnation", "species_elitism")?,
            },
            reproduction: ReproductionConfig {
                elitism: get_usize(reproduction, "DefaultReproduction", "elitism")?,
                survival_threshold: get_f64(
                    reproduction,
                    "DefaultReproduction",
                    "survival_threshold",
                )?,
                min_species_size: get_usize(
                    reproduction,
                    "DefaultReproduction",
                    "min_species_size",
                )?,
                fitness_sharing: FitnessSharingMode::from_raw(&get_optional_string_default(
                    reproduction,
                    "DefaultReproduction",
                    "fitness_sharing",
                    "normalized",
                )?),
                spawn_method: SpawnMethod::from_raw(&get_optional_string_default(
                    reproduction,
                    "DefaultReproduction",
                    "spawn_method",
                    "smoothed",
                )?),
                interspecies_crossover_prob: get_optional_f64_default(
                    reproduction,
                    "DefaultReproduction",
                    "interspecies_crossover_prob",
                    0.0,
                )?,
            },
        })
    }

    pub fn input_keys(&self) -> Vec<i64> {
        (1..=self.genome.num_inputs).map(|i| -(i as i64)).collect()
    }

    pub fn output_keys(&self) -> Vec<i64> {
        (0..self.genome.num_outputs).map(|i| i as i64).collect()
    }

    pub fn is_better_fitness(&self, candidate: f64, previous: f64) -> bool {
        self.neat.fitness_criterion.is_better(candidate, previous)
    }

    pub fn meets_threshold(&self, value: f64) -> bool {
        self.neat
            .fitness_criterion
            .meets_threshold(value, self.neat.fitness_threshold)
    }

    pub fn worst_fitness(&self) -> f64 {
        self.neat.fitness_criterion.worst_value()
    }
}
