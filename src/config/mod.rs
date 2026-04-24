use std::error::Error;
use std::fmt;
use std::fs;
use std::ops::{Add, AddAssign, Div};
use std::path::Path;

use serde::de::{self, Visitor};
use serde::Deserialize;

use crate::activation::ActivationFunction;
use crate::aggregation::AggregationFunction;

mod toml_config;

use toml_config::TomlConfigDocument;

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
}

impl FitnessCriterion {
    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "mean" => Some(Self::Mean),
            _ => None,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialConnection {
    pub mode: InitialConnectionMode,
    pub fraction: Probability,
}

impl InitialConnection {
    pub fn parse(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        let mut parts = trimmed.split_whitespace();
        let Some(mode_raw) = parts.next() else {
            return Some(Self::unconnected());
        };
        let fraction = match parts.next() {
            Some(raw) => match raw.parse::<f64>() {
                Ok(value) => Probability::new(value),
                Err(_) => return None,
            },
            None => Probability::one(),
        };
        let mode = match mode_raw.trim().to_ascii_lowercase().as_str() {
            "unconnected" => InitialConnectionMode::Unconnected,
            "full_nodirect" => InitialConnectionMode::FullNoDirect,
            "full_direct" => InitialConnectionMode::FullDirect,
            "full" => InitialConnectionMode::Full,
            "partial_nodirect" => InitialConnectionMode::PartialNoDirect,
            "partial_direct" => InitialConnectionMode::PartialDirect,
            "partial" => InitialConnectionMode::Partial,
            _ => return None,
        };
        Some(Self { mode, fraction })
    }

    pub fn unconnected() -> Self {
        Self {
            mode: InitialConnectionMode::Unconnected,
            fraction: Probability::one(),
        }
    }

    pub fn full_direct() -> Self {
        Self {
            mode: InitialConnectionMode::FullDirect,
            fraction: Probability::one(),
        }
    }

    pub fn partial_direct(fraction: f64) -> Self {
        Self {
            mode: InitialConnectionMode::PartialDirect,
            fraction: Probability::new(fraction),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    pub fn new(value: f64) -> Self {
        if value.is_finite() {
            Self(value.clamp(0.0, 1.0))
        } else {
            Self::zero()
        }
    }

    pub const fn zero() -> Self {
        Self(0.0)
    }

    pub const fn one() -> Self {
        Self(1.0)
    }

    pub fn parse(value: &str) -> Option<Self> {
        value.trim().parse::<f64>().ok().map(Self::new)
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ProbabilityVisitor;

        impl Visitor<'_> for ProbabilityVisitor {
            type Value = Probability;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a finite probability number")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.is_finite() {
                    Ok(Probability::new(value))
                } else {
                    Err(E::custom("probability must be finite"))
                }
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Probability::new(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Probability::new(value as f64))
            }
        }

        deserializer.deserialize_any(ProbabilityVisitor)
    }
}

impl From<Probability> for f64 {
    fn from(value: Probability) -> Self {
        value.0
    }
}

impl Add for Probability {
    type Output = f64;

    fn add(self, rhs: Self) -> Self::Output {
        self.0 + rhs.0
    }
}

impl Add<Probability> for f64 {
    type Output = f64;

    fn add(self, rhs: Probability) -> Self::Output {
        self + rhs.0
    }
}

impl AddAssign<Probability> for f64 {
    fn add_assign(&mut self, rhs: Probability) {
        *self += rhs.0;
    }
}

impl Div<f64> for Probability {
    type Output = f64;

    fn div(self, rhs: f64) -> Self::Output {
        self.0 / rhs
    }
}

impl PartialEq<f64> for Probability {
    fn eq(&self, other: &f64) -> bool {
        self.0 == *other
    }
}

impl PartialOrd<f64> for Probability {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialEq<Probability> for f64 {
    fn eq(&self, other: &Probability) -> bool {
        *self == other.0
    }
}

impl PartialOrd<Probability> for f64 {
    fn partial_cmp(&self, other: &Probability) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.0)
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralMutationSurer {
    Default,
    Enabled,
    Disabled,
}

impl StructuralMutationSurer {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "yes" | "true" | "on" => Some(Self::Enabled),
            "0" | "no" | "false" | "off" => Some(Self::Disabled),
            "default" => Some(Self::Default),
            _ => None,
        }
    }

    pub fn is_enabled(&self, default: bool) -> bool {
        match self {
            Self::Default => default,
            Self::Enabled => true,
            Self::Disabled => false,
        }
    }
}

impl fmt::Display for StructuralMutationSurer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Enabled => write!(f, "true"),
            Self::Disabled => write!(f, "false"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompatibilityExcessCoefficient {
    Auto,
    Value(f64),
}

impl CompatibilityExcessCoefficient {
    pub fn parse(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            Some(Self::Auto)
        } else if let Ok(value) = trimmed.parse::<f64>() {
            Some(Self::Value(value))
        } else {
            None
        }
    }

    pub fn resolve(&self, default: f64) -> f64 {
        match self {
            Self::Auto => default,
            Self::Value(value) => *value,
        }
    }
}

impl fmt::Display for CompatibilityExcessCoefficient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Value(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetNumSpecies {
    Disabled,
    Count(usize),
}

impl TargetNumSpecies {
    pub fn parse(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("none") || trimmed.is_empty() {
            Some(Self::Disabled)
        } else if let Ok(value) = trimmed.parse::<usize>() {
            Some(Self::Count(value))
        } else {
            None
        }
    }

    pub fn target_count(&self) -> Option<usize> {
        match self {
            Self::Count(value) => Some(*value),
            Self::Disabled => None,
        }
    }
}

impl fmt::Display for TargetNumSpecies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "none"),
            Self::Count(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeciesFitnessFunction {
    Mean,
    Max,
    Min,
    Median,
}

impl SpeciesFitnessFunction {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" => Some(Self::Mean),
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "median" => Some(Self::Median),
            _ => None,
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
            Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitnessSharingMode {
    Normalized,
    Canonical,
}

impl FitnessSharingMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "normalized" => Some(Self::Normalized),
            "canonical" => Some(Self::Canonical),
            _ => None,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnMethod {
    Smoothed,
    Proportional,
}

impl SpawnMethod {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "smoothed" => Some(Self::Smoothed),
            "proportional" => Some(Self::Proportional),
            _ => None,
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
    pub conn_add_prob: Probability,
    pub conn_delete_prob: Probability,
    pub node_add_prob: Probability,
    pub node_delete_prob: Probability,
    pub single_structural_mutation: bool,
    pub structural_mutation_surer: StructuralMutationSurer,
    pub activation: ActivationConfig,
    pub aggregation: AggregationConfig,
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
    pub survival_threshold: Probability,
    pub min_species_size: usize,
    pub fitness_sharing: FitnessSharingMode,
    pub spawn_method: SpawnMethod,
    pub interspecies_crossover_prob: Probability,
}

pub trait ConfigChoice: Copy + PartialEq + fmt::Display {
    fn name(self) -> &'static str;
}

impl ConfigChoice for ActivationFunction {
    fn name(self) -> &'static str {
        ActivationFunction::name(self)
    }
}

impl ConfigChoice for AggregationFunction {
    fn name(self) -> &'static str {
        AggregationFunction::name(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceAttributeDefault<T> {
    Random,
    Value(T),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceAttributeConfig<T> {
    pub default: ChoiceAttributeDefault<T>,
    pub mutate_rate: Probability,
    pub options: Vec<T>,
}

impl<T: ConfigChoice> ChoiceAttributeConfig<T> {
    pub fn default_label(&self) -> &'static str {
        match self.default {
            ChoiceAttributeDefault::Random => "random",
            ChoiceAttributeDefault::Value(value) => value.name(),
        }
    }

    pub fn options_label(&self) -> String {
        self.options
            .iter()
            .map(|value| value.name())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub type ActivationConfig = ChoiceAttributeConfig<ActivationFunction>;
pub type AggregationConfig = ChoiceAttributeConfig<AggregationFunction>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatInitType {
    Gaussian,
    Uniform,
}

impl FloatInitType {
    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        if normalized.contains("gauss") || normalized.contains("normal") {
            Some(Self::Gaussian)
        } else if normalized.contains("uniform") {
            Some(Self::Uniform)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Uniform => "uniform",
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
    pub mutate_rate: Probability,
    pub replace_rate: Probability,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoolAttributeConfig {
    pub default: bool,
    pub mutate_rate: Probability,
    pub rate_to_true_add: Probability,
    pub rate_to_false_add: Probability,
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
        Self::from_toml_str(&text)
    }

    pub fn from_str(text: &str) -> Result<Self, ConfigError> {
        Self::from_toml_str(text)
    }

    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        TomlConfigDocument::from_str(text)?.into_config()
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
