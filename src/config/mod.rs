use std::error::Error;
use std::fmt;
use std::fs;
use std::ops::{Add, AddAssign, Div};
use std::path::Path;
use std::str::FromStr;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeMemoryKind {
    None,
    NodeGru,
    Hebbian,
}

impl NodeMemoryKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "plain" => Some(Self::None),
            "node-gru" | "node_gru" | "gru" => Some(Self::NodeGru),
            "hebbian" | "fast-weight" | "fast_weight" => Some(Self::Hebbian),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NodeGru => "node-gru",
            Self::Hebbian => "hebbian",
        }
    }
}

impl fmt::Display for NodeMemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl ConfigChoice for NodeMemoryKind {
    fn name(self) -> &'static str {
        NodeMemoryKind::name(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeGruTopology {
    Standard,
    Minimal,
    Coupled,
    ResetOnly,
}

impl NodeGruTopology {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "standard" | "gru" => Some(Self::Standard),
            "minimal" | "update-only" | "update_only" => Some(Self::Minimal),
            "coupled" | "coupled-gate" | "coupled_gate" => Some(Self::Coupled),
            "reset-only" | "reset_only" => Some(Self::ResetOnly),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Minimal => "minimal",
            Self::Coupled => "coupled",
            Self::ResetOnly => "reset-only",
        }
    }
}

impl fmt::Display for NodeGruTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl ConfigChoice for NodeGruTopology {
    fn name(self) -> &'static str {
        NodeGruTopology::name(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHebbianRule {
    Plain,
    Oja,
    Bcm,
    OjaBcm,
}

impl NodeHebbianRule {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "plain" | "hebb" | "hebbian" => Some(Self::Plain),
            "oja" => Some(Self::Oja),
            "bcm" => Some(Self::Bcm),
            "oja-bcm" | "oja_bcm" | "bcm-oja" | "bcm_oja" => Some(Self::OjaBcm),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Oja => "oja",
            Self::Bcm => "bcm",
            Self::OjaBcm => "oja-bcm",
        }
    }
}

impl fmt::Display for NodeHebbianRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl ConfigChoice for NodeHebbianRule {
    fn name(self) -> &'static str {
        NodeHebbianRule::name(self)
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
                if sorted.len().is_multiple_of(2) {
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
    pub node_memory_kind: ChoiceAttributeConfig<NodeMemoryKind>,
    pub node_gru_topology: ChoiceAttributeConfig<NodeGruTopology>,
    pub node_gru_reset_bias: FloatAttributeConfig,
    pub node_gru_reset_response: FloatAttributeConfig,
    pub node_gru_reset_memory_weight: FloatAttributeConfig,
    pub node_gru_update_bias: FloatAttributeConfig,
    pub node_gru_update_response: FloatAttributeConfig,
    pub node_gru_update_memory_weight: FloatAttributeConfig,
    pub node_gru_candidate_memory_weight: FloatAttributeConfig,
    pub node_hebbian_rule: ChoiceAttributeConfig<NodeHebbianRule>,
    pub node_hebbian_decay: FloatAttributeConfig,
    pub node_hebbian_eta: FloatAttributeConfig,
    pub node_hebbian_key_weight: FloatAttributeConfig,
    pub node_hebbian_alpha: FloatAttributeConfig,
    pub node_hebbian_mod_bias: FloatAttributeConfig,
    pub node_hebbian_mod_response: FloatAttributeConfig,
    pub node_hebbian_theta_decay: FloatAttributeConfig,
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
    pub adaptive_mutation: AdaptiveMutationConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveMutationConfig {
    pub enabled: bool,
    pub start_after: usize,
    pub full_after: usize,
    pub max_multiplier: f64,
    pub caps: MutationRateCaps,
}

impl AdaptiveMutationConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            start_after: 0,
            full_after: 0,
            max_multiplier: 1.0,
            caps: MutationRateCaps::unlimited(),
        }
    }

    pub fn multiplier(&self, generations_without_improvement: usize) -> f64 {
        if !self.enabled || generations_without_improvement < self.start_after {
            return 1.0;
        }
        let max_multiplier = self.max_multiplier.max(1.0);
        if self.full_after <= self.start_after {
            return max_multiplier;
        }
        let span = self.full_after - self.start_after;
        let progressed = generations_without_improvement
            .saturating_sub(self.start_after)
            .min(span);
        1.0 + ((max_multiplier - 1.0) * progressed as f64 / span as f64)
    }

    pub fn adapted_genome_config(
        &self,
        base: &GenomeConfig,
        generations_without_improvement: usize,
    ) -> GenomeConfig {
        let multiplier = self.multiplier(generations_without_improvement);
        if (multiplier - 1.0).abs() <= f64::EPSILON {
            return base.clone();
        }

        let mut adapted = base.clone();
        adapted.conn_add_prob =
            scale_probability(base.conn_add_prob, multiplier, self.caps.conn_add_prob);
        adapted.conn_delete_prob = scale_probability(
            base.conn_delete_prob,
            multiplier,
            self.caps.conn_delete_prob,
        );
        adapted.node_add_prob =
            scale_probability(base.node_add_prob, multiplier, self.caps.node_add_prob);
        adapted.node_delete_prob = scale_probability(
            base.node_delete_prob,
            multiplier,
            self.caps.node_delete_prob,
        );
        adapted
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutationRateCaps {
    pub conn_add_prob: Probability,
    pub conn_delete_prob: Probability,
    pub node_add_prob: Probability,
    pub node_delete_prob: Probability,
}

impl MutationRateCaps {
    pub fn unlimited() -> Self {
        Self {
            conn_add_prob: Probability::one(),
            conn_delete_prob: Probability::one(),
            node_add_prob: Probability::one(),
            node_delete_prob: Probability::one(),
        }
    }
}

fn scale_probability(base: Probability, multiplier: f64, cap: Probability) -> Probability {
    Probability::new((base.value() * multiplier.max(1.0)).min(cap.value()))
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

    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let config = TomlConfigDocument::from_str(text)?.into_config()?;
        config.validate()?;
        Ok(config)
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

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_positive_usize("neat", "pop_size", self.neat.pop_size)?;
        validate_finite("neat", "fitness_threshold", self.neat.fitness_threshold)?;
        validate_positive_usize("genome", "num_inputs", self.genome.num_inputs)?;
        validate_positive_usize("genome", "num_outputs", self.genome.num_outputs)?;
        validate_non_negative_finite(
            "genome",
            "compatibility_disjoint_coefficient",
            self.genome.compatibility_disjoint_coefficient,
        )?;
        validate_non_negative_finite(
            "genome",
            "compatibility_enable_penalty",
            self.genome.compatibility_enable_penalty,
        )?;
        validate_non_negative_finite(
            "genome",
            "compatibility_weight_coefficient",
            self.genome.compatibility_weight_coefficient,
        )?;
        validate_choice_attribute("genome.activation", "activation", &self.genome.activation)?;
        validate_choice_attribute(
            "genome.aggregation",
            "aggregation",
            &self.genome.aggregation,
        )?;
        validate_float_attribute("genome.bias", "bias", &self.genome.bias)?;
        validate_float_attribute("genome.response", "response", &self.genome.response)?;
        validate_float_attribute(
            "genome.time_constant",
            "time_constant",
            &self.genome.time_constant,
        )?;
        validate_float_attribute("genome.iz_a", "iz_a", &self.genome.iz_a)?;
        validate_float_attribute("genome.iz_b", "iz_b", &self.genome.iz_b)?;
        validate_float_attribute("genome.iz_c", "iz_c", &self.genome.iz_c)?;
        validate_float_attribute("genome.iz_d", "iz_d", &self.genome.iz_d)?;
        validate_choice_attribute(
            "genome.node_memory_kind",
            "node_memory_kind",
            &self.genome.node_memory_kind,
        )?;
        validate_choice_attribute(
            "genome.node_gru_topology",
            "node_gru_topology",
            &self.genome.node_gru_topology,
        )?;
        validate_float_attribute(
            "genome.node_gru_reset_bias",
            "node_gru_reset_bias",
            &self.genome.node_gru_reset_bias,
        )?;
        validate_float_attribute(
            "genome.node_gru_reset_response",
            "node_gru_reset_response",
            &self.genome.node_gru_reset_response,
        )?;
        validate_float_attribute(
            "genome.node_gru_reset_memory_weight",
            "node_gru_reset_memory_weight",
            &self.genome.node_gru_reset_memory_weight,
        )?;
        validate_float_attribute(
            "genome.node_gru_update_bias",
            "node_gru_update_bias",
            &self.genome.node_gru_update_bias,
        )?;
        validate_float_attribute(
            "genome.node_gru_update_response",
            "node_gru_update_response",
            &self.genome.node_gru_update_response,
        )?;
        validate_float_attribute(
            "genome.node_gru_update_memory_weight",
            "node_gru_update_memory_weight",
            &self.genome.node_gru_update_memory_weight,
        )?;
        validate_float_attribute(
            "genome.node_gru_candidate_memory_weight",
            "node_gru_candidate_memory_weight",
            &self.genome.node_gru_candidate_memory_weight,
        )?;
        validate_choice_attribute(
            "genome.node_hebbian_rule",
            "node_hebbian_rule",
            &self.genome.node_hebbian_rule,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_decay",
            "node_hebbian_decay",
            &self.genome.node_hebbian_decay,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_eta",
            "node_hebbian_eta",
            &self.genome.node_hebbian_eta,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_key_weight",
            "node_hebbian_key_weight",
            &self.genome.node_hebbian_key_weight,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_alpha",
            "node_hebbian_alpha",
            &self.genome.node_hebbian_alpha,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_mod_bias",
            "node_hebbian_mod_bias",
            &self.genome.node_hebbian_mod_bias,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_mod_response",
            "node_hebbian_mod_response",
            &self.genome.node_hebbian_mod_response,
        )?;
        validate_float_attribute(
            "genome.node_hebbian_theta_decay",
            "node_hebbian_theta_decay",
            &self.genome.node_hebbian_theta_decay,
        )?;
        validate_float_attribute("genome.weight", "weight", &self.genome.weight)?;

        validate_non_negative_finite(
            "species_set",
            "compatibility_threshold",
            self.species_set.compatibility_threshold,
        )?;
        validate_non_negative_finite(
            "species_set",
            "threshold_adjust_rate",
            self.species_set.threshold_adjust_rate,
        )?;
        validate_non_negative_finite(
            "species_set",
            "threshold_min",
            self.species_set.threshold_min,
        )?;
        validate_non_negative_finite(
            "species_set",
            "threshold_max",
            self.species_set.threshold_max,
        )?;
        if self.species_set.threshold_min > self.species_set.threshold_max {
            return Err(invalid_config(
                "species_set",
                "threshold_min",
                self.species_set.threshold_min,
                "must be <= threshold_max",
            ));
        }
        if self.species_set.compatibility_threshold < self.species_set.threshold_min
            || self.species_set.compatibility_threshold > self.species_set.threshold_max
        {
            return Err(invalid_config(
                "species_set",
                "compatibility_threshold",
                self.species_set.compatibility_threshold,
                "must be inside [threshold_min, threshold_max]",
            ));
        }

        if self.reproduction.elitism > self.neat.pop_size {
            return Err(invalid_config(
                "reproduction",
                "elitism",
                self.reproduction.elitism,
                "must be <= neat.pop_size",
            ));
        }
        validate_positive_usize(
            "reproduction",
            "min_species_size",
            self.reproduction.min_species_size,
        )?;
        if self.reproduction.min_species_size > self.neat.pop_size {
            return Err(invalid_config(
                "reproduction",
                "min_species_size",
                self.reproduction.min_species_size,
                "must be <= neat.pop_size",
            ));
        }
        if self.reproduction.adaptive_mutation.enabled {
            validate_non_negative_finite(
                "reproduction.adaptive_mutation",
                "max_multiplier",
                self.reproduction.adaptive_mutation.max_multiplier,
            )?;
            if self.reproduction.adaptive_mutation.max_multiplier < 1.0 {
                return Err(invalid_config(
                    "reproduction.adaptive_mutation",
                    "max_multiplier",
                    self.reproduction.adaptive_mutation.max_multiplier,
                    "must be >= 1.0",
                ));
            }
        }
        Ok(())
    }
}

impl FromStr for Config {
    type Err = ConfigError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::from_toml_str(text)
    }
}

fn validate_choice_attribute<T: ConfigChoice>(
    section: &str,
    key: &str,
    config: &ChoiceAttributeConfig<T>,
) -> Result<(), ConfigError> {
    if config.options.is_empty() {
        return Err(invalid_config(section, "options", "", "must not be empty"));
    }
    if let ChoiceAttributeDefault::Value(value) = config.default {
        if !config.options.contains(&value) {
            return Err(invalid_config(
                section,
                key,
                value.name(),
                "default value must be included in options",
            ));
        }
    }
    Ok(())
}

fn validate_float_attribute(
    section: &str,
    key: &str,
    config: &FloatAttributeConfig,
) -> Result<(), ConfigError> {
    validate_finite(section, "init_mean", config.init_mean)?;
    validate_non_negative_finite(section, "init_stdev", config.init_stdev)?;
    validate_finite(section, "min_value", config.min_value)?;
    validate_finite(section, "max_value", config.max_value)?;
    validate_non_negative_finite(section, "mutate_power", config.mutate_power)?;
    if config.min_value > config.max_value {
        return Err(invalid_config(
            section,
            key,
            format!("{}..{}", config.min_value, config.max_value),
            "min_value must be <= max_value",
        ));
    }
    Ok(())
}

fn validate_positive_usize(section: &str, key: &str, value: usize) -> Result<(), ConfigError> {
    if value == 0 {
        Err(invalid_config(section, key, value, "must be > 0"))
    } else {
        Ok(())
    }
}

fn validate_finite(section: &str, key: &str, value: f64) -> Result<(), ConfigError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_config(section, key, value, "must be finite"))
    }
}

fn validate_non_negative_finite(section: &str, key: &str, value: f64) -> Result<(), ConfigError> {
    validate_finite(section, key, value)?;
    if value < 0.0 {
        Err(invalid_config(section, key, value, "must be >= 0"))
    } else {
        Ok(())
    }
}

fn invalid_config(section: &str, key: &str, value: impl ToString, message: &str) -> ConfigError {
    ConfigError::InvalidValue {
        section: section.to_string(),
        key: key.to_string(),
        value: value.to_string(),
        message: message.to_string(),
    }
}
